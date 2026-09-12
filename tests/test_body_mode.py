"""Orientation-free body surface mode (``extra.train.surface_mode = "body"``).

``sdf_body = min(sdf_in, -sdf_out)`` (:func:`tsm.data.body_sdf`) drops the recto/verso naming:
one scalar SDF that is positive inside the papyrus, 0 on both faces *and* on a labelled contact
plane, negative outside.  Everything here is CPU and tiny: the analytic definition, the
augmentation rules, the loss, one train + evaluate round trip, the prediction channels and the
viewer table.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from synth import AXIS_YX, make_synthetic
from tsm.config import parse_config
from tsm.data import CLIP, SURFACE_MODES, Augment, CropDataset, SpatialParams, body_sdf, target_keys
from tsm.infer import (
    BODY_PRED_CHANNELS,
    N_BODY_HEAD_CH,
    activate_heads,
    decode_pred,
    extract_surface,
    load_student,
    n_head_ch,
    pred_channels,
    run_infer,
    to_unit,
)
from tsm.labels import encode_sdf_u8
from tsm.student import TSMNet, build_model, heads_for
from tsm.train import (
    EMA,
    body_loss,
    compute_losses,
    evaluate,
    save_checkpoint,
    soft_body,
    surface_aux_opts,
    surface_loss,
    synthetic_batch,
    run_train,
)
from tsm.view import layer_for

P = 16


# --------------------------------------------------------------------------- #
# the definition
# --------------------------------------------------------------------------- #
def two_sheets_faces(shape=(24, 8, 8), sheets=((4, 12), (12, 20))):
    """Two stacked sheets that *touch* at z = 12: the analytic two-face SDFs (outward = +z).

    Each voxel is assigned to the nearest sheet and measured against *that* sheet's faces --
    the sawtooth the real label builder writes.  ``sdf_in`` is the signed distance to the IN
    face (positive on its outward side), ``sdf_out`` the same for the OUT face; the OUT face of
    the lower sheet and the IN face of the upper one are the same plane -- a labelled contact."""
    z = np.arange(shape[0], dtype=np.float32)
    centres = np.array([0.5 * (a + b) for a, b in sheets], np.float32)
    k = np.argmin(np.abs(z[:, None] - centres[None, :]), axis=1)
    faces = np.asarray(sheets, np.float32)

    def field(col):
        return np.broadcast_to((z - faces[k, col]).reshape(-1, 1, 1), shape).astype(np.float32).copy()

    return field(0), field(1)


def test_body_sdf_signs_and_contact_split():
    sdf_in, sdf_out = two_sheets_faces()
    b = body_sdf(sdf_in, sdf_out)
    col = b[:, 0, 0]
    assert col[8] == pytest.approx(4.0) and col[16] == pytest.approx(4.0)   # inside both sheets
    assert col[12] == pytest.approx(0.0)                                    # the contact plane
    assert col[4] == pytest.approx(0.0) and col[20] == pytest.approx(0.0)   # the outer faces
    assert col[2] == pytest.approx(-2.0) and col[22] == pytest.approx(-2.0)  # outside: -distance
    # two sheets that touch stay two 26-connected components of body > 0
    lab, n = ndi.label(b > 0, structure=np.ones((3, 3, 3), np.uint8))
    assert n == 2
    # ... while the naive union of the two "inside" half spaces would be a single one
    lab1, n1 = ndi.label((sdf_in > 0) & (sdf_out < 0), structure=np.ones((3, 3, 3), np.uint8))
    assert n1 == 2 and np.array_equal(lab > 0, lab1 > 0)
    # clipping commutes with the minimum
    np.testing.assert_allclose(np.clip(b, -3, 3), body_sdf(np.clip(sdf_in, -3, 3), np.clip(sdf_out, -3, 3)).clip(-3, 3))


def test_surface_mode_registration_and_target_keys():
    assert "body" in SURFACE_MODES
    t = target_keys("body")
    assert t["surface_sdf"] == 1 and t["surface_valid"] == 1
    assert t == target_keys("medial")
    assert heads_for("body")["surface"] == 2 and heads_for("body") == heads_for("medial")
    with pytest.raises(ValueError):
        heads_for("bogus")


def test_dataset_body_targets_come_from_the_faces_labels(tmp_path):
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    kw = dict(patch=16, stride=16, length=4, augment=False)
    ds_b = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, surface_mode="body", **kw)
    ds_f = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, surface_mode="faces", **kw)
    assert ds_b.valid_channel == "faces_valid" and ds_b.target_keys["surface_sdf"] == 1
    o = ds_b.origins[0]
    a, b = ds_b.load(o), ds_f.load(o)
    sv = b["surface_valid"][0]
    want = np.where(sv > 0, body_sdf(b["surface_sdf"][0], b["surface_sdf"][1]), 0.0)
    np.testing.assert_allclose(a["surface_sdf"][0], want, atol=1e-5)
    np.testing.assert_array_equal(a["surface_valid"], b["surface_valid"])


# --------------------------------------------------------------------------- #
# augmentation: the body SDF is a plain scalar SDF
# --------------------------------------------------------------------------- #
def _batch(sdf: torch.Tensor) -> dict[str, torch.Tensor]:
    n = sdf.shape[-1]
    one = torch.ones(1, 1, n, n, n)
    return {
        "input": torch.zeros(1, 2, n, n, n),
        "surface_sdf": sdf.clone(), "surface_valid": one.clone(),
        "ink_prob": torch.zeros(1, 1, n, n, n), "ink_valid": one.clone(),
        "winding": torch.zeros(1, 6, n, n, n), "winding_conf": torch.zeros(1, 1, n, n, n),
        "winding_valid": torch.zeros(1, 1, n, n, n),
    }


def _face_fields(n: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar sheets along z: (sdf_in, sdf_out) as (1, 1, n, n, n) tensors, nothing saturated."""
    z = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1, 1).expand(1, 1, n, n, n).clone()
    sdf_in = (z - 8.0).clamp(-CLIP, CLIP)
    sdf_out = (z - 16.0).clamp(-CLIP, CLIP)
    return sdf_in, sdf_out


def _transport(sdf: torch.Tensor, p: SpatialParams) -> torch.Tensor:
    return Augment("none", seed=0).apply_spatial(_batch(sdf), [p])["surface_sdf"]


def test_body_target_commutes_with_the_cube_rotations_and_flips():
    """min() and resampling commute exactly for the transforms that permute voxel centres."""
    from tests.test_fiber import cube_rotations

    sdf_in, sdf_out = _face_fields()
    body = torch.minimum(sdf_in, -sdf_out)
    cases = [SpatialParams(rot90=m) for m in cube_rotations()]
    cases += [SpatialParams(flips=(bool(a), bool(b), bool(c)))
              for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    assert len(cases) == 32
    for p in cases:
        got = _transport(body, p)
        pair = _transport(torch.cat([sdf_in, sdf_out], 1), p)
        want = torch.minimum(pair[:, 0:1], -pair[:, 1:2])
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


def test_body_target_commutes_with_a_random_so3_rotation_to_sub_voxel_accuracy():
    """A continuous rotation resamples with trilinear weights, so min() and interpolation
    commute only up to sub-voxel effects -- concentrated in the cells the medial ridge crosses."""
    from tsm.data import uniform_rotation_matrix

    sdf_in, sdf_out = _face_fields()
    body = torch.minimum(sdf_in, -sdf_out)
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        p = SpatialParams(rotation=uniform_rotation_matrix(gen))
        got = _transport(body, p)
        pair = _transport(torch.cat([sdf_in, sdf_out], 1), p)
        want = torch.minimum(pair[:, 0:1], -pair[:, 1:2])
        core = (slice(None), slice(None)) + (slice(6, -6),) * 3
        err = (got[core] - want[core]).abs()
        assert float(err.mean()) < 0.15
        assert float(err.max()) < 1.0  # at most about half a voxel of ridge kink


def test_isotropic_scale_multiplies_the_body_sdf_and_saturation_ignores():
    aug = Augment("none")
    b = _batch(torch.full((1, 1, P, P, P), 4.0))
    out = aug.apply_spatial(b, [SpatialParams(scale=torch.full((3,), 1.5))])
    core = out["surface_sdf"][:, :, 4:-4, 4:-4, 4:-4]
    assert torch.allclose(core, torch.full_like(core, 6.0), atol=1e-3)
    # a *shrunk* saturated label is unknown -> valid 2 (the same rule as the faces / medial SDFs)
    sat = synthetic_batch(1, P, seed=4, surface_mode="body")
    assert sat["surface_sdf"].shape[1] == 1
    sat["surface_valid"].fill_(1.0)
    sat["ink_valid"].fill_(1.0)
    sat["winding_valid"].fill_(1.0)
    sat["surface_sdf"].fill_(CLIP)
    o = aug.apply_spatial(sat, [SpatialParams(scale=torch.full((3,), 0.8))])
    inner = (slice(None), slice(None)) + (slice(3, -3),) * 3
    assert torch.all(o["surface_valid"][inner] == 2.0)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
AUX = {"gap": 1.0, "gap_radius": 3, "cldice": 1.0}


def _sdf_valid(seed: int = 0):
    torch.manual_seed(seed)
    t = torch.randn(2, 1, 12, 12, 12) * 4.0
    valid = torch.ones(2, 1, 12, 12, 12)
    pred = torch.cat([torch.randn(2, 1, 12, 12, 12) * 4.0, torch.full((2, 1, 12, 12, 12), 3.0)], 1)
    return pred, t, valid


def test_body_loss_is_the_medial_loss_with_zero_aux():
    pred, t, valid = _sdf_valid()
    a = body_loss(pred, t, valid)
    b = surface_loss(pred, t, valid)
    assert set(a) == set(b) == {"sdf_l1", "valid_bce", "band_dice"}
    for k in a:
        assert float(a[k]) == float(b[k])
    z = surface_aux_opts({"shell": 0.0}, "body")
    for k, v in body_loss(pred, t, valid, aux=z).items():
        assert float(v) == float(b[k])
    with pytest.raises(ValueError):
        body_loss(pred, torch.cat([t, t], 1), valid)


def test_body_loss_grows_with_the_gap_and_cldice_terms_and_backprops():
    pred, t, valid = _sdf_valid(1)
    pred = pred.clone().requires_grad_(True)
    aux = surface_aux_opts(AUX, "body")
    d = body_loss(pred, t, valid, aux=aux)
    assert {"gap", "cldice", "shell_body"} & set(d) == {"gap", "cldice"}
    assert float(d["gap"].detach()) > 0 and float(d["cldice"].detach()) > 0
    (d["gap"] + d["cldice"]).backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0
    # per-face aux terms name the body as their "face"
    full = body_loss(pred, t, valid, aux=surface_aux_opts({**AUX, "shell": 0.5, "crest": 0.5, "far": 0.1}, "body"))
    assert {"shell_body", "crest_body", "far_body"} <= set(full)


def test_soft_body_single_channel_is_the_half_space_indicator():
    p = torch.linspace(-6, 6, 13).view(1, 1, 13, 1, 1)
    s = soft_body(p, None, 2.0)
    torch.testing.assert_close(s, torch.sigmoid(p / 2.0))
    assert float(s[0, 0, 0, 0, 0]) < 0.05 and float(s[0, 0, -1, 0, 0]) > 0.95
    # the two-face form is unchanged
    torch.testing.assert_close(soft_body(p, -p, 2.0), torch.sigmoid(p / 2.0) ** 2)


def test_compute_losses_branches_on_the_surface_mode():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="body")
    assert model.heads["surface"] == 2
    batch = synthetic_batch(1, 16, surface_mode="body")
    out = model(batch["input"])
    loss, parts = compute_losses(out, batch, surface_mode="body")
    assert "surface/sdf_l1" in parts and np.isfinite(parts["total"])
    loss.backward()
    l2, p2 = compute_losses(model(batch["input"]), batch, surface_mode="body",
                            surface_aux=surface_aux_opts(AUX, "body"))
    assert "surface/gap" in p2 and "surface/cldice" in p2 and p2["surface"] > parts["surface"]


def test_train_opts_validates_the_surface_mode_and_allows_body_aux():
    from tsm.train import train_opts

    def cfg(train):
        return parse_config({
            "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
            "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
            "out_dir": "/tmp/tsm-none", "extra": {"train": train},
        })

    o = train_opts(cfg({"surface_mode": "body", "surface_aux": AUX}))
    assert o["surface_mode"] == "body" and o["surface_aux"]["gap"] == 1.0 and o["surface_aux"]["cldice"] == 1.0
    with pytest.raises(ValueError, match="surface_mode"):
        train_opts(cfg({"surface_mode": "bogus"}))
    with pytest.raises(ValueError, match="'faces' or 'body'"):
        train_opts(cfg({"surface_mode": "medial", "surface_aux": AUX}))


# --------------------------------------------------------------------------- #
# metrics: body overlap / topology and fibre exclusivity
# --------------------------------------------------------------------------- #
def _axis_json(root: str) -> str:
    path = os.path.join(root, "axis.json")
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -100000, "y": AXIS_YX[0], "x": AXIS_YX[1]},
                                      {"z": 100000, "y": AXIS_YX[0], "x": AXIS_YX[1]}]}, fh)
    return path


def test_two_train_steps_and_evaluate_in_body_mode(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True, fiber=True, rv=True)
    extra = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1,
             "coarse_store": s["coarse"], "fine_store": s["fine"], "augment": False,
             "surface_mode": "body", "heads": {"fiber": True}, "fiber_mode": "direction",
             "input_radial": False, "input_axis": True, "axis_tangent": True,
             "axis_path": _axis_json(root), "holdout_origins": {"z_frac": 0.5}, "eval_max_crops": 2}
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root, "extra": {"train": extra},
    })
    r = run_train(cfg)
    assert r["step"] == 2
    m = r["holdout"]
    for k in ("surface/sdf_mae", "surface/zc_dice", "surface/valid_auprc",
              "surface/body_dice", "surface/body_iou", "surface/body_vol_ratio",
              "surface/body_n_components_ratio", "fiber/overlap_frac", "fiber/firing_frac"):
        assert k in m, k
        assert np.isfinite(m[k]), k
    assert 0.0 <= m["surface/body_dice"] <= 1.0 and 0.0 <= m["surface/body_iou"] <= 1.0
    assert 0.0 <= m["fiber/overlap_frac"] <= 1.0 and 0.0 <= m["fiber/firing_frac"] <= 1.0
    assert "surface/thickness_mae" not in m  # the body SDF names no sides


def test_evaluate_body_metrics_are_exact_for_a_perfect_and_a_blank_prediction(tmp_path):
    """A model that copies the label body scores Dice/IoU/volume ratio 1; one that predicts
    nothing scores 0 -- and the component ratio counts 26-connected blobs of >= 32 voxels."""
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                     length=4, augment=False, surface_mode="body")
    origins = ds.origins[:2]

    class Copy(torch.nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale
            self.i = 0
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            n = int(x.shape[0])
            take = origins[self.i:self.i + n]
            self.i += n
            b = torch.stack([torch.from_numpy(np.asarray(ds.load(o)["surface_sdf"][0])) for o in take])
            sdf = (self.scale * b)[:, None]
            return {"surface": torch.cat([sdf, torch.zeros_like(sdf)], 1),
                    "ink": torch.zeros_like(sdf), "winding": torch.zeros(sdf.shape[0], 8, *sdf.shape[2:])}

    good = evaluate(Copy(1.0), ds, origins, batch=1, surface_mode="body")
    assert good["surface/body_dice"] == pytest.approx(1.0)
    assert good["surface/body_iou"] == pytest.approx(1.0)
    assert good["surface/body_vol_ratio"] == pytest.approx(1.0)
    assert good["surface/body_n_components_ratio"] == pytest.approx(1.0)
    bad = evaluate(Copy(-1.0), ds, origins, batch=1, surface_mode="body")
    assert bad["surface/body_dice"] == pytest.approx(0.0) and bad["surface/body_iou"] == pytest.approx(0.0)


def test_fiber_exclusivity_metric_extremes(tmp_path):
    """``fiber/overlap_frac`` = n(vt & hz) / n(vt | hz): 1.0 when both channels always fire,
    0.0 when they are disjoint."""
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32,
                       faces=True, fiber=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                     length=4, augment=False, surface_mode="body", fiber=True, fiber_mode="class")
    origins = ds.origins[:2]

    class Const(torch.nn.Module):
        def __init__(self, vt: float, hz: float) -> None:
            super().__init__()
            self.v, self.h = vt, hz
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            B = x.shape[0]
            sh = (B, 1) + tuple(x.shape[2:])
            zeros = torch.zeros(sh)
            fib = torch.cat([torch.full(sh, self.v), torch.full(sh, self.h)], 1)
            return {"surface": torch.cat([zeros, zeros], 1), "ink": zeros,
                    "winding": torch.zeros(B, 8, *x.shape[2:]), "fiber": fib}

    both = evaluate(Const(9.0, 9.0), ds, origins, batch=1, surface_mode="body")
    assert both["fiber/overlap_frac"] == pytest.approx(1.0)
    assert both["fiber/firing_frac"] == pytest.approx(1.0)
    # one channel on, the other off: the two sets are disjoint (and one of them empty)
    one = evaluate(Const(9.0, -9.0), ds, origins, batch=1, surface_mode="body")
    assert one["fiber/overlap_frac"] == pytest.approx(0.0)
    assert one["fiber/firing_frac"] == pytest.approx(1.0)
    none = evaluate(Const(-9.0, -9.0), ds, origins, batch=1, surface_mode="body")
    assert none["fiber/overlap_frac"] == pytest.approx(0.0) and none["fiber/firing_frac"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_pred_channels_and_activations_body():
    assert pred_channels("body") == BODY_PRED_CHANNELS and n_head_ch("body") == N_BODY_HEAD_CH
    assert len(BODY_PRED_CHANNELS) == N_BODY_HEAD_CH + 1
    ch = pred_channels("body", True, "direction")
    assert ch == (BODY_PRED_CHANNELS[:11] + ["fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"]
                  + ["surface_body1", "fiber_vt", "fiber_hz"])
    assert n_head_ch("body", True, "direction") == 15 and ch[:15][-1] == "fiber_strength"
    assert pred_channels("body", True, "class") == (BODY_PRED_CHANNELS[:11] + ["fiber_vt", "fiber_hz"]
                                                    + ["surface_body1"])
    assert n_head_ch("body", True, "class") == 13
    B, n = 1, 4
    out = {"surface": torch.randn(B, 2, n, n, n), "ink": torch.randn(B, 1, n, n, n),
           "winding": torch.randn(B, 8, n, n, n)}
    phys = activate_heads(out, "body")
    assert phys.shape == (B, N_BODY_HEAD_CH, n, n, n)
    torch.testing.assert_close(phys[:, 0:1], out["surface"][:, 0:1])
    torch.testing.assert_close(phys[:, 1:2], torch.sigmoid(out["surface"][:, 1:2]))
    u = to_unit(phys, CLIP, "body")
    assert float(u.min()) >= 0 and float(u.max()) <= 1
    q = np.round(u.numpy() * 255).astype(np.uint8)
    np.testing.assert_array_equal(q[:, 0], encode_sdf_u8(phys[:, 0].numpy(), CLIP))


def _body_checkpoint(path: str, widths=(8, 16, 16, 16, 16)) -> None:
    net = TSMNet(widths=widths, heads=heads_for("body"))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ema.update(net)
    save_checkpoint(path, net, ema, opt, None, 5,
                    {"widths": list(widths), "surface_mode": "body",
                     "train": {"widths": list(widths), "surface_mode": "body"}})


def test_run_infer_writes_the_derived_body_surface(tmp_path):
    import zarr

    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ckpt = os.path.join(root, "train", "latest.pt")
    _body_checkpoint(ckpt)
    _, info = load_student(ckpt)
    assert info["surface_mode"] == "body"
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"infer": {"patch": 32, "out_tile": 64, "chunk": 32, "stats_box": [32, 64, 64],
                            "prefetch": False, "device": "cpu"}},
    })
    summary = run_infer(cfg)
    assert summary["surface_mode"] == "body"
    pred = os.path.join(root, "student", "pred.zarr")
    arr = zarr.open_array(store=pred, mode="r")
    assert list(arr.attrs["channels"]) == BODY_PRED_CHANNELS
    a = np.asarray(arr[:])
    dec = decode_pred(a, CLIP, BODY_PRED_CHANNELS)
    assert "sdf_body" in dec and "surface_body1" in dec
    np.testing.assert_array_equal(
        dec["surface_body1"],
        extract_surface(a[BODY_PRED_CHANNELS.index("sdf_body")], a[BODY_PRED_CHANNELS.index("valid")], CLIP))
    sj = json.load(open(os.path.join(root, "student", "pred.summary.json")))
    assert sj["surface_mode"] == "body" and "surface_body1" in sj["surface1"]
    assert "thickness" not in sj["surface1"]  # no thickness pass in body mode
    assert os.path.exists(os.path.join(root, "student", "preview_sdf_body.png"))


# --------------------------------------------------------------------------- #
# viewer table / configs
# --------------------------------------------------------------------------- #
def test_view_layers_for_the_body_channels():
    lay = layer_for("sdf_body", "student", clip=CLIP)
    assert lay is not None and lay.kind == "sdf" and lay.clip == CLIP
    lay1 = layer_for("surface_body1", "student")
    assert lay1 is not None and lay1.kind == "prob"


@pytest.mark.parametrize("name", ["body30k", "body30k_bgf", "body30k_bgf_aux"])
def test_shipped_body_configs_load(name):
    from tsm.config import load_config
    from tsm.train import train_opts

    cfg = load_config(os.path.join("configs", f"{name}.json"))
    o = train_opts(cfg)
    assert o["surface_mode"] == "body" and o["steps"] == 30000 and o["ckpt_every"] == 2000
    assert cfg.out_dir == f"/home/forrest/tsm-output/{name}"
    bgf = name.endswith("bgf") or name.endswith("aux")
    assert o["input_axis"] is bgf and o["axis_tangent"] is bgf
    assert o["fiber_mode"] == ("direction" if bgf else "class")
    assert (o["surface_aux"]["gap"], o["surface_aux"]["cldice"]) == ((1.0, 1.0) if name.endswith("aux") else (0.0, 0.0))
    ev = load_config(os.path.join("configs", f"paris4_eval_{name}.json"))
    assert ev.out_dir == f"/home/forrest/tsm-output/eval_{name}"
    assert ev.extra["infer"]["checkpoint"] == f"/home/forrest/tsm-output/{name}/train/latest.pt"
    assert train_opts(ev)["surface_mode"] == "body"


def test_ablate_variant_config_accepts_a_surface_mode_override():
    from tsm.ablate import ablate_opts, variant_config
    from tsm.config import load_config
    from tsm.train import train_opts

    cfg = load_config(os.path.join("configs", "body30k.json"))
    opts = ablate_opts(cfg)
    vc = variant_config(cfg, "bgf", {"surface_mode": "body", "fiber_mode": "direction", "input_axis": True},
                        {**opts, "steps": 300}, str("/tmp/tsm-ablate-none"))
    o = train_opts(vc)
    assert o["surface_mode"] == "body" and o["fiber_mode"] == "direction" and o["input_axis"] is True
    assert o["steps"] == 300
