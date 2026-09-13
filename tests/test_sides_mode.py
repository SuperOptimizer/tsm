"""Orientation-free "sides" surface mode (``extra.train.surface_mode = "sides"``).

The signed body SDF of ``surface_mode="body"`` flips sign twice per sheet and its interior
magnitude is the (noisy) half-thickness, so a single L1 regression hedges near zero.  "sides"
splits the two questions apart: ``d_face = min(|sdf_in|, |sdf_out|)`` (:func:`tsm.data.face_dist`,
an UNSIGNED distance, regressed) and ``body = (sdf_in > 0) & (sdf_out < 0)``
(:func:`tsm.data.body_mask_np`, a 0/1 mask, classified).  Both are derived on the fly from the
existing two-face labels -- no relabelling.

Everything here is CPU and tiny: the analytic targets, the augmentation rules, the loss, one
train + evaluate round trip, the prediction channels, the viewer table and ``dev/eval_region.py``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import AXIS_YX, make_synthetic  # noqa: E402
from tsm.config import parse_config  # noqa: E402
from tsm.data import (  # noqa: E402
    CLIP,
    SURFACE_MODES,
    Augment,
    CropDataset,
    SpatialParams,
    body_mask_np,
    body_sdf,
    encode_prob,
    encode_sdf,
    encode_signed,
    face_dist,
    target_keys,
)
from tsm.infer import (  # noqa: E402
    N_SIDES_HEAD_CH,
    SIDES_PRED_CHANNELS,
    activate_heads,
    decode_pred,
    load_student,
    n_head_ch,
    pred_channels,
    run_infer,
    to_unit,
)
from tsm.labels import encode_sdf_u8  # noqa: E402
from tsm.student import TSMNet, build_model, heads_for  # noqa: E402
from tsm.train import (  # noqa: E402
    EMA,
    compute_losses,
    evaluate,
    run_train,
    save_checkpoint,
    sides_loss,
    surface_aux_opts,
    synthetic_batch,
)
from tsm.view import layer_for  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402

eval_region = pytest.importorskip("eval_region")

P = 16


# --------------------------------------------------------------------------- #
# the definition
# --------------------------------------------------------------------------- #
def two_sheets_faces(shape=(24, 8, 8), sheets=((4, 12), (12, 20))):
    """Two stacked sheets that *touch* at z = 12: the analytic two-face SDFs (outward = +z).

    Same fixture as ``tests/test_body_mode.py``: each voxel is measured against the faces of the
    nearest sheet, and the OUT face of the lower sheet is the IN face of the upper one -- a
    labelled contact plane."""
    z = np.arange(shape[0], dtype=np.float32)
    centres = np.array([0.5 * (a + b) for a, b in sheets], np.float32)
    k = np.argmin(np.abs(z[:, None] - centres[None, :]), axis=1)
    faces = np.asarray(sheets, np.float32)

    def field(col):
        return np.broadcast_to((z - faces[k, col]).reshape(-1, 1, 1), shape).astype(np.float32).copy()

    return field(0), field(1)


def test_face_dist_is_zero_on_both_faces_and_on_the_contact():
    sdf_in, sdf_out = two_sheets_faces()
    d = face_dist(sdf_in, sdf_out)
    col = d[:, 0, 0]
    assert (col >= 0).all()                                      # never negative
    assert col[4] == pytest.approx(0.0) and col[20] == pytest.approx(0.0)   # the outer faces
    assert col[12] == pytest.approx(0.0)                          # the contact plane
    assert col[8] == pytest.approx(4.0) and col[16] == pytest.approx(4.0)   # sheet centres
    assert col[2] == pytest.approx(2.0) and col[22] == pytest.approx(2.0)   # outside: +distance
    # the magnitude half of the body SDF, exactly
    np.testing.assert_allclose(d, np.abs(body_sdf(sdf_in, sdf_out)), atol=1e-6)
    # clipping commutes with min/abs on a symmetric clip
    np.testing.assert_allclose(np.clip(d, 0, 3),
                               face_dist(np.clip(sdf_in, -3, 3), np.clip(sdf_out, -3, 3)).clip(0, 3))


def test_body_mask_np_is_two_26_components_split_by_the_contact():
    sdf_in, sdf_out = two_sheets_faces()
    b = body_mask_np(sdf_in, sdf_out)
    np.testing.assert_array_equal(b, body_sdf(sdf_in, sdf_out) > 0)
    lab, n = ndi.label(b, structure=np.ones((3, 3, 3), np.uint8))
    assert n == 2
    col = b[:, 0, 0]
    assert col[8] and col[16] and not col[12] and not col[2] and not col[22]


def test_surface_mode_registration_and_head_widths():
    assert "sides" in SURFACE_MODES
    t = target_keys("sides")
    assert t["surface_sdf"] == 1 and t["surface_body"] == 1 and t["surface_valid"] == 1
    assert heads_for("sides") == heads_for("faces") and heads_for("sides")["surface"] == 3
    with pytest.raises(ValueError):
        heads_for("bogus")


def test_dataset_sides_targets_come_from_the_faces_labels(tmp_path):
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    kw = dict(patch=16, stride=16, length=4, augment=False)
    ds_s = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, surface_mode="sides", **kw)
    ds_f = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, surface_mode="faces", **kw)
    assert ds_s.valid_channel == "faces_valid" and ds_s.target_keys["surface_sdf"] == 1
    o = ds_s.origins[0]
    a, b = ds_s.load(o), ds_f.load(o)
    sv = b["surface_valid"][0]
    np.testing.assert_allclose(a["surface_sdf"][0],
                               np.where(sv > 0, face_dist(b["surface_sdf"][0], b["surface_sdf"][1]), 0.0),
                               atol=1e-5)
    np.testing.assert_allclose(a["surface_body"][0],
                               np.where(sv > 0, body_mask_np(b["surface_sdf"][0], b["surface_sdf"][1]), 0.0),
                               atol=1e-6)
    np.testing.assert_array_equal(a["surface_valid"], b["surface_valid"])
    # the targets survive the numpy -> tensor conversion
    assert "surface_body" in ds_s.to_tensors(a)


# --------------------------------------------------------------------------- #
# augmentation: a scalar distance + a probability-like mask, no sign, no swap
# --------------------------------------------------------------------------- #
def _batch(d: torch.Tensor, body: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    n = d.shape[-1]
    one = torch.ones(1, 1, n, n, n)
    b = {
        "input": torch.zeros(1, 2, n, n, n),
        "surface_sdf": d.clone(), "surface_valid": one.clone(),
        "ink_prob": torch.zeros(1, 1, n, n, n), "ink_valid": one.clone(),
        "winding": torch.zeros(1, 6, n, n, n), "winding_conf": torch.zeros(1, 1, n, n, n),
        "winding_valid": torch.zeros(1, 1, n, n, n),
    }
    if body is not None:
        b["surface_body"] = body.clone()
    return b


def _face_fields(n: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1, 1).expand(1, 1, n, n, n).clone()
    return (z - 8.0).clamp(-CLIP, CLIP), (z - 16.0).clamp(-CLIP, CLIP)


def _transport(d: torch.Tensor, body: torch.Tensor, p: SpatialParams):
    o = Augment("none", seed=0).apply_spatial(_batch(d, body), [p])
    return o["surface_sdf"], o["surface_body"]


def _all_32() -> list[SpatialParams]:
    from tests.test_fiber import cube_rotations

    cases = [SpatialParams(rot90=m) for m in cube_rotations()]
    cases += [SpatialParams(flips=(bool(a), bool(b), bool(c)))
              for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    assert len(cases) == 32
    return cases


def test_sides_targets_commute_with_the_cube_rotations_and_flips():
    """The 24 rotations and the 8 flips permute voxel centres, so deriving the targets and then
    transporting them equals transporting the faces and deriving afterwards -- exactly."""
    sdf_in, sdf_out = _face_fields()
    d = torch.minimum(sdf_in.abs(), sdf_out.abs())
    body = ((sdf_in > 0) & (sdf_out < 0)).float()
    for p in _all_32():
        got_d, got_b = _transport(d, body, p)
        pair = Augment("none", seed=0).apply_spatial(
            _batch(torch.cat([sdf_in, sdf_out], 1)), [p])["surface_sdf"]
        want_d = torch.minimum(pair[:, 0:1].abs(), pair[:, 1:2].abs())
        want_b = ((pair[:, 0:1] > 0) & (pair[:, 1:2] < 0)).float()
        torch.testing.assert_close(got_d, want_d, atol=1e-5, rtol=0)
        torch.testing.assert_close(got_b, want_b, atol=1e-5, rtol=0)


def test_sides_targets_commute_with_a_random_so3_rotation_to_sub_voxel_accuracy():
    from tsm.data import uniform_rotation_matrix

    sdf_in, sdf_out = _face_fields()
    d = torch.minimum(sdf_in.abs(), sdf_out.abs())
    body = ((sdf_in > 0) & (sdf_out < 0)).float()
    gen = torch.Generator().manual_seed(3)
    core = (slice(None), slice(None)) + (slice(6, -6),) * 3
    for _ in range(3):
        p = SpatialParams(rotation=uniform_rotation_matrix(gen))
        got_d, got_b = _transport(d, body, p)
        pair = Augment("none", seed=0).apply_spatial(
            _batch(torch.cat([sdf_in, sdf_out], 1)), [p])["surface_sdf"]
        want_d = torch.minimum(pair[:, 0:1].abs(), pair[:, 1:2].abs())
        want_b = ((pair[:, 0:1] > 0) & (pair[:, 1:2] < 0)).float()
        assert float((got_d[core] - want_d[core]).abs().mean()) < 0.15
        assert float((got_d[core] - want_d[core]).abs().max()) < 1.0
        # the resampled mask has a soft edge where the hard one has a step, so the two 0.5
        # level sets can differ -- but only SUB-VOXEL: every disagreeing voxel is within half a
        # voxel of a face (measured: max d_face over the disagreement < 0.5)
        diff = (got_b[core] > 0.5).float() != want_b[core]
        assert bool(diff.any())
        assert float(want_d[core][diff].max()) < 0.6
        # ... and the mask stays a probability, never a signed field
        assert float(got_b.min()) >= 0.0 and float(got_b.max()) <= 1.0


def test_isotropic_scale_multiplies_d_face_and_saturation_ignores():
    aug = Augment("none")
    b = _batch(torch.full((1, 1, P, P, P), 4.0), torch.ones(1, 1, P, P, P))
    out = aug.apply_spatial(b, [SpatialParams(scale=torch.full((3,), 1.5))])
    core = out["surface_sdf"][:, :, 4:-4, 4:-4, 4:-4]
    assert torch.allclose(core, torch.full_like(core, 6.0), atol=1e-3)
    bcore = out["surface_body"][:, :, 4:-4, 4:-4, 4:-4]
    assert torch.allclose(bcore, torch.ones_like(bcore), atol=1e-3)  # a mask does not scale
    # a *shrunk* saturated label is unknown -> valid 2 (the same rule as the faces / medial SDFs)
    sat = synthetic_batch(1, P, seed=4, surface_mode="sides")
    assert sat["surface_sdf"].shape[1] == 1 and float(sat["surface_sdf"].min()) >= 0.0
    for k in ("surface_valid", "ink_valid", "winding_valid"):
        sat[k].fill_(1.0)
    sat["surface_sdf"].fill_(CLIP)
    o = aug.apply_spatial(sat, [SpatialParams(scale=torch.full((3,), 0.8))])
    inner = (slice(None), slice(None)) + (slice(3, -3),) * 3
    assert torch.all(o["surface_valid"][inner] == 2.0)


def test_v1_cpu_augmentation_carries_the_body_mask():
    from tsm.data import augment

    rng = np.random.default_rng(0)
    sample = {"ct": np.zeros((1, 8, 8, 8), np.float32),
              "surface_sdf": np.random.default_rng(1).random((1, 8, 8, 8)).astype(np.float32),
              "surface_valid": np.ones((1, 8, 8, 8), np.float32),
              "surface_body": (np.random.default_rng(2).random((1, 8, 8, 8)) > 0.5).astype(np.float32)}
    out = augment(sample, rng, intensity=False)
    assert out["surface_body"].shape == (1, 8, 8, 8)
    assert set(np.unique(out["surface_body"])) <= {0.0, 1.0}     # flips/rot90 only permute


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
AUX = {"gap": 1.0, "gap_radius": 3, "cldice": 1.0}


def _case(seed: int = 0):
    torch.manual_seed(seed)
    d = torch.rand(2, 1, 12, 12, 12) * 8.0
    body = (torch.rand(2, 1, 12, 12, 12) > 0.5).float()
    valid = torch.ones(2, 1, 12, 12, 12)
    pred = torch.cat([torch.rand(2, 1, 12, 12, 12) * 8.0,
                      torch.randn(2, 1, 12, 12, 12),
                      torch.full((2, 1, 12, 12, 12), 3.0)], 1)
    return pred, d, body, valid


def test_sides_loss_terms_are_finite_and_the_head_shape_is_checked():
    pred, d, body, valid = _case()
    out = sides_loss(pred, d, valid, body)
    assert set(out) == {"sdf_l1", "band_dice", "body_bce", "body_dice", "valid_bce"}
    for k, v in out.items():
        assert np.isfinite(float(v)), k
    with pytest.raises(ValueError):
        sides_loss(pred[:, :2], d, valid, body)
    with pytest.raises(ValueError):
        sides_loss(pred, torch.cat([d, d], 1), valid, body)


def test_a_perfect_prediction_has_almost_no_body_loss_and_the_body_terms_backprop():
    _, d, body, valid = _case(1)
    big = 20.0
    perfect = torch.cat([d, (body * 2 - 1) * big, torch.full_like(d, big)], 1)
    good = sides_loss(perfect, d, valid, body)
    assert float(good["sdf_l1"]) == pytest.approx(0.0, abs=1e-5)
    assert float(good["body_bce"]) < 1e-6 and float(good["body_dice"]) < 1e-4
    # flipping the body logit is maximally wrong
    flipped = torch.cat([d, (1 - body) * 2 * big - big, torch.full_like(d, big)], 1)
    bad = sides_loss(flipped, d, valid, body)
    assert float(bad["body_bce"]) > 1.0 and float(bad["body_dice"]) > 0.9
    # ... and the body terms drive the body logit
    p = torch.cat([d, torch.zeros_like(d), torch.full_like(d, big)], 1).requires_grad_(True)
    t = sides_loss(p, d, valid, body)
    (t["body_bce"] + t["body_dice"]).backward()
    assert float(p.grad[:, 1:2].abs().sum()) > 0
    assert float(p.grad[:, 0:1].abs().sum()) == 0.0   # the body terms never touch d_face


def test_sides_aux_gap_and_cldice_run_and_backprop():
    pred, d, body, valid = _case(2)
    pred = pred.clone().requires_grad_(True)
    aux = surface_aux_opts(AUX, "sides")
    out = sides_loss(pred, d, valid, body, aux=aux)
    assert {"gap", "cldice"} <= set(out)
    assert float(out["gap"].detach()) > 0 and float(out["cldice"].detach()) > 0
    (out["gap"] + out["cldice"]).backward()
    assert pred.grad is not None and float(pred.grad[:, 1:2].abs().sum()) > 0
    full = sides_loss(pred, d, valid, body,
                      aux=surface_aux_opts({**AUX, "shell": 0.5, "crest": 0.5, "far": 0.1}, "sides"))
    assert {"shell_face", "crest_face", "far_face"} <= set(full)


def test_compute_losses_branches_on_the_sides_mode():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="sides")
    assert model.heads["surface"] == 3
    batch = synthetic_batch(1, 16, surface_mode="sides")
    loss, parts = compute_losses(model(batch["input"]), batch, surface_mode="sides")
    for k in ("surface/sdf_l1", "surface/body_bce", "surface/body_dice", "surface/valid_bce"):
        assert k in parts and np.isfinite(parts[k]), k
    loss.backward()
    _, p2 = compute_losses(model(batch["input"]), batch, surface_mode="sides",
                           surface_aux=surface_aux_opts(AUX, "sides"))
    assert "surface/gap" in p2 and "surface/cldice" in p2 and p2["surface"] > parts["surface"]
    # the body target is mandatory in this mode
    del batch["surface_body"]
    with pytest.raises(ValueError, match="surface_body"):
        compute_losses(model(batch["input"]), batch, surface_mode="sides")


def test_train_opts_validates_the_sides_mode_and_allows_the_aux_terms():
    from tsm.train import train_opts

    def cfg(train):
        return parse_config({
            "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
            "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
            "out_dir": "/tmp/tsm-none", "extra": {"train": train},
        })

    o = train_opts(cfg({"surface_mode": "sides", "surface_aux": AUX}))
    assert o["surface_mode"] == "sides" and o["surface_aux"]["gap"] == 1.0
    with pytest.raises(ValueError, match="surface_mode"):
        train_opts(cfg({"surface_mode": "bogus"}))
    with pytest.raises(ValueError, match="'sides'"):
        train_opts(cfg({"surface_mode": "medial", "surface_aux": AUX}))


# --------------------------------------------------------------------------- #
# train + evaluate
# --------------------------------------------------------------------------- #
def _axis_json(root: str) -> str:
    path = os.path.join(root, "axis.json")
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -100000, "y": AXIS_YX[0], "x": AXIS_YX[1]},
                                      {"z": 100000, "y": AXIS_YX[0], "x": AXIS_YX[1]}]}, fh)
    return path


def test_two_train_steps_and_evaluate_in_sides_mode(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True, fiber=True, rv=True)
    extra = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1,
             "coarse_store": s["coarse"], "fine_store": s["fine"], "augment": False,
             "surface_mode": "sides", "heads": {"fiber": True}, "fiber_mode": "direction",
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
              "surface/body_n_components_ratio", "fiber/overlap_frac"):
        assert k in m and np.isfinite(m[k]), k
    assert 0.0 <= m["surface/body_dice"] <= 1.0 and 0.0 <= m["surface/zc_dice"] <= 1.0
    assert "surface/thickness_mae" not in m


def test_evaluate_sides_metrics_are_exact_for_a_perfect_and_an_inverted_prediction(tmp_path):
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                     length=4, augment=False, surface_mode="sides")
    origins = ds.origins[:2]

    class Copy(torch.nn.Module):
        def __init__(self, sign: float) -> None:
            super().__init__()
            self.sign = sign
            self.i = 0
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            n = int(x.shape[0])
            take = origins[self.i:self.i + n]
            self.i += n
            items = [ds.load(o) for o in take]
            d = torch.stack([torch.from_numpy(np.asarray(t["surface_sdf"][0])) for t in items])[:, None]
            b = torch.stack([torch.from_numpy(np.asarray(t["surface_body"][0])) for t in items])[:, None]
            logit = self.sign * (b * 2.0 - 1.0) * 20.0
            return {"surface": torch.cat([d, logit, torch.zeros_like(d)], 1),
                    "ink": torch.zeros_like(d), "winding": torch.zeros(d.shape[0], 8, *d.shape[2:])}

    good = evaluate(Copy(1.0), ds, origins, batch=1, surface_mode="sides")
    assert good["surface/sdf_mae"] == pytest.approx(0.0, abs=1e-5)
    assert good["surface/zc_dice"] == pytest.approx(1.0)
    assert good["surface/body_dice"] == pytest.approx(1.0)
    assert good["surface/body_iou"] == pytest.approx(1.0)
    assert good["surface/body_vol_ratio"] == pytest.approx(1.0)
    assert good["surface/body_n_components_ratio"] == pytest.approx(1.0)
    bad = evaluate(Copy(-1.0), ds, origins, batch=1, surface_mode="sides")
    assert bad["surface/body_dice"] == pytest.approx(0.0)
    assert bad["surface/zc_dice"] == pytest.approx(1.0)   # the distance half is still perfect


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_pred_channels_and_activations_sides():
    assert pred_channels("sides") == SIDES_PRED_CHANNELS and n_head_ch("sides") == N_SIDES_HEAD_CH
    assert len(SIDES_PRED_CHANNELS) == N_SIDES_HEAD_CH + 1
    assert SIDES_PRED_CHANNELS[:3] == ["d_face", "body", "valid"]
    ch = pred_channels("sides", True, "direction")
    assert ch == (SIDES_PRED_CHANNELS[:12] + ["fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"]
                  + ["surface_side1", "fiber_vt", "fiber_hz"])
    assert n_head_ch("sides", True, "direction") == 16 and ch[:16][-1] == "fiber_strength"
    assert pred_channels("sides", True, "class") == (SIDES_PRED_CHANNELS[:12] + ["fiber_vt", "fiber_hz"]
                                                     + ["surface_side1"])
    assert n_head_ch("sides", True, "class") == 14

    B, n = 1, 4
    torch.manual_seed(0)
    out = {"surface": torch.randn(B, 3, n, n, n), "ink": torch.randn(B, 1, n, n, n),
           "winding": torch.randn(B, 8, n, n, n)}
    phys = activate_heads(out, "sides")
    assert phys.shape == (B, N_SIDES_HEAD_CH, n, n, n)
    torch.testing.assert_close(phys[:, 0:1], torch.relu(out["surface"][:, 0:1]))
    assert float(phys[:, 0:1].min()) >= 0.0                      # d_face >= 0
    torch.testing.assert_close(phys[:, 1:3], torch.sigmoid(out["surface"][:, 1:3]))
    u = to_unit(phys, CLIP, "sides")
    assert u.shape == phys.shape and float(u.min()) >= 0 and float(u.max()) <= 1
    q = np.round(u.numpy() * 255).astype(np.uint8)
    # the SAME sdf byte encoding, so every d_face byte is >= 128 and an sdf reader decodes it
    np.testing.assert_array_equal(q[:, 0], encode_sdf_u8(phys[:, 0].numpy(), CLIP))
    assert int(q[:, 0].min()) >= 128
    # the round trip: bytes -> fields
    dec = decode_pred(np.concatenate([q[0], np.zeros((1,) + q.shape[2:], np.uint8)]), CLIP,
                      SIDES_PRED_CHANNELS)
    assert "d_face" in dec and "body" in dec and "surface_side1" in dec
    assert dec["sdf"] is dec["d_face"] or np.array_equal(dec["sdf"], dec["d_face"])  # alias
    np.testing.assert_allclose(dec["d_face"], phys[0, 0].numpy(), atol=CLIP / 127.0)
    assert (dec["d_face"] >= 0).all()


def test_decode_pred_aliases_sdf_body_to_sdf():
    from tsm.infer import BODY_PRED_CHANNELS

    sh = (4, 4, 4)
    u8 = np.zeros((len(BODY_PRED_CHANNELS),) + sh, np.uint8)
    u8[BODY_PRED_CHANNELS.index("sdf_body")] = 200
    dec = decode_pred(u8, CLIP, BODY_PRED_CHANNELS)
    np.testing.assert_array_equal(dec["sdf"], dec["sdf_body"])


def _sides_checkpoint(path: str, widths=(8, 16, 16, 16, 16)) -> None:
    net = TSMNet(widths=widths, heads=heads_for("sides"))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ema.update(net)
    save_checkpoint(path, net, ema, opt, None, 5,
                    {"widths": list(widths), "surface_mode": "sides",
                     "train": {"widths": list(widths), "surface_mode": "sides"}})


def test_run_infer_writes_surface_side1_and_body(tmp_path):
    import zarr

    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ckpt = os.path.join(root, "train", "latest.pt")
    _sides_checkpoint(ckpt)
    _, info = load_student(ckpt)
    assert info["surface_mode"] == "sides"
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"infer": {"patch": 32, "out_tile": 64, "chunk": 32, "stats_box": [32, 64, 64],
                            "prefetch": False, "device": "cpu"}},
    })
    summary = run_infer(cfg)
    assert summary["surface_mode"] == "sides"
    arr = zarr.open_array(store=os.path.join(root, "student", "pred.zarr"), mode="r")
    assert list(arr.attrs["channels"]) == SIDES_PRED_CHANNELS
    a = np.asarray(arr[:])
    i_d, i_s = (SIDES_PRED_CHANNELS.index(k) for k in ("d_face", "surface_side1"))
    assert int(a[i_d][a[i_d] != 0].min()) >= 128         # the distance never encodes negative
    dec = decode_pred(a, CLIP, SIDES_PRED_CHANNELS)
    # the derived side channel is `extract_sides` (body skin U in-body distance valleys), not a
    # threshold on the compressed distance
    from tsm.infer import extract_sides

    want = extract_sides(dec["d_face"], dec["body"], dec["valid"]) & (a[i_d] != 0)
    np.testing.assert_array_equal(a[i_s] > 127, want)
    assert not (a[i_s] > 127)[dec["body"] < 0.5].any()   # sides lie on the papyrus side
    assert dict(arr.attrs["surface_side1_params"])["body_thr"] == 0.5
    assert "body" in dec and float(dec["body"].min()) >= 0.0 and float(dec["body"].max()) <= 1.0
    sj = json.load(open(os.path.join(root, "student", "pred.summary.json")))
    assert sj["surface_mode"] == "sides" and "surface_side1" in sj["surface1"]
    assert "thickness" not in sj["surface1"]
    assert os.path.exists(os.path.join(root, "student", "preview_d_face.png"))


def test_pred_dt_field_puts_the_sign_back_for_the_lasagna_export():
    from tsm.infer import pred_dt_field

    dec = {"d_face": np.array([0.0, 2.0, 5.0], np.float32),
           "body": np.array([0.9, 0.9, 0.1], np.float32)}
    np.testing.assert_allclose(pred_dt_field(dec, "d_face"), [0.0, 2.0, -5.0])
    np.testing.assert_allclose(pred_dt_field({"sdf": np.array([-1.0])}, "sdf"), [-1.0])


# --------------------------------------------------------------------------- #
# viewer table / configs / ablation
# --------------------------------------------------------------------------- #
def test_view_layers_for_the_sides_channels():
    lay = layer_for("d_face", "student", clip=CLIP)
    assert lay is not None and lay.kind == "sdf" and lay.clip == CLIP
    for ch in ("surface_side1", "body"):
        lay = layer_for(ch, "student")
        assert lay is not None and lay.kind == "prob", ch


@pytest.mark.parametrize("name", ["sides30k", "sides30k_aux"])
def test_shipped_sides_configs_load(name):
    from tsm.config import load_config
    from tsm.train import train_opts

    cfg = load_config(os.path.join("configs", f"{name}.json"))
    o = train_opts(cfg)
    assert o["surface_mode"] == "sides" and o["steps"] == 30000
    assert cfg.out_dir == f"/home/forrest/tsm-output/{name}"
    assert o["input_axis"] is True and o["axis_tangent"] is True and o["fiber_mode"] == "direction"
    want = (1.0, 1.0) if name.endswith("aux") else (0.0, 0.0)
    assert (o["surface_aux"]["gap"], o["surface_aux"]["cldice"]) == want
    ev = load_config(os.path.join("configs", f"paris4_eval_{name}.json"))
    assert ev.out_dir == f"/home/forrest/tsm-output/eval_{name}"
    assert ev.extra["infer"]["checkpoint"] == f"/home/forrest/tsm-output/{name}/train/latest.pt"
    assert train_opts(ev)["surface_mode"] == "sides"


def test_ablate_variant_config_accepts_the_sides_surface_mode():
    from tsm.ablate import ablate_opts, variant_config
    from tsm.config import load_config
    from tsm.train import train_opts

    cfg = load_config(os.path.join("configs", "sides30k.json"))
    opts = ablate_opts(cfg)
    vc = variant_config(cfg, "sides", {"surface_mode": "sides", "fiber_mode": "direction"},
                        {**opts, "steps": 300}, "/tmp/tsm-ablate-none")
    o = train_opts(vc)
    assert o["surface_mode"] == "sides" and o["fiber_mode"] == "direction" and o["steps"] == 300


# --------------------------------------------------------------------------- #
# dev/eval_region.py on a synthetic sides store
# --------------------------------------------------------------------------- #
SHAPE = (48, 32, 32)
SHEETS = ((9, 13), (25, 29))       # [z0, z1) of the 4-voxel upstream band of each sheet
CORE = ((10, 12), (26, 28))        # the 2-voxel sheet body inside it


def _write(path, channels, arrs, origin=(0, 0, 0), chunk=16):
    w = BrickWriter(path, list(channels), arrs[0].shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=2.4, scale=1.0)
    for ci, a in enumerate(arrs):
        w.write(ci, origin[0], origin[1], origin[2], np.ascontiguousarray(a))
    return path


def _budget(root):
    return parse_config({"volume": {"url": "x"},
                         "region": {"start_zyx": [0, 0, 0], "size_zyx": list(SHAPE)},
                         "out_dir": str(root)}).budget


def _sides_stores(root: str, weld: bool = False):
    """(rectoverso store, sides prediction store).  The prediction is exact unless ``weld``."""
    from tsm.rvfaces import thin_band

    rv = np.zeros(SHAPE, np.uint8)
    for z0, z1 in SHEETS:
        rv[z0:z0 + 2] = 1
        rv[z0 + 2:z1] = 2
    rv_path = _write(os.path.join(root, "rv.zarr"), ["rectoverso", "hzvt"], [rv, np.zeros_like(rv)])
    thin = thin_band(rv > 0, 32)
    body = np.zeros(SHAPE, bool)
    for z0, z1 in CORE:
        body[z0:z1] = True
    if weld:
        body[CORE[0][1] - 1:CORE[1][0] + 1, 0, 0] = True
    d = np.where(thin, 0.0, 5.0).astype(np.float32)
    ch = ["d_face", "body", "valid", "surface_side1"]
    pred_path = _write(os.path.join(root, "pred.zarr"), ch,
                       [encode_sdf(d), encode_prob(body.astype(np.float32)),
                        np.full(SHAPE, 255, np.uint8), thin.astype(np.uint8) * 255])
    return rv_path, pred_path


def test_upstream_body_pass_on_a_sides_store(tmp_path):
    rv_path, pred_path = _sides_stores(str(tmp_path))
    m = eval_region.upstream_body_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], list(SHAPE), _budget(tmp_path), brick=(16, 32, 32))
    assert m["prediction"] == "surface_side1" and m["unordered"] is True
    assert m["n_pred"] > 0 and m["n_band"] == m["n_pred"]
    assert m["dice_within2"] == pytest.approx(1.0)


def test_upstream_topology_body_entry_on_a_sides_store(tmp_path):
    budget = _budget(tmp_path)
    kw = dict(brick=(16, 32, 32), min_component=8, min_size=512, clip=CLIP)
    rv_path, pred_path = _sides_stores(str(tmp_path / "ok"))
    ok = eval_region.upstream_topology_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], list(SHAPE), budget, **kw)
    assert "in" not in ok and "out" not in ok and "inout_cross_links" not in ok
    b = ok["body"]
    assert b["n_ref_objects"] == 2 and b["n_student_components"] == 2
    assert (b["merges"], b["breaks"], b["missed"], b["spurious"]) == (0, 0, 0, 0)
    assert b["merges_per_100"] == 0.0

    rv2, pred2 = _sides_stores(str(tmp_path / "weld"), weld=True)
    bad = eval_region.upstream_topology_pass(
        eval_region.open_store(pred2), eval_region.open_store(rv2),
        [0, 0, 0], list(SHAPE), budget, **kw)["body"]
    assert bad["n_student_components"] == 1 and bad["merges"] == 1 and bad["merges_per_100"] > 0.0


E2E_SHAPE = (64, 64, 64)
E2E_ORIGIN = (16, 32, 32)


def _make_sides_case(root: str, invert_body: bool = False) -> str:
    """A sides student that copies ``face_dist`` / ``body_mask_np`` of the faces labels."""
    import zarr

    from tsm.data import decode_sdf

    s = make_synthetic(root, fine_shape=E2E_SHAPE, fine_origin=E2E_ORIGIN, faces=True)
    arr = zarr.open_array(store=s["fine"], mode="r")
    ch = list(arr.attrs["channels"])
    g = lambda n: np.asarray(arr[ch.index(n)])  # noqa: E731
    si, so = decode_sdf(g("sdf_in")), decode_sdf(g("sdf_out"))
    d_u8 = encode_sdf(face_dist(si, so))
    valid = np.full(E2E_SHAPE, 255, np.uint8)
    bod = body_mask_np(si, so)
    if invert_body:
        bod = ~bod
    z = np.arange(E2E_SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(E2E_SHAPE, np.float32)
    zer = np.zeros(E2E_SHAPE, np.float32)
    fields = {
        "d_face": d_u8, "body": encode_prob(bod.astype(np.float32)), "valid": valid, "ink": g("ink"),
        "sin": encode_signed(np.sin(0.3 * z)), "cos": encode_signed(np.cos(0.3 * z)),
        "density": np.full(E2E_SHAPE, 40, np.uint8),
        "nx": encode_signed(zer), "ny": encode_signed(zer), "nz": encode_signed(zer + 1),
        "conf": encode_prob(np.full(E2E_SHAPE, 0.7, np.float32)),
        "spare": np.zeros(E2E_SHAPE, np.uint8),
        "surface_side1": (((d_u8 != 0) & (decode_sdf(d_u8) <= 1.0) & (valid >= 128))
                          .astype(np.uint8) * 255),
    }
    _write(os.path.join(root, "student", "pred.zarr"), SIDES_PRED_CHANNELS,
           [fields[c] for c in SIDES_PRED_CHANNELS], E2E_ORIGIN, chunk=32)
    tdir = os.path.join(root, "teachers")
    _write(os.path.join(tdir, "ink.zarr"), ["ink"], [g("ink")], E2E_ORIGIN, chunk=32)
    cfg = {"volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
           "region": {"start_zyx": list(E2E_ORIGIN), "size_zyx": list(E2E_SHAPE)},
           "budget": {"ram_bytes": 2 << 30, "array_bytes": 1 << 30},
           "out_dir": root, "extra": {}}
    path = os.path.join(root, "eval.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def test_eval_region_sides_mode_end_to_end(tmp_path):
    root = str(tmp_path / "out")
    m = eval_region.main([_make_sides_case(root), "--no-gallery", "--brick", "32,32,32", "--axis", "none"])
    assert m["surface"]["student_surface"] == "d_face"
    assert m["surface"]["sdf_mae_band"] < 0.2
    assert m["surface"]["zc_dice_within1"] > 0.85
    b = m["body"]
    assert b["target"] == "(sdf_in > 0) & (sdf_out < 0)"
    assert b["n_faces_valid1"] > 0 and b["n_label_body"] > 0
    assert b["body_dice"] > 0.99 and b["body_iou"] > 0.98
    assert b["body_vol_ratio"] == pytest.approx(1.0, abs=0.02)
    assert "faces" not in m
    md = open(os.path.join(root, "eval", "metrics.md")).read()
    assert "## body" in md and "| body_dice |" in md


def test_eval_region_sides_mode_inverted_body_is_worse(tmp_path):
    a = eval_region.main([_make_sides_case(str(tmp_path / "a")), "--no-gallery",
                          "--brick", "32,32,32", "--axis", "none"])
    b = eval_region.main([_make_sides_case(str(tmp_path / "b"), invert_body=True), "--no-gallery",
                          "--brick", "32,32,32", "--axis", "none"])
    assert b["body"]["body_dice"] < a["body"]["body_dice"]
    # the distance half is untouched by the body error
    assert b["surface"]["sdf_mae_band"] == pytest.approx(a["surface"]["sdf_mae_band"], abs=1e-6)
