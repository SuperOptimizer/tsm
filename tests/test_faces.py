"""Two-face surface representation (``extra.labels.faces`` + ``extra.train.surface_mode``).

Everything runs on the CPU: the face label builder on a synthetic two-sheet volume, the
encodings, the dataset / augmentation rules, one training step and one inference + export.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import zarr

from synth import SHEET_HALF, make_synthetic
from tsm.config import parse_config
from tsm.data import CLIP, Augment, CropDataset, target_keys
from tsm.infer import (
    FACE_PRED_CHANNELS,
    N_FACE_HEAD_CH,
    activate_heads,
    decode_pred,
    export_lasagna,
    extract_surface,
    load_student,
    n_head_ch,
    pred_channels,
    run_infer,
    to_unit,
)
from tsm.labels import build_face_labels, decode_sdf_u8, encode_sdf_u8, fine_channels
from tsm.student import TSMNet, build_model, heads_for
from tsm.train import EMA, compute_losses, evaluate, faces_loss, save_checkpoint, synthetic_batch

# --------------------------------------------------------------------------- #
# synthetic two-sheet volume
# --------------------------------------------------------------------------- #
BODY_THRESHOLD = 120.0  # halfway between the synthetic air (40) and papyrus (200) levels
NO_CLOSE = dict(body_threshold=BODY_THRESHOLD, close_radius=0.0)  # geometry tests: leave the body as thresholded


def two_sheets(gap: int = 12, thickness: int = 20, shape=(72, 40, 40), band: int = 7, seed: int = 0):
    """Two bright slabs stacked along z with an air gap, recto band on the low-z (IN) face.

    Outward is +z (the coarse normal handed to the builder), so the IN face of each slab is
    the low-z one and the recto band sits on it, as in the real scrolls.
    """
    Z, Y, X = shape
    rng = np.random.default_rng(seed)
    ct = np.full(shape, 40.0)
    z0 = 10
    slabs = [(z0, z0 + thickness), (z0 + thickness + gap, z0 + 2 * thickness + gap)]
    body = np.zeros(shape, bool)
    for a, b in slabs:
        body[a:b] = True
    ct[body] = 200.0
    ct = np.clip(ct + rng.normal(0, 3, shape), 0, 255).astype(np.uint8)
    recto = np.zeros(shape, np.uint8)
    for a, _ in slabs:
        recto[a:a + band] = 255
    n_out = np.zeros((3,) + shape, np.float32)
    n_out[0] = 1.0  # outward = +z
    radial = np.zeros((3,) + shape, np.float32)
    radial[1] = 1.0  # deliberately wrong: the coarse normal must win
    return ct, recto, n_out, radial, slabs


def test_build_face_labels_two_sheets():
    ct, recto, n_out, radial, slabs = two_sheets()
    stats: dict = {}
    sdf_in_u8, sdf_out_u8, valid, thick = build_face_labels(
        recto, ct, n_out, radial, CLIP, **NO_CLOSE, stats=stats)
    for a in (sdf_in_u8, sdf_out_u8, valid, thick):
        assert a.shape == ct.shape and a.dtype == np.uint8
    # both sheets found, geometric thickness right, recto band lands on the IN face
    assert len(stats["components"]) == 2
    for c in stats["components"]:
        assert c["thickness_median"] == pytest.approx(20.0, abs=1.0)
        assert c["recto_is_in"] == 1.0
        assert c["ok"]
    # zero sets: exactly one voxel per (y, x) column per sheet, at the slab faces
    zin, zout = sdf_in_u8 == 128, sdf_out_u8 == 128
    assert set(np.unique(zin.sum(0))) == {2} and set(np.unique(zout.sum(0))) == {2}
    zi = np.flatnonzero(zin[:, 20, 20])
    zo = np.flatnonzero(zout[:, 20, 20])
    np.testing.assert_array_equal(zi, [s[0] for s in slabs])
    np.testing.assert_array_equal(zo, [s[1] - 1 for s in slabs])
    # the two zero sets of one sheet are separated by its thickness
    assert (zo[0] - zi[0]) == 19 and (zo[1] - zi[1]) == 19
    # signs: positive outward of each face
    sdf_in = decode_sdf_u8(sdf_in_u8, CLIP)
    sdf_out = decode_sdf_u8(sdf_out_u8, CLIP)
    assert sdf_in[zi[0] + 5, 20, 20] > 0 and sdf_in[zi[0] - 5, 20, 20] < 0
    assert sdf_out[zo[0] - 5, 20, 20] < 0 and sdf_out[zo[0] + 5, 20, 20] > 0
    # inside a sheet: sdf_in > 0 > sdf_out and their difference is the thickness
    inside = (sdf_in > 0) & (sdf_out < 0)
    assert inside[zi[0] + 10, 20, 20]
    assert (sdf_in - sdf_out)[zi[0] + 10, 20, 20] == pytest.approx(19.0, abs=1.0)
    assert thick[zi[0] + 10, 20, 20] == pytest.approx(20, abs=1)
    # well-separated sheets are supervised, nothing ignored
    assert (valid == 1).all()


def test_build_face_labels_contact_and_thickness_gates():
    # a gap smaller than gap_min: the out face of the first sheet is in contact -> ignore
    ct, recto, n_out, radial, _ = two_sheets(gap=4)
    _, _, valid, _ = build_face_labels(recto, ct, n_out, radial, CLIP, **NO_CLOSE, gap_min=6.0)
    assert (valid == 2).mean() > 0.2 and (valid == 1).any()
    # with gap_min and side_reach both below the 4-voxel gap it is real air: everything supervised
    _, _, valid2, _ = build_face_labels(recto, ct, n_out, radial, CLIP, **NO_CLOSE, gap_min=2.0, side_reach=3.0)
    assert not (valid2 == 2).any()
    # side_reach larger than the gap makes the local march see body on both sides -> ambiguous
    _, _, valid2b, _ = build_face_labels(recto, ct, n_out, radial, CLIP, **NO_CLOSE, gap_min=2.0, side_reach=6.0)
    assert (valid2b == 2).any()
    # thickness gate: 20-voxel sheets are outside [30, 90] -> all ignored
    _, _, valid3, _ = build_face_labels(recto, ct, n_out, radial, CLIP, **NO_CLOSE,
                                        gap_min=2.0, side_reach=3.0, min_thickness=30.0)
    assert (valid3 == 2).all()


def test_build_face_labels_geometry_ignores_the_teacher():
    """face_in/face_out are geometric: putting the recto band on the OUT face only changes
    the ``recto_is_in`` diagnostic, never the labels."""
    ct, recto, n_out, radial, slabs = two_sheets()
    a = build_face_labels(recto, ct, n_out, radial, CLIP, **NO_CLOSE)
    recto_flipped = np.zeros_like(recto)
    for _, b in slabs:
        recto_flipped[b - 7:b] = 255
    st: dict = {}
    b = build_face_labels(recto_flipped, ct, n_out, radial, CLIP, **NO_CLOSE, stats=st)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert all(c["recto_is_in"] == 0.0 for c in st["components"])
    # flipping the outward direction swaps which boundary is "in"
    c = build_face_labels(recto, ct, -n_out, radial, CLIP, **NO_CLOSE)
    assert np.abs(decode_sdf_u8(a[0], CLIP) + decode_sdf_u8(c[1], CLIP)).max() <= CLIP / 127 + 1e-5


def test_build_face_labels_no_body():
    shape = (24, 16, 16)
    ct = np.full(shape, 40, np.uint8)
    recto = np.zeros(shape, np.uint8)
    n_out = np.zeros((3,) + shape, np.float32)
    n_out[0] = 1.0
    si, so, valid, thick = build_face_labels(recto, ct, None, n_out, CLIP, **NO_CLOSE)
    assert (valid == 2).all() and not thick.any()
    assert decode_sdf_u8(si, CLIP).min() == pytest.approx(CLIP)
    assert (si == so).all()


def test_face_channel_encoding_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.uniform(-CLIP, CLIP, (4, 4, 4)).astype(np.float32)
    u = encode_sdf_u8(v, CLIP)
    assert u.min() >= 1
    assert np.abs(decode_sdf_u8(u, CLIP) - v).max() <= CLIP / 127 + 1e-5
    assert fine_channels(False) == ["sdf", "sdf_valid", "ink", "ink_valid"]
    assert fine_channels(True)[:4] == fine_channels(False)
    assert fine_channels(True)[4:] == ["sdf_in", "sdf_out", "faces_valid", "thickness"]


# --------------------------------------------------------------------------- #
# labels driver
# --------------------------------------------------------------------------- #
def test_run_labels_faces_option_validation():
    from tsm.labels import _faces_opts

    assert _faces_opts(None)["enabled"] is False
    assert _faces_opts(True)["enabled"] is True
    assert _faces_opts({"enabled": True, "body_threshold": 70})["body_threshold"] == 70
    with pytest.raises(ValueError):
        _faces_opts({"bogus": 1})
    with pytest.raises(ValueError):
        _faces_opts({"min_thickness": 100, "max_thickness": 50})


# --------------------------------------------------------------------------- #
# dataset / augmentation
# --------------------------------------------------------------------------- #
def test_dataset_faces_mode_shapes(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ds = CropDataset(_reader(s["ct"]), s["fine"], s["coarse"], patch=32, stride=32, augment=False, surface_mode="faces")
    assert ds.valid_channel == "faces_valid"
    item = ds.to_tensors(ds.load(ds.origins[0]))
    tk = target_keys("faces")
    assert tk["surface_sdf"] == 2
    for k, c in tk.items():
        assert item[k].shape == (c, 32, 32, 32), k
    # the two decoded faces are SHEET_HALF apart wherever the labels are valid
    m = item["surface_valid"][0] > 0
    d = (item["surface_sdf"][0] - item["surface_sdf"][1])[m]
    d = d[(item["surface_sdf"][0][m].abs() < CLIP - 1) & (item["surface_sdf"][1][m].abs() < CLIP - 1)]
    assert torch.allclose(d, torch.full_like(d, 2 * SHEET_HALF), atol=0.2)
    # the medial store is still readable in the default mode from the same store
    ds0 = CropDataset(_reader(s["ct"]), s["fine"], s["coarse"], patch=32, stride=32, augment=False)
    assert ds0.to_tensors(ds0.load(ds0.origins[0]))["surface_sdf"].shape == (1, 32, 32, 32)


def _reader(url: str):
    from tsm.data import open_reader_factory

    return open_reader_factory(url, 0, 2.4)


def test_augment_does_not_swap_faces():
    """in/out are defined by the physical outward direction, which moves with the geometry:
    a flip moves both fields with the grid and never exchanges the two channels."""
    B, P = 1, 16
    z = torch.arange(P, dtype=torch.float32).view(1, 1, P, 1, 1).expand(B, 1, P, P, P)
    sdf_in = (z - 4.0).clamp(-CLIP, CLIP)
    sdf_out = (z - 10.0).clamp(-CLIP, CLIP)
    batch = {
        "input": torch.zeros(B, 2, P, P, P),
        "surface_sdf": torch.cat([sdf_in, sdf_out], 1),
        "surface_valid": torch.ones(B, 1, P, P, P),
        "ink_prob": torch.zeros(B, 1, P, P, P), "ink_valid": torch.ones(B, 1, P, P, P),
        "winding": torch.zeros(B, 6, P, P, P), "winding_conf": torch.zeros(B, 1, P, P, P),
        "winding_valid": torch.zeros(B, 1, P, P, P),
    }
    aug = Augment({"preset": "none", "flip": {"p": 1.0}}, seed=0)
    params = [aug.sample_spatial() for _ in range(B)]
    assert any(params[0].flips)
    out = aug.apply_spatial(batch, params)
    got = out["surface_sdf"]
    assert got.shape == (B, 2, P, P, P)
    exp = batch["surface_sdf"]
    for a in range(3):
        if params[0].flips[a]:
            exp = exp.flip(a + 2)
    m = out["surface_valid"] == 1
    err = (got - exp).abs()[m.expand_as(got)]
    assert float(err.max()) < 1e-4
    # the in face stays "in": it is still the smaller (more negative) of the two everywhere
    assert bool((got[:, 0:1] >= got[:, 1:2]).all())


def test_augment_scales_both_faces():
    B, P = 1, 16
    base = torch.full((B, 1, P, P, P), 4.0)
    batch = {
        "input": torch.zeros(B, 2, P, P, P),
        "surface_sdf": torch.cat([base, -base], 1),
        "surface_valid": torch.ones(B, 1, P, P, P),
        "ink_prob": torch.zeros(B, 1, P, P, P), "ink_valid": torch.ones(B, 1, P, P, P),
        "winding": torch.zeros(B, 6, P, P, P), "winding_conf": torch.zeros(B, 1, P, P, P),
        "winding_valid": torch.zeros(B, 1, P, P, P),
    }
    aug = Augment({"preset": "none", "scale": {"p": 1.0, "lo": 1.5, "hi": 1.5}}, seed=0)
    params = [aug.sample_spatial() for _ in range(B)]
    out = aug.apply_spatial(batch, params)
    core = out["surface_sdf"][:, :, 4:-4, 4:-4, 4:-4]
    assert torch.allclose(core[:, 0], torch.full_like(core[:, 0], 6.0), atol=1e-3)
    assert torch.allclose(core[:, 1], torch.full_like(core[:, 1], -6.0), atol=1e-3)


# --------------------------------------------------------------------------- #
# model / losses / train step
# --------------------------------------------------------------------------- #
def test_heads_and_faces_loss():
    assert heads_for("faces")["surface"] == 3 and heads_for("medial")["surface"] == 2
    with pytest.raises(ValueError):
        heads_for("bogus")
    B, P = 2, 8
    tgt = torch.randn(B, 2, P, P, P) * 4.0
    valid = torch.ones(B, 1, P, P, P)
    pred = torch.cat([tgt, torch.full((B, 1, P, P, P), 10.0)], 1)
    d = faces_loss(pred, tgt, valid)
    assert set(d) == {"sdf_in_l1", "sdf_out_l1", "band_dice_in", "band_dice_out", "valid_bce"}
    for k in ("sdf_in_l1", "sdf_out_l1", "band_dice_in", "band_dice_out"):
        assert float(d[k]) == pytest.approx(0.0, abs=1e-5)
    assert float(d["valid_bce"]) < 1e-3
    # a wrong face pair costs something
    bad = torch.cat([tgt.flip(1), torch.full((B, 1, P, P, P), 10.0)], 1)
    d2 = faces_loss(bad, tgt, valid)
    assert float(d2["sdf_in_l1"]) > 0
    with pytest.raises(ValueError):
        faces_loss(pred[:, :2], tgt, valid)


def test_train_step_faces_mode():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="faces")
    assert model.heads["surface"] == 3
    batch = synthetic_batch(1, 16, surface_mode="faces")
    assert batch["surface_sdf"].shape[1] == 2
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    out = model(batch["input"])
    loss, parts = compute_losses(out, batch, surface_mode="faces")
    assert "surface/sdf_in_l1" in parts and "surface/band_dice_out" in parts
    loss.backward()
    opt.step()
    assert np.isfinite(parts["total"])
    # the medial path is untouched and still the default
    m2 = build_model(widths=(8, 16, 16))
    b2 = synthetic_batch(1, 16)
    l2, p2 = compute_losses(m2(b2["input"]), b2, surface_mode="medial")
    assert "surface/sdf_l1" in p2


def test_evaluate_faces_metrics(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ds = CropDataset(_reader(s["ct"]), s["fine"], s["coarse"], patch=32, stride=32, augment=False, surface_mode="faces")
    model = build_model(widths=(8, 16, 16), surface_mode="faces")
    res = evaluate(model, ds, ds.origins[:2], batch=1, surface_mode="faces")
    for k in ("surface/in_sdf_mae", "surface/out_sdf_mae", "surface/in_zc_dice", "surface/out_zc_dice",
              "surface/in_surf_sym_median", "surface/thickness_mae", "surface/valid_auprc"):
        assert k in res, k


# --------------------------------------------------------------------------- #
# inference / export
# --------------------------------------------------------------------------- #
def test_pred_channels_and_activations():
    assert pred_channels("faces") == FACE_PRED_CHANNELS and n_head_ch("faces") == N_FACE_HEAD_CH
    assert len(FACE_PRED_CHANNELS) == N_FACE_HEAD_CH + 3
    B, P = 1, 4
    out = {"surface": torch.randn(B, 3, P, P, P), "ink": torch.randn(B, 1, P, P, P), "winding": torch.randn(B, 8, P, P, P)}
    phys = activate_heads(out, "faces")
    assert phys.shape == (B, N_FACE_HEAD_CH, P, P, P)
    torch.testing.assert_close(phys[:, 0:2], out["surface"][:, 0:2])
    torch.testing.assert_close(phys[:, 2:3], torch.sigmoid(out["surface"][:, 2:3]))
    u = to_unit(phys, CLIP, "faces")
    assert float(u.min()) >= 0 and float(u.max()) <= 1
    q = np.round(u.numpy() * 255).astype(np.uint8)
    for c in (0, 1):
        np.testing.assert_array_equal(q[:, c], encode_sdf_u8(phys[:, c].numpy(), CLIP))


def _faces_checkpoint(path: str, widths=(8, 16, 16, 16, 16)) -> None:
    net = TSMNet(widths=widths, heads=heads_for("faces"))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ema.update(net)
    save_checkpoint(path, net, ema, opt, None, 7,
                    {"widths": list(widths), "surface_mode": "faces", "train": {"widths": list(widths), "surface_mode": "faces"}})


def test_run_infer_and_export_faces(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ckpt = os.path.join(root, "train", "latest.pt")
    _faces_checkpoint(ckpt)
    _, info = load_student(ckpt)
    assert info["surface_mode"] == "faces"
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"infer": {"patch": 32, "out_tile": 64, "chunk": 32, "stats_box": [32, 64, 64],
                            "prefetch": False, "device": "cpu"}},
    })
    summary = run_infer(cfg)
    assert summary["surface_mode"] == "faces"
    pred = os.path.join(root, "student", "pred.zarr")
    arr = zarr.open_array(store=pred, mode="r")
    assert list(arr.attrs["channels"]) == FACE_PRED_CHANNELS
    a = np.asarray(arr[:])
    ch = FACE_PRED_CHANNELS
    dec = decode_pred(a, CLIP, ch)
    for k in ("sdf_in", "sdf_out", "surface_in1", "surface_out1", "thickness"):
        assert k in dec
    np.testing.assert_array_equal(dec["surface_in1"], extract_surface(a[ch.index("sdf_in")], a[ch.index("valid")], CLIP))
    np.testing.assert_array_equal(dec["surface_out1"], extract_surface(a[ch.index("sdf_out")], a[ch.index("valid")], CLIP))
    data = a[0] != 0
    th = np.clip(np.rint(dec["sdf_in"] - dec["sdf_out"]), 0, 255)
    np.testing.assert_array_equal(dec["thickness"][data], th[data])
    sj = json.load(open(os.path.join(root, "student", "pred.summary.json")))
    assert sj["surface_mode"] == "faces" and "surface_in1" in sj["surface1"] and "thickness" in sj["surface1"]
    assert os.path.exists(os.path.join(root, "student", "preview_faces.png"))
    # lasagna export takes pred_dt from sdf_in by default
    res = export_lasagna(pred, str(tmp_path / "las"), name="tsm", level_shift=1, cos_scaledown=1, scaledown=2,
                         base_shape_zyx=[128, 128, 128], n_levels=4, clip=CLIP, brick=(32, 64, 64))
    assert res["pred_dt_channel"] == "sdf_in"


# --------------------------------------------------------------------------- #
# brick-wise consistency (the driver builds faces per haloed brick)
# --------------------------------------------------------------------------- #
def test_face_labels_brickwise_equals_whole():
    from tsm.labels import iter_cores, read_box_padded

    shape = (48, 40, 40)
    ct, recto, n_out, radial, _ = two_sheets(gap=12, thickness=12, shape=shape, band=4)
    clip, halo, brick = 8.0, 20, (16, 20, 20)  # halo covers the clip and the sheet half-thickness
    kw = dict(**NO_CLOSE, gap_min=2.0)

    def compute(lo, hi):
        r = read_box_padded(recto, lo, hi)
        c = read_box_padded(ct, lo, hi)
        nn = np.stack([read_box_padded(n_out[k], lo, hi) for k in range(3)])
        rr = np.stack([read_box_padded(radial[k], lo, hi) for k in range(3)])
        return build_face_labels(r, c, nn, rr, clip, **kw)

    whole = [a[halo:-halo, halo:-halo, halo:-halo]
             for a in compute((-halo,) * 3, tuple(s + halo for s in shape))]
    got = [np.zeros(shape, np.uint8) for _ in range(4)]
    for lo, hi in iter_cores(shape, brick):
        outs = compute([v - halo for v in lo], [v + halo for v in hi])
        core = tuple(slice(halo, halo + hi[a] - lo[a]) for a in range(3))
        for g, o in zip(got, outs):
            g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = o[core]
    for name, g, w in zip(("sdf_in", "sdf_out", "faces_valid", "thickness"), got, whole):
        assert np.array_equal(g, w), f"{name} differs brick-wise vs whole ({(g != w).sum()} voxels)"


# --------------------------------------------------------------------------- #
# bundle-level body: closing bridges intra-bundle gaps, not inter-sheet gaps
# --------------------------------------------------------------------------- #
def test_close_ball_and_normal_line_bridge_only_small_gaps():
    from tsm.labels import close_ball, close_normal_line

    m = np.zeros((40, 12, 12), bool)
    m[5:10] = True      # layer 1
    m[13:18] = True     # layer 2 of the same bundle (3-voxel gap)
    m[30:35] = True     # a neighbouring sheet (12-voxel gap)
    n = np.zeros((3,) + m.shape, np.float32)
    n[0] = 1.0          # normal along z, i.e. across the layers
    for close in (lambda a, r: close_ball(a, r), lambda a, r: close_normal_line(a, n, r)):
        c2 = close(m, 2.0)
        assert c2[10:13].all()                       # the 3-voxel intra-bundle gap is closed
        assert not c2[18:30].any()                   # the inter-sheet gap stays open
        assert not close(m, 0.0)[10].any()           # radius 0 = no closing
        assert close(m, 6.0)[20].all()               # 2r = 12 does bridge the wide gap: r < gap/2
        np.testing.assert_array_equal(c2[m], m[m])   # closing only ever adds voxels
    # a line element along the normal leaves in-plane structure alone, a ball rounds it off
    holed = np.zeros((20, 20, 20), bool)
    holed[8:12] = True
    holed[:, 9:11, 9:11] = False  # an in-plane hole through the sheet
    assert close_normal_line(holed, np.broadcast_to(n[:, :1, :1, :1], (3,) + holed.shape).copy(), 3.0)[10, 9, 9] == holed[10, 9, 9]
    assert close_ball(holed, 3.0)[10, 9, 9]


def test_build_face_labels_bundle_closing():
    """Two bundles of two thin layers each: with the default ball closing the builder sees two
    sheets (not four), the thickness is the bundle thickness and the recto band is on one face."""
    Z, Y, X = 80, 24, 24
    rng = np.random.default_rng(0)
    ct = np.full((Z, Y, X), 40.0)
    bundles = [(10, 34), (50, 74)]  # 24-voxel bundles, 16 voxels of air between them
    for a, b in bundles:
        ct[a:a + 10] = 200.0        # layer 1
        ct[a + 14:b] = 200.0        # layer 2 (4-voxel intra-bundle gap)
    ct = np.clip(ct + rng.normal(0, 3, ct.shape), 0, 255).astype(np.uint8)
    recto = np.zeros((Z, Y, X), np.uint8)
    for a, _ in bundles:
        recto[a:a + 8] = 255        # the innermost layer of each bundle
    n_out = np.zeros((3, Z, Y, X), np.float32)
    n_out[0] = 1.0
    st_layers: dict = {}
    build_face_labels(recto, ct, n_out, n_out, CLIP, body_threshold=BODY_THRESHOLD, close_radius=0.0,
                      min_thickness=1.0, stats=st_layers)
    assert len(st_layers["components"]) == 4  # layer-level: four thin bodies
    st: dict = {}
    _, _, valid, thick = build_face_labels(recto, ct, n_out, n_out, CLIP, body_threshold=BODY_THRESHOLD,
                                           close_radius=6.0, stats=st)
    assert len(st["components"]) == 2  # bundle-level: two sheets, the 16-voxel gap stays open
    for c in st["components"]:
        assert c["thickness_median"] == pytest.approx(24.0, abs=1.5)
        assert c["recto_is_in"] == 1.0  # the recto band is now adjacent to one face only
        assert c["ok"]  # 24 is inside the default [15, 90] gate
    assert thick[20, 12, 12] == pytest.approx(24, abs=2)
    assert (valid == 1).all()


# --------------------------------------------------------------------------- #
# local (marching) side test vs the old nearest-medial ("centroid") rule
# --------------------------------------------------------------------------- #
def concentric_shells(Z: int = 12, N: int = 160):
    """A thick sheet bundle and a thin delaminated layer as concentric cylinders about a z axis.

    Thick bundle r in [20, 50) (30 voxels), air gap r in [50, 58) (8 voxels), thin layer
    r in [58, 62) (4 voxels).  The nearest *medial* voxel of a point on the thick bundle's OUT
    face (r ~ 49) is the thin layer's ridge (8 + 2 = 10 voxels away) rather than its own ridge
    (15 voxels away), so the nearest-medial rule puts that face on the wrong side -- exactly the
    "two sheets facing the same gap are both labelled IN" failure seen on real data.
    """
    c = N / 2.0
    _, y, x = np.meshgrid(np.arange(Z), np.arange(N), np.arange(N), indexing="ij")
    dy, dx = y - c, x - c
    r = np.sqrt(dy ** 2 + dx ** 2)
    body = ((r >= 20) & (r < 50)) | ((r >= 58) & (r < 62))
    rad = np.zeros((3, Z, N, N), np.float32)
    rr = np.maximum(r, 1e-6)
    rad[1], rad[2] = dy / rr, dx / rr
    return body, rad, r


def test_local_side_beats_nearest_medial_on_concentric_shells():
    from scipy import ndimage as ndi

    from tsm.labels import _edt, _local_faces, _shift_any, _signed_side, medial_surface

    body, rad, r = concentric_shells()
    bnd = body & _shift_any(~body, False)
    fin, fout, amb = _local_faces(bnd, body, rad, 6)
    # ground truth: the IN face of a shell is its smaller-radius boundary
    truth_in = bnd & ((np.abs(r - 20) < 1.6) | (np.abs(r - 58) < 1.6))
    truth_out = bnd & ((np.abs(r - 49) < 1.6) | (np.abs(r - 61) < 1.6))
    tot = int((truth_in | truth_out).sum())
    assert tot == int(bnd.sum())
    local_ok = int((fin & truth_in).sum() + (fout & truth_out).sum()) / tot
    assert local_ok == 1.0 and not amb.any()
    # the old rule: side of the nearest medial voxel along n_out there
    d = _edt(body)
    _, midx = ndi.distance_transform_edt(~medial_surface(body, d), return_indices=True)
    midx = midx.astype(np.int32)
    side = _signed_side(midx, np.ascontiguousarray(rad[:, midx[0], midx[1], midx[2]]), body.shape)
    cin, cout = bnd & (side <= 0), bnd & (side > 0)
    centroid_ok = int((cin & truth_in).sum() + (cout & truth_out).sum()) / tot
    assert centroid_ok < 0.8  # measured 0.74: the whole out face of the thick bundle is flipped
    thick_out = truth_out & (np.abs(r - 49) < 1.6)
    assert bool(cin[thick_out].all()) and bool(fout[thick_out].all())


def test_build_face_labels_concentric_shells_end_to_end():
    body, rad, r = concentric_shells()
    ct = np.where(body, 200.0, 40.0)
    ct = np.clip(ct + np.random.default_rng(0).normal(0, 2, ct.shape), 0, 255).astype(np.uint8)
    recto = np.zeros(ct.shape, np.uint8)
    recto[(r >= 20) & (r < 28)] = 255  # band on the innermost part of the thick bundle
    st: dict = {}
    si, so, valid, thick = build_face_labels(recto, ct, rad, rad, CLIP, body_threshold=BODY_THRESHOLD,
                                             close_radius=0.0, gap_min=2.0, side_reach=6.0, stats=st)
    sdf_in, sdf_out = decode_sdf_u8(si, CLIP), decode_sdf_u8(so, CLIP)
    mid = ct.shape[0] // 2
    row = slice(int(ct.shape[1] // 2), int(ct.shape[1] // 2) + 1)
    # along +x from the axis the zero sets appear at the shell boundaries, in before out
    zi = np.flatnonzero(si[mid, ct.shape[1] // 2] == 128)
    zo = np.flatnonzero(so[mid, ct.shape[1] // 2] == 128)
    c = ct.shape[2] // 2
    zi, zo = zi[zi > c], zo[zo > c]  # the +x half
    assert len(zi) == 2 and len(zo) == 2
    assert zi[0] < zo[0] < zi[1] < zo[1]  # in, out, in, out going outward
    # the local disagreement diagnostic is recorded and large here
    assert st["local_vs_centroid_disagree_frac"][0] > 0.15
    assert st["face_ambiguous_voxels"] >= 0
    del row, sdf_in, sdf_out, thick, valid
