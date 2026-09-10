"""Smoke + sanity test of ``dev/eval_region.py`` on synthetic stores (CPU, no network)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import make_synthetic  # noqa: E402
from tsm.data import CLIP, encode_prob, encode_sdf, encode_signed  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402

eval_region = pytest.importorskip("eval_region")

ORIGIN = (16, 32, 32)
SHAPE = (64, 64, 64)


def _write(path, channels, arrs, origin, voxel_um=2.4, scale=1.0, chunk=32):
    w = BrickWriter(path, list(channels), arrs[0].shape, chunk=chunk, origin_zyx=origin, voxel_um=voxel_um, scale=scale)
    for ci, a in enumerate(arrs):
        w.write(ci, origin[0], origin[1], origin[2], np.ascontiguousarray(a))
    return path


def _pred_from_fine(fine_path: str, out: str, noise: float = 0.0, seed: int = 0):
    """A student store that copies the fine labels (plus optional sdf noise) so the metrics are known."""
    from tsm.infer import PRED_CHANNELS, extract_surface
    from tsm.labels import decode_sdf_u8

    arr = zarr.open_array(store=fine_path, mode="r")
    ch = list(arr.attrs["channels"])
    sdf_u8 = np.asarray(arr[ch.index("sdf")])
    sdf_valid = np.asarray(arr[ch.index("sdf_valid")])
    ink_u8 = np.asarray(arr[ch.index("ink")])
    rng = np.random.default_rng(seed)
    sdf = decode_sdf_u8(sdf_u8, CLIP) + (rng.normal(0, noise, sdf_u8.shape).astype(np.float32) if noise else 0.0)
    p_sdf = encode_sdf(sdf)
    p_sdf[sdf_u8 == 0] = 0
    valid = encode_prob((sdf_valid == 1).astype(np.float32) * 0.9 + 0.05)
    z = np.arange(SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE, np.float32)
    ang = 0.3 * z
    fields = {
        "sdf": p_sdf, "valid": valid, "ink": ink_u8,
        "sin": encode_signed(np.sin(ang)), "cos": encode_signed(np.cos(ang)),
        "density": np.full(SHAPE, 40, np.uint8),
        "nx": encode_signed(np.zeros(SHAPE, np.float32)), "ny": encode_signed(np.zeros(SHAPE, np.float32)),
        "nz": encode_signed(np.ones(SHAPE, np.float32)),
        "conf": encode_prob(np.full(SHAPE, 0.7, np.float32)), "spare": np.zeros(SHAPE, np.uint8),
    }
    fields["surface1"] = extract_surface(fields["sdf"], valid, CLIP).astype(np.uint8) * 255
    _write(out, PRED_CHANNELS, [fields[c] for c in PRED_CHANNELS], ORIGIN, chunk=32)
    return fields


def _make_case(root: str, noise: float = 0.0) -> dict:
    s = make_synthetic(root, fine_shape=SHAPE, fine_origin=ORIGIN)
    fields = _pred_from_fine(s["fine"], os.path.join(root, "student", "pred.zarr"), noise=noise)
    # teachers: recto = the label sdf band, ink = the label ink, lasagna/m7 at level 2
    from tsm.labels import decode_sdf_u8

    arr = zarr.open_array(store=s["fine"], mode="r")
    ch = list(arr.attrs["channels"])
    sdf = decode_sdf_u8(np.asarray(arr[ch.index("sdf")]), CLIP)
    band = (np.abs(sdf) <= 1.5) & (np.asarray(arr[ch.index("sdf")]) != 0)
    tdir = os.path.join(root, "teachers")
    _write(os.path.join(tdir, "recto.zarr"), ["recto"], [band.astype(np.uint8) * 255], ORIGIN)
    _write(os.path.join(tdir, "ink.zarr"), ["ink"], [np.asarray(arr[ch.index("ink")])], ORIGIN)
    cshape = tuple(v // 4 for v in SHAPE)
    corigin = tuple(v // 4 for v in ORIGIN)
    las = [np.full(cshape, 128, np.uint8) for _ in range(8)]
    _write(os.path.join(tdir, "lasagna.zarr"), ["cos", "grad_mag", "d0z", "d1z", "d0y", "d1y", "d0x", "d1x"],
           las, corigin, voxel_um=9.6, scale=4.0, chunk=8)
    _write(os.path.join(tdir, "m7_l2.zarr"), ["m7"], [np.full(cshape, 100, np.uint8)], corigin,
           voxel_um=9.6, scale=4.0, chunk=8)
    cfg = {
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 2 << 30, "array_bytes": 1 << 30},
        "out_dir": root,
        "extra": {},
    }
    path = os.path.join(root, "eval.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return {"cfg": path, "root": root, "fields": fields, **s}


def test_eval_region_perfect_student(tmp_path):
    """A student that copies the labels: sdf MAE ~ 0, ink dice 1, zero crossings coincide."""
    case = _make_case(str(tmp_path / "out"), noise=0.0)
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    surf = m["surface"]
    assert surf["sdf_mae_band"] is not None and surf["sdf_mae_band"] < 0.2
    assert surf["zc_dice_within1"] > 0.99 and surf["zc_n_pred"] > 0
    assert surf["valid_auprc"] is None or surf["valid_auprc"] > 0.9
    assert m["ink"]["dice_at_0.5"] > 0.99 and m["ink"]["auprc"] > 0.9
    # student surface1 vs the recto medial surface: the band is 3 voxels thick around the same zero set
    assert m["surface"]["student_to_recto"]["median"] <= 2.0
    assert m["surface"]["recto_to_student"]["median"] <= 2.0
    assert m["surface"]["n_student_surface1"] > 0 and m["surface"]["n_recto_medial"] > 0
    # winding is compared on coarse-valid voxels after the x4 downsample
    w = m["winding"]
    assert w["n_coarse_valid"] > 0 and w["phase_err_deg_n"] > 0
    assert 0.0 <= w["phase_err_deg_median"] <= 180.0 and 0.0 <= w["normal_err_deg_median"] <= 180.0
    assert w["density_mae"] is not None
    assert m["surface1_stats"]["n_voxels"] > 0
    for name in ("metrics.json", "metrics.md"):
        assert os.path.exists(os.path.join(case["root"], "eval", name))
    md = open(os.path.join(case["root"], "eval", "metrics.md")).read()
    assert "sdf_mae_band" in md and "## winding" in md


def test_eval_region_noisy_student_is_worse(tmp_path):
    a = eval_region.main([_make_case(str(tmp_path / "a"), noise=0.0)["cfg"], "--no-gallery", "--brick", "32,32,32"])
    b = eval_region.main([_make_case(str(tmp_path / "b"), noise=6.0)["cfg"], "--no-gallery", "--brick", "32,32,32"])
    assert b["surface"]["sdf_mae_band"] > a["surface"]["sdf_mae_band"]
    assert b["surface"]["zc_dice_within1"] < a["surface"]["zc_dice_within1"]


def test_eval_region_gallery(tmp_path):
    case = _make_case(str(tmp_path / "out"), noise=1.0)
    m = eval_region.main([case["cfg"], "--brick", "32,32,32", "--slices", "2"])
    paths = m["gallery"]
    assert len(paths) == 3 and all(os.path.exists(p) for p in paths)
    from PIL import Image

    im = np.asarray(Image.open(paths[0]))
    assert im.ndim == 3 and im.shape[2] == 3 and im.max() > 0
    assert os.path.basename(paths[-1]) == "gallery_yz.png"


# --------------------------------------------------------------------------- #
# two-face ("faces") surface mode
# --------------------------------------------------------------------------- #
def _shift_z(a: np.ndarray, k: int) -> np.ndarray:
    """Move the volume ``k`` voxels along +z, replicating the first plane (no wrap-around)."""
    if k == 0:
        return a
    b = np.empty_like(a)
    b[k:] = a[:-k]
    b[:k] = a[0]
    return b


def _faces_pred_from_fine(fine_path: str, out: str, shift: int = 0):
    """A two-face student store that copies the faces labels, optionally moved ``shift`` voxels along +z."""
    from tsm.infer import FACE_PRED_CHANNELS, extract_surface

    arr = zarr.open_array(store=fine_path, mode="r")
    ch = list(arr.attrs["channels"])
    g = lambda n: _shift_z(np.asarray(arr[ch.index(n)]), shift)  # noqa: E731
    sin_u8, sout_u8, thick = g("sdf_in"), g("sdf_out"), g("thickness")
    valid = np.full(SHAPE, 255, np.uint8)
    z = np.arange(SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE, np.float32)
    fields = {
        "sdf_in": sin_u8, "sdf_out": sout_u8, "valid": valid,
        "ink": np.asarray(arr[ch.index("ink")]),
        "sin": encode_signed(np.sin(0.3 * z)), "cos": encode_signed(np.cos(0.3 * z)),
        "density": np.full(SHAPE, 40, np.uint8),
        "nx": encode_signed(np.zeros(SHAPE, np.float32)), "ny": encode_signed(np.zeros(SHAPE, np.float32)),
        "nz": encode_signed(np.ones(SHAPE, np.float32)),
        "conf": encode_prob(np.full(SHAPE, 0.7, np.float32)), "spare": np.zeros(SHAPE, np.uint8),
        "surface_in1": extract_surface(sin_u8, valid, CLIP).astype(np.uint8) * 255,
        "surface_out1": extract_surface(sout_u8, valid, CLIP).astype(np.uint8) * 255,
        "thickness": thick,
    }
    _write(out, FACE_PRED_CHANNELS, [fields[c] for c in FACE_PRED_CHANNELS], ORIGIN, chunk=32)
    return fields


def _make_faces_case(root: str, shift: int = 0, winding_fine: bool = False) -> dict:
    from tsm.labels import decode_sdf_u8

    s = make_synthetic(root, fine_shape=SHAPE, fine_origin=ORIGIN, faces=True,
                       winding_fine=winding_fine)
    _faces_pred_from_fine(s["fine"], os.path.join(root, "student", "pred.zarr"), shift=shift)
    arr = zarr.open_array(store=s["fine"], mode="r")
    ch = list(arr.attrs["channels"])
    sdf_u8 = np.asarray(arr[ch.index("sdf")])
    band = (np.abs(decode_sdf_u8(sdf_u8, CLIP)) <= 1.5) & (sdf_u8 != 0)  # recto = the medial band
    tdir = os.path.join(root, "teachers")
    _write(os.path.join(tdir, "recto.zarr"), ["recto"], [band.astype(np.uint8) * 255], ORIGIN)
    _write(os.path.join(tdir, "ink.zarr"), ["ink"], [np.asarray(arr[ch.index("ink")])], ORIGIN)
    cfg = {
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 2 << 30, "array_bytes": 1 << 30},
        "out_dir": root,
        "extra": {},
    }
    path = os.path.join(root, "eval.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return {"cfg": path, "root": root, **s}


def test_eval_region_faces_perfect(tmp_path):
    """A two-face student that copies the faces labels: per-face metrics near perfect."""
    case = _make_faces_case(str(tmp_path / "out"), shift=0)
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    f = m["faces"]
    assert f["labels"].endswith(os.path.join("labels", "fine.zarr")) and f["n_faces_valid1"] > 0
    for name in ("in", "out"):
        r = f[name]
        assert r["sdf_mae_band"] is not None and r["sdf_mae_band"] < 0.2
        assert r["zc_dice_within1"] > 0.99 and r["zc_n_pred"] > 0 and r["zc_n_label"] > 0
        assert r["pred_to_label_median"] == 0.0 and r["label_to_pred_median"] == 0.0
        assert r["pred_to_label_frac_gt3"] == 0.0 and r["label_to_pred_frac_gt3"] == 0.0
    # the thickness gap (sdf_in - sdf_out) is compared on the unsaturated label sheet interior
    assert f["thickness_mae"] is not None and f["thickness_mae"] < 0.01
    assert f["thickness_median_abs_err"] < 0.01
    assert 0 < f["thickness_n"] < f["n_faces_valid1"]
    assert f["thickness_n"] == f["thickness_n_interior"]  # 6 voxels thick, far below 2*clip
    assert f["thickness_frac_saturated"] == 0.0
    # the medial derived from the two faces sits on the recto medial; the raw in face is half a
    # sheet (SHEET_HALF = 3 voxels) away from it
    med, raw = m["surface"], m["surface_inface_vs_recto"]
    # the label `sdf` channel is the medial sdf, so it must be compared with the derived medial
    assert med["student_surface"] == "medial_from_faces"
    assert med["sdf_mae_band"] < 0.2  # ~SHEET_HALF = 3 if the in face were used instead
    assert med["n_student_medial"] > 0 and med["recto_to_student"]["median"] <= 1.5
    assert med["student_to_recto"]["median"] <= 1.5
    assert raw["recto_to_student"]["median"] >= 2.0 and raw["student_to_recto"]["median"] >= 2.0
    md = open(os.path.join(case["root"], "eval", "metrics.md")).read()
    assert "## faces" in md and "in.zc_dice_within1" in md and "## surface_inface_vs_recto" in md


def test_eval_region_faces_shifted(tmp_path):
    """The same student moved 2 voxels along +z: the per-face surface distances are ~2 voxels."""
    case = _make_faces_case(str(tmp_path / "out"), shift=2)
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    for name in ("in", "out"):
        r = m["faces"][name]
        assert 1.5 <= r["pred_to_label_median"] <= 2.5, (name, r)
        assert 1.5 <= r["label_to_pred_median"] <= 2.5, (name, r)
        assert r["sdf_mae_band"] > 1.0
        assert r["zc_dice_within1"] < 0.5
    # a rigid shift keeps the gap between the two faces (up to the uint8 sdf quantisation,
    # ~2 * clip / 254 = 0.16 voxels per face)
    assert m["faces"]["thickness_mae"] < 0.2 and m["faces"]["thickness_n"] > 0
    assert m["faces"]["thickness_median_abs_err"] < 0.01


def test_eval_region_faces_gallery(tmp_path):
    case = _make_faces_case(str(tmp_path / "out"), shift=0)
    m = eval_region.main([case["cfg"], "--brick", "32,32,32", "--slices", "1"])
    from PIL import Image

    assert len(m["gallery"]) == 2 and all(os.path.exists(p) for p in m["gallery"])
    im = np.asarray(Image.open(m["gallery"][0]))
    assert im.ndim == 3 and im.max() > 0


def test_eval_region_faces_labels_option(tmp_path):
    """--faces-labels points at an explicit store; a non-faces store is rejected."""
    case = _make_faces_case(str(tmp_path / "out"), shift=0)
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32",
                          "--faces-labels", os.path.join(case["root"], "labels", "fine.zarr")])
    assert m["faces"]["in"]["zc_dice_within1"] > 0.99
    with pytest.raises(FileNotFoundError):
        eval_region.main([case["cfg"], "--no-gallery", "--faces-labels", str(tmp_path / "nope.zarr")])


# --------------------------------------------------------------------------- #
# R19: the validity AP keeps both validity classes
# --------------------------------------------------------------------------- #
def test_valid_auprc_keeps_both_validity_classes(tmp_path):
    """The evaluation domain is `sdf_valid != 2` intersected with the *prediction* data mask.
    Using the label SDF byte (0 = no data = validity 0) would delete every negative."""
    case = _make_case(str(tmp_path / "out"), noise=0.0)
    f = zarr.open_array(store=case["fine"], mode="r")
    fch = list(f.attrs["channels"])
    lv = np.asarray(f[fch.index("sdf_valid")])
    assert (lv == 0).any() and (lv == 1).any(), "the fixture must contain both validity classes"
    # a student that ran everywhere writes an sdf byte everywhere (no 0 = "no data")
    arr = zarr.open_array(store=os.path.join(case["root"], "student", "pred.zarr"), mode="a")
    pch = list(arr.attrs["channels"])
    sdf = np.asarray(arr[pch.index("sdf")])
    sdf[sdf == 0] = 255
    arr[pch.index("sdf")] = sdf

    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    ap = m["surface"]["valid_auprc"]
    assert isinstance(ap, float) and 0.0 <= ap <= 1.0
    n_pos, n_neg = int((lv == 1).sum()), int((lv == 0).sum())
    assert m["surface"]["valid_auprc_n"] == n_pos + n_neg  # negatives survived the selector
    assert ap > n_pos / (n_pos + n_neg)  # the student's valid head is better than the prevalence


# --------------------------------------------------------------------------- #
# R20: the reservoir samples the pooled voxel population
# --------------------------------------------------------------------------- #
def _blocks(rng, sizes, prevalences, shift=1.0):
    out = []
    for n, p in zip(sizes, prevalences):
        y = rng.random(n) < p
        out.append(((rng.normal(0, 1, n) + shift * y).astype(np.float32), y))
    return out


def test_reservoir_is_invariant_to_the_brick_partitioning():
    """Unequal occupancy + non-perfect scores: AP/AUROC of the sample follow the pooled population,
    whatever the brick sizes are (an equal quota per brick would not)."""
    rng = np.random.default_rng(12345)  # not the reservoir's own seed: independent label draws
    blocks = _blocks(rng, [100_000, 200, 5_000, 50], [0.01, 0.5, 0.02, 0.8])
    pool_s = np.concatenate([b[0] for b in blocks])
    pool_y = np.concatenate([b[1] for b in blocks])
    exact_ap = eval_region.average_precision(pool_s, pool_y)
    exact_auc = eval_region.auroc(pool_s, pool_y)

    def sampled(chunk: int | None):
        r = eval_region.Reservoir(cap=20_000, seed=0)
        if chunk is None:  # one add per (unequal) brick
            for s, y in blocks:
                r.add(s, y, len(blocks))
        else:              # the same voxels cut into equal bricks instead
            for i in range(0, pool_s.size, chunk):
                r.add(pool_s[i:i + chunk], pool_y[i:i + chunk], -(-pool_s.size // chunk))
        return r

    for chunk in (None, 1000, 7919, pool_s.size):
        r = sampled(chunk)
        s, y = r.arrays()
        assert s.size == 20_000 and r.n == pool_s.size
        assert y.mean() == pytest.approx(pool_y.mean(), abs=0.01)   # pooled prevalence
        assert r.auprc() == pytest.approx(exact_ap, abs=0.05)
        assert r.auroc() == pytest.approx(exact_auc, abs=0.02)
    # the old per-brick quota would have given the 200-voxel 50%-positive brick equal weight
    assert pool_y.mean() < 0.05


def test_reservoir_keeps_everything_below_the_cap():
    r = eval_region.Reservoir(cap=10, seed=0)
    r.add(np.zeros(0, np.float32), np.zeros(0, bool), 3)
    assert r.arrays()[0].size == 0 and r.auprc() is None and r.auroc() is None
    r.add(np.array([0.2, 0.8], np.float32), np.array([False, True]), 3)
    assert r.arrays()[0].size == 2 and r.n == 2 and r.auprc() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# R21: clipped faces cannot place a medial surface
# --------------------------------------------------------------------------- #
def _face_pair(x_in: float, x_out: float, clip: float, n: int = 100):
    from tsm.labels import encode_sdf_u8

    x = np.arange(n, dtype=np.float32)[None, None, :] * np.ones((1, 3, 1), np.float32)
    return encode_sdf_u8(x - x_in, clip), encode_sdf_u8(x - x_out, clip)


@pytest.mark.parametrize("thickness", [10.0, 30.0])
def test_medial_from_faces_is_exact_below_twice_the_clip(thickness):
    clip = 20.0
    x_in, x_out = 50.0 - thickness / 2, 50.0 + thickness / 2
    zc = eval_region.medial_zc_from_faces(*_face_pair(x_in, x_out, clip), clip)
    idx = np.flatnonzero(zc[0, 1])
    assert idx.tolist() == [50], (thickness, idx)


@pytest.mark.parametrize("thickness", [40.0, 60.0])
def test_doubly_clipped_faces_report_no_medial_surface(thickness):
    """At and above 2*clip the sum of the two face SDFs is a plateau: reporting its edge as the
    medial surface measured a clipping artefact (the review's 10/70 case returned x=50, not 40)."""
    clip = 20.0
    x_in, x_out = 50.0 - thickness / 2, 50.0 + thickness / 2
    si, so = _face_pair(x_in, x_out, clip)
    zc = eval_region.medial_zc_from_faces(si, so, clip)
    assert not zc.any(), np.flatnonzero(zc[0, 1]).tolist()
    _, usable = eval_region.medial_sdf_from_faces(si, so, clip)
    sat = eval_region.face_saturated(si, so)
    interior = (np.arange(100)[None, None, :] * np.ones((1, 3, 1)) >= x_in) & \
               (np.arange(100)[None, None, :] * np.ones((1, 3, 1)) <= x_out)
    assert sat[interior].any() and not (usable & sat).any()
    assert eval_region.face_saturated(*_face_pair(45.0, 55.0, clip)).sum() >= 0


def test_medial_coverage_frac_is_reported(tmp_path):
    case = _make_faces_case(str(tmp_path / "out"))
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    cov = m["surface"]["medial_coverage_frac"]
    assert cov is None or 0.0 <= cov <= 1.0
    assert m["surface"]["medial_coverage_n"] >= 0


# --------------------------------------------------------------------------- #
# teacher-independent fibre metrics (fiber.upstream, 2026-09-07)
# --------------------------------------------------------------------------- #
def test_fiber_upstream_pass_scores_the_human_bands(tmp_path):
    """A perfect direction prediction on a plane sheet: class accuracy 1, angular error 0."""
    from dev.eval_region import Store, fiber_upstream_pass
    from tsm.data import encode_signed
    from tsm.fiber import derive_direction_targets
    from tsm.infer import pred_channels

    P = 16
    root = str(tmp_path)
    n0 = np.array([0.0, 1.0, 0.0], np.float32)  # sheet normal along y: vt is along z, hz along x
    g = np.stack(np.meshgrid(*[np.arange(P, dtype=np.float32) - (P - 1) / 2.0] * 3, indexing="ij"))
    sdf = (g * n0.reshape(3, 1, 1, 1)).sum(0)[None]
    ones = np.ones((1, P, P, P), np.float32)
    # the human bands: hz in the first half of z, vt in the second
    band = np.zeros((1, P, P, P), np.uint8)
    band[:, : P // 2] = 1
    band[:, P // 2:] = 2
    t = derive_direction_targets(ones * 0.0, ones * 0.0, sdf, ones, band=band.astype(np.float32))
    ch = pred_channels("faces", True, "direction")
    fields = {c: np.zeros((P, P, P), np.uint8) for c in ch}
    fields["sdf_in"] = encode_sdf(sdf[0])
    fields["sdf_out"] = encode_sdf(sdf[0] - 4.0)
    for i, name in enumerate(("fiber_dz", "fiber_dy", "fiber_dx")):
        fields[name] = encode_signed(t["fiber_dir"][i])
    fields["fiber_strength"] = np.full((P, P, P), 255, np.uint8)
    pred_path = _write(os.path.join(root, "pred.zarr"), ch, [fields[c] for c in ch], (0, 0, 0), chunk=P)
    from tsm.infer import write_fiber_class_channels

    write_fiber_class_channels(pred_path, CLIP, brick=P)
    band_path = _write(os.path.join(root, "band.zarr"), ["rectoverso", "hzvt"],
                       [np.zeros((P, P, P), np.uint8), band[0]], (0, 0, 0), chunk=P)
    from tsm.config import load_config

    cfgp = os.path.join(root, "c.json")
    json.dump({"volume": {"url": "x", "level": 0, "voxel_um": 2.4},
               "region": {"start_zyx": [0, 0, 0], "size_zyx": [P, P, P]},
               "budget": {"ram_bytes": 1 << 30, "array_bytes": 1 << 29}, "out_dir": root, "extra": {}},
              open(cfgp, "w"))
    budget = load_config(cfgp).budget
    m = fiber_upstream_pass(Store(pred_path), Store(band_path), "hzvt", [0, 0, 0], [P, P, P], CLIP, budget,
                            brick=(P, P, P))
    assert m["teacher_independent"] and m["n_band"] > 0
    assert m["class_acc"] == pytest.approx(1.0)
    assert m["angle_deg_median"] < 1.0  # only the u8 quantisation of the direction channels


# --------------------------------------------------------------------------- #
# topology vs the upstream bands (upstream_topology, 2026-09-07)
# --------------------------------------------------------------------------- #
TOPO_SHAPE = (32, 24, 24)
TOPO_ZS = (8, 16, 24)
TOPO_MIN = 200          # the plane / band sizes here are ~576 / ~1728 voxels, not the 2000 of a
                        # real 1024^2 eval-region sheet


def _sheets(zs, shape=TOPO_SHAPE):
    """The thinned centre planes: one voxel thick."""
    a = np.zeros(shape, bool)
    for z in zs:
        a[z] = True
    return a


def _band3(zs, shape=TOPO_SHAPE):
    """The reference band as registered: 3 voxels thick around each centre plane."""
    a = np.zeros(shape, bool)
    for z in zs:
        a[z - 1:z + 2] = True
    return a


def _near_of(band, near=8.0):
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(~band) <= near


def _topo(stud, zs=TOPO_ZS, band=None, thin=None, **kw):
    """face_topology with the unthinned 3-voxel band as the reference, its centre planes as `thin`."""
    band = _band3(zs) if band is None else band
    thin = _sheets(zs) if thin is None else thin
    kw.setdefault("min_size", TOPO_MIN)
    return eval_region.face_topology(band, stud, _near_of(band), thin=thin, **kw)


def test_topology_perfect_prediction_has_no_defects():
    r = _topo(_sheets(TOPO_ZS))
    assert r["n_ref_objects"] == 3 and r["n_ref_objects_all"] == 3
    assert r["n_ref_thin_fragments"] == 3 and r["n_student_components"] == 3
    assert r["n_ref_voxels"] == 3 * 3 * 24 * 24 and r["n_band_voxels"] == 3 * 24 * 24
    assert not r["contact_split"]
    assert (r["merges"], r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0, 0)
    assert r["merged_extra"] == 0 and r["split_extra"] == 0
    assert r["missed_frac"] == 0.0 and r["merges_per_100"] == 0.0
    assert r["dice_within2"] == pytest.approx(1.0)   # measured on the thinned planes


def test_topology_bridge_between_two_sheets_is_one_merge():
    """One prediction spanning two labelled sheets: a merge, and no break/missed/spurious."""
    stud = _sheets(TOPO_ZS)
    stud[8:17, 0, 0] = True                      # weld the z=8 and z=16 sheets together
    r = _topo(stud)
    assert r["n_student_components"] == 2
    assert r["merges"] == 1 and r["merged_extra"] == 1
    assert (r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0)
    assert r["merges_per_100"] == pytest.approx(100.0 / 3)


def test_topology_split_sheet_is_one_break():
    """One labelled sheet covered by two disjoint predictions: a break, not a merge."""
    stud = _sheets([8, 24])
    stud[16, :11] = True                         # y = 12 left empty -> two 26-components
    stud[16, 13:] = True
    r = _topo(stud)
    assert r["n_student_components"] == 4
    assert r["breaks"] == 1 and r["split_extra"] == 1
    assert (r["merges"], r["missed"], r["spurious"]) == (0, 0, 0)
    assert r["breaks_per_100"] == pytest.approx(100.0 / 3)


def test_topology_absent_sheet_is_one_missed():
    r = _topo(_sheets([8, 24]))
    assert r["missed"] == 1 and r["missed_frac"] == pytest.approx(1.0 / 3)
    assert r["missed_per_100"] == pytest.approx(100.0 / 3)
    assert (r["merges"], r["breaks"], r["spurious"]) == (0, 0, 0)


def test_topology_floating_extra_sheet_is_one_spurious():
    """A predicted sheet 3 voxels off any band voxel: no band voxel is attributed to it."""
    r = _topo(_sheets([8, 12, 16, 24]))
    assert r["spurious"] == 1 and r["spurious_per_100"] == pytest.approx(100.0 / 3)
    assert (r["merges"], r["breaks"], r["missed"]) == (0, 0, 0)


def test_topology_ignores_specks_and_faraway_predictions():
    stud = _sheets(TOPO_ZS)
    stud[12, 0, 0] = True                        # a 1-voxel speck: below min_size
    r = _topo(stud)
    assert r["spurious"] == 0 and r["n_student_components"] == 4
    assert r["n_student_components_min_size"] == 3


def test_topology_reference_is_the_unthinned_band_not_its_shattered_thinning():
    """A band whose *thinning* falls apart is still ONE reference object: 0 merges, 0 missed.

    The 2026-09-07 eval run measured 108561 thinned in-face fragments against 3080 student
    components on the Paris 4 region: with the thinned planes as the reference, one honest
    student sheet "merges" hundreds of fragments and most of them are "missed".  Here the same
    situation in miniature -- a solid 3-voxel band with two rows of its centre plane missing, so
    `thin_band` returns seven pieces while the band itself is one 26-component.
    """
    from tsm.rvfaces import thin_band

    band = np.zeros(TOPO_SHAPE, bool)
    band[7:10] = True
    band[8, 8, :] = False                        # holes in the centre plane only: the band stays
    band[8, 16, :] = False                       # 26-connected through the z = 7 / z = 9 layers
    thin = thin_band(band, 1)
    stud = _sheets([8])                          # one honest prediction of that one sheet
    near = _near_of(band)

    r = _topo(stud, band=band, thin=thin, min_size=100)   # the thinned pieces are 144-192 voxels
    assert r["n_ref_objects"] == 1 and r["n_ref_objects_all"] == 1
    assert r["n_ref_thin_fragments"] == 7        # what the old, thinned reference counted as sheets
    assert (r["merges"], r["merged_extra"], r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0, 0, 0)

    old = eval_region.face_topology(thin, stud, near, min_size=100)
    assert old["merges"] == 1 and old["merged_extra"] == 2   # the metric this replaces


def test_topology_contact_plane_splits_two_sheets_into_two_objects():
    """`core` (the band minus the label contact voxels) keeps two sheets that touch apart.

    Upstream class 3 is *contact*: the recto of one sheet against the verso of the next, so it
    belongs to both bands and would weld them into a single reference object -- and then two
    correct predictions, one per sheet, count as a break of that phantom object.
    """
    band = np.zeros(TOPO_SHAPE, bool)
    band[8:11, :12] = True                       # sheet A, 3 voxels thick
    band[8:11, 13:] = True                       # sheet B
    contact = np.zeros(TOPO_SHAPE, bool)
    contact[8:11, 12] = True                     # the class-3 strip joining the two bands
    band |= contact
    stud = np.zeros(TOPO_SHAPE, bool)
    stud[9, :12] = True                          # one honest prediction per sheet
    stud[9, 13:] = True
    thin = np.zeros(TOPO_SHAPE, bool)            # the centre plane of the band, contact included
    thin[9] = band[9]
    kw = dict(near_mask=_near_of(band), thin=thin, min_size=TOPO_MIN)

    welded = eval_region.face_topology(band, stud, **kw)
    assert welded["n_ref_objects"] == 1 and not welded["contact_split"]
    assert welded["breaks"] == 1                 # one phantom "sheet" split across two predictions

    r = eval_region.face_topology(band, stud, core=band & ~contact, **kw)
    assert r["contact_split"] and r["n_ref_objects"] == 2 and r["n_ref_objects_all"] == 2
    assert (r["merges"], r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0, 0)


def test_topology_a_band_of_contact_only_is_still_one_object():
    """A band component made entirely of contact voxels is not silently dropped by the split."""
    band = np.zeros(TOPO_SHAPE, bool)
    band[8:11] = True
    contact = band.copy()                        # the whole band is class 3
    r = eval_region.face_topology(band, _sheets([9]), _near_of(band), thin=_sheets([9]),
                                  core=band & ~contact, min_size=TOPO_MIN)
    assert r["n_ref_objects"] == 1 and r["contact_split"]
    assert (r["merges"], r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0, 0)


def test_inout_cross_links_counts_in_faces_touching_two_out_faces():
    s_out = _sheets([10, 12])
    s_in = _sheets([11])
    r = eval_region.inout_cross_links(s_in, s_out)
    assert r["n_in_components"] == 1 and r["n_out_components"] == 2
    assert r["cross_links"] == 1 and r["cross_links_per_100"] == pytest.approx(100.0)
    assert eval_region.inout_cross_links(_sheets([11]), _sheets([13]))["cross_links"] == 0


def _rv_pred_stores(root):
    """A 2-sheet recto/verso slab (bands 3 voxels thick) and a student sitting on the thinned bands."""
    Z, Y, X = 48, 32, 32
    rv = np.zeros((Z, Y, X), np.uint8)
    for z0 in (10, 26):
        rv[z0:z0 + 3] = 1                        # recto -> in face
    for z0 in (18, 34):
        rv[z0:z0 + 3] = 2                        # verso -> out face
    rv_path = _write(os.path.join(root, "rv.zarr"), ["rectoverso", "hzvt"],
                     [rv, np.zeros_like(rv)], (0, 0, 0))
    s_in = np.zeros((Z, Y, X), np.uint8)
    s_out = np.zeros((Z, Y, X), np.uint8)
    s_in[[11, 27]] = 255
    s_out[[19, 35]] = 255
    ch = ["sdf_in", "sdf_out", "surface_in1", "surface_out1"]
    flat = np.full((Z, Y, X), 128, np.uint8)
    pred_path = _write(os.path.join(root, "pred.zarr"), ch, [flat, flat.copy(), s_in, s_out], (0, 0, 0))
    return rv_path, pred_path, (Z, Y, X)


def test_upstream_topology_pass_on_a_clean_two_sheet_slab(tmp_path):
    from tsm.config import parse_config

    rv_path, pred_path, (Z, Y, X) = _rv_pred_stores(str(tmp_path))
    budget = parse_config({"volume": {"url": "x"},
                           "region": {"start_zyx": [0, 0, 0], "size_zyx": [Z, Y, X]},
                           "out_dir": str(tmp_path)}).budget
    m = eval_region.upstream_topology_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], [Z, Y, X], budget, brick=(16, 32, 32), min_component=8, min_size=512)
    assert m["teacher_independent"] is True and m["connectivity"] == 26
    assert m["min_size"] == 512 and m["contact_split"] is False   # no class-3 voxel in this slab
    for face in ("in", "out"):
        r = m[face]
        # 2 sheets, each a 3-voxel thick band (3 * 32 * 32 voxels) thinned to one centre plane
        assert r["n_ref_objects"] == 2 and r["n_ref_objects_all"] == 2
        assert r["n_ref_thin_fragments"] == 2 and r["n_ref_voxels"] == 2 * 3 * 32 * 32
        assert r["n_student_components"] == 2 and not r["contact_split"]
        assert (r["merges"], r["breaks"], r["missed"], r["spurious"]) == (0, 0, 0, 0)
        assert r["dice_within2"] == pytest.approx(1.0)
    assert m["inout_cross_links"]["cross_links"] == 0
    md = eval_region.to_markdown({"region": {"start_zyx": [0, 0, 0], "size_zyx": [Z, Y, X]},
                                  "clip": 20.0, "seconds": 1.0, "upstream_topology": m})
    assert "## Topology vs upstream bands" in md
    assert "reference objects = 26-components of the UNTHINNED upstream band" in md
    assert "no contact (class 3) voxels in this region" in md
    assert "| n_ref_objects |" in md and "| n_ref_thin_fragments |" in md
    assert "| merges |" in md and "inout_cross_links.cross_links" in md
