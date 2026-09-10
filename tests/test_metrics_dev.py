"""Guards on the dev diagnostics: experiment manifests before cache reuse (R22) and the
rotation valid-support mask in the side / equivariance metrics (R23).  CPU only, no models."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

import tsm.equivariance as E  # noqa: E402

OB = pytest.importorskip("orientation_bias")
tta_side = pytest.importorskip("tta_side")
bench_student = pytest.importorskip("bench_student")


# --------------------------------------------------------------------------- #
# R22: never reuse a cached prediction from a different experiment
# --------------------------------------------------------------------------- #
def test_manifest_diff_reports_every_changed_property():
    want = {"a": 1, "b": [1, 2], "c": {"x": 1.0}}
    assert OB.manifest_diff(want, dict(want)) == []
    assert OB.manifest_diff(want, {**want, "extra": 9}) == []          # extra keys are ignored
    assert OB.manifest_diff(want, {**want, "a": 2}) == ["a"]
    assert OB.manifest_diff(want, {**want, "c": {"x": 1.5}}) == ["c"]
    assert OB.manifest_diff(want, {"a": 1}) == ["b", "c"]              # missing keys differ
    assert OB.manifest_diff(want, None) == ["a", "b", "c"]             # no manifest at all
    assert bench_student.manifest_diff(want, dict(want)) == []
    assert bench_student.manifest_diff(want, None) == ["a", "b", "c"]


def test_file_fingerprint_changes_when_a_checkpoint_is_rewritten(tmp_path):
    p = tmp_path / "latest.pt"
    p.write_bytes(b"a" * 10)
    fp = OB.file_fingerprint(str(p))
    assert fp and fp == OB.file_fingerprint(str(p)) == bench_student.fingerprint(str(p))
    os.utime(p, ns=(0, 0))
    p.write_bytes(b"b" * 20)  # same path, different content: the fingerprint must move
    assert OB.file_fingerprint(str(p)) != fp
    assert OB.file_fingerprint(str(tmp_path / "missing.pt")) is None
    assert OB.file_fingerprint(None) is None and bench_student.fingerprint(None) == "none"


def _args(**kw):
    base = dict(checkpoint=None, config="configs/paris4.json", clip=20.0, dtype="fp32", box_npy=None,
                group="flip8_rot4", random_rotations=0, max_deg=0.0, seed=7)
    base.update(kw)
    return Namespace(**base)


def _box_info(start=(0, 0, 0)):
    return {"level": 1, "factor": 2, "start_zyx": list(start), "size": 128, "voxel_um": 4.8,
            "shape": [128, 128, 128], "cached": True, "read_s": 1.2}


def _tf(name="flip0_rot1", matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    return [{"name": name, "matrix": [list(r) for r in matrix]}]


def test_orientation_manifest_rejects_a_changed_input(tmp_path):
    ck1, ck2 = tmp_path / "v1.pt", tmp_path / "v2.pt"
    ck1.write_bytes(b"1" * 8)
    ck2.write_bytes(b"2" * 16)
    base = OB.experiment_manifest(_args(checkpoint=str(ck1)), "student", "student", 0, _box_info(), _tf())
    same = OB.experiment_manifest(_args(checkpoint=str(ck1)), "student", "student", 0, _box_info(), _tf())
    assert OB.manifest_diff(base, same) == []
    # volatile box fields (cached / read_s) must not force a recompute
    assert OB.manifest_diff(base, OB.experiment_manifest(
        _args(checkpoint=str(ck1)), "student", "student", 0, {**_box_info(), "cached": False, "read_s": 9.9}, _tf())) == []
    for label, other in {
        "checkpoint": OB.experiment_manifest(_args(checkpoint=str(ck2)), "student", "student", 0, _box_info(), _tf()),
        "offset": OB.experiment_manifest(_args(checkpoint=str(ck1)), "student", "student", 0, _box_info((512, 0, 0)), _tf()),
        "level_shift": OB.experiment_manifest(_args(checkpoint=str(ck1)), "student", "student", 2, _box_info(), _tf()),
        "dtype": OB.experiment_manifest(_args(checkpoint=str(ck1), dtype="bf16"), "student", "student", 0, _box_info(), _tf()),
        "teacher": OB.experiment_manifest(_args(checkpoint=str(ck1)), "recto", "surface", 0, _box_info(), _tf()),
        "matrix": OB.experiment_manifest(_args(checkpoint=str(ck1)), "student", "student", 0, _box_info(),
                                         _tf(matrix=((-1, 0, 0), (0, 1, 0), (0, 0, 1)))),
    }.items():
        assert OB.manifest_diff(base, other), label
    # a retrained checkpoint at the same path is caught by the fingerprint
    os.utime(ck1, ns=(0, 0))
    ck1.write_bytes(b"3" * 8)
    assert OB.manifest_diff(base, OB.experiment_manifest(
        _args(checkpoint=str(ck1)), "student", "student", 0, _box_info(), _tf())) == ["checkpoint_fingerprint"]


def test_manifest_round_trip(tmp_path):
    p = str(tmp_path / "preds" / "manifest.json")
    assert OB.read_manifest(p) is None
    m = OB.experiment_manifest(_args(), "recto", "surface", 0, _box_info(), _tf())
    OB.write_manifest(p, m)
    assert OB.manifest_diff(m, OB.read_manifest(p)) == []
    open(p, "w").write("{not json")
    assert OB.read_manifest(p) is None and OB.manifest_diff(m, OB.read_manifest(p))


def test_tta_side_manifest_covers_the_faces_store_and_transforms(tmp_path):
    store = tmp_path / "faces.npz"
    store.write_bytes(b"x" * 32)
    args = Namespace(teacher="recto", config="configs/paris4_small.json", dtype=None,
                     models_dir=str(tmp_path), faces_store=str(store), group="oct48", rotations=0,
                     max_degs="30,90", seed=7)
    fw = {"origin_zyx": [0, 0, 0], "shape": [64, 64, 64]}
    a = tta_side.side_manifest(args, "recto", 0, _box_info(), fw, _tf())
    assert tta_side.side_manifest(args, "recto", 0, _box_info(), fw, _tf()) == a
    b = tta_side.side_manifest(args, "recto", 0, _box_info((16, 0, 0)), fw, _tf())
    assert "box" in OB.manifest_diff(a, b)
    os.utime(store, ns=(0, 0))
    store.write_bytes(b"y" * 64)  # the faces store was rebuilt: the cache is not reusable
    assert OB.manifest_diff(a, tta_side.side_manifest(args, "recto", 0, _box_info(), fw, _tf())) == ["faces_fingerprint"]
    c = tta_side.side_manifest(args, "recto", 0, _box_info(), fw, _tf(matrix=((0, 1, 0), (1, 0, 0), (0, 0, 1))))
    assert "transforms" in OB.manifest_diff(a, c)


def test_bench_student_output_cache_is_keyed_by_the_checkpoint(tmp_path):
    ck = tmp_path / "latest.pt"
    ck.write_bytes(b"a" * 8)
    fp1 = bench_student.fingerprint(str(ck))
    p1 = bench_student._out_path((64, 64, 64), "trt_p128_ov", fp1)
    assert fp1 in os.path.basename(p1)
    os.utime(ck, ns=(0, 0))
    ck.write_bytes(b"b" * 16)
    p2 = bench_student._out_path((64, 64, 64), "trt_p128_ov", bench_student.fingerprint(str(ck)))
    assert p1 != p2  # a retrained checkpoint cannot silently reuse the old outputs
    assert bench_student.compare((64, 64, 64), "a", "b", 20.0, fp1) == {}  # nothing cached -> no numbers


def test_bench_student_box_manifest_detects_another_volume(tmp_path):
    want = {"url": "s3://a", "level": 0}
    path = str(tmp_path / "box.npy.json")
    with open(path, "w") as fh:
        json.dump(want, fh)
    have = json.load(open(path))
    assert bench_student.manifest_diff(want, have) == []
    assert bench_student.manifest_diff({**want, "url": "s3://b"}, have) == ["url"]


# --------------------------------------------------------------------------- #
# R23: the rotation valid-support mask is used by the metrics
# --------------------------------------------------------------------------- #
def test_equivariance_metrics_ignore_differences_outside_the_rotation_support():
    rng = np.random.default_rng(0)
    shape = (48, 48, 48)
    R = E.random_rotation_matrix(rng, 25.0)
    mask = np.asarray(E.rotation_valid_mask(shape, R, margin=3), bool)
    assert 0.2 < mask.mean() < 1.0 and (~mask).any()
    ident = rng.random(shape).astype(np.float32)
    p = ident.copy()
    p[~mask] = 1.0 - p[~mask]  # equivariant everywhere it was actually predicted
    m_masked = E.metrics(ident, p, "surface", mask)
    m_none = E.metrics(ident, p, "surface", None)
    assert m_masked["dice"] == pytest.approx(1.0) and m_masked["mean_abs"] == pytest.approx(0.0)
    assert m_masked["max_abs"] == pytest.approx(0.0) and m_masked["medial"]["n_a"] == m_masked["medial"]["n_b"]
    assert m_none["dice"] < 1.0 and m_none["mean_abs"] > 0  # unmasked: interpolation scored as bias


def test_side_metrics_apply_the_support_mask():
    faces, prob = _slab()
    support = np.ones_like(prob, bool)
    support[:, :, :24] = False  # the right half of x is outside the rotated support
    bad = prob.copy()
    bad[~support] = 0.0         # only differs outside the support
    good = tta_side.side_metrics(prob, faces)
    assert tta_side.side_metrics(bad, faces)["in_cov1"] < good["in_cov1"]  # unmasked: looks broken
    m = tta_side.side_metrics(bad, faces, support=support)
    assert m["in_cov1"] == pytest.approx(1.0) and m["side1"] == pytest.approx(1.0)
    assert m["support_frac"] == pytest.approx(support.mean())
    assert m["n_face_in"] == int((faces["face_in"] & support).sum())
    # identical numbers whether the out-of-support voxels are wrong or right
    assert m == tta_side.side_metrics(prob, faces, support=support)


def test_mask_on_window_matches_pred_on_window():
    win = {"factor": 2, "level0_lo": [4, 4, 4], "level0_hi": [12, 12, 12], "box_b0": [0, 0, 0]}
    mask = np.zeros((8, 8, 8), bool)
    mask[2:6, 2:6, 2:6] = True
    out = tta_side.mask_on_window(mask, win)
    assert out.shape == (8, 8, 8) and out.dtype == bool
    assert out.all()  # the window sits entirely inside the True block after the x2 upsampling
    assert not tta_side.mask_on_window(np.zeros((8, 8, 8), bool), win).any()


def _slab(nz=16, ny=48, nx=48, y_in=16, y_out=28):
    yy = np.arange(ny)[None, :, None] * np.ones((nz, 1, nx))
    faces = {"sdf_in": (yy - y_in).astype(np.float32), "sdf_out": (yy - y_out).astype(np.float32),
             "face_in": np.zeros((nz, ny, nx), bool), "face_out": np.zeros((nz, ny, nx), bool),
             "supervise": np.ones((nz, ny, nx), bool)}
    faces["face_in"][:, y_in, :] = True
    faces["face_out"][:, y_out, :] = True
    prob = np.zeros((nz, ny, nx), np.float32)
    prob[:, y_in - 1:y_in + 2, :] = 1.0
    return faces, prob
