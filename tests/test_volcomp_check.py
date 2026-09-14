"""Guards on ``dev/volcomp_check.py``'s metric functions (CPU, synthetic arrays, no network,
no checkpoints): the lossless-vs-volcomp comparison is only worth reading if "identical
inputs give a perfect score" and "a known perturbation gives the score you can compute by
hand" both hold."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

VC = pytest.importorskip("volcomp_check")


def test_dice_is_one_for_identical_masks_and_zero_for_disjoint_ones():
    a = np.zeros((4, 4, 4), bool)
    a[:2] = True
    assert VC.dice(a, a) == 1.0
    assert VC.dice(a, ~a) == 0.0
    assert VC.dice(np.zeros_like(a), np.zeros_like(a)) == 1.0  # both empty: no disagreement
    b = a.copy()
    b[0] = False  # |a|=32, |b|=16, inter=16 -> 2*16/48
    assert VC.dice(a, b) == pytest.approx(2 * 16 / 48)
    with pytest.raises(ValueError):
        VC.dice(a, a[:2])


def test_delta_stats_on_a_known_difference():
    a = np.zeros((2, 4, 4), np.float32)
    b = a.copy()
    b[0, 0, 0] = 0.5
    st = VC.delta_stats(a, b)
    assert st["n"] == 32
    assert st["mean"] == pytest.approx(0.5 / 32)
    assert st["max"] == pytest.approx(0.5)
    assert st["rms"] == pytest.approx(np.sqrt(0.25 / 32))
    assert VC.delta_stats(a, a)["max"] == 0.0
    with pytest.raises(ValueError):
        VC.delta_stats(a, a[:1])


def test_delta_stats_mask_broadcasts_over_channels_and_tolerates_an_empty_mask():
    a = np.zeros((3, 2, 2, 2), np.float32)
    b = a + 1.0
    m = np.zeros((2, 2, 2), bool)
    m[0, 0, 0] = True
    st = VC.delta_stats(a, b, m)
    assert st["n"] == 3 and st["mean"] == pytest.approx(1.0)  # one voxel x 3 channels
    assert VC.delta_stats(a, b, np.zeros((2, 2, 2), bool))["n"] == 0


def test_argmax_change_counts_only_flipped_voxels():
    a = np.zeros((3, 8), np.float32)
    a[0] = 1.0
    b = a.copy()
    assert VC.argmax_change(a, b) == 0.0
    b[:, :2] = 0.0
    b[2, :2] = 1.0  # two of eight voxels now win on class 2
    assert VC.argmax_change(a, b) == pytest.approx(0.25)
    m = np.zeros(8, bool)
    m[0] = True
    assert VC.argmax_change(a, b, m) == 1.0
    assert VC.argmax_change(a, b, np.zeros(8, bool)) == 0.0
    with pytest.raises(ValueError):
        VC.argmax_change(a.ravel(), b.ravel())


def test_line_angle_is_modulo_pi_and_ignores_magnitude():
    n = np.zeros((3, 5), np.float64)
    n[0] = 1.0
    assert np.allclose(VC.line_angle_deg(n, n), 0.0)
    assert np.allclose(VC.line_angle_deg(n, -n), 0.0)      # a line field: sign is arbitrary
    assert np.allclose(VC.line_angle_deg(n, 7.0 * n), 0.0)  # magnitude is irrelevant
    m = np.zeros_like(n)
    m[1] = 1.0
    assert np.allclose(VC.line_angle_deg(n, m), 90.0)
    d = np.zeros_like(n)
    d[0] = d[1] = 1.0
    assert np.allclose(VC.line_angle_deg(n, d), 45.0)
    with pytest.raises(ValueError):
        VC.line_angle_deg(n, n[:2])


def test_angle_stats_summarises_and_masks():
    ang = np.array([0.0, 10.0, 20.0, 30.0])
    st = VC.angle_stats(ang)
    assert st["n"] == 4 and st["mean"] == pytest.approx(15.0) and st["max"] == 30.0
    assert VC.angle_stats(ang, np.array([True, False, False, False]))["max"] == 0.0
    assert VC.angle_stats(ang, np.zeros(4, bool))["n"] == 0


def test_face_distance_is_the_nearer_of_the_two_faces():
    sdf = np.stack([np.array([3.0, -1.0, 0.0]), np.array([-5.0, 4.0, 2.0])]).astype(np.float32)
    assert np.allclose(VC.face_dist_from_sdf(sdf), [3.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        VC.face_dist_from_sdf(sdf[:1])


def test_ct_stats_reports_an_infinite_psnr_for_identical_boxes():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (8, 8, 8), dtype=np.uint8)
    assert VC.ct_stats(a, a)["psnr_db"] == float("inf")
    b = a.astype(np.int16)
    b[0, 0, 0] = min(255, int(b[0, 0, 0]) + 51)
    st = VC.ct_stats(a, b.astype(np.uint8))
    assert st["max"] <= 51 and np.isfinite(st["psnr_db"])
    assert st["lossless_mean"] == pytest.approx(float(a.mean()))


def test_table_renders_a_row_per_model_without_touching_the_filesystem():
    res = [
        {"model": "recto", "fg": {"mean": 0.01, "p99": 0.1, "max": 0.9},
         "all_channels": {"mean": 0.01, "p99": 0.1, "max": 0.9},
         "dice@0.5": 0.99, "argmax_change": 0.004, "forward_s": [10.0, 11.0]},
        {"model": "student", "d_face": {"mean": 0.02, "p99": 0.2, "max": 1.5},
         "all_channels": {"mean": 0.01, "p99": 0.1, "max": 0.5},
         "zero_set_dice@1": 0.98, "body_dice": 0.99, "forward_s": [5.0, 5.0]},
    ]
    out = VC.table(res)
    assert "recto" in out and "student d_face" in out and "dice@0.5" in out
    assert len(out.splitlines()) == 6  # header, rule, recto + all-ch, student + all-heads


def test_triple_parses_one_or_three_ints():
    assert VC._triple("34432,15360,18688") == (34432, 15360, 18688)
    assert VC._triple("256") == (256, 256, 256)
    with pytest.raises(Exception):
        VC._triple("1,2")


def test_cuda_is_refused_without_the_local_gpu_override(monkeypatch):
    monkeypatch.delenv(VC.ALLOW_LOCAL_GPU_ENV, raising=False)
    with pytest.raises(SystemExit):
        VC.pick_device("cuda")
    assert VC.pick_device("cpu").type == "cpu"
