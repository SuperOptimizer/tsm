"""Proposal logic of dev/survey_regions.py on a synthetic occupancy profile (no network)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "survey_regions", os.path.join(ROOT, "dev", "survey_regions.py"))
sr = importlib.util.module_from_spec(_spec)
sys.modules["survey_regions"] = sr
_spec.loader.exec_module(sr)

SCALE = 32
NZ5 = 2369          # a Paris-4-shaped survey level
DEPTH = 256
XY = 8192


def fake_survey(occ_lo: int = 300, occ_hi: int = 2000, cell: int = 4,
                ny5: int = 1022, nx5: int = 1022) -> sr.Survey:
    """Occupancy is a bump over [occ_lo, occ_hi) survey-z; papyrus sits off-centre in xy."""
    occ = np.zeros(NZ5, np.float32)
    t = np.linspace(0, np.pi, occ_hi - occ_lo)
    occ[occ_lo:occ_hi] = (0.02 + 0.25 * np.sin(t)).astype(np.float32)
    cy, cx = (ny5 + cell - 1) // cell, (nx5 + cell - 1) // cell
    counts = np.zeros((NZ5, cy, cx), np.uint8)
    # a disc of papyrus centred 40 cells right of / 20 cells below the umbilicus
    umb_c = (cy // 2, cx // 2)
    yy, xx = np.mgrid[:cy, :cx]
    disc = ((yy - (umb_c[0] + 20)) ** 2 + (xx - (umb_c[1] + 40)) ** 2) < 60 ** 2
    counts[occ_lo:occ_hi] = (disc * (cell * cell)).astype(np.uint8)
    bbox = np.full((NZ5, 4), -1, np.int32)
    bbox[occ_lo:occ_hi] = (200, 800, 200, 800)
    umb = np.zeros((NZ5, 2), np.float32)
    umb[:, 0] = umb_c[0] * cell
    umb[:, 1] = umb_c[1] * cell
    return sr.Survey(level=5, scale=SCALE, shape0=(NZ5 * SCALE, ny5 * SCALE, nx5 * SCALE),
                     shape5=(NZ5, ny5, nx5), occ=occ, bbox=bbox, umb=umb, cell=cell,
                     counts=counts, threshold=60, sigma=1.0)


def test_exclusion_intervals_merge_and_pad():
    ex = sr.exclusion_intervals(DEPTH)
    assert ex == [(sr.EXISTING_SLAB_Z[0] - 2 * DEPTH, sr.EVAL_Z[1] + 2 * DEPTH)]
    ex2 = sr.exclusion_intervals(DEPTH, keep_clear=0, zones=((0, 100), (5000, 5100)))
    assert ex2 == [(0, 100), (5000, 5100)]


def test_proposals_are_aligned_spread_and_clear_of_the_exclusion():
    sv = fake_survey()
    props = sr.propose(sv, k=8, depth=DEPTH, xy=XY, floor=0.05)
    assert len(props) == 8
    excl = sr.exclusion_intervals(DEPTH)
    zs = [p["start_zyx"][0] for p in props]
    assert zs == sorted(zs)
    for p in props:
        z, y, x = p["start_zyx"]
        assert [v % 64 for v in p["start_zyx"]] == [0, 0, 0]
        assert [v % 64 for v in p["size_zyx"]] == [0, 0, 0]
        assert [v % 4 for v in p["start_zyx"] + p["size_zyx"]] == [0] * 6  # level-2 grid
        # inside the occupied band and clear of the exclusion
        assert p["slice_occupancy"] >= 0.05
        for lo, hi in excl:
            assert z >= hi or z + DEPTH <= lo
        assert z + DEPTH <= sv.shape0[0]
        assert y + XY <= sv.shape0[1] and x + XY <= sv.shape0[2]
    # spread: slabs are at least half a bin apart (not bunched at the bin edges)
    span = zs[-1] - zs[0]
    assert all(b - a >= min(DEPTH * 4, span // 16) for a, b in zip(zs, zs[1:]))
    assert all(b - a >= DEPTH for a, b in zip(zs, zs[1:]))
    assert max(zs) - min(zs) > 0.5 * (2000 - 300) * SCALE


def test_exclusion_zone_is_respected_even_when_it_is_the_best_z():
    """A spike of occupancy inside the excluded band must not be proposed."""
    sv = fake_survey()
    lo, hi = sr.EXISTING_SLAB_Z[0] // SCALE, sr.EVAL_Z[1] // SCALE
    sv.occ[lo - 20:hi + 20] = 0.9
    props = sr.propose(sv, k=4, depth=DEPTH, xy=XY, floor=0.05)
    for p in props:
        for a, b in sr.exclusion_intervals(DEPTH):
            assert p["start_zyx"][0] >= b or p["start_zyx"][0] + DEPTH <= a


def test_floor_excludes_empty_z():
    sv = fake_survey(occ_lo=1000, occ_hi=1100)
    props = sr.propose(sv, k=8, depth=DEPTH, xy=XY, floor=0.05)
    assert props, "expected at least one proposal in the occupied band"
    for p in props:
        z5 = p["start_zyx"][0] // SCALE
        assert 990 <= z5 <= 1105


def test_xy_window_shifts_toward_the_papyrus_but_keeps_the_umbilicus():
    sv = fake_survey()
    props = sr.propose(sv, k=3, depth=DEPTH, xy=XY, floor=0.05)
    for p in props:
        dy, dx = p["shift_yx_l0"]
        assert dx > 0 and dy > 0, "window should move toward the off-centre disc"
        assert abs(dy) <= XY // 2 and abs(dx) <= XY // 2
        uy, ux = p["umbilicus_yx_l0"]
        y, x = p["start_zyx"][1], p["start_zyx"][2]
        assert y <= uy <= y + XY and x <= ux <= x + XY
    # with max_shift_frac 0 the window stays umbilicus-centred
    p0 = sr.propose(sv, k=1, depth=DEPTH, xy=XY, floor=0.05, max_shift_frac=0.0)[0]
    assert p0["shift_yx_l0"] == [0, 0]


def test_coverage_is_reported_for_an_oversized_cross_section():
    sv = fake_survey()
    p = sr.propose(sv, k=1, depth=DEPTH, xy=XY, floor=0.05)[0]
    assert 0.0 < p["cross_section_coverage"] < 1.0  # 8192 cannot hold the whole disc
    big = sr.propose(sv, k=1, depth=DEPTH, xy=30 * 1024, floor=0.05)[0]
    assert big["cross_section_coverage"] > p["cross_section_coverage"]


def test_label_size_estimate_scales_with_volume():
    sv = fake_survey()
    a = sr.propose(sv, k=1, depth=256, xy=6144, floor=0.05)[0]
    b = sr.propose(sv, k=1, depth=256, xy=8192, floor=0.05)[0]
    assert a["label_gb"] == pytest.approx(40.0, abs=0.2)
    assert b["label_gb"] == pytest.approx(40.0 * (8192 / 6144) ** 2, rel=0.01)


def test_best_xy_start_without_counts_is_umbilicus_centred_and_clamped():
    start, shift, _ = sr.best_xy_start(None, 128, (5000.0, 5000.0), XY, (32768, 32768))
    assert shift == (0, 0) and start == (5000 - XY // 2 - (5000 - XY // 2) % 64,) * 2
    # near the volume edge the window is clamped inside
    start, _, _ = sr.best_xy_start(None, 128, (100.0, 32000.0), XY, (32768, 32768))
    assert start[0] == 0 and start[1] + XY <= 32768


def test_write_configs_clones_the_base_config(tmp_path):
    base = json.load(open(os.path.join(ROOT, "configs", "paris4_faces_rvonly.json")))
    bp = tmp_path / "base.json"
    bp.write_text(json.dumps(base))
    props = [{"k": 3, "start_zyx": [1024, 2048, 3072], "size_zyx": [256, 8192, 8192]}]
    paths = sr.write_configs(str(bp), props, "/root/regions", str(tmp_path / "cfg"), log=lambda *_: None)
    cfg = json.load(open(paths[0]))
    assert paths[0].endswith("paris4_r3.json")
    assert cfg["region"] == {"start_zyx": [1024, 2048, 3072], "size_zyx": [256, 8192, 8192]}
    assert cfg["out_dir"] == "/root/regions/paris4_r3"
    assert cfg["extra"]["labels"]["teachers_dir"] == "/root/regions/paris4_r3/teachers"
    assert (cfg["extra"]["labels"]["faces"]["rectoverso_store"]
            == "/root/regions/paris4_r3/rectoverso/rectoverso.zarr")
    assert cfg["extra"]["train"]["fine_store"].startswith("/root/regions/paris4_r3/")
    assert cfg["extra"]["train"]["coarse_store"].startswith("/root/regions/paris4_r3/")
    # everything else is untouched
    assert cfg["volume"] == base["volume"]
    assert cfg["extra"]["teacher"] == base["extra"]["teacher"]
