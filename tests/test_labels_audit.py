"""Synthetic checks for the audit primitives in tsm.labels (CPU, no data)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from tsm.labels import (
    axis_yx_at,
    decode_lasagna_normal,
    medial_thickness,
    neighbor_count6,
    radial_vectors,
    sdf_from_mask,
    surface_stats,
    thin_surface_from_sdf,
)


def _slab(shape=(24, 32, 32), z0=10, z1=15) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[z0:z1] = True  # 5 voxels thick along z
    return m


def test_sdf_sign_and_clip():
    m = _slab()
    sdf = sdf_from_mask(m, clip=4.0)
    assert sdf.dtype == np.float32
    assert (sdf[m] < 0).all() and (sdf[~m] > 0).all()
    assert sdf.min() >= -4.0 and sdf.max() <= 4.0
    # first voxel outside the slab is at distance 1, voxel just inside at -1, centre at -3 (clipped to -3)
    assert sdf[15, 5, 5] == pytest.approx(1.0)
    assert sdf[9, 5, 5] == pytest.approx(1.0)
    assert sdf[10, 5, 5] == pytest.approx(-1.0)
    assert sdf[12, 5, 5] == pytest.approx(-3.0)
    # unclipped: far field grows linearly
    raw = sdf_from_mask(m)
    assert raw[0, 5, 5] == pytest.approx(10.0)


def test_thin_surface_is_one_voxel_thick_and_6_connected():
    m = _slab()
    surf = thin_surface_from_sdf(sdf_from_mask(m))
    # both faces of the slab, each one voxel thick; box faces excluded
    assert surf[10].all() and surf[14].all()
    assert not surf[11:14].any() and not surf[:10].any() and not surf[15:].any()
    _, thick = medial_thickness(surf)
    assert np.allclose(thick, 1.0)
    # each face is a single 6-connected component; two faces -> two components
    n6 = ndi.label(surf, structure=ndi.generate_binary_structure(3, 1))[1]
    assert n6 == 2
    deg = neighbor_count6(surf)[surf]
    assert deg.min() >= 2 and (deg == 4).mean() > 0.8  # interior face voxels have 4 in-plane neighbours


def test_medial_thickness_of_5_voxel_slab():
    m = _slab()
    medial, thick = medial_thickness(m)
    assert medial.any() and (medial & ~m).sum() == 0
    assert medial[12].all() and not medial[10:12].any() and not medial[13:15].any()
    assert abs(np.median(thick) - 5.0) < 0.6
    # the medial surface is itself 1 voxel thick
    _, t2 = medial_thickness(medial)
    assert np.median(t2) == pytest.approx(1.0)
    # a tilted 5-voxel sheet: thickness still ~5 and the ridge stays a connected sheet
    zz, yy, xx = np.mgrid[0:40, 0:40, 0:40]
    tilted = np.abs(zz - 0.5 * yy - 8) <= 2.0 * np.sqrt(1.25)
    med_t, th_t = medial_thickness(tilted)
    assert 3.0 <= np.median(th_t) <= 6.0  # 2*EDT-1 under-reads tilted sheets by <= 1 voxel
    lab, n = ndi.label(med_t, structure=np.ones((3, 3, 3)))
    assert np.bincount(lab.ravel())[1:].max() >= 0.95 * med_t.sum()


def test_surface_stats_on_thin_sheet():
    m = np.zeros((16, 20, 20), dtype=bool)
    m[8] = True
    s = surface_stats(m, min_component=10)
    assert s["thickness"]["median"] == pytest.approx(1.0)
    assert s["frac_deg6_ge2"] == 1.0
    assert s["components_26"]["n_components_min_size"] == 1
    assert s["components_6"]["n_components_min_size"] == 1
    assert s["enclosed_cavity_voxels"] == 0


def _encode_lasagna(n_xyz: np.ndarray) -> np.ndarray:
    """Plan formulas: dir0 = 0.5+0.5cos2t, dir1 = 0.5+0.5(cos2t-sin2t)/sqrt2 per plane."""
    nx, ny, nz = n_xyz
    out = []
    for gx, gy in ((nx, ny), (nx, nz), (ny, nz)):  # z, y, x channel pairs
        t = np.arctan2(gy, gx)
        c2, s2 = np.cos(2 * t), np.sin(2 * t)
        out.append(0.5 + 0.5 * c2)
        out.append(0.5 + 0.5 * (c2 - s2) / np.sqrt(2.0))
    return np.stack(out)


@pytest.mark.parametrize("n_xyz", [(0.3, -0.5, 0.8), (1.0, 0.0, 0.0), (0.0, 0.7, 0.7), (-0.6, 0.6, -0.5)])
def test_decode_lasagna_normal_roundtrip(n_xyz):
    n = np.array(n_xyz, dtype=np.float64)
    n /= np.linalg.norm(n)
    enc = _encode_lasagna(n)  # (6,)
    ch = np.concatenate([np.ones((2, 3, 3)) * 0.5, np.repeat(enc[:, None, None], 3, 1).repeat(3, 2)])  # (8,3,3)
    dec = decode_lasagna_normal(ch)  # (3, 3, 3) in (z, y, x)
    assert dec.shape == (3, 3, 3)
    got = np.array([dec[2, 1, 1], dec[1, 1, 1], dec[0, 1, 1]])  # back to (x, y, z)
    assert abs(abs(float(got @ n)) - 1.0) < 1e-3  # sign is arbitrary
    # 8-bit quantised input (as stored in the zarr) still decodes within a couple of degrees
    q = np.round(ch * 255) / 255
    got_q = decode_lasagna_normal(q)[::-1, 1, 1]
    assert np.degrees(np.arccos(min(1.0, abs(float(got_q @ n))))) < 3.0


def test_radial_vectors_point_away_from_axis():
    axis = np.array([[0.0, 100.0, 100.0], [10.0, 100.0, 120.0]])  # (z, y, x) control points
    ay, ax = axis_yx_at(axis, 5.0)
    assert (ay, ax) == (100.0, 110.0)
    r = radial_vectors(axis, 5, shape=(4, 4), origin=(100, 110))  # box starts at the axis
    assert r.shape == (3, 4, 4)
    assert np.allclose(r[0], 0.0)
    assert np.allclose(np.hypot(r[1], r[2])[1:, 1:], 1.0)
    assert r[2, 0, 3] == pytest.approx(1.0) and r[1, 3, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- histogram percentiles (R07)
def test_hist_percentiles_matches_np_percentile_on_random_histograms():
    from tsm.labels import hist_percentiles

    rng = np.random.default_rng(0)
    for _ in range(40):
        h = rng.integers(0, 7, 256).astype(np.int64)
        h[rng.integers(0, 256, 200)] = 0
        vals = np.repeat(np.arange(256), h)
        pos = vals[vals > 0]
        got = hist_percentiles(h, (0, 10, 25, 50, 90, 100))
        if pos.size == 0:
            assert got == [None] * 6
            continue
        exp = [float(np.percentile(pos, p)) for p in (0, 10, 25, 50, 90, 100)]
        assert np.allclose(got, exp), (h.tolist(), got, exp)


def test_hist_percentiles_keeps_the_zero_bin_when_asked():
    from tsm.labels import hist_percentiles

    h = np.zeros(256, np.int64)
    h[0], h[10] = 3, 1
    assert hist_percentiles(h, (50,)) == [10.0]
    assert hist_percentiles(h, (50,), drop_zero=False) == [0.0]


def test_hist_percentiles_handles_more_than_2_32_counts_without_allocating():
    from tsm.labels import hist_percentiles

    h = np.zeros(256, np.int64)
    h[0] = 10 ** 12          # background dominates
    h[5] = 3 * 10 ** 9       # > 2**31 positives
    h[9] = 10 ** 9
    p10, med, p90 = hist_percentiles(h, (10, 50, 90))
    assert (p10, med) == (5.0, 5.0)
    assert p90 == 9.0
    assert hist_percentiles(np.zeros(256, np.int64), (50,)) == [None]
