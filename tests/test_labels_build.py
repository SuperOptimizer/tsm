"""Synthetic checks for the label build in tsm.labels (CPU, no data)."""

from __future__ import annotations

import numpy as np
import pytest

from tsm.labels import (
    build_coarse_field,
    build_fine_labels,
    decode_sdf_u8,
    decode_signed_u8,
    encode_sdf_u8,
    encode_signed_u8,
    iter_cores,
    medial_thickness,
    phase_from_cos,
    radial_field,
    read_box_padded,
    upsample_coarse_normal,
)


def _tilted_band(shape=(40, 48, 48), half=3.0, tilt=0.5, offset=None) -> tuple[np.ndarray, np.ndarray]:
    """Band |z - tilt*y - c| <= half*sqrt(1+tilt^2) (a plane tilted in y-z); returns (mask, unit normal zyx)."""
    zz, yy, _ = np.mgrid[: shape[0], : shape[1], : shape[2]]
    c = shape[0] / 2 - tilt * shape[1] / 2 if offset is None else offset
    n = np.array([1.0, -tilt, 0.0]) / np.sqrt(1 + tilt * tilt)
    mask = np.abs(zz - tilt * yy - c) <= half * np.sqrt(1 + tilt * tilt)
    return mask, n


def _const_field(vec, shape) -> np.ndarray:
    return np.broadcast_to(np.asarray(vec, dtype=np.float32)[:, None, None, None], (3,) + tuple(shape)).copy()


def _fine_haloed(recto, ink, ct, coarse_vec, radial_vec, halo=8, **kw):
    """build_fine_labels the way the driver runs it: edge-replicated halo, core cropped."""
    shape = recto.shape
    lo, hi = (-halo,) * 3, tuple(s + halo for s in shape)
    r, i, c = (read_box_padded(a, lo, hi) for a in (recto, ink, ct))
    cn = None if coarse_vec is None else _const_field(coarse_vec, r.shape)
    outs = build_fine_labels(r, i, c, cn, _const_field(radial_vec, r.shape), **kw)
    return tuple(a[halo:-halo, halo:-halo, halo:-halo] for a in outs)


# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #
def test_signed_and_sdf_roundtrip():
    v = np.linspace(-1, 1, 2001, dtype=np.float32)
    u = encode_signed_u8(v)
    assert u.dtype == np.uint8 and u.min() == 0 and u.max() == 255
    assert np.abs(decode_signed_u8(u) - v).max() <= 0.5 / 127.5 + 1e-6
    assert encode_signed_u8(np.array([0.0])) [0] in (127, 128)
    sdf = np.linspace(-30, 30, 601, dtype=np.float32)
    u = encode_sdf_u8(sdf, 20.0)
    assert u.min() == 1 and u.max() == 255 and (u > 0).all()  # 0 reserved
    assert encode_sdf_u8(np.array([0.0]), 20.0)[0] == 128
    d = decode_sdf_u8(u, 20.0)
    inside = np.abs(sdf) <= 20
    assert np.abs(d[inside] - sdf[inside]).max() <= 0.5 * 20 / 127 + 1e-5
    assert np.allclose(d[sdf > 20], 20.0) and np.allclose(d[sdf < -20], -20.0)


# --------------------------------------------------------------------------- #
# fine labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flip", [False, True])
def test_fine_sdf_sign_flips_across_surface_with_requested_orientation(flip):
    shape = (40, 48, 48)
    mask, n = _tilted_band(shape)
    recto = np.where(mask, 255, 0).astype(np.uint8)
    ink = np.zeros(shape, np.uint8)
    ct = np.full(shape, 200, np.uint8)
    n_out = -n if flip else n
    # radial orthogonal to the sheet: must not be used when the coarse normal is present
    sdf_u8, valid, ink_u8, ink_valid = _fine_haloed(recto, ink, ct, n_out, [0.0, 0.0, 1.0], clip=8.0, min_component=10)
    assert sdf_u8.shape == shape and valid.shape == shape
    assert (valid == 1).all() and (ink_valid == 1).all() and (ink_u8 == 0).all()
    sdf = decode_sdf_u8(sdf_u8, 8.0)
    zz, yy, _ = np.mgrid[: shape[0], : shape[1], : shape[2]]
    c = shape[0] / 2 - 0.5 * shape[1] / 2
    signed = (zz - 0.5 * yy - c) / np.sqrt(1.25)  # true signed distance along +n
    if flip:
        signed = -signed
    far = np.abs(signed) >= 1.5
    assert (np.sign(sdf[far]) == np.sign(signed[far])).all()
    inside = np.abs(signed) <= 7.0
    assert np.abs(sdf[inside] - signed[inside]).max() < 1.2  # voxel medial + u8 quantisation
    # zero set is 1 voxel thick and covers the centre plane
    zero = sdf_u8 == 128
    _, thick = medial_thickness(zero)
    assert np.median(thick) == pytest.approx(1.0)
    assert zero.any()
    # every voxel-column across the sheet changes sign exactly once: all voxels of one sign sit
    # below (in z) every voxel of the other sign, with only the zero set in between
    col = sdf[:, 10:38, 10:38]
    zi = np.arange(shape[0])[:, None, None]
    lo_sign, hi_sign = (col > 0, col < 0) if flip else (col < 0, col > 0)
    assert lo_sign.any(0).all() and hi_sign.any(0).all()
    assert (np.where(lo_sign, zi, -1).max(0) < np.where(hi_sign, zi, shape[0]).min(0)).all()


def test_fine_falls_back_to_radial_and_masks_air():
    shape = (32, 40, 40)
    mask, n = _tilted_band(shape)
    recto = np.where(mask, 255, 0).astype(np.uint8)
    ink = np.full(shape, 77, np.uint8)
    ct = np.full(shape, 200, np.uint8)
    ct[:, :, :10] = 0  # air
    sdf_u8, valid, ink_u8, ink_valid = _fine_haloed(recto, ink, ct, None, -n, clip=6.0, min_component=10,
                                                    ct_sigma=0.0, ct_mask_dilate=0)
    zz, yy, _ = np.mgrid[: shape[0], : shape[1], : shape[2]]
    signed = -(zz - 0.5 * yy - (shape[0] / 2 - 0.5 * shape[1] / 2)) / np.sqrt(1.25)
    far = (np.abs(signed) >= 1.5) & (valid == 1)
    assert (np.sign(decode_sdf_u8(sdf_u8, 6.0)[far]) == np.sign(signed[far])).all()
    assert (valid[:, :, :10] == 0).all() and (sdf_u8[:, :, :10] == 0).all() and (ink_valid[:, :, :10] == 0).all()
    assert (valid[:, :, 10:] == 1).all() and (ink_u8 == 77).all()
    # zero-normal coarse field -> radial fallback gives the same result
    sdf2, *_ = _fine_haloed(recto, ink, ct, [0.0, 0.0, 0.0], -n, clip=6.0, min_component=10, ct_sigma=0.0, ct_mask_dilate=0)
    assert np.array_equal(sdf2, sdf_u8)


def test_fine_ignore_band_and_small_components():
    shape = (32, 40, 40)
    mask, n = _tilted_band(shape)
    recto = np.where(mask, 255, 0).astype(np.uint8)
    recto[26:28, 2:4, 2:4] = 255  # 8-voxel blob far from the sheet: dropped
    recto[1:5, 30:38, 30:38] = 100  # uncertain patch (prob 0.39, below threshold) far from the sheet
    ct = np.full(shape, 200, np.uint8)
    sdf_u8, valid, _, _ = _fine_haloed(recto, np.zeros(shape, np.uint8), ct, None, n, clip=6.0, min_component=20)
    assert (valid[1:5, 30:38, 30:38] == 2).all()
    assert (valid[10:22, 4:20, 4:20] == 1).all()
    assert (valid != 0).all()
    assert sdf_u8[27, 3, 3] != 128  # not a surface voxel
    # no medial at all -> ignore everything, sdf = +clip
    sdf3, valid3, _, _ = build_fine_labels(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), ct, None,
                                           _const_field(n, shape), clip=6.0)
    assert (valid3 == 2).all() and (sdf3 == 255).all()


# --------------------------------------------------------------------------- #
# phase / coarse field
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_phase_from_cos_recovers_sin(sign):
    shape = (8, 8, 64)
    x = np.arange(shape[2], dtype=np.float32)
    w = sign * x / 16.0  # w increases along +x for sign=+1
    cosw = np.broadcast_to(np.cos(2 * np.pi * w), shape).astype(np.float32)
    normal_out = _const_field([0.0, 0.0, sign], shape)  # outward = the direction along which w increases
    sinw = phase_from_cos(cosw, normal_out, sigma=0.0)
    expect = np.broadcast_to(np.sin(2 * np.pi * w), shape)
    ok = np.abs(expect) > 0.15  # away from extrema (sign there is irrelevant: sin ~ 0)
    assert np.abs(sinw[ok] - expect[ok]).max() < 1e-4
    assert np.abs(sinw[~ok]).max() <= 0.15 + 1e-6
    assert np.allclose(sinw * sinw + cosw * cosw, 1.0, atol=1e-5)
    # 8-bit quantised cos channel + smoothing: right sign wherever |sin| is not tiny
    cos_u8 = np.rint(127.5 + 127.5 * cosw).astype(np.uint8)
    cosq = 2.0 * (cos_u8.astype(np.float32) / 255.0) - 1.0
    sinq = phase_from_cos(cosq, normal_out, sigma=1.0)
    assert (np.sign(sinq[ok]) == np.sign(expect[ok])).all()


def _encode_lasagna_u8(n_xyz, shape, cos_ch: np.ndarray, grad_u8: int = 8) -> np.ndarray:
    nx, ny, nz = n_xyz
    ch = [cos_ch, np.full(shape, grad_u8 / 255.0, np.float32)]
    for gx, gy in ((nx, ny), (nx, nz), (ny, nz)):
        t = np.arctan2(gy, gx)
        c2, s2 = np.cos(2 * t), np.sin(2 * t)
        ch.append(np.full(shape, 0.5 + 0.5 * c2, np.float32))
        ch.append(np.full(shape, 0.5 + 0.5 * (c2 - s2) / np.sqrt(2.0), np.float32))
    return np.rint(np.stack(ch) * 255).astype(np.uint8)


@pytest.mark.parametrize("n_xyz", [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.6, -0.6, 0.5), (-0.4, 0.85, 0.3)])
def test_coarse_field_normal_points_outward_and_phase_is_unit(n_xyz):
    shape = (12, 16, 48)
    n = np.array(n_xyz) / np.linalg.norm(n_xyz)
    x = np.arange(shape[2], dtype=np.float32)
    cos_ch = np.broadcast_to(0.5 + 0.5 * np.cos(2 * np.pi * x / 16.0), shape).astype(np.float32)
    las = _encode_lasagna_u8(n, shape, cos_ch)
    ct = np.full(shape, 120, np.uint8)
    radial = _const_field([0.0, 0.0, 1.0], shape)  # outward = +x
    out = build_coarse_field(las, ct, radial)
    assert out.shape == (8,) + shape and out.dtype == np.uint8
    nz, ny, nx = decode_signed_u8(out[5]), decode_signed_u8(out[4]), decode_signed_u8(out[3])
    got = np.stack([nx, ny, nz])
    dot = (got * n[:, None, None, None]).sum(0)
    assert (np.abs(np.abs(dot) - 1.0) < 0.03).all()  # decoded direction
    assert ((got * np.array([1.0, 0, 0])[:, None, None, None]).sum(0) > 0).all()  # oriented outward (+x)
    s, c = decode_signed_u8(out[0]), decode_signed_u8(out[1])
    assert np.abs(np.sqrt(s * s + c * c) - 1.0).max() < 0.02
    assert (out[2] == 8).all()
    assert (out[7] == 1).all()
    assert out[6].mean() > 80 and out[6].max() > 180  # 5^3 window on a 16-voxel period: decent contrast
    # phase sign: w = x/16 increases along +x, n_x>0 was the outward orientation
    expect = np.sin(2 * np.pi * x / 16.0)
    ok = np.abs(expect) > 0.3
    assert (np.sign(s[..., ok]) == np.sign(expect[ok])).all()


def test_coarse_field_valid_marks_air_and_inconsistent_dirs():
    shape = (8, 8, 8)
    las = _encode_lasagna_u8((0.0, 0.0, 1.0), shape, np.full(shape, 0.75, np.float32))
    ct = np.full(shape, 120, np.uint8)
    ct[:, :, :3] = 0
    radial = _const_field([1.0, 0.0, 0.0], shape)
    out = build_coarse_field(las, ct, radial, ct_sigma=0.0, ct_mask_dilate=0)
    assert (out[7][:, :, :3] == 0).all() and (out[6][:, :, :3] == 0).all() and (out[7][:, :, 3:] == 1).all()
    bad = las.copy()
    bad[2:8] = 128  # dir pairs all "0.5": no direction information -> inconsistent
    out2 = build_coarse_field(bad, np.full(shape, 120, np.uint8), radial, ct_sigma=0.0, ct_mask_dilate=0)
    assert (out2[7] == 2).all()
    empty = np.zeros_like(las)
    out3 = build_coarse_field(empty, np.full(shape, 120, np.uint8), radial, ct_sigma=0.0, ct_mask_dilate=0)
    assert (out3[7] == 0).all()


# --------------------------------------------------------------------------- #
# bricks
# --------------------------------------------------------------------------- #
def test_iter_cores_and_padded_read():
    cores = list(iter_cores((10, 7, 5), (4, 4, 4)))
    assert cores[0] == ((0, 0, 0), (4, 4, 4)) and cores[-1] == ((8, 4, 4), (10, 7, 5))
    seen = np.zeros((10, 7, 5), int)
    for lo, hi in cores:
        seen[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += 1
    assert (seen == 1).all()
    a = np.arange(2 * 4 * 5 * 6).reshape(2, 4, 5, 6).astype(np.uint8)
    b = read_box_padded(a, (-2, 1, 4), (5, 3, 9))
    assert b.shape == (2, 7, 2, 5)
    iz, iy, ix = np.clip(np.arange(-2, 5), 0, 3), np.arange(1, 3), np.clip(np.arange(4, 9), 0, 5)
    assert np.array_equal(b, a[:, iz][:, :, iy][:, :, :, ix])
    c = read_box_padded(a[0], (0, 0, 0), (4, 5, 6))
    assert np.array_equal(c, a[0])
    d = read_box_padded(a, (1, 1, 1), (2, 2, 2), channels=1)
    assert d.shape == (1, 1, 1, 1) and d[0, 0, 0, 0] == a[1, 1, 1, 1]


def test_upsample_coarse_normal_alignment_and_validity():
    coarse = np.zeros((8, 4, 6, 6), np.uint8)
    n = np.array([0.6, 0.0, 0.8])  # (x, y, z)
    coarse[3] = encode_signed_u8(np.full((4, 6, 6), n[0]))
    coarse[4] = encode_signed_u8(np.full((4, 6, 6), n[1]))
    coarse[5] = encode_signed_u8(np.full((4, 6, 6), n[2]))
    coarse[7] = 1
    coarse[7, :, :, :2] = 0  # invalid strip (fine x < 8)
    up = upsample_coarse_normal(coarse, (-3, 0, -4), (13, 24, 30), scale=4)
    assert up.shape == (3, 16, 24, 34)
    nrm = np.sqrt((up * up).sum(0))
    assert (nrm[:, :, 4:8] == 0).all()  # x in [-4, 4): fully invalid
    assert np.allclose(nrm[:, :, 14:], 1.0)
    assert np.allclose(up[:, :, :, 14:], np.array([n[2], n[1], n[0]])[:, None, None, None], atol=1e-2)


def test_brickwise_fine_labels_equal_whole_array():
    shape = (48, 64, 64)
    m1, n = _tilted_band(shape, half=2.0, tilt=0.5, offset=6.0)
    m2, _ = _tilted_band(shape, half=2.0, tilt=0.5, offset=18.0)
    m3, _ = _tilted_band(shape, half=2.0, tilt=0.5, offset=30.0)
    rng = np.random.default_rng(0)
    recto = np.where(m1 | m2 | m3, 255, 0).astype(np.uint8)
    recto[recto == 0] = rng.integers(0, 60, size=int((recto == 0).sum()), dtype=np.uint8)  # noise below threshold
    recto[m2 & (np.arange(shape[2])[None, None, :] > 40)] = 150  # a partly uncertain sheet
    ink = rng.integers(0, 255, size=shape, dtype=np.uint8)
    ct = np.full(shape, 150, np.uint8)
    ct[:, :6, :] = 0
    axis = np.array([[0.0, -200.0, 32.0], [100.0, -200.0, 32.0]])
    clip, halo, brick = 6.0, 10, (24, 32, 32)  # sheets 12 apart: every voxel within 6 < halo of a medial
    kw = dict(clip=clip, min_component=50, ct_sigma=1.0, ct_mask_dilate=1)

    def compute(lo, hi, use_coarse):
        r = read_box_padded(recto, lo, hi)
        i = read_box_padded(ink, lo, hi)
        c = read_box_padded(ct, lo, hi)
        rad = radial_field(axis, lo, r.shape, scale=1)
        cn = _const_field(n, r.shape) if use_coarse else None
        return build_fine_labels(r, i, c, cn, rad, **kw)

    for use_coarse in (False, True):
        whole = compute((-halo,) * 3, tuple(s + halo for s in shape), use_coarse)
        whole = [a[halo:-halo, halo:-halo, halo:-halo] for a in whole]
        got = [np.zeros(shape, np.uint8) for _ in range(4)]
        for lo, hi in iter_cores(shape, brick):
            outs = compute([l - halo for l in lo], [h + halo for h in hi], use_coarse)
            core = tuple(slice(halo, halo + hi[a] - lo[a]) for a in range(3))
            for g, o in zip(got, outs):
                g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = o[core]
        for name, g, w in zip(("sdf", "sdf_valid", "ink", "ink_valid"), got, whole):
            assert np.array_equal(g, w), f"{name} differs brick-wise vs whole ({(g != w).sum()} voxels)"
        assert (got[1] == 2).any() and (got[1] == 0).any() and (got[1] == 1).any()


def test_coarse_field_marks_tangential_normal_as_ignore():
    shape = (12, 16, 48)
    n = np.array([0.0, 1.0, 0.0])  # perpendicular to the +x radial direction
    x = np.arange(shape[2], dtype=np.float32)
    cos_ch = np.broadcast_to(0.5 + 0.5 * np.cos(2 * np.pi * x / 16.0), shape).astype(np.float32)
    las = _encode_lasagna_u8(n, shape, cos_ch)
    ct = np.full(shape, 120, np.uint8)
    out = build_coarse_field(las, ct, _const_field([0.0, 0.0, 1.0], shape))
    assert (out[7] == 2).all()
