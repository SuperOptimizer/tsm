"""Augmentation v2 (tsm.data.Augment): target-aware equivariance on analytic fields, CPU, tiny crops."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tsm.data import (
    CLIP,
    INTENSITY_NAMES,
    PRESETS,
    SCAN_NAMES,
    SPATIAL_NAMES,
    TRANSFORM_NAMES,
    Augment,
    AugmentConfig,
    SpatialParams,
    apply_overrides,
    as_augment_config,
    rot90_matrix_inplane,
    rotation_matrix,
    split_holdout,
)
from tsm.train import synthetic_batch

N = 24  # crop size
T = 12.0  # sheet period (voxels); |sdf| <= 6 < CLIP so nothing saturates


def _centred(n: int) -> torch.Tensor:
    ax = [torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0 for _ in range(3)]
    return torch.stack(torch.meshgrid(*ax, indexing="ij"), dim=-1)  # (n, n, n, 3) zyx


def sheet_fields(normal_xyz, period: float, w0: float = 0.2, n: int = N) -> dict[str, torch.Tensor]:
    """Planar sheets with unit normal ``normal_xyz`` every ``period`` voxels: the full target set (B=1)."""
    nx, ny, nz = (float(v) for v in normal_xyz)
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    c = _centred(n)
    proj = c[..., 0] * nz + c[..., 1] * ny + c[..., 2] * nx  # n . c
    w = proj / period + w0
    sdf = (proj - (torch.round(w) - w0) * period)  # signed distance to the nearest sheet along +n
    one = torch.ones(1, 1, n, n, n)
    normal = torch.stack([torch.full((n, n, n), nx), torch.full((n, n, n), ny), torch.full((n, n, n), nz)])[None]
    return {
        "input": torch.cat([(0.5 + 0.5 * torch.cos(2 * math.pi * w))[None, None], torch.zeros(1, 1, n, n, n)], 1),
        "surface_sdf": sdf[None, None].clone(), "surface_valid": one.clone(),
        "ink_prob": (0.5 + 0.5 * torch.sin(c[..., 2] / 5.0) * torch.cos(c[..., 1] / 7.0))[None, None], "ink_valid": one.clone(),
        "winding": torch.cat([torch.sin(2 * math.pi * w)[None, None], torch.cos(2 * math.pi * w)[None, None],
                              torch.full((1, 1, n, n, n), 1.0 / period), normal], 1),
        "winding_conf": one * 0.8, "winding_valid": one.clone(),
    }


def _params(flips=(False, False, False), rot90=None, rotation=None, scale=None, elastic=None) -> SpatialParams:
    return SpatialParams(tuple(flips), rot90, rotation, None if scale is None else torch.tensor(scale, dtype=torch.float32), elastic)


CASES = {
    "flip_z": _params(flips=(True, False, False)),
    "flip_y": _params(flips=(False, True, False)),
    "flip_x": _params(flips=(False, False, True)),
    "flip_all": _params(flips=(True, True, True)),
    "rot90_k1": _params(rot90=rot90_matrix_inplane(1)),
    "rot90_k2": _params(rot90=rot90_matrix_inplane(2)),
    "rot90_k3": _params(rot90=rot90_matrix_inplane(3)),
    "rot90_zx": _params(rot90=torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])),  # about y
    "rot90_zy": _params(rot90=torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),  # about x
    "rotate_z10": _params(rotation=rotation_matrix(torch.tensor([1.0, 0.0, 0.0]), math.radians(10))),
    "rotate_axis15": _params(rotation=rotation_matrix(torch.tensor([0.3, -0.8, 0.5]), math.radians(15))),
    "rotate_axis-12": _params(rotation=rotation_matrix(torch.tensor([-0.6, 0.2, 0.75]), math.radians(-12))),
    "scale_0.85": _params(scale=[0.85, 0.85, 0.85]),
    "scale_1.2": _params(scale=[1.2, 1.2, 1.2]),
    "scale_aniso": _params(scale=[0.9, 1.15, 1.0]),
    "combo": _params(flips=(False, True, False), rot90=rot90_matrix_inplane(1),
                     rotation=rotation_matrix(torch.tensor([0.2, 0.9, -0.4]), math.radians(8)), scale=[1.1, 1.1, 1.1]),
    "combo_aniso": _params(flips=(True, False, True), rotation=rotation_matrix(torch.tensor([1.0, 1.0, 1.0]), math.radians(-14)),
                           scale=[0.9, 1.05, 1.15]),
}
NORMALS = [(0.0, 0.0, 1.0), (0.6, -0.8, 0.0), (0.36, 0.48, 0.8)]


@pytest.mark.parametrize("nrm", NORMALS, ids=["nz", "nxy", "nxyz"])
@pytest.mark.parametrize("case", list(CASES))
def test_spatial_equivariance_on_planar_sheets(case, nrm):
    p = CASES[case]
    b = sheet_fields(nrm, T)
    aug = Augment("none")
    out = aug.apply_spatial(b, [p])
    L = p.matrix().float()
    LinvT = torch.linalg.inv(L).T
    n_zyx = torch.tensor([nrm[2], nrm[1], nrm[0]]) / math.sqrt(sum(v * v for v in nrm))
    cov = LinvT @ n_zyx  # transformed (unnormalised) normal in zyx
    k = float(cov.norm())
    n2 = cov / k
    ref = sheet_fields((float(n2[2]), float(n2[1]), float(n2[0])), T / k)
    # sampled inside the crop (no OOB) and away from the folds of the triangle-wave sdf
    c = _centred(N)
    x_in = torch.einsum("ij,dhwj->dhwi", torch.linalg.inv(L), c)
    inside = (x_in.abs() <= (N - 1) / 2.0 - 1.0).all(-1)[None, None]
    away = ref["surface_sdf"].abs() < (T / k) / 2.0 - 1.5
    m = inside & away
    assert m.float().mean() > 0.2
    for key in ("input", "surface_sdf", "ink_prob", "winding", "winding_conf", "surface_valid", "ink_valid", "winding_valid"):
        assert torch.isfinite(out[key]).all(), key
    # masks stay binary / ternary, OOB -> 0 (we are within the crop here -> 1)
    for key in ("surface_valid", "ink_valid", "winding_valid"):
        assert set(out[key].unique().tolist()) <= {0.0, 1.0, 2.0}
        assert torch.all(out[key][m] == 1.0)
    # sdf: distances scale with the transform (isotropic: x s; anisotropic: det^(1/3) approximation)
    s_iso = p.scale_iso()
    tol = 0.25 if case != "scale_aniso" and case != "combo_aniso" else 1.0
    assert (out["surface_sdf"][m] * (k * s_iso) / s_iso - ref["surface_sdf"][m] * k).abs().max() < tol * 3  # sanity on the scale
    assert (out["surface_sdf"][m] - ref["surface_sdf"][m] * (s_iso * k)).abs().max() < tol, case
    # phase is a scalar invariant: sin/cos at the output voxel equal the analytic phase there
    ang_out = torch.atan2(out["winding"][:, 0], out["winding"][:, 1])
    ang_ref = torch.atan2(ref["winding"][:, 0], ref["winding"][:, 1])
    d = torch.atan2(torch.sin(ang_out - ang_ref), torch.cos(ang_out - ang_ref))
    assert d[m[:, 0]].abs().max() < 0.12, case
    assert torch.allclose(out["winding"][:, 0:2].norm(dim=1)[m[:, 0]], torch.ones(1), atol=1e-4)
    # density: wraps per voxel scale by |L^-T n| (1/s for isotropic scaling)
    torch.testing.assert_close(out["winding"][:, 2][m[:, 0]], torch.full_like(out["winding"][:, 2][m[:, 0]], k / T), atol=2e-4, rtol=0)
    # normal: rotated by the exact rotation part (covector rule), renormalised
    n_out = out["winding"][:, 3:6].permute(0, 2, 3, 4, 1)[m[:, 0]]
    exp = torch.tensor([float(n2[2]), float(n2[1]), float(n2[0])])
    assert (n_out - exp).abs().max() < 1e-4, case
    assert torch.allclose(n_out.norm(dim=-1), torch.ones(1), atol=1e-4)
    # ink / ct / conf: plain interpolation of the scalar field at L^-1 x
    assert (out["winding_conf"][m] - 0.8).abs().max() < 1e-4
    assert (out["input"][:, 0:1][m] - ref["input"][:, 0:1][m]).abs().max() < 0.08
    ink_in = b["ink_prob"][0, 0]
    # reference by direct trilinear lookup at x_in (interior only)
    g = (2.0 * x_in / N)[..., [2, 1, 0]][None]
    ink_ref = torch.nn.functional.grid_sample(ink_in[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=False)
    assert (out["ink_prob"][m] - ink_ref[m]).abs().max() < 1e-5


def test_flip_negates_normal_component_and_rot90_matches_numpy():
    b = sheet_fields((0.36, 0.48, 0.8), T)
    aug = Augment("none")
    for a, comp in ((0, 2), (1, 1), (2, 0)):  # spatial axis (z, y, x) <-> normal component index (nz, ny, nx)
        f = [False, False, False]
        f[a] = True
        o = aug.apply_spatial(b, [_params(flips=tuple(f))])
        exp = torch.flip(b["winding"], dims=(a + 2,)).clone()
        exp[:, 3 + comp] *= -1
        torch.testing.assert_close(o["winding"], exp, atol=1e-4, rtol=0)
        torch.testing.assert_close(o["surface_sdf"], torch.flip(b["surface_sdf"], dims=(a + 2,)), atol=1e-4, rtol=0)
        torch.testing.assert_close(o["input"], torch.flip(b["input"], dims=(a + 2,)), atol=1e-4, rtol=0)
    for k in range(4):
        o = aug.apply_spatial(b, [_params(rot90=rot90_matrix_inplane(k))])
        ref = np.rot90(b["surface_sdf"].numpy(), k, axes=(3, 4))
        np.testing.assert_allclose(o["surface_sdf"].numpy(), ref, atol=1e-4)
        # normal: np.rot90(k=1, axes=(y, x)) maps the point (x, y) -> (y, -x), so (nx, ny) -> (ny, -nx) (as v1)
        nref = np.rot90(b["winding"].numpy(), k, axes=(3, 4)).copy()
        for _ in range(k):
            nx, ny = nref[:, 3].copy(), nref[:, 4].copy()
            nref[:, 3], nref[:, 4] = ny, -nx
        np.testing.assert_allclose(o["winding"].numpy(), nref, atol=1e-4)


def test_oob_voxels_become_invalid_and_saturated_labels_ignored():
    b = synthetic_batch(1, N, seed=4)
    b["surface_valid"].fill_(1.0)
    b["ink_valid"].fill_(1.0)
    b["winding_valid"].fill_(1.0)
    b["surface_sdf"].fill_(CLIP)  # everything saturated
    aug = Augment("none")
    o = aug.apply_spatial(b, [_params(scale=[0.8, 0.8, 0.8])])  # zoom out: the corners sample outside the crop
    c = _centred(N)
    oob = (c.abs() / 0.8 > (N - 1) / 2.0 + 1e-4).any(-1)[None, None]
    assert oob.any() and (~oob).any()
    for key in ("surface_valid", "ink_valid", "winding_valid"):
        assert torch.all(o[key][oob] == 0.0), key
    assert torch.all(o["ink_valid"][~oob] == 1.0) and torch.all(o["winding_valid"][~oob] == 1.0)
    # the shrunk saturated sdf (16 < CLIP) is unknown -> valid 2, and the winding/ink targets are 0 where invalid
    assert torch.all(o["surface_valid"][~oob] == 2.0)
    assert torch.all(o["surface_sdf"][oob] == 0.0) and torch.all(o["winding"][:, :, oob[0, 0]] == 0.0)
    # zoom in never leaves the crop
    o2 = aug.apply_spatial(b, [_params(scale=[1.2, 1.2, 1.2])])
    assert torch.all(o2["ink_valid"] == 1.0) and torch.all(o2["surface_sdf"] == CLIP) and torch.all(o2["surface_valid"] == 1.0)


def test_mask_aware_interpolation_ignores_invalid_neighbours():
    b = sheet_fields((0, 0, 1), T)
    b["surface_sdf"][:, :, :, :, N // 2:] = 0.0  # garbage where invalid
    b["surface_valid"][:, :, :, :, N // 2:] = 0.0
    b["winding"][:, :, :, :, N // 2:] = -3.0
    b["winding_valid"][:, :, :, :, N // 2:] = 2.0
    o = Augment("none").apply_spatial(b, [_params(rotation=rotation_matrix(torch.tensor([1.0, 0.0, 0.0]), math.radians(7)))])
    m1 = o["surface_valid"] == 1
    assert m1.any()
    assert (o["surface_sdf"][m1].abs() <= T / 2 + 1e-3).all()  # never mixed with the zeroed half
    mw = (o["winding_valid"] == 1)[:, 0]
    assert torch.all(o["winding"][:, 2][mw] > 0) and (o["winding"][:, 2][mw] - 1.0 / T).abs().max() < 1e-4
    assert torch.all(o["winding"][:, 3:6].permute(0, 2, 3, 4, 1)[~mw] == 0.0)


def test_elastic_is_small_and_finite_and_rot90_all_axes_are_rotations():
    b = sheet_fields((0.6, -0.8, 0.0), T)
    aug = Augment({"elastic": {"p": 1.0, "grid": 3, "magnitude": 1.5}})
    p = aug.sample_spatial()
    assert p.elastic is not None and tuple(p.elastic.shape) == (3, 3, 3, 3)
    o = aug.apply_spatial(b, [_params(elastic=p.elastic)])
    assert torch.isfinite(o["surface_sdf"]).all() and torch.isfinite(o["winding"]).all()
    m = o["surface_valid"] == 1
    assert (o["surface_sdf"][m] - b["surface_sdf"][m]).abs().median() < 1.5  # displacement ~1.5 vox
    torch.testing.assert_close(o["winding"][:, 3:6][o["winding_valid"].expand(-1, 3, -1, -1, -1) == 1],
                               b["winding"][:, 3:6][o["winding_valid"].expand(-1, 3, -1, -1, -1) == 1], atol=1e-4, rtol=0)
    aug = Augment({"rot90": {"p": 1.0, "all_axes": True}}, seed=3)
    seen = set()
    for _ in range(40):
        q = aug.sample_spatial().rot90
        assert q is not None and abs(float(torch.det(q)) - 1.0) < 1e-6 and torch.equal(q.abs().sum(0), torch.ones(3))
        seen.add(tuple(q.flatten().tolist()))
    assert len(seen) > 10  # many of the 23 non-identity cube rotations


def test_none_preset_is_identity_and_strong_has_no_nans():
    b = synthetic_batch(2, 16, seed=7)
    o, fired = Augment("none", seed=1)(b)
    assert all(v == 0 for v in fired.values())
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, o[k]), k
    aug = Augment("strong", seed=2)
    tot = {k: 0 for k in TRANSFORM_NAMES}
    for _ in range(12):
        o, fired = aug(b)
        for k, v in fired.items():
            tot[k] += v
        for k, v in o.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), k
                assert tuple(v.shape) == tuple(b[k].shape), k
        for key in ("surface_valid", "ink_valid", "winding_valid"):
            assert set(o[key].unique().tolist()) <= {0.0, 1.0, 2.0}
        assert torch.equal(o["input"][:, 1], b["input"][:, 1])  # scale channel untouched
        assert o["surface_sdf"].abs().max() <= CLIP
    assert tot["artefact"] == 0 and sum(tot[k] for k in SPATIAL_NAMES) > 0 and sum(tot[k] for k in INTENSITY_NAMES) > 0
    assert set(fired) == set(TRANSFORM_NAMES)


def test_augment_is_seeded_and_deterministic():
    b = synthetic_batch(2, 16, seed=8)
    a1, a2 = Augment("strong", seed=5), Augment("strong", seed=5)
    for _ in range(3):
        o1, f1 = a1(b)
        o2, f2 = a2(b)
        assert f1 == f2
        for k in o1:
            if isinstance(o1[k], torch.Tensor):
                torch.testing.assert_close(o1[k], o2[k])


@pytest.mark.parametrize("name", INTENSITY_NAMES)
def test_each_intensity_transform_alone(name):
    b = synthetic_batch(2, 16, seed=9)
    cfg = {n: {"p": 0.0} for n in TRANSFORM_NAMES}
    cfg[name] = {"p": 1.0}
    aug = Augment(cfg, seed=11)
    o, fired = aug(b)
    assert fired[name] == 2 and sum(fired.values()) == 2
    for k in b:
        if isinstance(b[k], torch.Tensor) and k != "input":
            assert torch.equal(b[k], o[k]), k  # targets untouched by intensity ops
    ct = o["input"][:, 0]
    assert torch.isfinite(ct).all() and ct.shape == b["input"][:, 0].shape
    assert not torch.equal(ct, b["input"][:, 0])
    assert torch.equal(o["input"][:, 1], b["input"][:, 1])


def test_blur_axis_only_and_lowres_and_cutout_semantics():
    n = 16
    c = _centred(n)
    ramp = (c[..., 2] / n + 0.5)[None, None]  # varies along x only
    aug = Augment("none")
    torch.testing.assert_close(aug._blur(ramp, [1.5, 0.0, 0.0]), ramp, atol=1e-6, rtol=0)  # z blur of an x ramp is the identity
    bx = aug._blur(ramp, [0.0, 0.0, 1.5])
    assert (bx[..., 6:-6] - ramp[..., 6:-6]).abs().max() < 1e-5 and (bx - ramp).abs().max() > 1e-3  # edges replicate
    noise = torch.rand(1, 1, n, n, n)
    a = Augment({**{k: {"p": 0.0} for k in TRANSFORM_NAMES}, "lowres": {"p": 1.0, "factor_lo": 2.0, "factor_hi": 2.0}})
    lo, _ = a.apply_intensity(noise)
    assert lo.std() < 0.6 * noise.std()  # coarser scan: high frequencies gone
    a = Augment({**{k: {"p": 0.0} for k in TRANSFORM_NAMES}, "cutout": {"p": 1.0, "n_lo": 1, "n_hi": 1, "size_lo": 0.5, "size_hi": 0.5}}, seed=1)
    cut, _ = a.apply_intensity(noise)
    changed = cut != noise
    assert 0.1 < changed.float().mean() < 0.15 and cut[changed].std() < 1e-6  # one 8^3 box of a constant


def test_config_validation_presets_and_overrides():
    with pytest.raises(ValueError, match="unknown augment keys"):
        AugmentConfig.from_dict({"rotat": {"p": 1}})
    with pytest.raises(ValueError, match="unknown augment.rotate keys"):
        AugmentConfig.from_dict({"rotate": {"deg": 1}})
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"rotate": {"p": 1.5}})
    with pytest.raises(ValueError):
        AugmentConfig.from_dict({"oob_fill": "mirror"})
    with pytest.raises(ValueError):
        as_augment_config("huge")
    for name in PRESETS:
        cfg = AugmentConfig.preset(name)
        assert AugmentConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert AugmentConfig.preset("none").is_identity() and not AugmentConfig.preset("strong").is_identity()
    strong = AugmentConfig.preset("strong")
    assert strong.rotate.in_plane is False and strong.rotate.max_deg == 30.0 and strong.artefact.p == 0.0
    d = apply_overrides(strong.to_dict(), {"rotate.p": 0, "blur.sigma_hi": 3.0})
    cfg = AugmentConfig.from_dict(d)
    assert cfg.rotate.p == 0 and cfg.blur.sigma_hi == 3.0 and cfg.scale.p == strong.scale.p
    cfg2 = as_augment_config({"preset": "none", "flip": {"p": 1.0}})
    assert cfg2.flip.p == 1.0 and cfg2.rotate.p == 0.0


def _assert_boxes_disjoint(tr: np.ndarray, ho: np.ndarray, patch: int) -> None:
    """No training crop [o, o+patch)^3 may intersect any holdout crop, on any axis."""
    tr = np.asarray(tr, np.int64).reshape(-1, 3)
    ho = np.asarray(ho, np.int64).reshape(-1, 3)
    if not len(tr) or not len(ho):
        return
    d = np.abs(tr[:, None, :] - ho[None, :, :])
    inter = (d < patch).all(-1)  # 1-D overlap on every axis == 3-D box intersection
    assert not inter.any(), f"{int(inter.sum())} overlapping train/holdout crop pairs"


def test_split_holdout_interior_range_leaves_no_overlapping_train_crop():
    for axis, key in enumerate(("z_range", "y_range", "x_range")):
        for patch, stride in ((32, 16), (32, 8), (16, 16)):
            grid = list(range(0, 128 - patch + 1, stride))
            o = np.array([[a if k == axis else b for k in range(3)]
                          for a in grid for b in (0, stride)], np.int32)
            tr, ho = split_holdout(o, patch, {key: [32, 64]})
            assert len(ho) and len(tr)
            assert set(ho[:, axis].tolist()) == {g for g in grid if 32 <= g < 64}
            _assert_boxes_disjoint(tr, ho, patch)


def test_split_holdout_modes():
    o = np.array([[z, y, 0] for z in range(0, 128, 16) for y in (0, 16)], np.int32)
    tr, ho = split_holdout(o, 32, None)
    assert len(tr) == len(o) and len(ho) == 0
    tr, ho = split_holdout(o, 32, {"z_frac": 0.25})
    assert len(ho) > 0 and ho[:, 0].min() >= 112 - 32 and tr[:, 0].max() + 32 <= ho[:, 0].min()
    tr, ho = split_holdout(o, 32, {"z_range": [32, 64]})
    # an interior band: the last held origin (48) covers z [48, 80), so 64 must NOT be a train origin
    assert set(ho[:, 0].tolist()) == {32, 48}
    assert 64 not in set(tr[:, 0].tolist())
    _assert_boxes_disjoint(tr, ho, 32)
    tr, ho = split_holdout(o, 32, [[64, 0, 0]])
    assert len(ho) == 1 and not any((abs(tr[:, 0] - 64) < 32) & (abs(tr[:, 1]) < 32))
    with pytest.raises(ValueError):
        split_holdout(o, 32, {"bogus": 1})


def test_uniform_rotation_matrix_is_proper_rotation():
    import torch
    from tsm.data import uniform_rotation_matrix
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        R = uniform_rotation_matrix(g)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert abs(float(torch.det(R)) - 1.0) < 1e-4


def test_strong_preset_defaults_cover_all_axes():
    from tsm.data import AugmentConfig
    c = AugmentConfig.preset("strong")
    assert c.rot90.all_axes is True and c.rotate.in_plane is False and c.rotate.full_p > 0


# --------------------------------------------------------------------------- #
# scroll-axis input channels (extra.train.input_axis, 2026-09-07)
# --------------------------------------------------------------------------- #
def test_axis_input_channels_rotate_with_the_volume_for_all_cube_rotations():
    """``input_axis``: the constant scroll-axis field (1, 0, 0) must come back as ``R a`` --
    the same vector rule the radial channels follow -- for every one of the 24 cube rotations,
    otherwise the fibre classes ("vertical" = along the axis) are unlearnable after a rotation."""
    from tsm.student import in_channels, input_vec_slices
    from tests.test_fiber import cube_rotations

    P = 8
    assert in_channels(True, True) == 8 and input_vec_slices(True, True) == [(2, 5), (5, 8)]
    rad = torch.zeros(1, 3, P, P, P)
    rad[:, 2] = 1.0  # a constant radial field along +x, to check both slices move independently
    ax = torch.zeros(1, 3, P, P, P)
    ax[:, 0] = 1.0
    base = {"input": torch.cat([torch.zeros(1, 2, P, P, P), rad, ax], dim=1)}
    aug = Augment("none", seed=0, input_vec=input_vec_slices(True, True))
    for m in cube_rotations():
        out = aug.apply_spatial({k: v.clone() for k, v in base.items()}, [SpatialParams(rot90=m)])
        got_r, got_a = out["input"][:, 2:5], out["input"][:, 5:8]
        want_r = torch.einsum("ij,bjdhw->bidhw", m, rad)
        want_a = torch.einsum("ij,bjdhw->bidhw", m, ax)
        torch.testing.assert_close(got_r, want_r, atol=1e-5, rtol=0)
        torch.testing.assert_close(got_a, want_a, atol=1e-5, rtol=0)
        assert float(got_a.norm(dim=1).min()) > 1 - 1e-5  # still a unit vector everywhere
    # without the axis channels the historical 5-channel layout is untouched
    aug5 = Augment("none", seed=0)
    b5 = {"input": torch.cat([torch.zeros(1, 2, P, P, P), rad], dim=1)}
    m = cube_rotations()[3]
    o5 = aug5.apply_spatial({k: v.clone() for k, v in b5.items()}, [SpatialParams(rot90=m)])
    torch.testing.assert_close(o5["input"][:, 2:5], torch.einsum("ij,bjdhw->bidhw", m, rad), atol=1e-5, rtol=0)


def test_tta_moves_both_input_vector_slices():
    """The inference-side analogue: ``tta_forward`` accepts several vector slices."""
    from tsm.infer import tta_forward

    P = 6
    rad = torch.zeros(1, 3, P, P, P)
    rad[:, 2] = 1.0
    ax = torch.zeros(1, 3, P, P, P)
    ax[:, 0] = 1.0
    x = torch.cat([torch.zeros(1, 2, P, P, P), rad, ax], dim=1)
    out = tta_forward(x, (0,), True, vec=[(2, 5), (5, 8)])
    # flip z then transpose y<->x: the axis component (z) is negated, the radial x component moves to y
    assert float(out[0, 5].mean()) == -1.0 and float(out[0, 3].mean()) == 1.0
    one = tta_forward(x, (0,), True, vec=(2, 5))  # the single-slice form still works
    torch.testing.assert_close(one[:, 2:5], out[:, 2:5])


# --------------------------------------------------------------------------- #
# scan-domain family (2026-09-07): window, class_contrast, aniso_blur, ring, stripe,
# spectral_noise -- CT only, off in "strong", on in "strong_scan"
# --------------------------------------------------------------------------- #

# sha256 of four consecutive strong-preset batches (tensors + fired counts), computed with the
# implementation as it stood BEFORE the scan-domain family was added.  The new transforms all
# default to p=0 and ``_fires`` short-circuits without drawing, so the generator stream -- and
# therefore every strong-preset run -- must be bit-identical.
STRONG_DIGEST = "418a17d85a3516d190ff149259dc4393ccb848bdd7440bde2136b8efb030f58d"


def _digest(preset: str, seed: int = 23) -> str:
    import hashlib

    b = synthetic_batch(3, 16, seed=17)
    a = Augment(preset, seed=seed)
    h = hashlib.sha256()
    for _ in range(4):
        o, f = a(b)
        for k in sorted(o):
            if isinstance(o[k], torch.Tensor):
                h.update(o[k].float().contiguous().numpy().tobytes())
        h.update(repr(sorted(f.items())).encode())
    return h.hexdigest()


def _scan_cfg(name: str, p: float, **kw) -> dict:
    return {**{n: {"p": 0.0} for n in TRANSFORM_NAMES}, name: {"p": p, **kw}}


def test_strong_preset_is_unchanged_by_the_scan_family():
    assert _digest("strong") == STRONG_DIGEST
    assert SCAN_NAMES == ("aniso_blur", "class_contrast", "spectral_noise", "ring", "stripe", "window")
    strong, scan = AugmentConfig.preset("strong"), AugmentConfig.preset("strong_scan")
    for n in SCAN_NAMES:
        assert float(getattr(strong, n).p) == 0.0, n  # off in strong
        assert float(getattr(scan, n).p) > 0.0, n  # on in strong_scan
    for n in TRANSFORM_NAMES:
        if n not in SCAN_NAMES:  # strong_scan is strong + the scan family, nothing else
            assert getattr(scan, n) == getattr(strong, n), n
    assert scan.oob_fill == strong.oob_fill and scan.clip == strong.clip


@pytest.mark.parametrize("name", SCAN_NAMES)
def test_scan_transform_ct_only_p0_p1_and_in_range(name):
    b = synthetic_batch(2, 16, seed=9)
    ct0 = b["input"][:, 0].clone()
    o, fired = Augment(_scan_cfg(name, 0.0), seed=3)(b)  # p = 0: never fires, nothing moves
    assert fired[name] == 0 and sum(fired.values()) == 0
    for k, v in b.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, o[k]), k
    o, fired = Augment(_scan_cfg(name, 1.0), seed=3)(b)  # p = 1: fires on every sample
    assert fired[name] == 2 and sum(fired.values()) == 2
    for k, v in b.items():
        if isinstance(v, torch.Tensor) and k != "input":
            assert torch.equal(v, o[k]), k  # targets bit-identical
    ct = o["input"][:, 0]
    assert torch.equal(o["input"][:, 1], b["input"][:, 1])  # scale channel untouched
    assert torch.isfinite(ct).all() and not torch.equal(ct, ct0)
    assert float(ct.min()) >= 0.0 and float(ct.max()) <= 1.0  # [0, 255] in 8-bit units


def test_strong_scan_is_deterministic_and_finite():
    b = synthetic_batch(2, 16, seed=8)
    assert _digest("strong_scan", seed=5) == _digest("strong_scan", seed=5)
    aug = Augment("strong_scan", seed=4)
    tot = {k: 0 for k in TRANSFORM_NAMES}
    for _ in range(20):
        o, fired = aug(b)
        for k, v in fired.items():
            tot[k] += v
        for k, v in o.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), k
        assert o["surface_sdf"].abs().max() <= CLIP
    assert all(tot[n] > 0 for n in SCAN_NAMES if n != "stripe") and tot["artefact"] == 0


def _two_mode_crops(n: int, seed: int = 0, air: float = 42.0, pap: float = 112.0,
                    s_air: float = 6.0, s_pap: float = 20.0, shape=(8, 16, 16)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((n, 1, *shape), generator=g)
    which = (torch.rand((n, 1, *shape), generator=g) < 0.5).float()  # ~half air, half papyrus
    x = which * (pap + s_pap * z) + (1 - which) * (air + s_air * z)
    return (x / 255.0).clamp(0.0, 1.0)


def test_class_contrast_spans_the_requested_papyrus_spread():
    """Over 200 crops the papyrus-mode std is rescaled across the whole [s_lo, s_hi] range."""
    n = 200
    x = _two_mode_crops(n, seed=5)
    aug = Augment(_scan_cfg("class_contrast", 1.0), seed=7)
    y, fired = aug.apply_intensity(x)
    assert fired["class_contrast"] == n
    m = x > 100.0 / 255.0  # well above the air/papyrus midpoint + blend band: fully rescaled
    ratios = []
    for b in range(n):
        sel = m[b]
        ratios.append(float(y[b][sel].std() / x[b][sel].std()))
    r = np.array(ratios)
    assert (r >= 0.68).all() and (r <= 1.42).all()  # inside [s_lo, s_hi] up to the [0, 1] clip
    assert r.min() < 0.78 and r.max() > 1.32  # and it spans essentially the whole range
    assert abs(float(r.mean()) - 1.05) < 0.06  # uniform in [0.7, 1.4]
    # air is left alone: below the midpoint minus the blend band nothing changes
    air = x < 60.0 / 255.0
    assert float((y - x)[air].abs().max()) < 1e-6


def test_window_is_an_affine_remap_and_class_contrast_keeps_the_air_mode():
    x = _two_mode_crops(4, seed=11)
    a = Augment(_scan_cfg("window", 1.0, lo_lo=10.0, lo_hi=10.0, hi_lo=200.0, hi_hi=200.0), seed=1)
    y, fired = a.apply_intensity(x)
    assert fired["window"] == 4
    torch.testing.assert_close(y, ((x - 10.0 / 255.0) / (190.0 / 255.0)).clamp(0.0, 1.0), atol=1e-6, rtol=0)
    a = Augment(_scan_cfg("window", 1.0, lo_lo=0.0, lo_hi=0.0, hi_lo=255.0, hi_hi=255.0), seed=1)
    y, _ = a.apply_intensity(x)  # the identity window is the identity map
    torch.testing.assert_close(y, x, atol=1e-6, rtol=0)


def test_aniso_blur_smooths_z_and_yx_independently():
    n = 16
    c = _centred(n)
    ramp_z = (torch.sin(c[..., 0]) * 0.2 + 0.5)[None, None]  # varies along z only
    a = Augment(_scan_cfg("aniso_blur", 1.0, z_lo=1.5, z_hi=1.5, yx_lo=0.0, yx_hi=0.0), seed=2)
    y, _ = a.apply_intensity(ramp_z)
    assert y.std() < 0.5 * ramp_z.std()  # z blur flattens a z ripple
    a = Augment(_scan_cfg("aniso_blur", 1.0, z_lo=0.0, z_hi=0.0, yx_lo=1.5, yx_hi=1.5), seed=2)
    y, _ = a.apply_intensity(ramp_z)
    torch.testing.assert_close(y, ramp_z, atol=1e-6, rtol=0)  # in-plane blur of a z ripple: identity


def test_ring_and_stripe_are_constant_along_z():
    x = torch.full((1, 1, 6, 32, 32), 0.5)
    for name in ("ring", "stripe"):
        a = Augment(_scan_cfg(name, 1.0), seed=13)
        y, fired = a.apply_intensity(x)
        assert fired[name] == 1
        assert float((y - y[:, :, 0:1]).abs().max()) < 1e-6, name  # every z slice identical
        d = (y - x)[0, 0, 0]
        assert float(d.abs().max()) > 1e-4, name
        assert float(d.abs().max()) < 0.5 * 0.5, name  # low amplitude (< 50% of the level)


def test_spectral_noise_slope_follows_beta():
    def power_ratio(beta: float) -> float:
        a = Augment(_scan_cfg("spectral_noise", 1.0, beta_lo=beta, beta_hi=beta), seed=19)
        e = a._spectral_noise((24, 24, 24), torch.device("cpu"), beta)
        f = torch.fft.fftn(e[0, 0]).abs() ** 2
        k = torch.fft.fftfreq(24)
        kk = (k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2).sqrt()
        lo = f[(kk > 0) & (kk < 0.12)].mean()
        hi = f[kk > 0.3].mean()
        return float(lo / hi)
    assert power_ratio(-1.0) > 3.0 * power_ratio(0.0) > 3.0 * power_ratio(1.0)
    a = Augment(_scan_cfg("spectral_noise", 1.0, sigma_lo=6.0, sigma_hi=6.0, beta_lo=0.0, beta_hi=0.0), seed=21)
    x = torch.full((2, 1, 12, 16, 16), 0.5)
    y, _ = a.apply_intensity(x)
    assert 0.5 * 6.0 / 255.0 < float((y - x).std()) < 1.5 * 6.0 / 255.0  # requested grey-level std
