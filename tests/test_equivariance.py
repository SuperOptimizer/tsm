"""CPU checks for tsm.equivariance: group algebra, per-channel inverse rules, rotations, metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tsm import equivariance as E
from tsm.labels import decode_lasagna_normal


def _key(m: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.round(m).astype(int).ravel())


def test_group_closure_inverse_and_subgroups():
    els = {_key(g) for g in E.OCT48}
    assert len(els) == 48
    for g in E.OCT48:
        assert _key(g @ E.inverse(g)) == _key(np.eye(3))
        assert _key(E.inverse(g)) in els
        for h in E.OCT48:
            assert _key(g @ h) in els
    assert len(E.group_elements("flip8")) == 8 and len(E.group_elements("rot90z4")) == 4
    assert len(E.group_elements("rot24")) == 24 and len(E.group_elements("inplane16")) == 16
    for name in E.GROUPS:
        assert E.classify(E.group_elements(name)[0])["identity"]
    tags = [E.classify(g) for g in E.OCT48]
    assert sum(t["swaps_z"] for t in tags) == 32 and sum(t["pure_flip"] for t in tags) == 7
    assert sum(t["det"] == 1 for t in tags) == 24
    with pytest.raises(ValueError):
        E.group_elements("nope")


def test_apply_inverse_roundtrip_all_48_random_volume():
    torch.manual_seed(0)
    v = torch.randn(2, 3, 12, 12, 12)
    vn = v.numpy()
    for g in E.OCT48:
        assert torch.equal(E.apply(E.apply(v, g), E.inverse(g)), v), E.element_name(g)
        assert np.array_equal(E.apply(E.apply(vn, g), E.inverse(g)), vn)
        # composition: apply(apply(v, h), g) == apply(v, g h)
        h = E.OCT48[7]
        assert torch.equal(E.apply(E.apply(v, h), g), E.apply(v, g @ h))


def test_apply_moves_plane_wave_normal_by_g():
    """apply(V, g) of f(n . p) is f((g n) . p): the matrix really is the geometric action."""
    p = 16
    zz, yy, xx = np.meshgrid(*(np.arange(p) - (p - 1) / 2,) * 3, indexing="ij")
    rng = np.random.default_rng(0)
    for g in E.OCT48:
        n = rng.normal(size=3)
        vol = np.sin(0.7 * (n[0] * zz + n[1] * yy + n[2] * xx))
        n2 = g @ n
        expect = np.sin(0.7 * (n2[0] * zz + n2[1] * yy + n2[2] * xx))
        assert np.allclose(E.apply(vol, g), expect, atol=1e-9), E.element_name(g)


# --------------------------------------------------------------------------- #
# synthetic exactly-equivariant "models"
# --------------------------------------------------------------------------- #
def _sheet_ct(p: int, n: np.ndarray, width: float = 3.0) -> torch.Tensor:
    """Signed step-like field tanh((n . p) / width) of a planar sheet through the centre: (Z, Y, X)."""
    zz, yy, xx = np.meshgrid(*(np.arange(p, dtype=np.float64) - (p - 1) / 2,) * 3, indexing="ij")
    return torch.from_numpy(np.tanh((n[0] * zz + n[1] * yy + n[2] * xx) / width))


def _analytic_model(v: torch.Tensor, width: float = 3.0) -> dict[str, torch.Tensor]:
    """sdf = atanh(v) * width (scalar), normal = grad v / |grad v| (vector, (z, y, x)) -- both equivariant."""
    sdf = torch.atanh(v.clamp(-0.999999, 0.999999)) * width
    gz, gy, gx = torch.gradient(v, dim=(0, 1, 2), edge_order=1)
    n = torch.stack([gz, gy, gx])
    n = n / n.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return {"sdf": sdf, "normal_zyx": n, "normal_xyz": n.flip(0)}


def test_vector_rule_on_planar_sheet_all_48():
    p = 14
    n = np.array([0.3, -0.5, 0.8])
    n /= np.linalg.norm(n)
    v = _sheet_ct(p, n)
    ref = _analytic_model(v)
    for g in E.OCT48:
        out = _analytic_model(E.apply(v, g))
        sdf_back = E.invert_prediction(out["sdf"], g, "scalar")
        assert torch.allclose(sdf_back, ref["sdf"], atol=1e-9), E.element_name(g)
        nz_back = E.invert_prediction(out["normal_zyx"], g, "vector_zyx")
        assert torch.allclose(nz_back, ref["normal_zyx"], atol=1e-9), E.element_name(g)
        nx_back = E.invert_prediction(out["normal_xyz"], g, "vector_xyz")
        assert torch.allclose(nx_back, ref["normal_xyz"], atol=1e-9), E.element_name(g)
        # the transformed prediction itself has the rotated normal g n at the rotated point
        assert torch.allclose(out["normal_zyx"], E.apply_vector(ref["normal_zyx"], g), atol=1e-9)
        # wrong rule (treating the normal as scalars) must fail for anything that flips or permutes
        if not E.classify(g)["identity"]:
            assert not torch.allclose(E.invert_prediction(out["normal_zyx"], g, "scalar"), ref["normal_zyx"], atol=1e-6)


def test_student_rule_mixes_only_the_normal_channels():
    p = 12
    n = np.array([-0.6, 0.6, -0.5])
    n /= np.linalg.norm(n)
    v = _sheet_ct(p, n)

    def student(vol: torch.Tensor) -> torch.Tensor:
        m = _analytic_model(vol)
        s = m["sdf"][None]
        return torch.cat([s, torch.sigmoid(s), 1 - torch.sigmoid(s), torch.sin(s), torch.cos(s), s.abs(),
                          m["normal_xyz"], torch.sigmoid(-s), torch.zeros_like(s)], dim=0)  # (11, Z, Y, X)

    ref = student(v)
    for g in E.OCT48:
        back = E.invert_prediction(student(E.apply(v, g)), g, "student")
        assert back.shape == ref.shape
        assert torch.allclose(back, ref, atol=1e-6), E.element_name(g)


# --------------------------------------------------------------------------- #
# lasagna encoded directions
# --------------------------------------------------------------------------- #
def test_lasagna_dir_roundtrip_under_all_48():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(3, 300))
    n /= np.linalg.norm(n, axis=0)
    ch = np.concatenate([np.ones((1, 300)), np.ones((1, 300)) * 0.4, E.encode_lasagna_dirs(n)])
    # decode -> encode is the identity on consistent channels (sign-free)
    dec = decode_lasagna_normal(ch[:, :, None, None])[:, :, 0, 0]
    assert np.abs(np.abs((dec * n).sum(0)) - 1).max() < 1e-5
    assert np.allclose(E.encode_lasagna_dirs(dec), ch[2:], atol=1e-5)
    for g in E.OCT48:
        got = E.transform_lasagna_channels(ch, g)
        expect = np.concatenate([ch[:2], E.encode_lasagna_dirs(g @ n)])
        assert np.allclose(got, expect, atol=1e-5), E.element_name(g)
        # the flipped-sign normal encodes identically
        assert np.allclose(E.encode_lasagna_dirs(-(g @ n)), expect[2:], atol=1e-12)


class _GradientLasagna:
    """Exactly equivariant lasagna: encodes the CT gradient line as the 8 channels."""

    def __call__(self, v: torch.Tensor) -> np.ndarray:
        gz, gy, gx = torch.gradient(v, dim=(0, 1, 2), edge_order=1)
        n = torch.stack([gz, gy, gx]).numpy()
        mag = np.linalg.norm(n, axis=0)
        return np.concatenate([np.ones((1,) + v.shape), mag[None], E.encode_lasagna_dirs(n)]).astype(np.float32)


def test_invert_lasagna_on_a_sheet_all_48():
    p = 12
    n = np.array([0.2, 0.9, -0.4])
    n /= np.linalg.norm(n)
    v = _sheet_ct(p, n, width=4.0)
    model = _GradientLasagna()
    ref = model(v)
    for g in E.OCT48:
        back = E.invert_prediction(model(E.apply(v, g)), g, "lasagna")
        assert back.shape == ref.shape
        assert np.allclose(back, ref, atol=1e-4), E.element_name(g)


# --------------------------------------------------------------------------- #
# arbitrary rotations
# --------------------------------------------------------------------------- #
def test_random_rotation_matrix_is_proper_orthogonal():
    for seed in range(5):
        for max_deg in (None, 15.0):
            R = E.random_rotation_matrix(seed, max_deg)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
            if max_deg is not None:
                ang = math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2)))
                assert ang <= max_deg + 1e-9
    assert not E.is_signed_permutation(E.random_rotation_matrix(0, 15.0))


def test_apply_rotation_roundtrip_and_vector_rule():
    p = 24
    n = np.array([0.3, 0.5, 0.8])
    n /= np.linalg.norm(n)
    v = _sheet_ct(p, n, width=6.0).float()
    R = E.random_rotation_matrix(3, 12.0)
    mask = E.rotation_valid_mask((p, p, p), R, margin=3)
    assert 0.2 < mask.mean() < 1.0
    back = E.apply_rotation(E.apply_rotation(v, R), E.inverse(R))
    assert (back - v).abs()[torch.from_numpy(mask)].max() < 0.02
    # nearest keeps integer labels
    lab = (v > 0).to(torch.uint8)
    rl = E.apply_rotation(lab, R, mode="nearest")
    assert rl.dtype == torch.uint8 and set(rl.unique().tolist()) <= {0, 1}
    # the analytic model on the rotated sheet, inverted with the general-matrix path
    ref = _analytic_model(v, width=6.0)
    out = _analytic_model(E.apply_rotation(v, R), width=6.0)
    nb = E.invert_prediction(out["normal_zyx"], R, "vector_zyx")
    inner = torch.from_numpy(mask) & (ref["sdf"].abs() < 4)
    cos = (nb * ref["normal_zyx"]).sum(0)[inner]
    assert cos.min() > 0.995
    sb = E.invert_prediction(out["sdf"], R, "scalar")
    assert (sb - ref["sdf"]).abs()[inner].max() < 0.5


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _surface_prob(p: int) -> np.ndarray:
    zz = np.arange(p)[:, None, None] * np.ones((p, p, p))
    return np.exp(-((zz - p / 2) ** 2) / 4.0).astype(np.float32)


def test_metrics_identical_inputs_are_zero():
    p = 24
    prob = _surface_prob(p)
    m = E.metrics(prob, prob, "surface", min_component=10)
    assert m["dice"] == 1.0 and m["mean_abs"] == 0.0
    assert m["medial"]["symmetric"]["median"] == 0.0 and m["medial"]["symmetric"]["frac_gt3"] == 0.0
    assert E.primary_metric("surface", m)[1] == 0.0

    ink = (prob * 0.9).astype(np.float32)
    mi = E.metrics(ink, ink, "ink")
    assert mi["dice"] == 1.0 and mi["auprc"] == pytest.approx(1.0) and mi["mean_abs"] == 0.0

    n = np.array([0.2, 0.9, -0.4])
    n /= np.linalg.norm(n)
    las = _GradientLasagna()(_sheet_ct(p, n, width=4.0))
    ml = E.metrics(las, las, "lasagna")
    assert ml["cos_mae"] == 0.0 and ml["normal_angle"]["mean"] == 0.0 and ml["normal_angle"]["n"] > 0

    am = _analytic_model(_sheet_ct(p, n, width=3.0))
    s = am["sdf"][None]
    st = torch.cat([s, torch.sigmoid(s) * 0 + 0.9, torch.sigmoid(s), torch.sin(s), torch.cos(s), s.abs(),
                    am["normal_xyz"], torch.sigmoid(-s), torch.zeros_like(s)], dim=0)
    ms = E.metrics(st, st, "student", clip=20.0)
    assert ms["sdf_mae_band"] == 0.0 and ms["zero_crossing"]["symmetric"]["median"] == 0.0
    assert ms["normal_angle"]["mean"] == 0.0 and ms["phase_err_deg"]["mean"] == 0.0 and ms["ink_dice"] == 1.0


def test_metrics_detect_a_shift():
    p = 24
    prob = _surface_prob(p)
    shifted = np.roll(prob, 4, axis=0)
    m = E.metrics(prob, shifted, "surface", min_component=10)
    assert m["dice"] < 0.5 and m["medial"]["symmetric"]["median"] == pytest.approx(4.0)
    assert m["medial"]["symmetric"]["frac_gt3"] == 1.0 and m["medial"]["symmetric"]["frac_gt5"] == 0.0
    mi = E.metrics(prob, shifted, "ink")
    assert mi["auprc"] is not None and mi["auprc"] < 0.6
    assert E.average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert E.average_precision(np.array([0.1, 0.8, 0.9]), np.array([1, 1, 0])) == pytest.approx((0.5 + 2 / 3) / 2)
