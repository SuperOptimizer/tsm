"""Local Shape Descriptors for sheets (``extra.train.heads.lsd``, :func:`tsm.data.lsd_targets`).

Sheridan et al., "Local Shape Descriptors for Neuron Segmentation" (Nat Methods 2023), adapted
to papyrus sheets: an optional auxiliary head that regresses, per body voxel, the offset to the
centre plane of its own sheet (a vector), the windowed sheet thickness (a length) and the
windowed normal incoherence (a rotation-invariant scalar).

Everything here is CPU and tiny: the analytic descriptors, the augmentation transport rules,
the loss, one train + evaluate round trip, the prediction channels, the viewer table, and the
proof that a run with the head off is bit-identical to one that has never heard of it.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from synth import AXIS_YX, make_synthetic
from tsm.config import parse_config
from tsm.data import (
    CLIP,
    LSD_CHANNELS,
    LSD_SIGMA,
    Augment,
    CropDataset,
    SpatialParams,
    augment,
    lsd_coarse_factor,
    rot90_matrix_inplane,
    lsd_targets,
    rotation_matrix,
    target_keys,
)
from tsm.equivariance import group_elements
from tsm.infer import (
    LSD_PRED_CHANNELS,
    activate_heads,
    checkpoint_lsd,
    decode_pred,
    load_student,
    n_head_ch,
    pred_channels,
    to_unit,
    tta_inverse,
)
from tsm.student import build_model, heads_for
from tsm.train import (
    compute_losses,
    downsample_targets,
    evaluate,
    lsd_loss,
    run_train,
    synthetic_batch,
    train_opts,
)
from tsm.view import layer_for


# --------------------------------------------------------------------------- #
# analytic fixtures
# --------------------------------------------------------------------------- #
def flat_sheet(n: int = 48, centre: float = 24.0, thick: float = 10.0):
    """One flat sheet perpendicular to +z: ``(sdf_in, sdf_out, valid, normal)``."""
    z = np.arange(n, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, n, n), np.float32)
    si = np.clip(z - (centre - thick / 2.0), -CLIP, CLIP)
    so = np.clip(z - (centre + thick / 2.0), -CLIP, CLIP)
    nrm = np.zeros((3, n, n, n), np.float32)
    nrm[0] = 1.0
    return si, so, np.ones((n, n, n), np.float32), nrm


def crossing_sheets(n: int = 64, thick: float = 10.0):
    """Two perpendicular sheets (one normal to z, one normal to y) that cross at the centre."""
    g = np.arange(n, dtype=np.float32)
    z = g.reshape(-1, 1, 1) * np.ones((1, n, n), np.float32)
    y = g.reshape(1, -1, 1) * np.ones((n, 1, n), np.float32)
    c = n / 2.0

    def faces(coord):
        return (np.clip(coord - (c - thick / 2.0), -CLIP, CLIP),
                np.clip(coord - (c + thick / 2.0), -CLIP, CLIP))

    az, ao = faces(z)
    by, bo = faces(y)
    use_z = np.minimum(az, -ao) >= np.minimum(by, -bo)  # the nearer sheet owns the voxel
    si = np.where(use_z, az, by).astype(np.float32)
    so = np.where(use_z, ao, bo).astype(np.float32)
    nrm = np.zeros((3, n, n, n), np.float32)
    nrm[0] = use_z
    nrm[1] = ~use_z
    return si, so, np.ones((n, n, n), np.float32), nrm


# --------------------------------------------------------------------------- #
# the descriptors themselves
# --------------------------------------------------------------------------- #
def test_flat_sheet_descriptors_are_exact():
    n, c, t = 48, 24.0, 10.0
    si, so, valid, nrm = flat_sheet(n, c, t)
    lsd, lv = lsd_targets(si, so, valid, nrm)
    assert lsd.shape == (5, n, n, n) and lv.shape == (1, n, n, n)
    body = lv[0] > 0
    z = np.arange(n, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, n, n), np.float32)
    # the body is exactly the open slab (sdf_in > 0 > sdf_out)
    np.testing.assert_array_equal(body, (si > 0) & (so < 0))
    # off = the signed distance to the mid-plane along the (outward) normal: n * (c - z)
    np.testing.assert_allclose(lsd[0][body], (c - z)[body], atol=1e-5)
    assert np.abs(lsd[1:3][:, body]).max() == 0.0          # no y / x component
    np.testing.assert_allclose(lsd[3][body], t, atol=1e-4)  # thickness
    assert np.abs(lsd[4][body]).max() < 1e-5                # a flat sheet is coherent
    # everything is 0 off the body
    assert np.abs(lsd[:, ~body]).max() == 0.0


def test_offset_points_at_the_centre_plane_from_both_halves():
    si, so, valid, nrm = flat_sheet(48, 24.0, 10.0)
    lsd, lv = lsd_targets(si, so, valid, nrm)
    col = lsd[0][:, 0, 0]
    assert col[21] > 0 and col[27] < 0                      # below the centre -> +n, above -> -n
    assert abs(float(col[24])) < 1e-6                       # zero on the medial plane
    assert abs(float(col[20]) - 4.0) < 1e-5                 # |off| <= half the thickness
    assert float(np.abs(lsd[0:3][:, lv[0] > 0]).max()) <= 5.0 + 1e-5


def test_crossing_sheets_are_incoherent_and_flat_regions_are_not():
    si, so, valid, nrm = crossing_sheets(64)
    lsd, lv = lsd_targets(si, so, valid, nrm)
    body = lv[0] > 0
    ncov = lsd[4]
    assert ncov[32, 32, 32] > 0.3                            # at the crossing
    assert ncov[32, 8, 8] < 0.02 and ncov[8, 32, 8] < 0.02   # far along either sheet
    assert 0.0 <= ncov[body].min() and ncov[body].max() <= 2.0 / 3.0 + 1e-3


def test_valid_mask_follows_the_label_body_and_the_valid_channel():
    n = 32
    si, so, valid, nrm = flat_sheet(n, 16.0, 8.0)
    valid = valid.copy()
    valid[:, :8] = 2.0  # "ignore" half the crop
    lsd, lv = lsd_targets(si, so, valid, nrm)
    assert lv[:, :, :8].max() == 0.0
    assert lv[:, :, 8:].max() == 1.0
    # a missing normal also drops the voxel (nothing to project on)
    nrm2 = nrm.copy()
    nrm2[:, :, :, :4] = 0.0
    _, lv2 = lsd_targets(si, so, np.ones_like(valid), nrm2)
    assert lv2[:, :, :, :4].max() == 0.0 and lv2[:, :, :, 4:].max() == 1.0


def test_thickness_is_the_window_average_over_body_voxels():
    """Two sheets of different thickness in one crop: the window average sits between them."""
    n = 64
    g = np.arange(n, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, n, n), np.float32)
    # sheet A: z in (10, 20) (thick 10); sheet B: z in (30, 34) (thick 4)
    si = np.where(g < 25, g - 10.0, g - 30.0)
    so = np.where(g < 25, g - 20.0, g - 34.0)
    si, so = np.clip(si, -CLIP, CLIP).astype(np.float32), np.clip(so, -CLIP, CLIP).astype(np.float32)
    nrm = np.zeros((3, n, n, n), np.float32)
    nrm[0] = 1.0
    lsd, lv = lsd_targets(si, so, np.ones((n, n, n), np.float32), nrm, sigma=6.0)
    th = lsd[3]
    assert 4.0 < th[15, 0, 0] < 10.0        # pulled down by the thin sheet 10 voxels away
    assert 4.0 < th[32, 0, 0] < 10.0        # ... and up by the thick one
    assert th[15, 0, 0] > th[32, 0, 0]


def test_coarse_factor_follows_the_window_and_the_result_barely_depends_on_it():
    assert lsd_coarse_factor(6.0) == 4 and lsd_coarse_factor(3.0) == 2 and lsd_coarse_factor(1.0) == 1
    si, so, valid, nrm = crossing_sheets(64)
    ref, lv = lsd_targets(si, so, valid, nrm, factor=1)
    got, _ = lsd_targets(si, so, valid, nrm)  # the auto factor (4 at sigma 6)
    body = lv[0] > 0
    assert np.abs(got[0:3] - ref[0:3]).max() == 0.0            # the offset is never coarsened
    assert np.abs(got[3][body] - ref[3][body]).max() < 1.0     # < 1 voxel of thickness
    # the incoherence is block-replicated from the coarse grid: a few hundredths on average,
    # up to ~0.15 on the steepest transition (at the crossing itself)
    d = np.abs(got[4][body] - ref[4][body])
    assert d.mean() < 0.05 and d.max() < 0.2


def test_descriptors_commute_with_the_cube_symmetries():
    """The derivation itself is equivariant: rotating the labels rotates the descriptors."""
    si, so, valid, nrm = crossing_sheets(32)
    ref, rv = lsd_targets(si, so, valid, nrm)
    for g in group_elements("oct48")[:12]:
        gi = np.asarray(g, np.int64)
        perm = [int(np.argmax(np.abs(gi[i]))) for i in range(3)]        # axis i of the output
        sign = [float(gi[i, perm[i]]) for i in range(3)]

        def move(a):  # (..., Z, Y, X) -> the transformed volume
            b = np.transpose(a, list(range(a.ndim - 3)) + [a.ndim - 3 + perm.index(i) for i in range(3)])
            for i, s in enumerate(sign):
                if s < 0:
                    b = np.flip(b, axis=b.ndim - 3 + perm.index(i))
            return np.ascontiguousarray(b)

        n2 = np.einsum("ij,jzyx->izyx", gi.astype(np.float32), move(nrm))
        got, gv = lsd_targets(move(si), move(so), move(valid), n2)
        want = np.concatenate([np.einsum("ij,jzyx->izyx", gi.astype(np.float32), move(ref[0:3])),
                               move(ref[3:5])], 0)
        np.testing.assert_array_equal(gv, move(rv))
        assert np.abs(got - want).max() < 1e-5


# --------------------------------------------------------------------------- #
# target plumbing
# --------------------------------------------------------------------------- #
def test_target_keys_and_head_widths():
    t = target_keys("sides", lsd=True)
    assert t["lsd"] == 5 and t["lsd_valid"] == 1
    assert "lsd" not in target_keys("sides")
    with pytest.raises(ValueError, match="sides"):
        target_keys("faces", lsd=True)
    h = heads_for("sides", fiber=True, fiber_mode="direction", gap_class=True, lsd=True)
    assert h == {"surface": 4, "ink": 1, "winding": 8, "fiber": 4, "lsd": 5}
    assert "lsd" not in heads_for("sides", fiber=True, fiber_mode="direction")
    with pytest.raises(ValueError, match="sides"):
        heads_for("body", lsd=True)
    assert LSD_CHANNELS == ["lsd_off_z", "lsd_off_y", "lsd_off_x", "lsd_thick", "lsd_ncov"]


def test_dataset_derives_the_targets_and_to_tensors_carries_them(tmp_path):
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    kw = dict(patch=16, stride=16, length=4, augment=False, surface_mode="sides")
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"], lsd=True, **kw)
    assert ds.target_keys["lsd"] == 5 and ds.lsd_sigma == LSD_SIGMA
    t = ds.load(ds.origins[0])
    assert t["lsd"].shape == (5, 16, 16, 16) and t["lsd_valid"].shape == (1, 16, 16, 16)
    body = t["lsd_valid"][0] > 0
    assert body.any()
    # the mask is the label body of the sides target, restricted to valid == 1
    np.testing.assert_array_equal(body, (t["surface_body"][0] > 0) & (t["surface_valid"][0] == 1))
    # the thickness is positive on the body and everything is finite
    assert np.isfinite(t["lsd"]).all() and t["lsd"][3][body].min() > 0
    tt = ds.to_tensors(t)
    assert tt["lsd"].shape == (5, 16, 16, 16) and tt["lsd_valid"].dtype == torch.float32
    # ... and a dataset without the flag has neither key
    ds0 = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"], **kw)
    assert "lsd" not in ds0.load(ds0.origins[0])
    with pytest.raises(ValueError, match="sides"):
        CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                    length=2, augment=False, surface_mode="faces", lsd=True)


# --------------------------------------------------------------------------- #
# augmentation: off is a vector and a length, thick a length, ncov invariant
# --------------------------------------------------------------------------- #
def _lsd_batch(P: int = 12, off=(3.0, -1.0, 2.0), thick: float = 14.0, ncov: float = 0.3):
    """A constant-field batch carrying the lsd target (so resampling is exact)."""
    one = torch.ones(1, 1, P, P, P)
    o = torch.tensor(off, dtype=torch.float32).view(1, 3, 1, 1, 1).expand(1, 3, P, P, P)
    lsd = torch.cat([o, one * thick, one * ncov], dim=1).contiguous()
    return {
        "input": torch.zeros(1, 2, P, P, P),
        "surface_sdf": torch.zeros(1, 1, P, P, P), "surface_valid": one.clone(),
        "surface_body": one.clone(),
        "ink_prob": torch.zeros(1, 1, P, P, P), "ink_valid": one.clone(),
        "winding": torch.zeros(1, 6, P, P, P), "winding_conf": one.clone(), "winding_valid": one.clone(),
        "lsd": lsd, "lsd_valid": one.clone(),
    }


CUBE = ([SpatialParams(rot90=torch.tensor(np.asarray(g, np.float64)).float())
         for g in group_elements("rot24")]
        + [SpatialParams(flips=tuple(bool(v) for v in f))
           for f in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]])


@pytest.mark.parametrize("p", CUBE, ids=[str(i) for i in range(len(CUBE))])
def test_apply_spatial_transports_the_descriptors_under_every_cube_symmetry(p):
    off = torch.tensor([3.0, -1.0, 2.0])
    b = _lsd_batch(off=tuple(off.tolist()))
    out = Augment("none").apply_spatial(b, [p])
    L = p.matrix().float()
    c = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))
    got = out["lsd"][c]
    want = L @ off
    assert (got[:, 0:3] - want.view(1, 3, 1, 1, 1)).abs().max() < 1e-5
    assert (got[:, 3:4] - 14.0).abs().max() < 1e-5   # a cube symmetry is an isometry
    assert (got[:, 4:5] - 0.3).abs().max() < 1e-5    # ... and ncov is invariant anyway
    assert set(out["lsd_valid"].unique().tolist()) <= {0.0, 1.0}
    assert torch.all(out["lsd_valid"][c] == 1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_apply_spatial_transports_the_offset_under_sub_voxel_rotations(seed):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(3, generator=g)
    R = rotation_matrix(axis, math.radians(float(torch.rand(1, generator=g)) * 40.0 - 20.0))
    off = torch.tensor([3.0, -1.0, 2.0])
    b = _lsd_batch(P=16, off=tuple(off.tolist()))
    out = Augment("none").apply_spatial(b, [SpatialParams(rotation=R)])
    c = (slice(None), slice(None), slice(3, -3), slice(3, -3), slice(3, -3))
    got = out["lsd"][c]
    want = R @ off
    assert (got[:, 0:3] - want.view(1, 3, 1, 1, 1)).abs().max() < 1e-3
    assert (got[:, 3:4] - 14.0).abs().max() < 1e-3
    assert (got[:, 4:5] - 0.3).abs().max() < 1e-3


def test_isotropic_scale_multiplies_the_two_lengths_and_leaves_ncov_alone():
    off = torch.tensor([3.0, -1.0, 2.0])
    for s in (0.8, 1.25):
        b = _lsd_batch(P=16, off=tuple(off.tolist()))
        p = SpatialParams(scale=torch.tensor([s, s, s]))
        out = Augment("none").apply_spatial(b, [p])
        c = (slice(None), slice(None), slice(3, -3), slice(3, -3), slice(3, -3))
        got = out["lsd"][c]
        assert (got[:, 0:3] - (off * s).view(1, 3, 1, 1, 1)).abs().max() < 1e-4
        assert (got[:, 3:4] - 14.0 * s).abs().max() < 1e-4
        assert (got[:, 4:5] - 0.3).abs().max() < 1e-5


def test_v1_augmentation_flips_and_rotates_the_offset_vector():
    P = 8
    rng = np.random.default_rng(3)
    off = np.array([3.0, -1.0, 2.0], np.float32)
    lsd = np.concatenate([np.broadcast_to(off.reshape(3, 1, 1, 1), (3, P, P, P)),
                          np.full((1, P, P, P), 14.0, np.float32),
                          np.full((1, P, P, P), 0.3, np.float32)], 0).copy()
    sample = {"ct": np.zeros((1, P, P, P), np.float32), "lsd": lsd,
              "lsd_valid": np.ones((1, P, P, P), np.float32)}
    for _ in range(8):
        out = augment(dict(sample), rng, intensity=False)
        flips, k = out["aug"]["flips"], out["aug"]["rot_k"]
        L = (rot90_matrix_inplane(k).double() @ SpatialParams(flips=tuple(flips)).matrix()).float()
        want = (L @ torch.from_numpy(off)).numpy()
        got = out["lsd"][:, 2:-2, 2:-2, 2:-2]
        assert np.abs(got[0:3] - want.reshape(3, 1, 1, 1)).max() < 1e-5
        assert np.abs(got[3] - 14.0).max() < 1e-5 and np.abs(got[4] - 0.3).max() < 1e-5


def test_downsample_targets_masked_average():
    P = 4
    lsd = torch.zeros(1, 5, P, P, P)
    lsd[:, 0] = 2.0
    valid = torch.ones(1, 1, P, P, P)
    valid[:, :, 0::2] = 0.0     # only half the voxels of each 2-block are supervised
    lsd = lsd * valid
    t = {"surface_sdf": torch.zeros(1, 1, P, P, P), "surface_valid": torch.ones(1, 1, P, P, P),
         "ink_prob": torch.zeros(1, 1, P, P, P), "ink_valid": torch.ones(1, 1, P, P, P),
         "winding": torch.zeros(1, 6, P, P, P), "winding_conf": torch.zeros(1, 1, P, P, P),
         "winding_valid": torch.ones(1, 1, P, P, P), "lsd": lsd, "lsd_valid": valid}
    d = downsample_targets(t)
    assert d["lsd"].shape == (1, 5, 2, 2, 2) and d["lsd_valid"].shape == (1, 1, 2, 2, 2)
    # the masked average is the value of the supervised half, not half of it
    assert torch.allclose(d["lsd"][:, 0], torch.full((1, 2, 2, 2), 2.0))
    assert torch.all(d["lsd_valid"] == 0.0)  # decimation keeps the [0] plane, which is masked


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def test_lsd_loss_is_zero_for_a_perfect_prediction_and_backprops():
    b = synthetic_batch(2, 8, surface_mode="sides", lsd=True)
    t = b["lsd"]
    perfect = torch.cat([t[:, 0:4], torch.logit(t[:, 4:5].clamp(1e-4, 1 - 1e-4))], 1)
    d = lsd_loss(perfect, t, b["lsd_valid"])
    assert set(d) == {"off_l1", "thick_l1", "ncov_l1"}
    for k, v in d.items():
        assert float(v) == pytest.approx(0.0, abs=1e-5), k
    p = (perfect + 1.0).detach().requires_grad_(True)
    d2 = lsd_loss(p, t, b["lsd_valid"])
    tot = sum(d2.values())
    assert float(tot.detach()) > 0.0
    tot.backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    # masked: a voxel outside lsd_valid == 1 never contributes
    bad = perfect.clone()
    bad[b["lsd_valid"].expand_as(bad) != 1] += 100.0
    for v in lsd_loss(bad, t, b["lsd_valid"]).values():
        assert float(v) == pytest.approx(0.0, abs=1e-5)
    with pytest.raises(ValueError, match="5-channel"):
        lsd_loss(perfect[:, 0:3], t[:, 0:3], b["lsd_valid"])


def test_compute_losses_wires_the_head_under_its_own_weight():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16, 16, 16), in_ch=2, surface_mode="sides", lsd=True)
    b = synthetic_batch(1, 16, surface_mode="sides", lsd=True)
    out = model(b["input"])
    loss, parts = compute_losses(out, b, surface_mode="sides")
    for k in ("lsd/off_l1", "lsd/thick_l1", "lsd/ncov_l1", "lsd"):
        assert k in parts and np.isfinite(parts[k]), k
    loss.backward()
    # the loss weight scales exactly that head's contribution
    _, p2 = compute_losses(out, b, {"lsd": 0.0}, surface_mode="sides")
    assert p2["lsd"] == pytest.approx(parts["lsd"])
    assert p2["total"] == pytest.approx(parts["total"] - parts["lsd"], rel=1e-5)
    # and the balancer sees it as a head of its own
    _, p3 = compute_losses(out, b, surface_mode="sides", head_scale={"lsd": 2.0})
    assert p3["balance/lsd"] == 2.0


def test_defaults_off_is_bit_identical_on_a_synthetic_batch():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16, 16, 16), in_ch=2, surface_mode="sides")
    assert "lsd" not in {k: v for k, v in model.head.items()}
    b = synthetic_batch(1, 16, surface_mode="sides")
    assert "lsd" not in b and "lsd_valid" not in b
    out = model(b["input"])
    loss, parts = compute_losses(out, b, surface_mode="sides")
    assert not [k for k in parts if k.startswith("lsd")]
    # a batch that happens to carry the target but a model without the head changes nothing
    b2 = synthetic_batch(1, 16, surface_mode="sides", lsd=True)
    for k in ("input", "surface_sdf", "surface_valid", "surface_body", "ink_prob", "ink_valid",
              "winding", "winding_conf", "winding_valid"):
        b2[k] = b[k]
    loss2, parts2 = compute_losses(model(b2["input"]), b2, surface_mode="sides")
    assert float(loss2.detach()) == float(loss.detach()) and parts2 == parts


def test_train_opts_defaults_and_validation():
    def cfg(train):
        return parse_config({
            "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
            "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
            "out_dir": "/tmp/tsm-none", "extra": {"train": train},
        })

    o = train_opts(cfg({}))
    assert o["heads"] == {"fiber": False, "lsd": False} and o["lsd_sigma"] == 6.0
    assert "lsd" not in o["loss_weights"]  # the historical weight set, untouched
    o = train_opts(cfg({"surface_mode": "sides", "heads": {"lsd": True}}))
    assert o["heads"]["lsd"] and o["loss_weights"]["lsd"] == 1.0
    o = train_opts(cfg({"surface_mode": "sides", "heads": {"lsd": True},
                        "loss_weights": {"lsd": 0.25}, "lsd_sigma": 4.0}))
    assert o["loss_weights"]["lsd"] == 0.25 and o["lsd_sigma"] == 4.0
    with pytest.raises(ValueError, match="sides"):
        train_opts(cfg({"heads": {"lsd": True}}))
    with pytest.raises(ValueError, match="lsd_sigma"):
        train_opts(cfg({"lsd_sigma": 0}))


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


def test_two_train_steps_and_evaluate_with_the_lsd_head(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    extra = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1,
             "coarse_store": s["coarse"], "fine_store": s["fine"], "augment": False,
             "surface_mode": "sides", "heads": {"lsd": True}, "lsd_sigma": 4.0,
             "input_radial": False, "input_axis": False,
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
    for k in ("lsd/off_mae", "lsd/thick_mae", "lsd/ncov_mae"):
        assert k in m and np.isfinite(m[k]), k
    assert 0.0 <= m["lsd/ncov_mae"] <= 1.0
    ck = torch.load(os.path.join(root, "train", "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["config"]["lsd"] is True and checkpoint_lsd(ck["config"])
    model, info = load_student(os.path.join(root, "train", "latest.pt"))
    assert info["lsd"] is True and model.head["lsd"][0].out_channels == 5


def test_evaluate_lsd_metrics_are_exact_for_a_perfect_prediction(tmp_path):
    from tsm.volume import VolumeReader

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                     length=4, augment=False, surface_mode="sides", lsd=True, lsd_sigma=3.0)
    origins = ds.origins[:2]

    class Copy(torch.nn.Module):
        def __init__(self, err: float = 0.0) -> None:
            super().__init__()
            self.err = err
            self.i = 0
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            n = int(x.shape[0])
            items = [ds.load(o) for o in origins[self.i:self.i + n]]
            self.i += n
            t = torch.stack([torch.from_numpy(np.asarray(it["lsd"])) for it in items])
            d = torch.zeros(t.shape[0], 1, *t.shape[2:])
            lsd = torch.cat([t[:, 0:4] + self.err,
                             torch.logit(t[:, 4:5].clamp(1e-4, 1 - 1e-4))], 1)
            return {"surface": torch.cat([d, d, d], 1), "ink": d,
                    "winding": torch.zeros(d.shape[0], 8, *d.shape[2:]), "lsd": lsd}

    m = evaluate(Copy(0.0), ds, origins, batch=1, surface_mode="sides")
    assert m["lsd/off_mae"] == pytest.approx(0.0, abs=1e-4)
    assert m["lsd/thick_mae"] == pytest.approx(0.0, abs=1e-4)
    assert m["lsd/ncov_mae"] == pytest.approx(0.0, abs=1e-4)
    m2 = evaluate(Copy(2.0), ds, origins, batch=1, surface_mode="sides")
    assert m2["lsd/off_mae"] == pytest.approx(2.0, abs=1e-3)     # voxels, not normalised
    assert m2["lsd/thick_mae"] == pytest.approx(2.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# inference / export / viewer
# --------------------------------------------------------------------------- #
def test_pred_channels_and_head_count():
    ch = pred_channels("sides", True, "direction", False, True)
    assert LSD_PRED_CHANNELS == LSD_CHANNELS
    head = ch[: n_head_ch("sides", True, "direction", False, True)]
    assert head[-5:] == LSD_PRED_CHANNELS                 # the head channels stay contiguous
    assert ch[: ch.index("lsd_off_z")] == pred_channels("sides", True, "direction")[: ch.index("lsd_off_z")]
    assert n_head_ch("sides", True, "direction", False, True) == n_head_ch("sides", True, "direction") + 5
    assert pred_channels("sides") == pred_channels("sides", lsd=False)
    with pytest.raises(ValueError, match="sides"):
        pred_channels("body", lsd=True)
    with pytest.raises(ValueError, match="sides"):
        n_head_ch("faces", lsd=True)


def test_activate_heads_to_unit_and_decode_round_trip():
    B, P = 1, 4
    clip = CLIP
    off = torch.tensor([5.0, -12.0, 0.0]).view(1, 3, 1, 1, 1).expand(B, 3, P, P, P)
    raw = {
        "surface": torch.cat([torch.full((B, 1, P, P, P), 3.0), torch.zeros(B, 2, P, P, P)], 1),
        "ink": torch.zeros(B, 1, P, P, P),
        "winding": torch.cat([torch.ones(B, 2, P, P, P), torch.full((B, 1, P, P, P), 0.05),
                              torch.ones(B, 3, P, P, P), torch.zeros(B, 2, P, P, P)], 1),
        "lsd": torch.cat([off, torch.full((B, 1, P, P, P), 17.0), torch.zeros(B, 1, P, P, P)], 1),
    }
    phys = activate_heads(raw, "sides")
    assert phys.shape[1] == n_head_ch("sides", lsd=True)
    assert torch.allclose(phys[:, -5:-2], off)             # the offset stays raw voxels
    assert torch.allclose(phys[:, -2], torch.full((B, P, P, P), 17.0))   # relu'd thickness
    assert torch.allclose(phys[:, -1], torch.full((B, P, P, P), 0.5))    # sigmoid(0) incoherence
    u = to_unit(phys, clip, "sides", lsd=True)
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0
    u8 = torch.round(u * 255).to(torch.uint8)[0].numpy()
    # the head channels only (the derived surface_side1 is written later, by write_surface_channel)
    dec = decode_pred(u8, clip, pred_channels("sides", lsd=True)[: n_head_ch("sides", lsd=True)])
    assert dec["lsd_off_z"].mean() == pytest.approx(5.0, abs=0.2)
    assert dec["lsd_off_y"].mean() == pytest.approx(-12.0, abs=0.2)
    assert dec["lsd_thick"].mean() == pytest.approx(17.0, abs=0.5)
    assert dec["lsd_ncov"].mean() == pytest.approx(0.5, abs=0.01)
    # the channels before the lsd block are byte-identical to a run without it
    u_no = to_unit(activate_heads({k: v for k, v in raw.items() if k != "lsd"}, "sides"), clip, "sides")
    assert torch.allclose(u_no, u[:, : u_no.shape[1]])


def test_load_student_with_and_without_the_flag(tmp_path):
    from tsm.train import EMA, save_checkpoint

    for on in (False, True):
        model = build_model(widths=(8, 16, 16, 16, 16), in_ch=2, surface_mode="sides", lsd=on)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        path = str(tmp_path / f"ck_{on}.pt")
        save_checkpoint(path, model, EMA(model), opt, sched, 1,
                        {"widths": [8, 16, 16, 16, 16], "surface_mode": "sides", "in_ch": 2, "lsd": on,
                         "train": {"heads": {"fiber": False, "lsd": on}}})
        m2, info = load_student(path)
        assert info["lsd"] is on
        assert ("lsd" in dict(m2.head)) is on
    # a checkpoint blob from before the flag existed builds the historical head set
    assert checkpoint_lsd({"train": {"surface_mode": "sides"}}) is False
    assert checkpoint_lsd({"train": {"heads": {"lsd": True}}}) is True


def test_tta_inverse_moves_the_offset_like_a_vector():
    P = 4
    off = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1, 1).expand(1, 3, P, P, P)
    v = torch.cat([off, torch.full((1, 1, P, P, P), 7.0), torch.zeros(1, 1, P, P, P)], 1).contiguous()
    got = tta_inverse({"lsd": v}, (0,), False)["lsd"]
    assert torch.allclose(got[:, 0], torch.full((1, P, P, P), -1.0))   # the z flip negates off_z
    assert torch.allclose(got[:, 1:3], off[:, 1:3])
    assert torch.allclose(got[:, 3], torch.full((1, P, P, P), 7.0))
    got = tta_inverse({"lsd": v}, (), True)["lsd"]                     # the y<->x transpose
    assert torch.allclose(got[:, 1], torch.full((1, P, P, P), 3.0))
    assert torch.allclose(got[:, 2], torch.full((1, P, P, P), 2.0))


def test_viewer_kinds():
    assert layer_for("lsd_thick", "student").kind == "count"
    assert layer_for("lsd_ncov", "student").kind == "prob"
    for ch, axis in (("lsd_off_x", 0), ("lsd_off_y", 1), ("lsd_off_z", 2)):
        l = layer_for(ch, "student")
        assert l.kind == "signed" and l.vec == "lsd_off" and l.axis == axis


def test_shipped_configs_load():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    for name in ("sides30k_lsd.json", "paris4_eval_sides30k_lsd.json"):
        with open(os.path.join(root, name)) as fh:
            raw = json.load(fh)
        cfg = parse_config(raw)
        o = train_opts(cfg)
        assert o["surface_mode"] == "sides" and o["heads"]["lsd"] is True
        assert o["loss_weights"]["lsd"] == 1.0
    with open(os.path.join(root, "paris4_eval_sides30k_lsd.json")) as fh:
        ev = json.load(fh)
    assert ev["out_dir"].endswith("eval_sides30k_lsd")
    assert ev["extra"]["infer"]["checkpoint"].endswith("sides30k_lsd/train/latest.pt")
