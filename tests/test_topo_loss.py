"""``extra.train.surface_aux``: the two *body* topology terms (gap / cldice).

Both look at the sheet BODY -- the slab between the two faces, ``(sdf_in > 0) & (sdf_out < 0)`` --
rather than at one zero set at a time, so they are computed once per crop in ``faces_loss``.
The synthetic labels here are stacks of z-parallel slabs: a pair separated by a 2-voxel air gap
(the thing a weld destroys) and a pair separated by a 12-voxel one (real space between wraps).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsm.ablate import REPORT_COLUMNS, _metric
from tsm.train import (
    SURFACE_AUX_DEFAULTS,
    body_mask,
    faces_loss,
    narrow_gap_mask,
    soft_body,
    soft_skel,
    surface_aux_active,
    surface_aux_opts,
    surface_body_terms,
    compute_losses,
    synthetic_batch,
)

D, H, W = 64, 6, 6
CLIP = 20.0

# slab A: z in [4, 12), slab B: z in [14, 22)   -> 2-voxel gap  (12, 13)
# slab C: z in [32, 40), slab D: z in [52, 60)  -> 12-voxel gap (40 .. 51)
SLABS = [(4, 12), (14, 22), (32, 40), (52, 60)]
NARROW = list(range(12, 14))
WIDE = list(range(40, 52))
# a denser stack: three 2-voxel gaps (12-13, 22-23, 32-33), for the per-sample test
DENSE_SLABS = [(4, 12), (14, 22), (24, 32), (34, 42)]


def _faces_from_slabs(slabs=SLABS) -> tuple[torch.Tensor, torch.Tensor]:
    """(sdf target (1, 2, D, H, W), valid (1, 1, D, H, W)) for a stack of z-parallel slabs.

    ``sdf_in`` is the distance to the nearest in-face (positive above it), ``sdf_out`` to the
    nearest out-face (negative above it), so ``sdf_in > 0 and sdf_out < 0`` exactly inside a slab.
    """
    z = np.arange(D, dtype=np.float32)
    a = np.array([z0 - 0.5 for z0, _ in slabs], np.float32)      # in-faces (lower boundaries)
    b = np.array([z1 - 0.5 for _, z1 in slabs], np.float32)      # out-faces (upper boundaries)
    inside = np.zeros(D, bool)
    for z0, z1 in slabs:
        inside |= (z >= z0) & (z < z1)
    d_a = np.abs(z[:, None] - a[None]).min(1)
    d_b = np.abs(z[:, None] - b[None]).min(1)
    ins = np.where(inside, d_a, -d_a).astype(np.float32)         # > 0 exactly inside the body
    outs = np.where(inside, -d_b, d_b).astype(np.float32)        # < 0 exactly inside the body
    t = torch.from_numpy(np.stack([ins, outs])).clamp(-CLIP, CLIP)[None, :, :, None, None]
    return t.expand(1, 2, D, H, W).clone(), torch.ones(1, 1, D, H, W)


def _pred(sdf: torch.Tensor) -> torch.Tensor:
    """[sdf_in, sdf_out, valid logit] from a 2-channel sdf."""
    return torch.cat([sdf, torch.full_like(sdf[:, 0:1], 10.0)], 1)


def _gap_mask(radius: int = 3, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    sdf, v = _faces_from_slabs()
    if valid is not None:
        v = valid
    body = body_mask(sdf[:, 0:1], sdf[:, 1:2], v)
    return narrow_gap_mask(body, v, radius), body


# --------------------------------------------------------------------------- #
# the narrow-gap mask
# --------------------------------------------------------------------------- #
def test_body_mask_is_the_slab_interior():
    sdf, v = _faces_from_slabs()
    body = body_mask(sdf[:, 0:1], sdf[:, 1:2], v)[0, 0, :, 0, 0]
    want = torch.zeros(D)
    for z0, z1 in SLABS:
        want[z0:z1] = 1.0
    torch.testing.assert_close(body, want)


def test_narrow_gap_holds_the_2vox_gap_and_not_the_12vox_one():
    gap, body = _gap_mask(radius=3)
    col = gap[0, 0, :, 0, 0]
    assert col[NARROW].min() == 1.0                      # the 2-voxel gap is protected
    assert col[WIDE].max() == 0.0                        # a 12-voxel gap is real space, not a gap
    assert float((col * body[0, 0, :, 0, 0]).sum()) == 0.0   # never the body itself
    assert float(col.sum()) == len(NARROW)               # nothing else in a flat stack


def test_gap_radius_controls_which_gaps_are_protected():
    assert float(_gap_mask(radius=1)[0][0, 0, :, 0, 0].sum()) == len(NARROW)  # 2 <= 2*1
    wide7 = _gap_mask(radius=7)[0][0, 0, :, 0, 0]
    assert wide7[WIDE].min() == 1.0                      # 12 <= 2*7: now closed as well


def test_unsupervised_voxels_are_excluded():
    v = torch.ones(1, 1, D, H, W)
    v[:, :, :, :, 0] = 0.0                               # one x column unsupervised
    gap, _ = _gap_mask(radius=3, valid=v)
    assert float(gap[0, 0, :, :, 0].sum()) == 0.0
    assert float(gap[0, 0, NARROW, :, 1:].min()) == 1.0


# --------------------------------------------------------------------------- #
# the gap term
# --------------------------------------------------------------------------- #
def _weld(sdf: torch.Tensor) -> torch.Tensor:
    """``sdf`` with the narrow gap filled in: body everywhere between the two slabs."""
    w = sdf.clone()
    w[:, 0:1, NARROW[0]:NARROW[-1] + 1] = CLIP
    w[:, 1:2, NARROW[0]:NARROW[-1] + 1] = -CLIP
    return w


def _aux(**kw) -> dict:
    return {**SURFACE_AUX_DEFAULTS, **kw}


def _body_terms(pred: torch.Tensor, sdf: torch.Tensor, v: torch.Tensor, aux: dict) -> dict[str, float]:
    d = surface_body_terms(pred[:, 0:1], pred[:, 1:2], sdf[:, 0:1], sdf[:, 1:2], v, v, aux)
    return {k: float(x) for k, x in d.items()}


def test_gap_is_near_zero_for_a_perfect_prediction_and_saturates_on_a_weld():
    sdf, v = _faces_from_slabs()
    weld = _weld(sdf)
    for tau, ceiling in ((2.0, 0.2), (0.25, 0.01)):
        aux = _aux(gap=1.0, gap_radius=3, gap_tau=tau)
        perfect = _body_terms(_pred(sdf), sdf, v, aux)["gap"]
        welded = _body_terms(_pred(weld), sdf, v, aux)["gap"]
        # the label sdf is only ~1 voxel from the faces inside a 2-voxel gap, so at the default
        # tau = 2 even a perfect prediction has ~0.14 of soft body there; it -> 0 as tau -> 0
        assert perfect < ceiling, (tau, perfect)
        assert welded > 0.99 and welded > 5 * perfect, (tau, welded, perfect)


def test_gap_gradient_reaches_both_faces():
    sdf, v = _faces_from_slabs()
    p = torch.zeros_like(sdf, requires_grad=True)         # p_body = 1/4 everywhere
    d = surface_body_terms(p[:, 0:1], p[:, 1:2], sdf[:, 0:1], sdf[:, 1:2], v, v,
                           _aux(gap=1.0, gap_radius=3))
    d["gap"].backward()
    g = p.grad[0, :, NARROW[0], 0, 0]
    assert float(g[0]) > 0 and float(g[1]) < 0           # push sdf_in down and sdf_out up: no body
    assert float(p.grad[0, :, SLABS[0][0] + 2].abs().sum()) == 0.0  # nothing outside the gap mask


def test_gap_is_per_sample_so_a_sparse_crop_is_not_drowned():
    """A welded crop with 2 gap voxels next to a clean crop with 6 must count for half the loss."""
    a, v = _faces_from_slabs()                              # 2 narrow-gap voxels
    b, _ = _faces_from_slabs(DENSE_SLABS)                   # 6 narrow-gap voxels
    aux = _aux(gap=1.0, gap_radius=3, gap_tau=0.25)
    p = torch.cat([_weld(a), b])                            # crop 0 welded, crop 1 perfect
    t = torch.cat([a, b])
    vv = v.repeat(2, 1, 1, 1, 1)
    two = surface_body_terms(p[:, 0:1], p[:, 1:2], t[:, 0:1], t[:, 1:2], vv, vv, aux)["gap"]
    one = _body_terms(_pred(_weld(a)), a, v, aux)["gap"]
    assert float(one) > 0.99
    assert float(two) == pytest.approx(float(one) / 2, abs=0.02)   # per sample
    assert float(two) > 0.4                                        # a voxel mean would give ~0.25


def test_soft_body_is_a_probability():
    p = soft_body(torch.tensor([[10.0, -10.0, 0.0]]), torch.tensor([[-10.0, 10.0, 0.0]]), 2.0)
    assert float(p[0, 0]) > 0.98 and float(p[0, 1]) < 0.01 and float(p[0, 2]) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# soft skeletonisation
# --------------------------------------------------------------------------- #
def test_soft_skel_of_a_thick_slab_is_thin_and_inside_it():
    x = torch.zeros(1, 1, 32, 16, 16)
    x[:, :, 8:24, 4:12, 4:12] = 1.0
    s = soft_skel(x, 5)
    assert float(s.sum()) < 0.5 * float(x.sum())          # much thinner than the block
    assert float((s * (1.0 - x)).sum()) == 0.0            # and entirely inside it
    assert float(s.max()) <= 1.0 + 1e-6


def test_soft_skel_of_an_empty_map_is_empty():
    s = soft_skel(torch.zeros(1, 1, 16, 8, 8), 5)
    assert float(s.abs().sum()) == 0.0


# --------------------------------------------------------------------------- #
# the clDice term
# --------------------------------------------------------------------------- #
def _cldice(pred_sdf: torch.Tensor, tgt_sdf: torch.Tensor, v: torch.Tensor, tau: float = 0.5) -> float:
    """The clDice term at a sharp ``gap_tau`` so a perfect prediction really is ~0 (the soft body
    of a *correct* SDF still bleeds a little across the faces at the default tau = 2)."""
    return _body_terms(_pred(pred_sdf), tgt_sdf, v, _aux(cldice=1.0, gap_tau=tau))["cldice"]


def test_cldice_is_near_zero_when_the_prediction_is_the_label():
    sdf, v = _faces_from_slabs()
    assert _cldice(sdf, sdf, v) < 0.01
    assert _cldice(sdf, sdf, v, tau=2.0) < 0.2      # softer body: a small constant floor


def test_cldice_penalises_a_spurious_extra_slab():
    sdf, v = _faces_from_slabs()
    extra, _ = _faces_from_slabs(SLABS + [(42, 50)])      # an invented sheet in the wide gap
    bad = _cldice(extra, sdf, v)
    assert bad > 0.05 and bad > 10 * max(_cldice(sdf, sdf, v), 1e-9)


def test_cldice_penalises_a_weld_between_two_slabs():
    sdf, v = _faces_from_slabs()
    bad = _cldice(_weld(sdf), sdf, v)
    assert bad > 0.02 and bad > 10 * max(_cldice(sdf, sdf, v), 1e-9)


def test_cldice_penalises_a_missing_slab():
    sdf, v = _faces_from_slabs()
    missing, _ = _faces_from_slabs(SLABS[:-1])            # slab D is not predicted at all
    bad = _cldice(missing, sdf, v)
    assert bad > 0.05 and bad > 10 * max(_cldice(sdf, sdf, v), 1e-9)


def test_cldice_is_zero_and_finite_with_an_empty_label_body():
    sdf, v = _faces_from_slabs()
    empty = torch.cat([torch.full_like(sdf[:, 0:1], -CLIP), torch.full_like(sdf[:, 1:2], CLIP)], 1)
    val = _cldice(sdf, empty, v)
    assert val == 0.0 and np.isfinite(val)


def test_cldice_backprops():
    sdf, v = _faces_from_slabs()
    p = sdf.clone().requires_grad_(True)
    d = surface_body_terms(p[:, 0:1], p[:, 1:2], sdf[:, 0:1], sdf[:, 1:2], v, v, _aux(cldice=1.0))
    d["cldice"].backward()
    assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# wiring: off by default, on when asked
# --------------------------------------------------------------------------- #
HISTORICAL = {"sdf_in_l1", "band_dice_in", "sdf_out_l1", "band_dice_out", "valid_bce"}


def test_faces_loss_has_no_body_keys_when_off():
    sdf, v = _faces_from_slabs()
    pred = _pred(sdf * 0.5)
    a = faces_loss(pred, sdf, v, aux=None)
    b = faces_loss(pred, sdf, v, aux=surface_aux_opts(None))
    assert set(a) == set(b) == HISTORICAL
    for k in a:
        assert float(a[k]) == float(b[k])


def test_faces_loss_adds_gap_and_cldice_when_on():
    sdf, v = _faces_from_slabs()
    pred = _pred(sdf * 0.5)
    d = faces_loss(pred, sdf, v, aux=surface_aux_opts({"gap": 0.5, "cldice": 0.5}))
    assert set(d) == HISTORICAL | {"gap", "cldice"}
    assert all(np.isfinite(float(x)) for x in d.values())
    only_gap = faces_loss(pred, sdf, v, aux=surface_aux_opts({"gap": 0.5}))
    assert set(only_gap) == HISTORICAL | {"gap"}


def test_surface_aux_active_truth_table():
    assert not surface_aux_active(None)
    assert not surface_aux_active({})
    assert not surface_aux_active(surface_aux_opts(None))
    for k in ("shell", "crest", "far", "gap", "cldice"):
        assert surface_aux_active(surface_aux_opts({k: 0.5})), k
        assert not surface_aux_active(surface_aux_opts({k: 0.0})), k
    assert not surface_aux_active(surface_aux_opts({"gap_radius": 5, "cldice_iters": 3}))


def test_surface_aux_opts_validation():
    o = surface_aux_opts({"gap": 0.5, "gap_radius": 5, "gap_tau": 1.5, "cldice": 2, "cldice_iters": 3})
    assert o["gap"] == 0.5 and o["gap_radius"] == 5 and isinstance(o["gap_radius"], int)
    assert o["gap_tau"] == 1.5 and o["cldice"] == 2.0 and o["cldice_iters"] == 3
    d = surface_aux_opts(None)
    assert d["gap"] == 0.0 and d["cldice"] == 0.0 and d["gap_radius"] == 3 and d["cldice_iters"] == 5
    for bad in ({"gap": -1}, {"cldice": "x"}, {"gap_radius": 0}, {"cldice_iters": 0},
                {"gap_radius": True}, {"gap": 0.5, "gap_tau": 0}, {"gapp": 1}):
        with pytest.raises(ValueError):
            surface_aux_opts(bad)
    with pytest.raises(ValueError):
        surface_aux_opts({"gap": 0.5}, surface_mode="medial")


def test_compute_losses_includes_the_body_terms_in_the_surface_total():
    from tsm.student import build_model

    model = build_model(widths=(8, 16, 16), surface_mode="faces")
    batch = synthetic_batch(2, 16, surface_mode="faces", seed=7)
    out = model(batch["input"])
    ds = (1.0, 0.0)  # full-res level only, so parts["surface"] is exactly the level-0 term sum
    l0, p0 = compute_losses(out, batch, surface_mode="faces", ds_weights=ds)
    lz, pz = compute_losses(out, batch, surface_mode="faces", ds_weights=ds, surface_aux=surface_aux_opts(None))
    assert float(l0.detach()) == float(lz.detach()) and p0 == pz and "surface/gap" not in p0
    la, pa = compute_losses(out, batch, surface_mode="faces", ds_weights=ds,
                            surface_aux=surface_aux_opts({"gap": 1.0, "cldice": 1.0}))
    assert "surface/gap" in pa and "surface/cldice" in pa
    assert pa["surface"] == pytest.approx(p0["surface"] + pa["surface/gap"] + pa["surface/cldice"], rel=1e-4)
    assert float(la.detach()) > float(l0.detach())
    # and they are deep-supervised like every other surface term
    _, pb = compute_losses(out, batch, surface_mode="faces", ds_weights=(1.0, 0.5),
                           surface_aux=surface_aux_opts({"gap": 1.0, "cldice": 1.0}))
    assert pb["surface"] > pa["surface"]


# --------------------------------------------------------------------------- #
# ablation report
# --------------------------------------------------------------------------- #
def test_ablate_reports_the_topology_columns():
    heads = {h for _, h, _ in REPORT_COLUMNS}
    assert {"spurious", "missed"} <= heads
    keys = {k for k, _, _ in REPORT_COLUMNS}
    assert {"surface/spurious_surfaces", "surface/missed_surfaces"} <= keys
    faces = {"surface/in_spurious_surfaces": 3.0, "surface/in_missed_surfaces": 1.0}
    assert _metric(faces, "surface/spurious_surfaces") == 3.0     # two-face fallback
    assert _metric(faces, "surface/missed_surfaces") == 1.0
    medial = {"surface/spurious_surfaces": 2.0}
    assert _metric(medial, "surface/spurious_surfaces") == 2.0    # single-SDF key, direct
    assert _metric({}, "surface/missed_surfaces") is None
    for _, _, lower in REPORT_COLUMNS:
        assert isinstance(lower, bool)
    assert dict((k, lo) for k, _, lo in REPORT_COLUMNS)["surface/spurious_surfaces"] is True
