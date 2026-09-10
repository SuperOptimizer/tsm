"""``extra.train.surface_aux``: the optional zero-set surface terms (shell / crest / far).

The synthetic label is one sheet *pair* -- an IN face at z = ZIN and an OUT face at z = ZOUT,
each channel the signed distance to its own plane -- which is exactly what the two-face store
holds.  Against that we score three predictions: the perfect one, one that grows a spurious
extra face beside the true IN face (the measured failure mode: good recall, poor precision) and
one that misses the IN face entirely.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tsm.config import parse_config
from tsm.data import CLIP
from tsm.train import (
    SURFACE_AUX_DEFAULTS,
    compute_losses,
    faces_loss,
    masked_mean,
    per_sample_mean,
    surface_aux_active,
    surface_aux_opts,
    surface_aux_terms,
    synthetic_batch,
    train_opts,
)

D, H, W = 48, 8, 8
ZIN, ZOUT = 24.0, 34.0
SPUR = 31.0        # z of the spurious extra face (7 vox from the IN face)
AUX = {"shell": 1.0, "shell_radius": 8, "shell_margin": 3.0,
       "crest": 1.0, "crest_tol": 1.0, "far": 1.0, "far_margin": 6.0}


def _z() -> torch.Tensor:
    z = torch.arange(D, dtype=torch.float32)[None, None, :, None, None]
    return z.expand(1, 1, D, H, W).clone()


def _labels() -> tuple[torch.Tensor, torch.Tensor]:
    """(sdf target (1, 2, D, H, W), valid (1, 1, D, H, W)) for the sheet pair."""
    z = _z()
    sdf = torch.cat([(z - ZIN).clamp(-CLIP, CLIP), (z - ZOUT).clamp(-CLIP, CLIP)], dim=1)
    return sdf, torch.ones(1, 1, D, H, W)


def _pred(sdf_in: torch.Tensor, sdf_out: torch.Tensor) -> torch.Tensor:
    return torch.cat([sdf_in, sdf_out, torch.full_like(sdf_in, 10.0)], dim=1)


def _preds() -> dict[str, torch.Tensor]:
    sdf, _ = _labels()
    t_in, t_out = sdf[:, 0:1], sdf[:, 1:2]
    z = _z()
    # a second zero crossing at z = SPUR, replacing the true field in a +-5 vox band
    spurious = torch.where((z - SPUR).abs() <= 5.0, z - SPUR, t_in)
    return {
        "perfect": _pred(t_in, t_out),
        "spurious": _pred(spurious, t_out),
        "missing": _pred(torch.full_like(t_in, 10.0), t_out),  # never crosses zero
    }


def _terms(name: str, aux: dict | None = None) -> dict[str, float]:
    sdf, valid = _labels()
    out = faces_loss(_preds()[name], sdf, valid, aux=aux or AUX)
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# the three terms
# --------------------------------------------------------------------------- #
def test_perfect_prediction_scores_zero_on_every_aux_term():
    t = _terms("perfect")
    for k in ("shell_in", "crest_in", "far_in", "shell_out", "crest_out", "far_out"):
        assert t[k] == pytest.approx(0.0, abs=1e-6), k
    # ...and the main terms are (near) zero too, so the aux terms are not hiding a broken label
    assert t["sdf_in_l1"] == pytest.approx(0.0, abs=1e-6)


def test_spurious_extra_face_raises_far_and_shell_but_not_crest():
    t = _terms("spurious")
    assert t["far_in"] > 0.05, t
    assert t["shell_in"] > 0.05, t
    assert t["crest_in"] == pytest.approx(0.0, abs=1e-6)  # the true face is still predicted
    # the untouched OUT face stays at zero: the terms are strictly per face
    for k in ("shell_out", "crest_out", "far_out"):
        assert t[k] == pytest.approx(0.0, abs=1e-6), k


def test_missing_face_raises_crest_only():
    t = _terms("missing")
    assert t["crest_in"] == pytest.approx(10.0 - AUX["crest_tol"], abs=1e-5)
    assert t["far_in"] == pytest.approx(0.0, abs=1e-6)   # |sdf_pred| = 10 >= far_margin / 2
    assert t["shell_in"] == pytest.approx(0.0, abs=1e-6)  # ...and >= shell_margin
    for k in ("shell_out", "crest_out", "far_out"):
        assert t[k] == pytest.approx(0.0, abs=1e-6), k


def test_shell_ignores_the_face_itself_and_its_immediate_neighbourhood():
    """A prediction that is merely *thick* about the true face costs nothing below the margin."""
    sdf, valid = _labels()
    z = _z()
    thick = torch.where((z - ZIN).abs() <= 2.0, torch.zeros_like(z), sdf[:, 0:1])  # flat 0 core
    t = faces_loss(_pred(thick, sdf[:, 1:2]), sdf, valid, aux=AUX)
    assert float(t["shell_in"]) == pytest.approx(0.0, abs=1e-6)
    assert float(t["crest_in"]) == pytest.approx(0.0, abs=1e-6)


def test_far_only_looks_beyond_far_margin():
    """Moving the spurious face inside far_margin removes the far cost but keeps the shell one."""
    sdf, valid = _labels()
    z = _z()
    near = torch.where((z - (ZIN + 4.0)).abs() <= 1.0, z - (ZIN + 4.0), sdf[:, 0:1])
    t = faces_loss(_pred(near, sdf[:, 1:2]), sdf, valid, aux=AUX)
    assert float(t["far_in"]) == pytest.approx(0.0, abs=1e-6)
    assert float(t["shell_in"]) > 0.0


def test_crest_is_normalised_per_sample_not_per_voxel():
    """A batch with one dense and one sparse face: crest is the mean of the per-sample means."""
    x = torch.zeros(2, 1, 4, 4, 4)
    m = torch.zeros(2, 1, 4, 4, 4)
    m[0, :, 0, 0, 0] = 1.0     # one voxel
    x[0, :, 0, 0, 0] = 4.0
    m[1] = 1.0                 # 64 voxels
    x[1] = 0.0
    assert float(per_sample_mean(x, m)) == pytest.approx(2.0)          # (4 + 0) / 2
    assert float(masked_mean(x, m)) == pytest.approx(4.0 / 65.0)       # drowned by the dense sample
    # and an all-empty mask is 0, not nan
    assert float(per_sample_mean(x, torch.zeros_like(m))) == 0.0


def test_valid_mask_and_weight_gate_the_aux_terms():
    sdf, valid = _labels()
    pred = _preds()["spurious"]
    on = faces_loss(pred, sdf, valid, aux=AUX)
    # valid == 2 (ignore) everywhere the spurious face lives -> nothing to complain about
    z = _z()
    v2 = torch.where((z - SPUR).abs() <= 5.0, torch.full_like(z, 2.0), valid)
    off = faces_loss(pred, sdf, v2, aux=AUX)
    assert float(on["far_in"]) > 0 and float(off["far_in"]) == pytest.approx(0.0, abs=1e-6)
    assert float(off["shell_in"]) < float(on["shell_in"])
    # a faces_weight of exactly 1 changes nothing
    w1 = faces_loss(pred, sdf, valid, weight=torch.ones_like(valid), aux=AUX)
    for k, v in on.items():
        assert torch.equal(w1[k], v), k


# --------------------------------------------------------------------------- #
# off by default / bit-identical
# --------------------------------------------------------------------------- #
def test_zero_weights_leave_the_loss_bit_identical():
    sdf, valid = _labels()
    pred = _preds()["spurious"]
    base = faces_loss(pred, sdf, valid)
    for aux in (None, dict(SURFACE_AUX_DEFAULTS), surface_aux_opts(None),
                surface_aux_opts({"shell_radius": 8, "far_margin": 3.0})):
        got = faces_loss(pred, sdf, valid, aux=aux)
        assert set(got) == set(base)
        for k, v in base.items():
            assert torch.equal(got[k], v), k
    assert not surface_aux_active(surface_aux_opts(None))
    assert surface_aux_active(surface_aux_opts({"far": 0.1}))


def test_compute_losses_is_bit_identical_with_zero_weights_and_grows_with_them():
    from tsm.student import build_model

    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="faces")
    batch = synthetic_batch(2, 16, surface_mode="faces", seed=3)
    with torch.no_grad():
        out = model(batch["input"])
    l0, p0 = compute_losses(out, batch, surface_mode="faces")
    lz, pz = compute_losses(out, batch, surface_mode="faces", surface_aux=surface_aux_opts(None))
    assert torch.equal(l0, lz) and p0 == pz
    assert not any(k.startswith("surface/shell") for k in p0)
    la, pa = compute_losses(out, batch, surface_mode="faces", surface_aux=surface_aux_opts(AUX))
    assert {"surface/shell_in", "surface/crest_in", "surface/far_in",
            "surface/shell_out", "surface/crest_out", "surface/far_out"} <= set(pa)
    assert float(la) > float(l0)


def test_aux_terms_are_deep_supervised_like_the_main_sdf_loss():
    """Both resolution levels contribute: dropping the half-res weight lowers the head loss."""
    from tsm.student import build_model

    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="faces").train()
    batch = synthetic_batch(2, 16, surface_mode="faces", seed=5)
    out = model(batch["input"])
    assert isinstance(out["surface"], (list, tuple)) and len(out["surface"]) == 2
    aux = surface_aux_opts(AUX)
    _, both = compute_losses(out, batch, ds_weights=(1.0, 0.5), surface_mode="faces", surface_aux=aux)
    _, one = compute_losses(out, batch, ds_weights=(1.0, 0.0), surface_mode="faces", surface_aux=aux)
    assert both["surface"] > one["surface"]
    # level 0 is what is logged, so the per-term numbers must not move
    for k in ("surface/shell_in", "surface/crest_in", "surface/far_in"):
        assert both[k] == pytest.approx(one[k])


def test_aux_terms_backprop():
    sdf, valid = _labels()
    pred = _preds()["spurious"].clone().requires_grad_(True)
    t = faces_loss(pred, sdf, valid, aux={**SURFACE_AUX_DEFAULTS, "shell": 1.0, "shell_radius": 8,
                                          "crest": 1.0, "far": 1.0})
    sum(t[k] for k in ("shell_in", "crest_in", "far_in")).backward()
    g = pred.grad[:, 0:1]
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
    assert float(pred.grad[:, 1:2].abs().sum()) == 0.0  # the OUT channel is untouched


# --------------------------------------------------------------------------- #
# config plumbing
# --------------------------------------------------------------------------- #
def _cfg(train: dict):
    return parse_config({
        "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": "/tmp/tsm-none", "extra": {"train": train},
    })


def test_train_opts_defaults_and_validation():
    o = train_opts(_cfg({}))
    assert o["surface_aux"] == SURFACE_AUX_DEFAULTS and not surface_aux_active(o["surface_aux"])
    o = train_opts(_cfg({"surface_mode": "faces", "surface_aux": {"shell": 0.5, "crest": 0.5, "far": 0.1}}))
    assert o["surface_aux"]["shell"] == 0.5 and o["surface_aux"]["shell_margin"] == 3.0
    with pytest.raises(ValueError, match="unknown extra.train.surface_aux keys"):
        train_opts(_cfg({"surface_mode": "faces", "surface_aux": {"shel": 1}}))
    with pytest.raises(ValueError, match="non-negative"):
        train_opts(_cfg({"surface_mode": "faces", "surface_aux": {"crest": -1}}))
    with pytest.raises(ValueError, match="shell_radius"):
        train_opts(_cfg({"surface_mode": "faces", "surface_aux": {"shell_radius": 0}}))
    with pytest.raises(ValueError, match="surface_mode='faces'"):
        train_opts(_cfg({"surface_mode": "medial", "surface_aux": {"crest": 0.5}}))
    # ...but a zero-weight block in medial mode is a harmless no-op
    train_opts(_cfg({"surface_mode": "medial", "surface_aux": {"crest_tol": 2.0}}))


def test_shipped_configs_load():
    from tsm.ablate import ablate_opts, variant_config
    from tsm.config import load_config

    cfg = load_config("configs/paris4_faces_rvfw_aux.json")
    o = train_opts(cfg)
    assert o["surface_aux"]["shell"] == 0.5 and o["surface_aux"]["crest"] == 0.5
    assert o["surface_aux"]["far"] == 0.1 and o["core_radius_vox"] == 400.0
    assert cfg.out_dir.endswith("slab_faces_rvfw_aux")

    acfg = load_config("configs/ablate_surface_aux.json")
    ao = ablate_opts(acfg)
    assert ao["steps"] == 6000 and ao["include_base"] and set(ao["variants"]) == {
        "shell_crest", "shell_crest_far", "core400"}
    assert ao["out_dir"].endswith("slab_faces_rvfw/ablation_surface_aux")
    got = {n: train_opts(variant_config(acfg, n, ov, ao, ao["out_dir"])) for n, ov in ao["variants"].items()}
    assert got["shell_crest"]["surface_aux"]["shell"] == 0.5 and got["shell_crest"]["surface_aux"]["far"] == 0.0
    assert got["shell_crest_far"]["surface_aux"]["far"] == 0.1
    assert got["core400"]["core_radius_vox"] == 400.0 and not surface_aux_active(got["core400"]["surface_aux"])


def test_surface_aux_terms_shape_agnostic():
    """The helper works on any (B, 1, Z, Y, X) pair -- deep supervision hands it half-res crops."""
    t = torch.randn(3, 1, 8, 8, 8) * 5
    p = torch.randn(3, 1, 8, 8, 8) * 5
    m = torch.ones(3, 1, 8, 8, 8)
    got = surface_aux_terms(p, t, m, m, surface_aux_opts(AUX), "in")
    assert set(got) == {"shell_in", "crest_in", "far_in"}
    assert all(np.isfinite(float(v)) and float(v) >= 0 for v in got.values())
