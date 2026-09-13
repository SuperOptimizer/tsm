"""Opt-in extras of ``surface_mode="sides"``: the explicit air-gap class, the eikonal
regulariser and the CT-gated body label.

All three are off by default -- ``extra.train.surface_aux.{gap_dilate, gap_class,
gap_border_weight, eikonal}`` are 0 and ``extra.train.body_ct_gate`` is null -- and the last
test of every section asserts that the defaults leave the loss and the dataset targets exactly
as they were before these knobs existed.

1. ``gap_dilate`` widens the label air-gap mask (:func:`tsm.train.narrow_gap_mask`) without ever
   covering labelled body; ``gap_class`` classifies a 4th surface head channel against it, and
   ``gap_border_weight`` adds Ronneberger-style weight on the gap voxels that touch a body.
2. ``eikonal`` scores ``(|grad d_pred| - 1)^2`` away from the faces, the label clip and the
   medial ridge: 0 for a true unsigned distance, > 0 for a compressed one.
3. ``body_ct_gate`` turns the CT-dark part of the label body into ``surface_valid = 2`` (ignore)
   on the fly, without touching the label store.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import make_synthetic  # noqa: E402
from tsm.data import CLIP, CropDataset  # noqa: E402
from tsm.infer import (  # noqa: E402
    N_SIDES_HEAD_CH,
    SIDES_PRED_CHANNELS,
    activate_heads,
    checkpoint_gap_class,
    decode_pred,
    load_student,
    n_head_ch,
    pred_channels,
    to_unit,
)
from tsm.student import TSMNet, build_model, heads_for  # noqa: E402
from tsm.train import (  # noqa: E402
    EMA,
    eikonal_term,
    gap_border_mask,
    narrow_gap_mask,
    save_checkpoint,
    sides_loss,
    surface_aux_opts,
    train_opts,
)
from tsm.volume import VolumeReader  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def two_wraps(n: int = 24, gap: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Two flat sheets separated by a ``gap``-voxel air layer: (body mask, unsigned face distance).

    Sheets occupy z in [4, 4+t) and [4+t+gap, 4+2t+gap); the distance is the exact unsigned
    distance to the nearest face plane, so it satisfies |grad d| = 1 away from faces / ridges.
    """
    t = 6
    z = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1, 1).expand(1, 1, n, n, n).clone()
    faces = [4.0, 4.0 + t, 4.0 + t + gap, 4.0 + 2 * t + gap]
    d = torch.stack([(z - f).abs() for f in faces], 0).min(0).values
    body = (((z >= faces[0]) & (z < faces[1])) | ((z >= faces[2]) & (z < faces[3]))).float()
    return body, d.clamp(0.0, CLIP)


def two_wraps_finite(n: int = 24, gap: int = 3) -> torch.Tensor:
    """The body of :func:`two_wraps`, cut to a finite extent in x so the air around it is open
    (a dilated gap mask then has somewhere to grow)."""
    body, _ = two_wraps(n=n, gap=gap)
    x = torch.arange(n).view(1, 1, 1, 1, n)
    return body * ((x >= 2) & (x < n - 2)).float()


def _valid(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


AUX_GAP = {"gap": 1.0, "gap_radius": 3, "gap_dilate": 1, "gap_class": 1.0, "gap_border_weight": 3.0}


# --------------------------------------------------------------------------- #
# 1. the air gap: dilation, the new head channel, the border weight
# --------------------------------------------------------------------------- #
def test_gap_dilation_grows_the_mask_and_never_covers_the_body():
    body = two_wraps_finite()
    m1 = _valid(body)
    g0 = narrow_gap_mask(body, m1, 3, 0)
    assert float(g0.sum()) > 0
    prev = g0
    for dil in (1, 2, 3):
        g = narrow_gap_mask(body, m1, 3, dil)
        assert float(g.sum()) > float(prev.sum()), dil          # strictly grows
        assert float((g * body).sum()) == 0.0                   # never enters the label body
        assert float((g * (1 - m1)).sum()) == 0.0               # never leaves the supervised set
        assert float(((g0 > 0) & (g == 0)).sum()) == 0.0        # and always contains the original
        prev = g
    # dilate=0 is exactly the historical mask
    torch.testing.assert_close(narrow_gap_mask(body, m1, 3), g0, rtol=0, atol=0)


def test_gap_border_mask_is_the_gap_voxels_that_touch_a_body():
    body = two_wraps_finite(gap=5)
    m1 = _valid(body)
    gap = narrow_gap_mask(body, m1, 3, 1)
    border = gap_border_mask(body, gap, 1)
    assert float(border.sum()) > 0
    assert float(border.sum()) < float(gap.sum())        # the middle of a 5-voxel gap is excluded
    assert float((border * body).sum()) == 0.0           # still only air
    assert float(((border > 0) & (gap == 0)).sum()) == 0.0


def _sides_pred(d: torch.Tensor, body: torch.Tensor, gap_logit: torch.Tensor) -> torch.Tensor:
    big = torch.full_like(d, 20.0)
    return torch.cat([d, (body * 2 - 1) * 20.0, big, gap_logit], 1)


def test_gap_class_is_near_zero_for_a_perfect_gap_logit_and_positive_otherwise():
    body, d = two_wraps()
    valid = _valid(d)
    aux = surface_aux_opts(AUX_GAP, "sides")
    gap_t = narrow_gap_mask(body, valid, int(aux["gap_radius"]), int(aux["gap_dilate"]))
    perfect = sides_loss(_sides_pred(d, body, (gap_t * 2 - 1) * 20.0), d, valid, body, aux=aux)
    assert "gap_class" in perfect
    assert float(perfect["gap_class"]) < 1e-3
    wrong = sides_loss(_sides_pred(d, body, (1 - gap_t) * 40.0 - 20.0), d, valid, body, aux=aux)
    assert float(wrong["gap_class"]) > 1.0
    # and it is a gradient on the gap channel only
    p = _sides_pred(d, body, torch.zeros_like(d)).requires_grad_(True)
    out = sides_loss(p, d, valid, body, aux=aux)
    out["gap_class"].backward()
    assert float(p.grad[:, 3:4].abs().sum()) > 0
    assert float(p.grad[:, 0:3].abs().sum()) == 0.0


def test_gap_class_needs_the_fourth_head_channel():
    body, d = two_wraps()
    aux = surface_aux_opts(AUX_GAP, "sides")
    three = _sides_pred(d, body, torch.zeros_like(d))[:, :3]
    with pytest.raises(ValueError, match="gap_class"):
        sides_loss(three, d, _valid(d), body, aux=aux)
    with pytest.raises(ValueError, match="gap_class"):
        surface_aux_opts({"gap_class": 1.0}, "body")
    with pytest.raises(ValueError, match="gap_dilate"):
        surface_aux_opts({"gap_dilate": -1}, "sides")


def test_gap_border_weight_reweights_the_dense_terms_only():
    body, d = two_wraps()
    valid = _valid(d)
    torch.manual_seed(3)
    # a wrong, spatially VARYING prediction, so re-weighting a subset actually moves the mean
    pred = torch.cat([d + torch.rand_like(d) * 4, torch.randn_like(d) * 3,
                      torch.full_like(d, 20.0), torch.zeros_like(d)], 1)
    base = sides_loss(pred, d, valid, body, aux=surface_aux_opts({**AUX_GAP, "gap_border_weight": 0.0}, "sides"))
    hot = sides_loss(pred, d, valid, body, aux=surface_aux_opts(AUX_GAP, "sides"))
    # the border voxels are wrong too, so up-weighting them moves the two dense terms ...
    assert float(hot["sdf_l1"]) != float(base["sdf_l1"])
    assert float(hot["body_bce"]) != float(base["body_bce"])
    # ... and leaves the set-shaped ones alone
    assert float(hot["band_dice"]) == float(base["band_dice"])
    assert float(hot["body_dice"]) == float(base["body_dice"])


# --------------------------------------------------------------------------- #
# 2. the 4-channel head plumbing
# --------------------------------------------------------------------------- #
def test_sides_head_plumbing_round_trips_with_and_without_the_gap_channel():
    ch = pred_channels("sides", gap_class=True)
    assert ch == SIDES_PRED_CHANNELS[:3] + ["gap"] + SIDES_PRED_CHANNELS[3:]
    assert ch[:4] == ["d_face", "body", "valid", "gap"]
    assert n_head_ch("sides", gap_class=True) == N_SIDES_HEAD_CH + 1
    assert pred_channels("sides") == SIDES_PRED_CHANNELS      # default unchanged
    assert heads_for("sides", gap_class=True)["surface"] == 4
    assert heads_for("sides")["surface"] == 3
    with pytest.raises(ValueError):
        pred_channels("body", gap_class=True)
    with pytest.raises(ValueError):
        heads_for("faces", gap_class=True)

    B, n = 1, 4
    torch.manual_seed(0)
    out = {"surface": torch.randn(B, 4, n, n, n), "ink": torch.randn(B, 1, n, n, n),
           "winding": torch.randn(B, 8, n, n, n)}
    phys = activate_heads(out, "sides")
    assert phys.shape == (B, N_SIDES_HEAD_CH + 1, n, n, n)
    torch.testing.assert_close(phys[:, 1:4], torch.sigmoid(out["surface"][:, 1:4]))
    u = to_unit(phys, CLIP, "sides", gap_class=True)
    assert u.shape == phys.shape and float(u.min()) >= 0 and float(u.max()) <= 1
    # the gap byte is a plain probability and every other channel keeps its own encoding
    torch.testing.assert_close(u[:, 3], phys[:, 3])
    torch.testing.assert_close(u[:, 1:3], phys[:, 1:3])
    q = np.round(u.numpy() * 255).astype(np.uint8)
    dec = decode_pred(np.concatenate([q[0], np.zeros((1,) + q.shape[2:], np.uint8)]), CLIP, ch)
    assert "gap" in dec
    np.testing.assert_allclose(dec["gap"], phys[0, 3].numpy(), atol=1 / 255.0)


def _ckpt(path: str, gap_class: bool, widths=(8, 16, 16, 16, 16)) -> None:
    net = TSMNet(widths=widths, heads=heads_for("sides", gap_class=gap_class))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ema = EMA(net, 0.5)
    ema.update(net)
    cfg = {"widths": list(widths), "surface_mode": "sides",
           "train": {"widths": list(widths), "surface_mode": "sides"}}
    if gap_class:
        cfg["gap_class"] = True
    save_checkpoint(path, net, ema, opt, None, 5, cfg)


def test_load_student_builds_the_right_surface_head(tmp_path):
    p4, p3 = str(tmp_path / "gap.pt"), str(tmp_path / "plain.pt")
    _ckpt(p4, True)
    _ckpt(p3, False)
    m4, i4 = load_student(p4)
    m3, i3 = load_student(p3)
    assert m4.heads["surface"] == 4 and i4["gap_class"] is True
    assert m3.heads["surface"] == 3 and i3["gap_class"] is False   # a checkpoint without the key
    # the flag can also be inferred from the training options of an older blob
    assert checkpoint_gap_class({"train": {"surface_aux": {"gap_class": 1.0}}}) is True
    assert checkpoint_gap_class({"train": {"surface_aux": {"gap_class": 0.0}}}) is False
    assert checkpoint_gap_class({}) is False


def test_the_four_channel_head_trains_end_to_end():
    torch.manual_seed(0)
    model = build_model(widths=(8, 16, 16), surface_mode="sides", gap_class=True)
    assert model.heads["surface"] == 4
    x = torch.randn(1, 2, 16, 16, 16)
    s = model(x)["surface"]
    s = s[0] if isinstance(s, (list, tuple)) else s
    assert s.shape[1] == 4
    body, d = two_wraps(n=16)
    out = sides_loss(s, d[:, :, :16, :16, :16], _valid(d)[:, :, :16, :16, :16],
                     body[:, :, :16, :16, :16], aux=surface_aux_opts(AUX_GAP, "sides"))
    sum(out.values()).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


# --------------------------------------------------------------------------- #
# 3. the eikonal regulariser
# --------------------------------------------------------------------------- #
def test_eikonal_is_zero_for_a_true_unsigned_distance_and_positive_when_compressed():
    body, d = two_wraps(n=32, gap=7)
    m1 = _valid(d)
    exact = float(eikonal_term(d, d, m1, 1.0, 3.0))
    assert exact == pytest.approx(0.0, abs=1e-6)
    squashed = float(eikonal_term(0.2 * d, d, m1, 1.0, 3.0))
    assert squashed == pytest.approx((0.2 - 1.0) ** 2, abs=1e-3)   # |grad| = 0.2 on the mask
    assert float(eikonal_term(0.2 * d, d, m1, 0.5, 3.0)) == pytest.approx(0.5 * squashed, rel=1e-5)
    # a constant prediction has no gradient at all: the term is the full (0 - 1)^2
    # (the gradient magnitude is clamped at sqrt(EPS) to stay differentiable at 0)
    assert float(eikonal_term(torch.full_like(d, 5.0), d, m1, 1.0, 3.0)) == pytest.approx(1.0, abs=5e-3)
    # ... and it backprops into d_face
    p = (0.2 * d).clone().requires_grad_(True)
    eikonal_term(p, d, m1, 1.0, 3.0).backward()
    assert float(p.grad.abs().sum()) > 0


def test_eikonal_mask_excludes_the_faces_the_ridge_and_the_unsupervised_voxels():
    body, d = two_wraps(n=32, gap=7)
    m1 = _valid(d)
    squashed = 0.2 * d
    # eikonal_min_d above every label distance leaves the mask empty -> exactly 0 (never a nan)
    assert float(eikonal_term(squashed, d, m1, 1.0, 100.0)) == 0.0
    # ... and so does an empty supervised set
    assert float(eikonal_term(squashed, d, torch.zeros_like(d), 1.0, 3.0)) == 0.0
    # the ridge (and the faces, another sign change of the label gradient) really is dropped:
    # with min_d = 0 the LABEL itself still scores 0, although |grad d_label| = 0 on both
    assert float(eikonal_term(d, d, m1, 1.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    # the mask never covers a voxel the label saturated: a field clipped flat everywhere has no
    # unsaturated voxels left at all
    flat = torch.full_like(d, CLIP)
    assert float(eikonal_term(squashed, flat, m1, 1.0, 3.0)) == 0.0


def test_sides_loss_reports_eikonal_only_when_asked():
    body, d = two_wraps(n=32, gap=7)
    valid = _valid(d)
    pred = torch.cat([0.2 * d, (body * 2 - 1) * 20.0, torch.full_like(d, 20.0)], 1)
    off = sides_loss(pred, d, valid, body, aux=surface_aux_opts({"gap": 1.0}, "sides"))
    assert "eikonal" not in off
    on = sides_loss(pred, d, valid, body,
                    aux=surface_aux_opts({"gap": 1.0, "eikonal": 0.1, "eikonal_min_d": 3.0}, "sides"))
    assert float(on["eikonal"]) > 0
    assert float(on["sdf_l1"]) == float(off["sdf_l1"])       # the primary terms are untouched


# --------------------------------------------------------------------------- #
# 4. the CT-gated body label
# --------------------------------------------------------------------------- #
def _gate(ct: np.ndarray, sv: np.ndarray, body: np.ndarray, thr: float, mode: str = "sides"):
    """``CropDataset._ct_gate`` on a bare namespace (no store / reader needed)."""
    import types

    obj = types.SimpleNamespace(body_ct_gate=float(thr), surface_mode=mode, _ct_gate_crops=0,
                                CT_GATE_LOG_EVERY=CropDataset.CT_GATE_LOG_EVERY)
    key = "surface_body" if mode == "sides" else "surface_sdf"
    t = {key: body[None].astype(np.float32)}
    return CropDataset._ct_gate(obj, ct, sv, t)


def test_ct_gate_ignores_dark_body_voxels_and_keeps_the_bright_ones():
    n = 8
    body = np.zeros((n, n, n), bool)
    body[2:6] = True                       # a slab of "papyrus"
    ct = np.full((n, n, n), 200, np.uint8)
    ct[:, :, :4] = 10                      # the left half of the crop is air
    sv = np.ones((n, n, n), np.float32)
    from scipy.ndimage import gaussian_filter

    out = _gate(ct, sv, body, 60.0)
    dark = body & (gaussian_filter(ct.astype(np.float32), 2.0) < 60)   # the gate smooths first
    assert 0 < dark.sum() < body.sum()
    np.testing.assert_array_equal(out == 2, dark)          # dark body -> ignore
    np.testing.assert_array_equal(out[~dark], sv[~dark])   # everything else untouched
    assert (out[~body] != 2).all()                         # non-body voxels are never gated
    # body mode gates on surface_sdf > 0 instead
    sdf = np.where(body, 3.0, -3.0).astype(np.float32)
    out_b = _gate(ct, sv, sdf, 60.0, mode="body")
    np.testing.assert_array_equal(out_b == 2, dark)
    # a threshold below every CT value gates nothing; unsupervised voxels stay unsupervised
    np.testing.assert_array_equal(_gate(ct, sv, body, 0.0), sv)
    sv0 = np.zeros((n, n, n), np.float32)
    np.testing.assert_array_equal(_gate(ct, sv0, body, 255.0), sv0)


@pytest.fixture(scope="module")
def synth_store(tmp_path_factory):
    return make_synthetic(str(tmp_path_factory.mktemp("gate")), fine_shape=(32, 64, 64),
                          fine_origin=(0, 0, 0), chunk=32, faces=True)


def _ds(s: dict, **kw) -> CropDataset:
    return CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                       length=4, augment=False, surface_mode="sides", **kw)


def test_ct_gate_in_the_dataset_only_changes_surface_valid(synth_store):
    base = _ds(synth_store)
    gated = _ds(synth_store, body_ct_gate=255.0)   # gate EVERY body voxel
    o = base.origins[0]
    a, b = base.load(o), gated.load(o)
    for k in ("surface_sdf", "surface_body", "ink_prob", "ink_valid", "ct"):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)
    body = a["surface_body"][0] > 0
    sup = a["surface_valid"][0] > 0
    # gate everything: the ignore set grows by exactly the supervised label body
    np.testing.assert_array_equal(b["surface_valid"][0] == 2, (a["surface_valid"][0] == 2) | (body & sup))
    np.testing.assert_array_equal(b["surface_valid"][0][~body], a["surface_valid"][0][~body])
    assert int((body & sup).sum()) > 0
    assert gated._ct_gate_crops == 1


def test_ct_gate_defaults_leave_the_targets_bit_identical(synth_store):
    base = _ds(synth_store)
    off = _ds(synth_store, body_ct_gate=None)
    o = base.origins[0]
    a, b = base.load(o), off.load(o)
    assert set(a) == set(b)
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(v, b[k], err_msg=k)
    assert off.body_ct_gate is None and off._ct_gate_crops == 0


def test_train_opts_validates_body_ct_gate():
    from tsm.config import parse_config

    def cfg(train: dict):
        return parse_config({"volume": {"url": "u", "voxel_um": 2.4},
                             "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
                             "out_dir": "/tmp/x", "extra": {"train": train}})

    base = {"surface_mode": "sides", "fine_store": "f"}
    assert train_opts(cfg(base))["body_ct_gate"] is None
    assert train_opts(cfg({**base, "body_ct_gate": 60}))["body_ct_gate"] == 60.0
    with pytest.raises(ValueError, match="body_ct_gate"):
        train_opts(cfg({**base, "body_ct_gate": 300}))
    with pytest.raises(ValueError, match="body_ct_gate"):
        train_opts(cfg({**base, "surface_mode": "faces", "body_ct_gate": 60}))


# --------------------------------------------------------------------------- #
# 5. the defaults change nothing, and the shipped configs load
# --------------------------------------------------------------------------- #
def test_all_new_weights_default_to_off_and_the_loss_is_bit_identical():
    body, d = two_wraps()
    valid = (torch.rand_like(d) * 3).floor()     # a mix of 0 / 1 / 2 so every mask is exercised
    torch.manual_seed(0)
    pred = torch.cat([torch.rand_like(d) * 8, torch.randn_like(d), torch.randn_like(d)], 1)
    ref = sides_loss(pred, d, valid, body)
    defaults = surface_aux_opts(None, "sides")
    assert defaults["gap_class"] == 0.0 and defaults["gap_dilate"] == 0
    assert defaults["gap_border_weight"] == 0.0 and defaults["eikonal"] == 0.0
    assert defaults["eikonal_min_d"] == 3.0
    off = sides_loss(pred, d, valid, body, aux=defaults)
    assert set(off) == set(ref)
    for k in ref:
        assert float(off[k]) == float(ref[k]), k
    # the historical aux pair is unchanged too
    old = sides_loss(pred, d, valid, body, aux=surface_aux_opts({"gap": 1.0, "cldice": 1.0}, "sides"))
    assert set(old) == set(ref) | {"gap", "cldice"}


def test_shipped_gap_configs_load():
    from tsm.config import load_config

    cfg = load_config(os.path.join("configs", "sides30k_gap.json"))
    o = train_opts(cfg)
    assert cfg.out_dir == "/home/forrest/tsm-output/sides30k_gap"
    assert o["surface_mode"] == "sides" and o["steps"] == 30000 and o["body_ct_gate"] == 60.0
    aux = o["surface_aux"]
    assert (aux["gap"], aux["gap_radius"], aux["gap_dilate"]) == (1.0, 3, 1)
    assert (aux["gap_class"], aux["gap_border_weight"], aux["cldice"]) == (1.0, 3.0, 1.0)
    assert (aux["eikonal"], aux["eikonal_min_d"]) == (0.1, 3.0)
    ev = load_config(os.path.join("configs", "paris4_eval_sides30k_gap.json"))
    assert ev.out_dir == "/home/forrest/tsm-output/eval_sides30k_gap"
    assert ev.extra["infer"]["checkpoint"] == "/home/forrest/tsm-output/sides30k_gap/train/latest.pt"
    assert train_opts(ev)["surface_mode"] == "sides"
