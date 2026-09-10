"""winding_model_9um reimplementation: key/shape parity, tiny forward, ray sampler, real load (slow)."""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pytest
import torch

from tsm.winding import (
    WINDING_ENABLED,
    WINDING_FILE,
    ArrayReader,
    Axis,
    WindingArch,
    WindingNet,
    WindingTeacher,
    build_winding_net,
    decode_outputs,
    infer_winding_arch,
    load_winding_net,
    prepare_input,
    ray_frame,
    sample_ray_tile,
    windings_per_ct_period,
)
from tsm.teachers import DEFAULT_MODELS_DIR

INVENTORY = sorted(glob.glob(os.path.expanduser("~/.cache/tsm-production/models/winding_model_9um/*/*.tsm-model/model.json")))


def load_inventory(path: str) -> dict[str, tuple[int, ...]]:
    d = json.load(open(path))
    t = d["tensors"]
    items = t.items() if isinstance(t, dict) else ((x["name"], x) for x in t)
    return {k: tuple(int(i) for i in v["shape"]) for k, v in items}


def tiny_arch(**kw) -> WindingArch:
    base = dict(encoder_channels=(4, 8, 12), trunk_dim=24, trunk_stem_blocks=1, axial_attention_blocks=2,
                attention_heads=2, decoder_dim=12, max_relative_distance=8, transverse_tokens=4)
    base.update(kw)
    return WindingArch(**base)


# --------------------------------------------------------------------------- #
# key / shape parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not INVENTORY, reason="winding inventory not present")
def test_inventory_key_parity():
    expected = load_inventory(INVENTORY[0])
    net = build_winding_net(expected)
    got = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    wrong = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    assert not missing and not unexpected and not wrong, (missing[:10], unexpected[:10], wrong[:10])
    assert sum(p.numel() for p in net.parameters()) == 9_135_724
    arch = net.arch
    assert arch.encoder_channels == (32, 64, 96) and arch.trunk_dim == 192 and arch.trunk_stem_blocks == 2
    assert arch.axial_attention_blocks == 6 and arch.attention_heads == 6 and arch.decoder_dim == 96
    assert arch.max_relative_distance == 128 and arch.transverse_size == 128 and arch.out_channels == 4


@pytest.mark.skipif(not INVENTORY, reason="winding inventory not present")
def test_infer_arch_accepts_tensors_and_prefixes():
    inv = load_inventory(INVENTORY[0])
    a = infer_winding_arch(inv)
    b = infer_winding_arch({k: torch.empty(s, device="meta") for k, s in inv.items()})
    c = infer_winding_arch({"module." + k: v for k, v in inv.items()})
    assert a == b == c


def test_tiny_arch_roundtrip_keys():
    net = WindingNet(tiny_arch())
    keys = set(net.state_dict())
    assert "stages.0.layers.0.weight" in keys and "stages.0.layers.2.weight" not in keys
    assert "attention_blocks.1.transverse_attention.relative_bias" in keys
    assert net.state_dict()["attention_blocks.0.ray_attention.relative_bias"].shape == (2, 17)
    assert net.state_dict()["attention_blocks.0.transverse_attention.relative_bias"].shape == (2, 7, 7)
    assert net.state_dict()["full_resolution_head.3.weight"].shape == (4, 4, 1, 1, 1)
    assert infer_winding_arch(net.state_dict()) == tiny_arch()


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ray_strides", [(1, 1, 1), (1, 2, 1), (2, 1, 1)])
def test_tiny_forward_finite_shape_monotone(ray_strides):
    torch.manual_seed(0)
    net = WindingNet(tiny_arch(ray_strides=ray_strides)).eval()
    ct = torch.rand(2, 32, 32, 24) * 255
    valid = torch.ones(2, 32, 32, 24, dtype=torch.bool)
    valid[1, :, :, 20:] = False
    x = prepare_input(ct, valid)
    assert x.shape == (2, 2, 32, 32, 24)
    assert torch.allclose(x[1, 1, :, :, 20:], torch.zeros(1)) and torch.all(x[1, 0, :, :, 20:] == 0)
    with torch.inference_mode():
        out = decode_outputs(net(x))
    assert out["logits"].shape == (2, 32, 32, 24)
    assert all(torch.isfinite(v).all() for v in out.values())
    assert (out["density"] >= 0).all()
    assert (out["phase"].diff(dim=-1) >= -1e-6).all()


def test_prepare_input_modes_and_errors():
    ct = torch.rand(1, 8, 8, 16) * 255
    v = torch.ones(1, 8, 8, 16, dtype=torch.bool)
    z = prepare_input(ct, v, "zscore_masked")
    assert abs(float(z[0, 0].mean())) < 1e-4 and abs(float(z[0, 0].std(unbiased=False)) - 1.0) < 1e-3
    assert torch.allclose(prepare_input(ct, v, "div255")[0, 0], ct[0] / 255)
    with pytest.raises(ValueError):
        prepare_input(ct, v, "bogus")
    with pytest.raises(ValueError):
        WindingNet(tiny_arch())(torch.zeros(1, 3, 32, 32, 24))


# --------------------------------------------------------------------------- #
# ray sampler on a synthetic cylinder volume
# --------------------------------------------------------------------------- #
def cylinder_volume(shape_zyx=(24, 160, 160), center_xy=(80.0, 80.0), radii=(20, 32, 44, 56), thick=1.2):
    z, y, x = np.indices(shape_zyx).astype(np.float64)
    r = np.hypot(x - center_xy[0], y - center_xy[1])
    v = np.zeros(shape_zyx, np.float64)
    for rr in radii:
        v += 200.0 * np.exp(-0.5 * ((r - rr) / thick) ** 2)
    v += 20.0
    return np.clip(v, 0, 255).astype(np.uint8)


def test_sample_ray_tile_cylinder_profile_and_validity():
    vol = cylinder_volume()
    reader = ArrayReader(vol)
    center = (80.0, 80.0, 12.0)
    for angle in (0.0, math.pi / 3, math.pi, 4.7):
        frame = ray_frame(center, angle, r0=10.0, ray_length=60, transverse=8, spacing=1.0)
        ct, valid = sample_ray_tile(reader, frame)
        assert ct.shape == (8, 8, 60) and valid.shape == (8, 8, 60)
        assert valid.all(), "whole tile inside the volume"
        prof = ct[4, 4].numpy()
        peaks = [i for i in range(1, 59) if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 100]
        # sheets at r = 20, 32, 44, 56 -> ray samples k = r - 10
        assert peaks == [10, 22, 34, 46], (angle, peaks)
        # samples between sheets are dark, sheets bright
        assert prof[10] > 180 and prof[16] < 40
    # the frame's own geometry: centre ray points sit on the ray, transverse basis spans z and the tangent
    pts = frame.center_points()
    np.testing.assert_allclose(pts[0], np.array(center) + 10.0 * np.array(frame.direction_xyz), atol=1e-5)
    assert abs(np.dot(frame.axis_a_xyz, frame.direction_xyz)) < 1e-9 and abs(np.dot(frame.axis_b_xyz, frame.direction_xyz)) < 1e-9
    assert np.allclose(frame.axis_a_xyz, (0, 0, 1))


def test_sample_ray_tile_out_of_bounds_marks_invalid():
    vol = cylinder_volume()
    reader = ArrayReader(vol)
    frame = ray_frame((80.0, 80.0, 12.0), 0.0, r0=40.0, ray_length=80, transverse=8)
    ct, valid = sample_ray_tile(reader, frame)
    # x = 80 + 40 + k runs out of the 160-wide volume at k >= 40 (x > 159)
    assert valid[4, 4, :39].all() and not valid[4, 4, 41:].any()
    assert float(ct[4, 4, 60]) == 0.0
    # ArrayReader sub-block with an origin behaves like the full volume
    sub = ArrayReader(vol[:, 60:100, 100:160], origin_zyx=(0, 60, 100), full_shape=vol.shape)
    frame2 = ray_frame((80.0, 80.0, 12.0), 0.0, r0=25.0, ray_length=30, transverse=4)
    a, va = sample_ray_tile(reader, frame2)
    b, vb = sample_ray_tile(sub, frame2)
    assert torch.equal(va, vb) and torch.allclose(a, b)


def test_axis_interpolation_and_levels():
    axis = Axis([(100.0, 200.0, 0.0), (140.0, 240.0, 400.0), (140.0, 200.0, 800.0)])
    assert axis.xy_at(200.0) == (120.0, 220.0)
    assert axis.xy_at(50.0, level=2) == (30.0, 55.0)  # z=200 full-res, /4
    assert axis.xy_at(-10.0) == (100.0, 200.0) and axis.xy_at(10_000.0) == (140.0, 200.0)
    assert np.allclose(axis.xyz_at(600.0), (140.0, 220.0, 600.0))


def test_windings_per_period_metric_on_ideal_rate():
    n = 384
    k = np.arange(n)
    for period in (8, 16, 24):
        ct = 60 + 100 * np.cos(2 * np.pi * k / period)
        dens = np.full(n, 1.0 / period)
        w = windings_per_ct_period(ct, dens, np.ones(n, bool))
        assert len(w) > 0 and np.allclose(w, 1.0, atol=0.05), (period, w)


def test_flat_ct_is_rejected_by_the_periodicity_check():
    """R24: for a constant profile the FFT peak and median are both 0, so the ratio test alone
    accepts the first in-band bin and turns a constant density into perfect windings per period."""
    n = 384
    valid = np.ones(n, bool)
    dens = np.full(n, 1.0 / 32.0)
    for name, ct in (("flat zero", np.zeros(n)),
                     ("flat 128", np.full(n, 128.0)),
                     ("tiny noise", np.random.default_rng(0).normal(0, 1e-6, n)),
                     ("low-amplitude ripple", 128 + 0.01 * np.cos(2 * np.pi * np.arange(n) / 16)),
                     ("linear ramp", np.linspace(0, 255, n))):
        w = windings_per_ct_period(ct, dens, valid)
        assert len(w) == 0, (name, w)
    # a real sheet train is still accepted (and still measured correctly)
    ct = 128 + 60 * np.cos(2 * np.pi * np.arange(n) / 16)
    w = windings_per_ct_period(ct, np.full(n, 1.0 / 16.0), valid)
    assert len(w) >= 5 and np.allclose(w, 1.0, atol=0.05)
    # ... including with noise on top of it
    ct2 = ct + np.random.default_rng(1).normal(0, 5, n)
    w2 = windings_per_ct_period(ct2, np.full(n, 1.0 / 16.0), valid)
    assert len(w2) >= 5 and np.allclose(w2, 1.0, atol=0.05)


def test_kill_switch_is_off_until_validation_passes():
    assert WINDING_ENABLED is False


# --------------------------------------------------------------------------- #
# slow: real checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_checkpoint_loads_strictly_and_runs():
    path = os.path.join(DEFAULT_MODELS_DIR, WINDING_FILE)
    if not os.path.exists(path):
        pytest.skip("winding checkpoint not present")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_winding_net(device=device)
    assert cfg["model"]["trunk_dim"] == 192 and cfg["ray_length"] == 384
    teacher = WindingTeacher(net, device=device)
    torch.manual_seed(0)
    ct = torch.rand(128, 128, 384) * 255
    valid = torch.ones(128, 128, 384, dtype=torch.bool)
    out = teacher.predict_tile(ct, valid)
    assert out["density"].shape == (1, 128, 128, 384)
    assert torch.isfinite(out["density"]).all() and (out["density"] >= 0).all()
    # the trained head bias corresponds to ~0.057 windings/voxel; a random tile stays in that regime
    assert 0.01 < float(out["density"].mean()) < 0.3
