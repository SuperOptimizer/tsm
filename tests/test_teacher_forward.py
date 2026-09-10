"""Forward-pass sanity checks on tiny synthetic specs, plus one slow real-load test."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tsm.teachers import (
    TEACHERS,
    ArchSpec,
    DecoderSpec,
    NNUNetResEncUNet,
    Normalizer,
    VesuviusUNet,
    apply_activation,
    infer_nnunet_arch,
    infer_vesuvius_arch,
    load_teacher,
)


def tiny_spec(se=None, upsample="transpconv", norm="instance", residual_decoder=False):
    return ArchSpec(
        in_channels=1, features_per_stage=[8, 16, 32], n_blocks_per_stage=[1, 1, 1], strides=[1, 2, 2],
        conv_bias=True, norm=norm, squeeze_excitation=se,
        task_decoders={"surface": DecoderSpec(2, [1, 1], residual=residual_decoder, upsample_mode=upsample)},
        shared_decoder=DecoderSpec(None, [1, 1], upsample_mode=upsample),
        task_heads={"ink": 1},
    )


@pytest.mark.parametrize("se", [None, "scse"])
@pytest.mark.parametrize("upsample", ["transpconv", "trilinear", "pixelshuffle"])
def test_vesuvius_forward_shapes(se, upsample):
    torch.manual_seed(0)
    net = VesuviusUNet(tiny_spec(se=se, upsample=upsample)).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.inference_mode():
        out = net(x)
    assert set(out) == {"surface", "ink"}
    assert out["surface"].shape == (1, 2, 32, 32, 32)
    assert out["ink"].shape == (1, 1, 32, 32, 32)
    assert all(torch.isfinite(v).all() for v in out.values())


@pytest.mark.parametrize("norm", ["instance", "group", "none"])
def test_norm_variants(norm):
    spec = tiny_spec(norm=norm)
    spec.num_groups = 4
    net = VesuviusUNet(spec)
    keys = net.state_dict().keys()
    has_norm = any(k.endswith("stem.convs.0.norm.weight") for k in keys)
    assert has_norm == (norm != "none")
    n_all = len([k for k in keys if k.startswith("shared_encoder.stem.convs.0.all_modules.")])
    assert n_all == (4 if norm != "none" else 2)
    with torch.inference_mode():
        out = net(torch.randn(1, 1, 16, 16, 16))
    assert out["surface"].shape == (1, 2, 16, 16, 16)


def test_state_dict_round_trip_and_shape_inference():
    torch.manual_seed(1)
    spec = tiny_spec(se="scse", residual_decoder=True)
    a = VesuviusUNet(spec)
    sd = a.state_dict()
    # duplicate registrations point at the same storage
    assert sd["shared_encoder.stem.convs.0.conv.weight"].data_ptr() == \
        sd["shared_encoder.stem.convs.0.all_modules.0.weight"].data_ptr()
    inferred = infer_vesuvius_arch(sd)
    assert inferred == spec
    b = VesuviusUNet(inferred)
    missing, unexpected = b.load_state_dict(sd, strict=True), None
    x = torch.randn(1, 1, 16, 16, 16)
    a.eval(), b.eval()
    with torch.inference_mode():
        ya, yb = a(x), b(x)
    for k in ya:
        assert torch.equal(ya[k], yb[k])


def test_nnunet_forward_and_round_trip():
    torch.manual_seed(2)
    spec = ArchSpec(in_channels=1, features_per_stage=[8, 16, 32], n_blocks_per_stage=[1, 2, 1], strides=[1, 2, 2],
                    decoder=DecoderSpec(3, [1, 1]))
    net = NNUNetResEncUNet(spec).eval()
    sd = net.state_dict()
    assert "decoder.encoder.stem.convs.0.conv.weight" in sd and "encoder.stem.convs.0.conv.weight" in sd
    assert infer_nnunet_arch(sd) == spec
    net2 = NNUNetResEncUNet(infer_nnunet_arch(sd)).eval()
    net2.load_state_dict(sd, strict=True)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.inference_mode():
        y = net(x)
        assert torch.equal(y, net2(x))
    assert y.shape == (1, 3, 32, 32, 32)
    assert apply_activation(y, "softmax").sum(1).allclose(torch.ones(1, 32, 32, 32))


def test_skip_index_shifts_when_strided():
    net = VesuviusUNet(tiny_spec())
    keys = set(net.state_dict())
    # stage 0: same channels, stride 1 -> identity skip, no keys
    assert not any(k.startswith("shared_encoder.stages.0.blocks.0.skip") for k in keys)
    # stage 1: strided + projection -> AvgPool at skip.0, conv at skip.1 (no bias)
    assert "shared_encoder.stages.1.blocks.0.skip.1.conv.weight" in keys
    assert "shared_encoder.stages.1.blocks.0.skip.1.conv.bias" not in keys
    assert "shared_encoder.stages.1.blocks.0.skip.1.norm.weight" in keys


def test_normalizers_match_reference_formulas():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(16, 16, 16), dtype=np.uint8)
    xf = x.astype(np.float32)
    z = Normalizer("zscore_instance")(x)
    assert z.dtype == torch.float32
    np.testing.assert_allclose(z.numpy(), (xf - xf.mean()) / max(xf.std(), 1e-8), rtol=1e-5, atol=1e-5)
    ct = Normalizer("ct_clip", mean=87.54424285888672, std=47.74376678466797, lo=0.0, hi=212.0)(x)
    np.testing.assert_allclose(ct.numpy(), (np.clip(xf, 0, 212) - 87.54424285888672) / 47.74376678466797, rtol=1e-5)
    pm = Normalizer("percentile_minmax")(torch.from_numpy(x))
    lo, hi = np.percentile(xf, (1.0, 99.0))
    np.testing.assert_allclose(pm.numpy(), (np.clip(xf, lo, hi) - lo) / (hi - lo), rtol=1e-5, atol=1e-6)
    d = Normalizer("div255")(x)
    np.testing.assert_allclose(d.numpy(), xf / 255.0)
    assert torch.all(Normalizer("percentile_minmax")(np.full((4, 4, 4), 7, np.uint8)) == 0)


def test_teacher_registry():
    assert set(TEACHERS) == {"recto", "m7", "ink", "lasagna", "fiber"}
    for name, spec in TEACHERS.items():
        assert spec.name == name
        assert spec.kind in ("vesuvius", "nnunet")
        assert (spec.target is None) == (spec.kind == "nnunet")


@pytest.mark.slow
def test_real_recto_load_and_forward():
    path = os.path.join(os.path.expanduser("~/.cache/tsm-models"), TEACHERS["recto"].file)
    if not os.path.exists(path):
        pytest.skip("recto checkpoint not present")
    net, spec = load_teacher("recto")
    x = spec.normalizer(np.zeros((128, 128, 128), np.uint8))[None, None]
    with torch.inference_mode():
        out = spec.select(net(x))
    assert out.shape == (1, 2, 128, 128, 128)
    assert torch.isfinite(out).all()
