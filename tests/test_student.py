import pytest
import torch

from tsm.student import HEADS, TSMNet, make_input, normalize_ct, scale_channel_value


def test_scale_channel():
    assert scale_channel_value(2.4) == 0.0
    assert scale_channel_value(9.6) == pytest.approx(2.0)


def test_normalize_ct_per_crop():
    ct = torch.randint(0, 256, (3, 8, 8, 8), dtype=torch.uint8)
    x = normalize_ct(ct)
    assert x.shape == ct.shape
    assert torch.allclose(x.mean(dim=(1, 2, 3)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(x.std(dim=(1, 2, 3)), torch.ones(3), atol=1e-4)
    inp = make_input(ct, 9.6)
    assert inp.shape == (3, 2, 8, 8, 8) and torch.all(inp[:, 1] == 2.0)
    inp2 = make_input(ct, torch.tensor([2.4, 9.6, 2.4]))
    assert inp2[0, 1, 0, 0, 0] == 0.0 and inp2[1, 1, 0, 0, 0] == pytest.approx(2.0)


def test_param_count_in_target_range():
    n = TSMNet().num_params()
    assert 5e6 <= n <= 25e6, n  # 19.96M with the FP8-friendly widths (32, 64, 128, 256, 256)


@torch.no_grad()
def test_shapes_64_all_heads_and_deep_supervision():
    net = TSMNet()
    x = torch.randn(1, 2, 64, 64, 64)
    net.train()
    out = net(x)
    assert set(out) == set(HEADS)
    for name, c in HEADS.items():
        assert len(out[name]) == 2
        assert tuple(out[name][0].shape) == (1, c, 64, 64, 64)
        assert tuple(out[name][1].shape) == (1, c, 32, 32, 32)
    net.eval()
    out_e = net(x)
    for name, c in HEADS.items():
        assert isinstance(out_e[name], torch.Tensor)
        assert tuple(out_e[name].shape) == (1, c, 64, 64, 64)
        assert torch.isfinite(out_e[name]).all()


@torch.no_grad()
def test_aux_channels_refiner_mode():
    net = TSMNet(aux_ch=11, widths=(8, 16, 32)).eval()
    assert net.in_ch == 13
    x = torch.randn(1, 2, 16, 16, 16)
    aux = torch.randn(1, 11, 16, 16, 16)
    out = net(x, aux=aux)
    assert tuple(out["winding"].shape) == (1, 8, 16, 16, 16)
    with pytest.raises(ValueError):
        net(x)  # aux missing
    with pytest.raises(ValueError):
        net(torch.randn(1, 2, 14, 14, 14), aux=torch.randn(1, 11, 14, 14, 14))  # not divisible by 4


def test_widths_multiple_of_groups_and_channels_last():
    with pytest.raises(ValueError):
        TSMNet(widths=(12, 24, 48))  # 12 is not a multiple of 8
    net = TSMNet(widths=(8, 16, 16)).to(memory_format=torch.channels_last_3d).eval()
    x = torch.randn(1, 2, 16, 16, 16).contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        out = net(x)
    assert tuple(out["surface"].shape) == (1, 2, 16, 16, 16) and torch.isfinite(out["surface"]).all()
    assert net.stem[0].conv1.weight.is_contiguous(memory_format=torch.channels_last_3d)


def test_custom_heads_and_ds_levels():
    net = TSMNet(widths=(8, 16, 32, 32), heads={"a": 3}, ds_levels=3).train()
    out = net(torch.randn(1, 2, 16, 16, 16))
    assert [tuple(t.shape) for t in out["a"]] == [(1, 3, 16, 16, 16), (1, 3, 8, 8, 8), (1, 3, 4, 4, 4)]


# --------------------------------------------------------------------------- act_ckpt + BatchNorm (R17)
def _tensors(out):
    """Flatten TSMNet's {head: [level tensors]} output."""
    if torch.is_tensor(out):
        return [out]
    vals = out.values() if isinstance(out, dict) else out
    return [t for v in vals for t in _tensors(v)]


def _loss(out):
    return sum(t.float().square().mean() for t in _tensors(out))


def _bn_net(act_ckpt: int) -> TSMNet:
    torch.manual_seed(0)
    return TSMNet(widths=(4, 8, 16), blocks=1, dec_blocks=1, ds_levels=1, norm="batch", act_ckpt=act_ckpt)


def test_act_ckpt_does_not_update_batchnorm_stats_twice():
    """R17: the backward recompute used to bump every checkpointed BN a second time."""
    x = torch.randn(2, 2, 16, 16, 16)
    for ck in (0, 1, 2):
        net = _bn_net(ck).train()
        out = net(x)
        loss = _loss(out)
        loss.backward()
        tracked = {n: int(m.num_batches_tracked) for n, m in net.named_modules()
                   if isinstance(m, torch.nn.modules.batchnorm._NormBase)}
        assert tracked, "no BatchNorm layers in the probe net"
        assert set(tracked.values()) == {1}, (ck, tracked)


def test_act_ckpt_matches_the_uncheckpointed_reference():
    x = torch.randn(2, 2, 16, 16, 16)
    ref, ckn = _bn_net(0).train(), _bn_net(2).train()
    ckn.load_state_dict(ref.state_dict())
    outs = []
    for net in (ref, ckn):
        o = net(x)
        _loss(o).backward()
        outs.append([t.detach().clone() for t in _tensors(o)])
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-5)
    g0 = {n: p.grad for n, p in ref.named_parameters()}
    for n, p in ckn.named_parameters():
        assert torch.allclose(g0[n], p.grad, atol=1e-4), n
    b0 = dict(ref.named_buffers())
    for n, b in ckn.named_buffers():
        assert torch.allclose(b0[n].float(), b.float(), atol=1e-5), n
