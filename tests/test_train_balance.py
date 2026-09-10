"""Per-head loss balancing (``extra.train.balance``): LossBalancer, gradient-norm logging."""

import json
import os

import pytest
import torch

from synth import make_synthetic
from tsm.config import parse_config
from tsm.student import TSMNet
from tsm.train import (
    LossBalancer,
    compute_losses,
    head_grad_norms,
    model_input,
    run_train,
    synthetic_batch,
    train_opts,
)

P = 16


def _cfg(root, ct_url, extra):
    return parse_config({
        "volume": {"url": ct_url, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": extra},
    })


def _base_extra(s, **kw):
    e = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
         "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "augment": False,
         "coarse_store": s["coarse"], "fine_store": s["fine"], "eval_holdout": False}
    e.update(kw)
    return e


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_balance_option_validation(tmp_path):
    root = str(tmp_path)
    o = train_opts(_cfg(root, "x", {}))
    assert o["balance"] is None and o["balance_warmup"] == 200 and o["balance_every"] == 50
    assert o["balance_freeze"] is True and o["grad_log_every"] == 100
    assert train_opts(_cfg(root, "x", {"balance": "scale"}))["balance"] == "scale"
    with pytest.raises(ValueError, match="balance must be one of"):
        train_opts(_cfg(root, "x", {"balance": "gradnorm"}))
    with pytest.raises(ValueError, match="balance_decay"):
        train_opts(_cfg(root, "x", {"balance_decay": 1.0}))
    with pytest.raises(ValueError, match="balance_warmup"):
        train_opts(_cfg(root, "x", {"balance_every": 0}))
    with pytest.raises(ValueError, match="grad_log_every"):
        train_opts(_cfg(root, "x", {"grad_log_every": -1}))


# --------------------------------------------------------------------------- #
# "scale"
# --------------------------------------------------------------------------- #
def test_scale_is_off_during_warmup_then_equalises_two_heads():
    w = {"big": 1.0, "small": 1.0}
    bal = LossBalancer("scale", w, warmup=5, decay=0.5)
    parts = {"big": 100.0, "small": 1.0}
    for _ in range(4):
        bal.update(parts)
        assert bal.scales() == w  # warmup: unbalanced, exactly loss_weights
    for _ in range(60):
        bal.update(parts)
    m = bal.scales()
    contrib = {h: m[h] * parts[h] for h in parts}
    assert contrib["big"] == pytest.approx(contrib["small"], rel=1e-3)
    # the total keeps the magnitude of the unbalanced sum (sum_h w_h * loss_h)
    assert sum(contrib.values()) == pytest.approx(sum(w[h] * parts[h] for h in parts), rel=1e-3)


def test_scale_respects_loss_weights_and_zero_weight_heads():
    bal = LossBalancer("scale", {"a": 1.0, "b": 3.0, "off": 0.0}, warmup=0, decay=0.5)
    parts = {"a": 100.0, "b": 1.0, "off": 7.0}
    for _ in range(60):
        bal.update(parts)
    m = bal.scales()
    assert m["off"] == 0.0
    assert (m["b"] * parts["b"]) / (m["a"] * parts["a"]) == pytest.approx(3.0, rel=1e-3)


def test_scale_freeze_vs_continuous():
    parts = {"a": 100.0, "b": 1.0}
    frozen = LossBalancer("scale", {"a": 1.0, "b": 1.0}, warmup=3, decay=0.5)
    live = LossBalancer("scale", {"a": 1.0, "b": 1.0}, warmup=3, decay=0.5, freeze=False)
    for _ in range(20):
        frozen.update(parts)
        live.update(parts)
    m0 = frozen.scales()
    drift = {"a": 10.0, "b": 1.0}  # head "a" improves 10x
    for _ in range(50):
        frozen.update(drift)
        live.update(drift)
    assert frozen.scales() == m0  # frozen: constant multipliers, a fixed reweighting
    # continuous: the ratio of the multipliers tracks the 10x change in the ratio of the losses
    # (the overall scale stays pinned by the sum-of-weights rescale, so compare ratios)
    assert live.scales()["a"] / live.scales()["b"] == pytest.approx(10 * m0["a"] / m0["b"], rel=0.05)
    lc = {h: live.scales()[h] * drift[h] for h in drift}
    assert lc["a"] == pytest.approx(lc["b"], rel=1e-2)


def test_null_balance_multiplies_by_the_plain_loss_weights():
    bal = LossBalancer(None, {"surface": 1.0, "ink": 6.0})
    assert not bal.enabled and bal.scales() is None
    bal.update({"surface": 1.0, "ink": 100.0})
    assert bal.scales() is None and bal.log() == {}
    b = synthetic_batch(1, P, seed=1)
    net = TSMNet(widths=(8, 16, 16)).train()
    out = net(model_input(b))
    w = {"surface": 1.0, "ink": 6.0, "winding": 1.0}
    a, _ = compute_losses(out, b, w)
    c, parts = compute_losses(out, b, w, head_scale=w)
    assert float(a.detach()) == pytest.approx(float(c.detach()), rel=1e-6)
    assert parts["balance/ink"] == 6.0 and parts["balance/ink_contrib"] == pytest.approx(6.0 * parts["ink"])


# --------------------------------------------------------------------------- #
# "gradnorm_lite"
# --------------------------------------------------------------------------- #
def test_gradnorm_lite_equalises_the_measured_trunk_norms():
    torch.manual_seed(0)
    b = synthetic_batch(1, P, seed=4)
    net = TSMNet(widths=(8, 16, 16)).train()
    weights = {"surface": 1.0, "ink": 2.0, "winding": 1.0}
    bal = LossBalancer("gradnorm_lite", weights, every=1, decay=0.5)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    head_params = {h: list(net.head[h].parameters()) for h in net.head}
    trunk = list(net.dec[0].parameters())
    ratios = []
    for it in range(10):
        opt.zero_grad()
        hl: dict[str, torch.Tensor] = {}
        loss, parts = compute_losses(net(model_input(b)), b, weights, head_scale=bal.scales(), head_losses=hl)
        norms = head_grad_norms(hl, head_params, trunk)
        assert set(norms) == set(weights) and all(n > 0 for _, n in norms.values())
        mult = bal.scales()
        # post-multiplier norms, normalised by loss_weights: equal => balanced
        eq = {h: mult[h] * n_all / weights[h] for h, (_, n_all) in norms.items()}
        ratios.append(max(eq.values()) / min(eq.values()))
        bal.measure({h: n_all for h, (_, n_all) in norms.items()})
        loss.backward()
        opt.step()
    assert ratios[0] > 1.5, ratios  # the raw heads are far apart to begin with
    assert ratios[-1] == pytest.approx(1.0, abs=0.15), ratios
    assert all(v > 0 for v in bal.norms.values()) and "balance/ink_gnorm" in bal.log()


def test_head_grad_norms_do_not_disturb_the_backward():
    torch.manual_seed(0)
    b = synthetic_batch(1, P, seed=6)
    net = TSMNet(widths=(8, 16, 16)).train()
    ref = {}
    for measure in (False, True):
        net.zero_grad(set_to_none=True)
        hl: dict[str, torch.Tensor] = {}
        loss, _ = compute_losses(net(model_input(b)), b, head_losses=hl)
        if measure:
            head_grad_norms(hl, {h: list(net.head[h].parameters()) for h in net.head},
                            list(net.dec[0].parameters()))
        loss.backward()
        g = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        if not measure:
            ref = g
        else:
            assert set(g) == set(ref) and all(torch.equal(g[k], ref[k]) for k in g)


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def test_balance_null_run_is_bit_identical_with_and_without_grad_logging(tmp_path):
    s = make_synthetic(str(tmp_path / "data"))
    outs = []
    for name, gle in (("a", 0), ("b", 1)):
        root = str(tmp_path / name)
        r = run_train(_cfg(root, s["ct"], _base_extra(s, steps=3, ckpt_every=3, grad_log_every=gle)))
        outs.append((r, torch.load(os.path.join(root, "train", "latest.pt"), weights_only=False)))
        lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
        keys = set().union(*(set(l) for l in lines))
        assert not any(k.startswith("balance/") for k in keys)
        assert (any(k.startswith("gradnorm/") for k in keys)) == (gle > 0)
    (r0, c0), (r1, c1) = outs
    assert r0["losses"]["total"] == r1["losses"]["total"]
    assert all(torch.equal(c0["model"][k], c1["model"][k]) for k in c0["model"])


def test_run_train_logs_balance_multipliers_and_gradnorms(tmp_path):
    s = make_synthetic(str(tmp_path / "data"))
    root = str(tmp_path / "bal")
    extra = _base_extra(s, steps=4, ckpt_every=4, balance="scale", balance_warmup=1, balance_decay=0.5,
                        grad_log_every=1, loss_weights={"surface": 1.0, "ink": 2.0, "winding": 1.0})
    run_train(_cfg(root, s["ct"], extra))
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    assert len(lines) == 4
    for h in ("surface", "ink", "winding"):
        assert all(f"balance/{h}" in l and f"balance/{h}_contrib" in l and f"gradnorm/{h}" in l for l in lines)
        assert lines[0][f"balance/{h}"] == pytest.approx(extra["loss_weights"][h])  # warmup step: unbalanced
    last = lines[-1]
    contrib = {h: last[f"balance/{h}_contrib"] for h in ("surface", "ink", "winding")}
    raw = {h: last[h] for h in contrib}
    assert max(raw.values()) / min(raw.values()) > 1.5             # the raw head losses differ a lot
    assert max(contrib.values()) / min(contrib.values()) < 2.5     # the contributions much less


def test_gradnorm_lite_runs_and_logs(tmp_path):
    s = make_synthetic(str(tmp_path / "data"))
    root = str(tmp_path / "gn")
    extra = _base_extra(s, steps=3, ckpt_every=3, balance="gradnorm_lite", balance_every=1, grad_log_every=0)
    r = run_train(_cfg(root, s["ct"], extra))
    assert r["step"] == 3
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    for h in ("surface", "ink", "winding"):
        assert all(l[f"balance/{h}"] > 0 and l[f"balance/{h}_gnorm"] > 0 for l in lines[1:])
