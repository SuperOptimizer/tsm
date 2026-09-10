import json
import os

import numpy as np
import pytest
import torch

from synth import make_synthetic
from tsm.config import parse_config
from tsm.student import TSMNet
from tsm.train import (
    EMA,
    augment_mode,
    auroc,
    average_precision,
    compute_losses,
    evaluate,
    train_opts,
    zero_crossing,
    downsample_targets,
    ink_loss,
    load_checkpoint,
    lr_lambda,
    model_input,
    run_train,
    save_checkpoint,
    surface_band_dice,
    surface_loss,
    synthetic_batch,
    winding_loss,
)

P = 16


def test_surface_ignore_voxels_get_zero_gradient():
    b = synthetic_batch(2, P, seed=1)
    valid = b["surface_valid"]
    pred = torch.randn(2, 2, P, P, P, requires_grad=True)
    d = surface_loss(pred, b["surface_sdf"], valid)
    sum(d.values()).backward()
    g = pred.grad
    assert torch.all(g[:, 0:1][valid == 2] == 0) and torch.all(g[:, 1:2][valid == 2] == 0)
    assert torch.all(g[:, 0:1][valid == 0] == 0)  # sdf only where valid==1
    assert torch.any(g[:, 1:2][valid == 0] != 0)  # valid logit trained on valid!=2
    assert torch.any(g[:, 0:1][valid == 1] != 0)


def test_ink_and_winding_masks():
    b = synthetic_batch(2, P, seed=2)
    pi = torch.randn(2, 1, P, P, P, requires_grad=True)
    sum(ink_loss(pi, b["ink_prob"], b["ink_valid"]).values()).backward()
    assert torch.all(pi.grad[b["ink_valid"] != 1] == 0) and torch.any(pi.grad[b["ink_valid"] == 1] != 0)
    pw = torch.randn(2, 8, P, P, P, requires_grad=True)
    d = winding_loss(pw, b["winding"], b["winding_conf"], b["winding_valid"])
    sum(d.values()).backward()
    inval = (b["winding_valid"] != 1).expand(-1, 8, -1, -1, -1)
    assert torch.all(pw.grad[inval] == 0)
    assert torch.all(pw.grad[:, 7] == 0)  # spare channel unsupervised
    assert torch.any(pw.grad[:, 0][b["winding_valid"][:, 0] == 1] != 0)


def test_per_sample_masks_isolate_samples():
    b = synthetic_batch(2, P, seed=3)
    b["surface_valid"][1].fill_(2.0)  # 2 = ignore everywhere (0 would still train the valid logit as negative)
    b["ink_valid"][1].zero_()
    b["winding_valid"][0].zero_()
    net = TSMNet(widths=(8, 16, 16)).train()
    x = model_input(b).requires_grad_(True)
    loss, parts = compute_losses(net(x), b)
    loss.backward()
    assert loss.item() > 0 and all(v >= 0 for v in parts.values())
    # surface/ink supervise sample 0 only, winding sample 1 only (GroupNorm keeps samples independent)
    x = model_input(b).requires_grad_(True)
    ls, _ = compute_losses({"surface": net(x)["surface"]}, b)
    ls.backward()
    assert x.grad[1].abs().sum() == 0 and x.grad[0].abs().sum() > 0
    x = model_input(b).requires_grad_(True)
    lw, _ = compute_losses({"winding": net(x)["winding"]}, b)
    lw.backward()
    assert x.grad[0].abs().sum() == 0 and x.grad[1].abs().sum() > 0


def test_empty_mask_gives_finite_zero_loss():
    b = synthetic_batch(1, P)
    b["surface_valid"].zero_()
    b["ink_valid"].zero_()
    b["winding_valid"].zero_()
    net = TSMNet(widths=(8, 16, 16)).train()
    loss, parts = compute_losses(net(model_input(b)), b)
    assert torch.isfinite(loss)
    assert parts["surface/sdf_l1"] == 0 and parts["winding/phase"] == 0 and parts["ink/ink_bce"] == 0


def test_band_dice_and_downsample():
    t = torch.rand(1, 1, 8, 8, 8) * 20 - 10
    m = torch.ones_like(t)
    assert surface_band_dice(t, t, m).item() < 1e-4
    assert surface_band_dice(t, -t, m).item() < 1e-4  # sign-agnostic
    assert surface_band_dice(t, t + 30, m).item() > 0.5
    b = synthetic_batch(2, P)
    d = downsample_targets({k: b[k] for k in ("surface_sdf", "surface_valid", "ink_prob", "ink_valid", "winding", "winding_conf", "winding_valid")})
    assert tuple(d["surface_sdf"].shape) == (2, 1, 8, 8, 8) and tuple(d["winding"].shape) == (2, 6, 8, 8, 8)
    assert set(d["surface_valid"].unique().tolist()) <= {0.0, 1.0, 2.0}


def test_eval_output_form_accepted():
    b = synthetic_batch(1, P)
    net = TSMNet(widths=(8, 16, 16)).eval()
    with torch.no_grad():
        loss, parts = compute_losses(net(model_input(b)), b)
    assert torch.isfinite(loss) and "total" in parts


def test_optimizer_steps_decrease_loss():
    torch.manual_seed(0)
    b = synthetic_batch(2, P, seed=5)
    net = TSMNet(widths=(8, 16, 16)).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss, _ = compute_losses(net(model_input(b)), b)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses


def test_lr_schedule():
    assert lr_lambda(0, 10, 100) == pytest.approx(0.1)
    assert lr_lambda(9, 10, 100) == pytest.approx(1.0)
    assert lr_lambda(55, 10, 100) == pytest.approx(0.01 + 0.99 * 0.5, abs=1e-6)
    assert lr_lambda(100, 10, 100) == pytest.approx(0.01)


def test_checkpoint_round_trip(tmp_path):
    net = TSMNet(widths=(8, 16, 16))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, 2, 10))
    ema = EMA(net, 0.5)
    b = synthetic_batch(1, P)
    for _ in range(3):
        opt.zero_grad()
        loss, _ = compute_losses(net.train()(model_input(b)), b)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(net)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, net, ema, opt, sched, 3, {"x": 1})
    net2 = TSMNet(widths=(8, 16, 16))
    opt2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: lr_lambda(s, 2, 10))
    ema2 = EMA(net2, 0.5)
    step = load_checkpoint(path, net2, ema2, opt2, sched2)
    assert step == 3
    for p, q in zip(net.parameters(), net2.parameters()):
        torch.testing.assert_close(p, q)
    for p, q in zip(ema.params, ema2.params):
        torch.testing.assert_close(p, q)
    assert sched2.last_epoch == sched.last_epoch and opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert torch.load(path, weights_only=False)["config"] == {"x": 1}
    # ema params differ from the live ones but copy_to installs them
    ema2.copy_to(net2)
    for p, e in zip(net2.parameters(), ema2.params):
        torch.testing.assert_close(p, e)


def _cfg(root, ct_url, extra):
    return parse_config({
        "volume": {"url": ct_url, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": extra},
    })


def test_run_train_dry_run_and_resume_cpu(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    extra = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 1,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "coarse_store": s["coarse"],
             "fine_store": s["fine"]}
    cfg = _cfg(root, s["ct"], extra)
    r = run_train(cfg, dry_run=True)
    assert r["params"] > 0 and "total" in r["losses"]
    assert not os.path.exists(os.path.join(root, "train"))
    r1 = run_train(cfg)
    assert r1["step"] == 2 and os.path.exists(os.path.join(root, "train", "ckpt_2.pt"))
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    assert [l["step"] for l in lines] == [1, 2] and all("surface" in l and "winding" in l and "it_s" in l for l in lines)
    assert all(l["winding"] > 0 for l in lines)  # winding supervised from the upsampled coarse store
    with pytest.raises(FileExistsError):
        run_train(cfg)
    extra["steps"] = 3
    r2 = run_train(_cfg(root, s["ct"], extra), resume=True)
    assert r2["step"] == 3 and os.path.exists(os.path.join(root, "train", "ckpt_3.pt"))
    ck = torch.load(os.path.join(root, "train", "latest.pt"), weights_only=False)
    assert ck["step"] == 3 and set(ck) >= {"model", "ema", "opt", "sched", "step", "config"}


def test_augment_mode_and_train_opts_validation(tmp_path):
    assert augment_mode({"augment": True}) == ("v2", "strong")
    assert augment_mode({"augment": False})[0] == "off" and augment_mode({"augment": "none"})[0] == "off"
    assert augment_mode({"augment": "v1"}) == ("v1", None)
    mode, cfg = augment_mode({"augment": {"preset": "strong", "rotate": {"p": 0.0}}})
    assert mode == "v2" and cfg.rotate.p == 0.0 and cfg.flip.p == 0.5
    with pytest.raises(ValueError):
        augment_mode({"augment": {"rotat": {"p": 1}}})
    with pytest.raises(ValueError):
        augment_mode({"augment": 3})
    root = str(tmp_path)
    with pytest.raises(ValueError, match="must be in"):
        train_opts(_cfg(root, "x", {"augment": {"rotate": {"p": 2}}}))
    with pytest.raises(ValueError, match="unknown augment keys"):
        train_opts(_cfg(root, "x", {"augment": {"rotat": {"p": 1}}}))
    o = train_opts(_cfg(root, "x", {"augment": "light", "holdout_origins": {"z_frac": 0.2}}))
    assert o["augment"] == "light" and o["holdout_origins"] == {"z_frac": 0.2}


def test_metric_helpers():
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    y = np.array([1, 0, 1, 0, 0], bool)
    assert average_precision(s, y) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(s, y > 5) != average_precision(s, y) and np.isnan(average_precision(s, np.zeros(5, bool)))
    assert auroc(s, y) == pytest.approx(5 / 6) and auroc(np.array([0.0, 1.0]), np.array([0, 1], bool)) == 1.0
    sdf = torch.linspace(-3, 3, 8).view(1, 1, 1, 1, 8).expand(1, 1, 4, 4, 8)
    zc = zero_crossing(sdf, torch.ones_like(sdf, dtype=torch.bool))
    assert zc.sum() == 16 and zc[0, 0, :, :, 3].all()  # the last non-positive column


def test_run_train_v2_augment_logs_counts_and_evaluates_holdout(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    extra = {"steps": 2, "patch": 32, "batch": 2, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "coarse_store": s["coarse"], "fine_store": s["fine"],
             "augment": {"preset": "strong", "flip": {"p": 1.0}, "artefact": {"p": 1.0}}, "holdout_origins": {"z_frac": 0.5},
             "eval_max_crops": 3}
    cfg = _cfg(root, s["ct"], extra)
    d = run_train(cfg, dry_run=True)
    assert d["augment_time_s"] > 0
    r = run_train(cfg)
    assert r["n_holdout_origins"] > 0 and r["n_train_origins"] > 0
    assert os.path.exists(os.path.join(root, "train", "holdout.npy"))
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    assert all(l["aug/flip"] == 2 and l["aug/artefact"] == 2 for l in lines) and "aug/rotate" in lines[0]
    m = r["holdout"]
    mj = json.load(open(os.path.join(root, "train", "holdout_metrics.json")))
    assert set(m) == set(mj) and all(mj[k] == v or (np.isnan(v) and np.isnan(mj[k])) for k, v in m.items())
    for k in ("surface/sdf_mae", "surface/zc_dice", "surface/surf_sym_median", "surface/surf_p2t_p90", "surface/surf_t2p_frac_gt3",
              "surface/valid_auprc", "ink/ink_auprc", "winding/phase_err_deg", "winding/density_mae", "winding/normal_err_deg",
              "winding/conf_auroc"):
        assert k in m, k
    assert m["n_crops"] == 3 and np.isfinite(m["surface/sdf_mae"]) and 0 <= m["surface/zc_dice"] <= 1
    assert 0 <= m["winding/phase_err_deg"] <= 180 and 0 <= m["winding/normal_err_deg"] <= 180
    # v1 path still runs
    extra2 = {**extra, "augment": "v1", "holdout_origins": None, "steps": 1}
    r2 = run_train(_cfg(str(tmp_path / "v1"), s["ct"], {**extra2}), force=True)
    assert r2["step"] == 1 and "holdout" not in r2


def test_evaluate_is_deterministic_and_rewards_the_right_answer(tmp_path):
    from tsm.data import CropDataset
    from tsm.volume import VolumeReader

    root = str(tmp_path / "out")
    s = make_synthetic(root)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"], patch=32, stride=16, length=4, augment=False)
    origins = ds.origins[:2]

    class Oracle(torch.nn.Module):
        """Emits the targets of the crop it is asked about (looked up by the CT), i.e. a perfect student."""

        def __init__(self, ds, origins):
            super().__init__()
            self.items = [ds.to_tensors(ds.load(o)) for o in origins]

        def forward(self, x):
            outs = []
            for b in range(x.shape[0]):
                it = min(self.items, key=lambda it: float((model_input({"input": it["input"][None]})[0] - x[b]).abs().sum()))
                w = it["winding"]
                conf = (it["winding_conf"] - 0.5) * 20
                outs.append({"surface": torch.cat([it["surface_sdf"], (it["surface_valid"] == 1).float() * 20 - 10]),
                             "ink": (it["ink_prob"] - 0.5) * 20, "winding": torch.cat([w, conf, torch.zeros_like(conf)])})
            return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}

    m1 = evaluate(Oracle(ds, origins), ds, origins, batch=2)
    m2 = evaluate(Oracle(ds, origins), ds, origins, batch=1)
    for k, v in m1.items():
        if isinstance(v, float) and np.isfinite(v):
            assert m2[k] == pytest.approx(v, abs=1e-5), k
    assert m1["surface/sdf_mae"] < 1e-5 and m1["surface/zc_dice"] == 1.0 and m1["surface/surf_sym_median"] == 0.0
    assert m1["surface/valid_auprc"] == pytest.approx(1.0) and m1["ink/ink_auprc"] == pytest.approx(1.0)
    assert np.isnan(m1["winding/conf_auroc"])  # the synthetic coarse conf is > 0.5 everywhere: one class -> NaN by design
    assert m1["winding/phase_err_deg"] < 1e-2 and m1["winding/normal_err_deg"] < 1e-2 and m1["winding/density_mae"] < 1e-6
    net = TSMNet(widths=(8, 16, 16, 16, 16)).eval()
    m3 = evaluate(net, ds, origins)
    assert m3["surface/sdf_mae"] > m1["surface/sdf_mae"] and m3["n_crops"] == 2


# --------------------------------------------------------------------------- #
# R09: legacy augmentation cannot be combined with cached feature supervision
# --------------------------------------------------------------------------- #
def test_v1_augment_with_cached_features_is_rejected(tmp_path):
    from tsm.train import check_augment_feat_distill

    root = str(tmp_path)
    fd = {"enabled": True, "teachers": ["recto"], "source": "cached"}
    with pytest.raises(ValueError, match="unsupported combination"):
        train_opts(_cfg(root, "x", {"augment": "v1", "feat_distill": dict(fd)}))
    with pytest.raises(ValueError, match="unsupported combination"):  # per-teacher override
        train_opts(_cfg(root, "x", {"augment": "v1",
                                    "feat_distill": {"enabled": True, "teachers": ["recto", "ink"],
                                                     "sources": {"ink": "cached"}}}))
    with pytest.raises(ValueError, match="dino"):  # dino features are always read from a cache
        train_opts(_cfg(root, "x", {"augment": "v1",
                                    "feat_distill": {"enabled": True, "teachers": ["dino"], "pairs": []}}))
    # allowed: live teachers with v1, cached teachers with v2 / off, and cached with feat_distill off
    train_opts(_cfg(root, "x", {"augment": "v1", "feat_distill": {"enabled": True, "teachers": ["recto"]}}))
    train_opts(_cfg(root, "x", {"augment": True, "feat_distill": dict(fd)}))
    train_opts(_cfg(root, "x", {"augment": False, "feat_distill": dict(fd)}))
    train_opts(_cfg(root, "x", {"augment": "v1", "feat_distill": {**fd, "enabled": False}}))
    # the helper is also usable directly on a resolved options dict
    check_augment_feat_distill(train_opts(_cfg(root, "x", {"augment": True, "feat_distill": dict(fd)})))


# --------------------------------------------------------------------------- #
# R16: a completely missed surface must not vanish from the distance summaries
# --------------------------------------------------------------------------- #
def test_missed_surface_is_censored_not_dropped():
    from tsm.train import _dist_stats, _surface_distances

    tgt = np.zeros((8, 8, 8), bool)
    tgt[:, :, 4] = True
    empty = np.zeros((8, 8, 8), bool)
    a, c = _surface_distances(empty, tgt)
    diag = float(np.sqrt(3 * 8.0 ** 2))
    assert a.size == 0 and c.size == int(tgt.sum()) and np.allclose(c, diag)
    a2, c2 = _surface_distances(tgt, empty)  # symmetric: a spurious surface with no label
    assert c2.size == 0 and np.allclose(a2, diag)
    perfect = _surface_distances(tgt, tgt)
    assert perfect[0].max() == 0.0
    # pooled over [perfect crop, total miss] the percentiles must not look perfect
    pooled = np.concatenate([perfect[1], c])
    st = _dist_stats(pooled, "t2p")
    assert st["t2p_median"] > 0 and st["t2p_p90"] > 3 and st["t2p_frac_gt3"] == pytest.approx(0.5)
    assert _surface_distances(empty, empty) == pytest.approx((np.zeros(0), np.zeros(0)))
    assert _surface_distances(empty, tgt, censor=99.0)[1][0] == 99.0


def test_evaluate_counts_and_censors_a_missed_crop(tmp_path):
    """One crop predicted perfectly, one with no surface at all: distances are not 0 and the
    miss is counted."""
    from tsm.data import CropDataset
    from tsm.volume import VolumeReader

    root = str(tmp_path / "out")
    s = make_synthetic(root)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"], patch=32, stride=16,
                     length=4, augment=False)
    origins = ds.origins[:2]

    class HalfBlind(torch.nn.Module):
        """Perfect on one crop (the one with the smaller label surface), empty (sdf = +clip) on the other."""

        def __init__(self, ds, origins):
            super().__init__()
            self.items = [ds.to_tensors(ds.load(o)) for o in origins]

        def forward(self, x):
            outs = []
            for b in range(x.shape[0]):
                ii = min(range(len(self.items)),
                         key=lambda i: float((model_input({"input": self.items[i]["input"][None]})[0] - x[b]).abs().sum()))
                it = self.items[ii]
                w = it["winding"]
                conf = (it["winding_conf"] - 0.5) * 20
                sdf = it["surface_sdf"] if ii == 1 else torch.full_like(it["surface_sdf"], 20.0)
                outs.append({"surface": torch.cat([sdf, (it["surface_valid"] == 1).float() * 20 - 10]),
                             "ink": (it["ink_prob"] - 0.5) * 20,
                             "winding": torch.cat([w, conf, torch.zeros_like(conf)])})
            return {k: torch.stack([o[k] for o in outs]) for k in outs[0]}

    m = evaluate(HalfBlind(ds, origins), ds, origins, batch=1)
    assert m["surface/missed_surfaces"] == 1.0 and m["surface/spurious_surfaces"] == 0.0
    assert m["surface/surf_t2p_median"] > 0 and m["surface/surf_t2p_p90"] > 3
    assert m["surface/surf_t2p_frac_gt3"] > 0.4  # the missed crop's voxels are all counted as far
    assert m["surface/zc_dice"] < 1.0


# --------------------------------------------------------------------------- #
# R20: the eval sampler is a pooled reservoir, not a per-crop quota
# --------------------------------------------------------------------------- #
def test_sampler_is_independent_of_the_eval_batch_size():
    from tsm.train import _Sampler

    rng = np.random.default_rng(0)
    n_crops, per = 16, 500
    scores = rng.random((n_crops, per)).astype(np.float32)
    labels = rng.random((n_crops, per)) < np.linspace(0.02, 0.6, n_crops)[:, None]  # unequal occupancy
    exact = average_precision(scores.ravel(), labels.ravel())
    got = {}
    for B in (1, 2, 4, 8, 16):
        smp = _Sampler(4000, seed=0)
        for i in range(0, n_crops, B):
            smp.add(torch.from_numpy(scores[i:i + B]), torch.from_numpy(labels[i:i + B]), n_crops)
        sc, lb = smp.arrays()
        assert sc.size == 4000 and smp.n == n_crops * per  # sample size never scales with the batch
        got[B] = (float(lb.mean()), average_precision(sc, lb))
    assert len({round(v[0], 12) for v in got.values()}) == 1  # identical sample, any partitioning
    assert got[1][0] == pytest.approx(labels.mean(), abs=0.02)  # pooled prevalence, not per-crop quotas
    assert got[1][1] == pytest.approx(exact, abs=0.03)


def test_sampler_keeps_everything_below_the_cap_and_handles_empty_adds():
    from tsm.train import _Sampler

    smp = _Sampler(100, seed=1)
    smp.add(torch.zeros(0), torch.zeros(0, dtype=torch.bool), 4)
    assert smp.arrays()[0].size == 0 and np.isnan(average_precision(*smp.arrays()))
    smp.add(torch.tensor([0.1, 0.9, 0.4]), torch.tensor([False, True, True]), 4)
    sc, lb = smp.arrays()
    assert sc.size == 3 and sorted(sc.tolist()) == pytest.approx([0.1, 0.4, 0.9])
    assert average_precision(sc, lb) == pytest.approx(1.0)  # both positives rank above the negative
