"""Ablation harness on the synthetic stores (CPU, 1-step runs)."""

from __future__ import annotations

import json
import os

import pytest

from synth import make_synthetic
from tsm.ablate import ablate_opts, apply_train_overrides, run_ablation, variant_config
from tsm.config import parse_config


def _cfg(root, ct_url, train, ablate):
    return parse_config({
        "volume": {"url": ct_url, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": train, "ablate": ablate},
    })


def test_overrides_expand_presets_and_validate():
    t = apply_train_overrides({"augment": True, "lr": 1e-3}, {"augment.rotate.p": 0, "lr": 2e-3, "loss_weights.ink": 0.5})
    assert t["augment"] == {"preset": "strong", "rotate": {"p": 0}} and t["lr"] == 2e-3 and t["loss_weights"] == {"ink": 0.5}
    t = apply_train_overrides({"augment": {"preset": "light", "flip": {"p": 1}}}, {"augment.flip.axes": [True, False, False]})
    assert t["augment"]["flip"] == {"p": 1, "axes": [True, False, False]} and t["augment"]["preset"] == "light"
    assert apply_train_overrides({"augment": "strong"}, {"augment": "none"})["augment"] == "none"
    with pytest.raises(ValueError):
        apply_train_overrides({"augment": "v1"}, {"augment.rotate.p": 0})


def test_ablation_end_to_end(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    train = {"patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "num_workers": 0, "stride": 16,
             "log_every": 1, "coarse_store": s["coarse"], "fine_store": s["fine"]}
    ablate = {"steps": 1, "seed": 3, "holdout_origins": {"z_frac": 0.5}, "eval_max_crops": 2,
              "variants": {"no_rot": {"augment.rotate.p": 0.0}, "no_aug": {"augment": "none"}, "broken": {"augment": {"nope": 1}}}}
    cfg = _cfg(root, s["ct"], train, ablate)
    with pytest.raises(ValueError):  # the broken variant is rejected before anything trains
        run_ablation(cfg, dry_run=True)
    del ablate["variants"]["broken"]
    cfg = _cfg(root, s["ct"], train, ablate)
    o = ablate_opts(cfg)
    assert o["include_base"] and set(o["variants"]) == {"no_rot", "no_aug"}
    vc = variant_config(cfg, "no_rot", {"augment.rotate.p": 0.0}, o, os.path.join(root, "ablation"))
    assert vc.extra["train"]["augment"] == {"preset": "strong", "rotate": {"p": 0.0}} and vc.extra["train"]["steps"] == 1
    assert vc.out_dir == os.path.join(root, "ablation", "no_rot") and vc.extra["train"]["holdout_origins"] == {"z_frac": 0.5}
    d = run_ablation(cfg, dry_run=True)
    assert set(d["variants"]) == {"base", "no_rot", "no_aug"} and not os.path.exists(os.path.join(root, "ablation", "base"))
    r = run_ablation(cfg)
    res = r["results"]
    assert set(res) == {"base", "no_rot", "no_aug"} and all("metrics" in v and "error" not in v for v in res.values())
    assert all(v["metrics"]["n_crops"] == 2 and v["step"] == 1 for v in res.values())
    md = open(os.path.join(root, "ablation", "ablation.md")).read()
    assert "| base |" in md and "| no_rot |" in md and "sdf MAE" in md and "phase deg" in md
    js = json.load(open(os.path.join(root, "ablation", "ablation.json")))
    assert js["opts"]["steps"] == 1 and set(js["results"]) == set(res)
    for n in res:
        assert os.path.exists(os.path.join(root, "ablation", n, "train", "holdout_metrics.json"))
    with pytest.raises(ValueError):
        ablate_opts(_cfg(root, s["ct"], train, {"bogus": 1}))
    with pytest.raises(ValueError):
        ablate_opts(_cfg(root, s["ct"], train, {"variants": {"base": {}}}))


def test_cli_has_ablate_stage():
    from tsm.cli import STAGES

    assert "ablate" in STAGES


def test_variant_seed_override_wins_over_the_ablation_seed(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    train = {"patch": 32, "batch": 1, "widths": [8, 16, 16, 16, 16], "num_workers": 0, "stride": 16,
             "coarse_store": s["coarse"], "fine_store": s["fine"]}
    ablate = {"steps": 1, "seed": 3, "variants": {"seed1": {"seed": 1}}}
    cfg = _cfg(root, s["ct"], train, ablate)
    o = ablate_opts(cfg)
    assert variant_config(cfg, "base", {}, o, root).extra["train"]["seed"] == 3
    assert variant_config(cfg, "seed1", {"seed": 1}, o, root).extra["train"]["seed"] == 1


def test_ablate_augment_config_loads():
    """configs/ablate_augment.json: the augmentation sweep (none / geo_only / strong_scan / seed)."""
    import glob

    from tsm.config import load_config
    from tsm.data import INTENSITY_NAMES, SCAN_NAMES

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ablate_augment.json")
    assert glob.glob(path)
    cfg = load_config(path)
    o = ablate_opts(cfg)
    assert o["steps"] == 6000 and o["include_base"] and o["eval_max_crops"] == 64
    assert o["holdout_origins"] == {"y_frac": 0.15} and o["out_dir"].endswith("ablation_augment")
    assert set(o["variants"]) == {"none", "geo_only", "strong_scan", "strong_seed1"}
    # variant_config validates extra.train for every variant (nothing is written here)
    out = {n: variant_config(cfg, n, ov, o, "/tmp/ablate_augment_check") for n, ov in o["variants"].items()}
    assert out["none"].extra["train"]["augment"] == "none"
    assert out["strong_scan"].extra["train"]["augment"] == "strong_scan"
    assert out["strong_seed1"].extra["train"]["seed"] == 1 and out["strong_seed1"].extra["train"]["augment"] == "strong"
    geo = out["geo_only"].extra["train"]["augment"]
    assert geo["preset"] == "strong" and all(geo[n] == {"p": 0.0} for n in INTENSITY_NAMES)
    assert all(n in geo for n in SCAN_NAMES) and "flip" not in geo and "rotate" not in geo
