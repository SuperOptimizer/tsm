#!/usr/bin/env python
"""Augmentation / training ablations from one base config.

    uv run python dev/ablate.py configs/paris4.json --steps 300 --seed 0 \
        --variants '{"no_rot": {"augment.rotate.p": 0}, "no_aug": {"augment": "none"}, "v1": {"augment": "v1"}}' \
        [--variants-file variants.json] [--out ~/tsm-output/ablation] [--holdout-z-frac 0.15] [--crops 32] [--dry-run] [--force]

Override keys are dotted paths into ``extra.train`` (``augment.<transform>.<field>``, ``lr``,
``loss_weights.ink``, ...).  Every variant trains ``--steps`` optimizer steps with the same seed
and holdout split and is scored on the held-out crops; see ``<out>/ablation.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tsm.ablate import run_ablation  # noqa: E402
from tsm.config import RunCfg, load_config  # noqa: E402
from tsm.limits import MemWatchdog, peak_rss_mb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--variants", default=None, help="JSON object {name: {dotted train key: value}}")
    p.add_argument("--variants-file", default=None, help="JSON file with the same structure")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output root (default <out_dir>/ablation)")
    p.add_argument("--holdout-z-frac", type=float, default=None)
    p.add_argument("--crops", type=int, default=None, help="holdout crops evaluated per variant")
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    ab = dict(cfg.extra.get("ablate", {}))
    variants = dict(ab.get("variants", {}))
    if a.variants_file:
        with open(a.variants_file) as fh:
            variants.update(json.load(fh))
    if a.variants:
        variants.update(json.loads(a.variants))
    ab["variants"] = variants
    if a.steps is not None:
        ab["steps"] = a.steps
    if a.seed is not None:
        ab["seed"] = a.seed
    if a.out:
        ab["out_dir"] = os.path.expanduser(a.out)
    if a.holdout_z_frac is not None:
        ab["holdout_origins"] = {"z_frac": a.holdout_z_frac}
    if a.crops is not None:
        ab["eval_max_crops"] = a.crops
    if a.no_base:
        ab["include_base"] = False
    cfg = RunCfg(cfg.volume, cfg.region, cfg.budget, cfg.out_dir, {**cfg.extra, "ablate": ab})
    with MemWatchdog(cfg.budget):
        run_ablation(cfg, dry_run=a.dry_run, force=a.force)
    print(f"[ablate] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
