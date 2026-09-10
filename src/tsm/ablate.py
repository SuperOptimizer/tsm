"""Ablation harness: train short runs from one base config with named overrides, score each on a
fixed held-out crop set (``tsm.train.evaluate``), write ``ablation.md`` + ``ablation.json``.

    tsm ablate config.json            # extra.ablate = {variants, steps, seed, holdout_origins, ...}
    uv run python dev/ablate.py config.json --steps 300 --variants '{"no_rot": {"augment.rotate.p": 0}}'

Every variant is ``run_train`` on a copy of the base config with ``extra.train`` overrides
applied by dotted path (``"augment.rotate.p"``, ``"lr"``, ``"loss_weights.ink"``); ``augment``
given as a preset name is expanded to ``{"preset": name}`` first so per-transform keys apply.
The step budget, seed and holdout split are the same for every variant; the EMA checkpoint of
each run is evaluated on the holdout crops (``<out_dir>/<name>/train/holdout_metrics.json``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Any

from tsm.config import RunCfg

ABLATE_DEFAULTS: dict[str, Any] = {
    "variants": {},  # {name: {"<train dotted key>": value, ...}}
    "include_base": True,
    "steps": 300,
    "seed": 0,
    "holdout_origins": {"z_frac": 0.15},
    "eval_max_crops": 32,
    "out_dir": None,  # default <out_dir>/ablation
    "train": {},  # extra.train overrides applied to every variant (dotted keys)
}

# columns of the markdown table (metric key, short header, lower-is-better). Two-face (faces) runs
# report ``surface/in_*`` / ``surface/out_*`` instead of the single-SDF keys; ``_metric`` maps them.
REPORT_COLUMNS: list[tuple[str, str, bool]] = [
    ("surface/sdf_mae", "sdf MAE", True),
    ("surface/zc_dice", "zc Dice", False),
    ("surface/surf_sym_median", "surf med", True),
    ("surface/surf_sym_p90", "surf p90", True),
    ("surface/surf_sym_frac_gt3", "surf >3", True),
    ("surface/out_surf_sym_median", "out med", True),
    ("surface/out_surf_sym_frac_gt3", "out >3", True),
    ("surface/valid_auprc", "valid AUPRC", False),
    ("ink/ink_auprc", "ink AUPRC", False),
    ("winding/phase_err_deg", "phase deg", True),
    ("winding/density_mae", "dens MAE", True),
    ("winding/normal_err_deg", "normal deg", True),
    ("winding/conf_auroc", "conf AUROC", False),
    # fibre head (only present when heads.fiber is on; the direction mode adds the angular error
    # and the teacher-independent human-band numbers)
    ("fiber/vt_auprc", "vt AUPRC", False),
    ("fiber/hz_auprc", "hz AUPRC", False),
    ("fiber/angle_deg", "fibre deg", True),
    ("fiber/band_angle_deg", "band deg", True),
    ("fiber/band_class_acc", "band acc", False),
]


def ablate_opts(cfg: RunCfg) -> dict[str, Any]:
    raw = cfg.extra.get("ablate", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.ablate must be an object")
    unknown = sorted(set(raw) - set(ABLATE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.ablate keys: {unknown}")
    opts = copy.deepcopy(ABLATE_DEFAULTS)
    opts.update(copy.deepcopy(raw))
    if not isinstance(opts["variants"], dict) or not all(isinstance(v, dict) for v in opts["variants"].values()):
        raise ValueError("extra.ablate.variants must map names to {dotted train key: value}")
    if "base" in opts["variants"]:
        raise ValueError("'base' is reserved for the unmodified config")
    if int(opts["steps"]) < 1:
        raise ValueError("extra.ablate.steps must be >= 1")
    return opts


def apply_train_overrides(train: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Copy of the ``extra.train`` dict with dotted-path overrides applied."""
    from tsm.data import apply_overrides

    t = copy.deepcopy(train)
    if any(str(k).startswith("augment.") for k in overrides):
        a = t.get("augment", True)
        if a is True:
            a = "strong"
        if isinstance(a, str) and a != "v1":
            t["augment"] = {"preset": a}
        elif not isinstance(a, dict):
            raise ValueError(f"cannot apply augment.* overrides to augment={a!r}")
    return apply_overrides(t, overrides)


def variant_config(cfg: RunCfg, name: str, overrides: dict[str, Any], opts: dict[str, Any], out_root: str) -> RunCfg:
    """The base config with the ablation budget, the holdout split, the label stores pinned and ``overrides``."""
    from tsm.train import train_opts

    base_train = dict(cfg.extra.get("train", {}))
    if not base_train.get("stores"):  # extra.train.stores is self-contained and replaces these
        base_train.setdefault("fine_store", os.path.join(cfg.out_dir, "labels", "fine.zarr"))
        base_train.setdefault("coarse_store", os.path.join(cfg.out_dir, "labels", "coarse.zarr"))
    base_train.update({
        "steps": int(opts["steps"]), "seed": int(opts["seed"]), "holdout_origins": opts["holdout_origins"],
        "eval_holdout": True, "eval_max_crops": int(opts["eval_max_crops"]),
    })
    base_train.setdefault("ckpt_every", int(opts["steps"]))
    base_train.setdefault("warmup", max(1, min(int(base_train.get("warmup", 500)), int(opts["steps"]) // 10)))
    train = apply_train_overrides(base_train, {**opts.get("train", {}), **overrides})
    extra = copy.deepcopy(cfg.extra)
    extra.pop("ablate", None)
    extra["train"] = train
    vc = RunCfg(volume=cfg.volume, region=cfg.region, budget=cfg.budget, out_dir=os.path.join(out_root, name), extra=extra)
    train_opts(vc)  # validate now, before any training starts
    return vc


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _metric(m: dict[str, Any], key: str) -> Any:
    """Report value for ``key``: the single-SDF key, else the in-face (``surface/in_*``) equivalent."""
    if key in m:
        return m[key]
    head, _, name = key.partition("/")
    if head == "surface" and not name.startswith(("in_", "out_")):
        return m.get(f"surface/in_{name}")
    return None


def write_report(results: dict[str, dict[str, Any]], out_dir: str, opts: dict[str, Any]) -> tuple[str, str]:
    """``ablation.json`` (everything) + ``ablation.md`` (table of REPORT_COLUMNS, delta vs base)."""
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "ablation.json")
    with open(jp, "w") as fh:
        json.dump({"opts": opts, "results": results}, fh, indent=2)
    names = list(results)
    base = results.get("base", {}).get("metrics") or {}
    lines = [
        f"# Ablation ({opts['steps']} steps, seed {opts['seed']}, holdout {json.dumps(opts['holdout_origins'])}, "
        f"{opts['eval_max_crops']} crops)", "",
        "| variant | overrides | " + " | ".join(h for _, h, _ in REPORT_COLUMNS) + " | it/s |",
        "|---|---|" + "|".join("---:" for _ in REPORT_COLUMNS) + "|---:|",
    ]
    for n in names:
        r = results[n]
        m = r.get("metrics") or {}
        cells = []
        for key, _, lower in REPORT_COLUMNS:
            v = _metric(m, key)
            bv = _metric(base, key)
            cell = _fmt(v)
            if n != "base" and isinstance(v, float) and isinstance(bv, float) and not (math.isnan(v) or math.isnan(bv)):
                d = v - bv
                better = (d < 0) == lower if d != 0 else None
                cell += f" ({'+' if d >= 0 else ''}{d:.4f}{'' if better is None else (' +' if better else ' -')})"
            cells.append(cell)
        ov = json.dumps(r.get("overrides", {})) if n != "base" else ""
        lines.append(f"| {n} | `{ov}` | " + " | ".join(cells) + f" | {_fmt(r.get('it_s'))} |")
    lines += ["", "Deltas are vs `base`; a trailing `+` marks an improvement, `-` a regression.", ""]
    if any("error" in r for r in results.values()):
        lines.append("Failed variants: " + ", ".join(f"{n} ({r['error']})" for n, r in results.items() if "error" in r))
    mp = os.path.join(out_dir, "ablation.md")
    with open(mp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return mp, jp


def run_ablation(cfg: RunCfg, dry_run: bool = False, force: bool = False, variants: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    from tsm.train import run_train

    opts = ablate_opts(cfg)
    if variants is not None:
        opts["variants"] = dict(variants)
    out_root = opts["out_dir"] or os.path.join(cfg.out_dir, "ablation")
    plan: dict[str, dict[str, Any]] = ({"base": {}} if opts["include_base"] else {})
    plan.update(opts["variants"])
    if not plan:
        raise ValueError("nothing to run: no variants and include_base is false")
    cfgs = {n: variant_config(cfg, n, ov, opts, out_root) for n, ov in plan.items()}
    print(f"[ablate] {len(cfgs)} runs x {opts['steps']} steps (seed {opts['seed']}) -> {out_root}", flush=True)
    for n, vc in cfgs.items():
        print(f"[ablate]   {n}: {json.dumps(plan[n])} augment={json.dumps(vc.extra['train'].get('augment'))}", flush=True)
    if dry_run:
        return {"out_dir": out_root, "variants": {n: vc.extra["train"] for n, vc in cfgs.items()}}
    results: dict[str, dict[str, Any]] = {}
    for n, vc in cfgs.items():
        t = time.perf_counter()
        rec: dict[str, Any] = {"overrides": plan[n], "out_dir": vc.out_dir}
        try:
            # a variant whose training already finished (or was cut short) resumes from its
            # latest.pt instead of being refused; a completed run just re-runs the holdout eval
            latest = os.path.join(vc.out_dir, "train", "latest.pt")
            s = run_train(vc, resume=(not force) and os.path.exists(latest), force=force)
            rec.update({"metrics": s.get("holdout"), "losses": s.get("losses"), "step": s.get("step"),
                        "peak_vram_mb": s.get("peak_vram_mb"), "seconds": time.perf_counter() - t})
            log = os.path.join(vc.out_dir, "train", "log.jsonl")
            if os.path.exists(log):
                its = [json.loads(l).get("it_s", 0.0) for l in open(log)]
                rec["it_s"] = float(sum(its[1:]) / max(1, len(its) - 1)) if len(its) > 1 else (its[0] if its else None)
        except Exception as exc:  # keep going: one broken variant should not kill the sweep
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[ablate] {n} FAILED: {rec['error']}", flush=True)
        results[n] = rec
        write_report(results, out_root, opts)  # incremental: readable while the sweep runs
    mp, jp = write_report(results, out_root, opts)
    print(f"[ablate] wrote {mp} and {jp}", flush=True)
    print(open(mp).read(), flush=True)
    return {"out_dir": out_root, "results": results, "md": mp, "json": jp}


def rewrite_report(out_dir: str) -> str:
    """Regenerate ``ablation.md`` from an existing ``ablation.json`` (e.g. after a column change)."""
    with open(os.path.join(out_dir, "ablation.json")) as fh:
        d = json.load(fh)
    return write_report(d["results"], out_dir, d["opts"])[0]


if __name__ == "__main__":  # python -m tsm.ablate <ablation_dir> ...  → rewrite the markdown tables
    import sys
    for a in sys.argv[1:]:
        print(rewrite_report(a))


__all__ = ["ABLATE_DEFAULTS", "REPORT_COLUMNS", "rewrite_report", "ablate_opts", "apply_train_overrides", "variant_config", "write_report", "run_ablation"]
