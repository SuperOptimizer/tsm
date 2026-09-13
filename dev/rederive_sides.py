#!/usr/bin/env python3
"""Re-derive ``surface_side1`` in an existing sides ``pred.zarr``, in place.

The derived side channel of a ``surface_mode="sides"`` run used to be the pointwise shell
``d_face <= 1``, which selects nothing once the regressed unsigned distance is compressed (it
saturates a couple of voxels above zero and never reaches the faces).  :func:`tsm.infer.extract_sides`
rebuilds it from the body mask and the in-body face-distance valleys instead; this script applies
that to a store that is already on disk, so an evaluated run can be re-scored without re-running
inference.

    uv run python dev/rederive_sides.py configs/paris4_eval.json
    uv run python dev/rederive_sides.py /path/to/out/student/pred.zarr --valley-max 2.5

The argument is either a run config (the store is ``<out_dir>/student/pred.zarr`` and the clip
comes from ``pred.summary.json``) or a pred.zarr path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.data import CLIP  # noqa: E402
from tsm.infer import SIDES_EXTRACT_DEFAULTS, rederive_sides  # noqa: E402


def resolve(target: str) -> tuple[str, float]:
    """(pred.zarr path, clip) from a config path or a store path."""
    t = os.path.expanduser(str(target))
    if os.path.isdir(t) and (os.path.exists(os.path.join(t, "zarr.json"))
                             or os.path.exists(os.path.join(t, ".zarray"))):
        pred = t
    else:
        from tsm.config import load_config

        pred = os.path.join(os.path.expanduser(load_config(t).out_dir), "student", "pred.zarr")
    summary = os.path.join(os.path.dirname(pred), "pred.summary.json")
    clip = float(json.load(open(summary)).get("clip", CLIP)) if os.path.exists(summary) else float(CLIP)
    return pred, clip


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run config JSON, or a sides pred.zarr path")
    ap.add_argument("--brick", type=int, default=128)
    ap.add_argument("--body-thr", type=float, default=SIDES_EXTRACT_DEFAULTS["body_thr"])
    ap.add_argument("--valley-tol", type=float, default=SIDES_EXTRACT_DEFAULTS["valley_tol"])
    ap.add_argument("--valley-max", type=float, default=SIDES_EXTRACT_DEFAULTS["valley_max"])
    ap.add_argument("--min-component", type=int, default=100,
                    help="component pruning of the STATS pass only (the channel itself is not pruned)")
    a = ap.parse_args(argv)
    pred, clip = resolve(a.target)
    print(f"[rederive] {pred} (clip={clip})", flush=True)
    out = rederive_sides(pred, clip, brick=int(a.brick), min_component=int(a.min_component),
                         body_thr=float(a.body_thr), valley_tol=float(a.valley_tol),
                         valley_max=float(a.valley_max))
    print(json.dumps({k: v for k, v in out.items() if k != "surface_side1"}, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()
