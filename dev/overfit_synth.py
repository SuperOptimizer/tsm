"""GPU smoke: overfit one fixed batch (one fine + one coarse crop) on synthetic stores.

    uv run python dev/overfit_synth.py [--steps 200] [--patch 128] [--root DIR]

Builds a 192^3 fine store (+ 48^3 coarse store, same footprint at 9.6 um) and a CT
volume (tests/synth.py), then runs ``tsm.train.run_train`` with ``overfit_one`` and
prints the per-head loss curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

from synth import make_synthetic  # noqa: E402
from tsm.config import parse_config  # noqa: E402
from tsm.train import run_train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--root", default=os.path.expanduser("~/tsm-output/dev_overfit"))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    root = a.root
    if a.fresh and os.path.exists(root):
        import shutil

        shutil.rmtree(root)
    t = time.perf_counter()
    if not os.path.exists(os.path.join(root, "labels", "fine.zarr", "zarr.json")):
        s = make_synthetic(root, (a.size,) * 3, (0, 0, 0), chunk=64)
        print(f"[dev] synthetic stores written in {time.perf_counter() - t:.1f}s")
    else:
        s = {"fine": os.path.join(root, "labels", "fine.zarr"), "coarse": os.path.join(root, "labels", "coarse.zarr"),
             "ct": os.path.join(root, "ct.zarr")}
    train_dir = os.path.join(root, "train")
    if os.path.exists(train_dir):
        import shutil

        shutil.rmtree(train_dir)
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [a.size] * 3},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30, "vram_frac": 0.8},
        "out_dir": root,
        "extra": {"train": {"steps": a.steps, "patch": a.patch, "batch": a.batch, "accum": 1, "overfit_one": True,
                            "num_workers": 0, "ckpt_every": a.steps, "log_every": 10, "warmup": 20,
                            "fine_store": s["fine"], "coarse_store": s["coarse"]}},
    })
    summary = run_train(cfg)
    log = [json.loads(l) for l in open(os.path.join(train_dir, "log.jsonl"))]
    print("\n[dev] step   total   surface   ink   winding   it/s   vram_peak")
    for r in log:
        if r["step"] in (1, 10, 25, 50, 100, 150, 200) or r["step"] == a.steps:
            print(f"[dev] {r['step']:5d}  {r['total']:.4f}  {r['surface']:.4f}  {r['ink']:.4f}  {r['winding']:.4f}  {r['it_s']:.2f}  {r['vram_peak_mb']}")
    print("[dev] final per-term:", json.dumps({k: round(v, 4) for k, v in summary["losses"].items() if "/" in k}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
