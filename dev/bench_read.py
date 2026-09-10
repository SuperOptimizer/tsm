"""Time VolumeReader.read of one box from S3 / HTTPS at several chunk concurrencies.

    uv run python dev/bench_read.py --box 512,640,640 --conc 10,32,64
"""
from __future__ import annotations

import argparse
import time

from tsm import volume
from tsm.config import load_config
from tsm.volume import VolumeReader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paris4_roi.json")
    ap.add_argument("--box", default="512,640,640")
    ap.add_argument("--conc", default="10,32,64")
    ap.add_argument("--source", default="alt", choices=["primary", "alt"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    url = cfg.volume.url if a.source == "primary" else cfg.volume.alt_url
    shape = tuple(int(v) for v in a.box.split(","))
    z0, y0, x0 = cfg.region.start_zyx
    print(f"| source | concurrency | box | seconds | MiB/s |\n|---|---|---|---|---|")
    for i, c in enumerate(int(v) for v in a.conc.split(",")):
        volume.set_read_concurrency(c)
        r = VolumeReader(url, cfg.volume.level, cfg.volume.voxel_um, cfg.budget)
        # shift the box per run so the S3 edge cache cannot serve a repeat
        zz = z0 + 128 * i
        t = time.perf_counter()
        box = r.read(zz, zz + shape[0], y0, y0 + shape[1], x0, x0 + shape[2])
        dt = time.perf_counter() - t
        print(f"| {a.source} | {c} | {shape} | {dt:.1f} | {box.nbytes / (1 << 20) / dt:.1f} |", flush=True)
        del box, r
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
