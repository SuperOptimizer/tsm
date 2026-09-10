"""Insert the fibre channels into an existing fine label store.

    uv run python dev/add_fiber_channels.py configs/paris4_faces_rv.json [--teachers-dir D]

``tsm labels`` writes the fine store as ``fine_channels(faces, fiber, rv)`` -- the faces
block, then the fibre block, then the rectoverso block (see :func:`tsm.labels.fine_channels`).
A store built before ``fiber.zarr`` existed therefore has the *rv* block sitting where the
fibre block belongs, so the fibre channels cannot simply be appended: they have to be
inserted and the rv block shifted right.

This script does that out of place and brick-wise:

1. read the existing ``<out_dir>/labels/fine.zarr`` and check its channel list is exactly
   ``fine_channels(faces, fiber=False, rv)`` for this config;
2. create ``fine.zarr.new`` with ``fine_channels(faces, fiber=True, rv)``;
3. for every core brick of ``labels.brick``, copy each existing channel into its new index
   and fill ``fiber_vt / fiber_hz / fiber_valid`` from ``<teachers_dir>/fiber.zarr`` through
   :func:`tsm.labels.fiber_block` -- the same code path (and uint8 encoding, and
   "``fiber_valid`` = the CT data mask" rule, taken from the store's own ``ink_valid``) that
   ``tsm labels`` uses;
4. copy the old array attrs (summary fields and all) over the new ones;
5. swap ``fine.zarr`` -> ``fine.zarr.prev`` and ``fine.zarr.new`` -> ``fine.zarr``.

The new store is written by :class:`tsm.volume.BrickWriter` with the same identity
``tsm labels`` would use, so its ``done.json`` fingerprint validates for later resumes and
readers.  That ``done.json`` is also the progress file: killing the script mid-copy and
re-running it resumes at the first unfinished brick.

The script refuses to run while a build is in progress on the store (a ``tsm labels``
process holding the same config, or a ``done.json`` touched less than ``--stale-minutes``
ago); ``--force`` skips those checks and the "old store is complete" check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
import zarr

from tsm.config import load_config
from tsm.labels import _label_opts, fine_channels, fiber_block, iter_cores
from tsm.limits import estimate_and_assert, peak_rss_mb
from tsm.rvfaces import RV_CHANNELS
from tsm.volume import BrickWriter

MARKER = "add_fiber_channels.json"


def _log(msg: str) -> None:
    print(f"[add-fiber] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# safety checks
# --------------------------------------------------------------------------- #
def labels_processes(config_path: str) -> list[str]:
    """``pgrep -f`` lines of running ``tsm labels`` processes using this config."""
    base = os.path.basename(config_path)
    try:
        out = subprocess.run(["pgrep", "-af", base], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        _log(f"WARNING could not run pgrep ({type(exc).__name__}: {exc}); skipping the process check")
        return []
    me = str(os.getpid())
    hits = []
    for line in out.stdout.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid in (me, str(os.getppid())) or "add_fiber_channels" in cmd:
            continue
        if "labels" in cmd:
            hits.append(line)
    return hits


def check_not_building(fine_path: str, config_path: str, stale_minutes: float) -> None:
    hits = labels_processes(config_path)
    if hits:
        raise SystemExit("[add-fiber] a label build looks active for this config:\n  "
                         + "\n  ".join(hits) + "\n  refusing to touch the store (--force overrides)")
    done = os.path.join(fine_path, "done.json")
    if os.path.exists(done):
        age = (time.time() - os.path.getmtime(done)) / 60.0
        if age < stale_minutes:
            raise SystemExit(f"[add-fiber] {done} was modified {age:.1f} min ago (< {stale_minutes:g}); "
                             "a build is probably still running -- refusing (--force overrides)")


def check_old_complete(fine_path: str, channels: list[str], cores: list[tuple], origin: tuple) -> None:
    """Every core brick of every channel must already be written in the old store."""
    try:
        with open(os.path.join(fine_path, "done.json")) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[add-fiber] cannot read {fine_path}/done.json ({exc}); "
                         "the old store cannot be proven complete (--force overrides)")
    bricks = d.get("bricks", d) if isinstance(d, dict) else {}
    missing = [lo for lo, _ in cores
               if len(bricks.get(",".join(str(origin[a] + lo[a]) for a in range(3)), [])) != len(channels)]
    if missing:
        raise SystemExit(f"[add-fiber] the old store is incomplete: {len(missing)}/{len(cores)} bricks are "
                         f"unfinished (e.g. {missing[0]}) -- finish `tsm labels` first (--force overrides)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="insert the fibre channels into an existing fine label store")
    ap.add_argument("config")
    ap.add_argument("--teachers-dir", default=None, help="override extra.labels.teachers_dir")
    ap.add_argument("--fine-store", default=None, help="override <out_dir>/labels/fine.zarr")
    ap.add_argument("--stale-minutes", type=float, default=5.0)
    ap.add_argument("--force", action="store_true",
                    help="skip the in-progress and completeness checks")
    ap.add_argument("--dry-run", action="store_true", help="plan and budget only, no I/O")
    args = ap.parse_args(argv[1:])

    cfg = load_config(args.config)
    opts = _label_opts(cfg)
    tdir = args.teachers_dir or opts["teachers_dir"]
    fine_path = os.path.abspath(os.path.expanduser(
        args.fine_store or os.path.join(cfg.out_dir, "labels", "fine.zarr")))
    new_path = fine_path + ".new"
    prev_path = fine_path + ".prev"
    fiber_path = os.path.join(tdir, "fiber.zarr")
    shape = tuple(int(s) for s in cfg.region.size_zyx)
    origin = tuple(int(s) for s in cfg.region.start_zyx)
    brick = [int(v) for v in opts["brick"]]
    um = float(cfg.volume.voxel_um)

    if not os.path.exists(os.path.join(fine_path, "zarr.json")):
        raise SystemExit(f"[add-fiber] no fine store at {fine_path}")
    if not os.path.exists(os.path.join(fiber_path, "zarr.json")):
        raise SystemExit(f"[add-fiber] no fibre teacher store at {fiber_path} (--teachers-dir?)")

    old = zarr.open_array(store=fine_path, mode="r")
    old_attrs = dict(old.attrs)
    old_channels = [str(c) for c in old_attrs.get("channels", [])]
    do_faces = bool(opts["faces"]["enabled"])
    do_rv = do_faces and str(opts["faces"]["source"]) != "ct"
    want_old = fine_channels(do_faces, False, do_rv)
    want_new = fine_channels(do_faces, True, do_rv)
    if old_channels != want_old:
        raise SystemExit(f"[add-fiber] {fine_path} channels {old_channels} != {want_old} expected for this "
                         f"config (faces={do_faces}, rv={do_rv}); nothing to do if they already match "
                         f"{want_new}")
    if tuple(int(s) for s in old.shape[1:]) != shape:
        raise SystemExit(f"[add-fiber] store shape {tuple(old.shape[1:])} != config region size {shape}")
    if tuple(int(v) for v in old_attrs.get("origin_zyx", origin)) != origin:
        raise SystemExit(f"[add-fiber] store origin {old_attrs.get('origin_zyx')} != config region start {origin}")
    chunk = int(old.chunks[1])
    ink_valid_idx = want_old.index("ink_valid")
    # old index -> new index (every old channel keeps its name; the rv block shifts right by 3)
    remap = [(want_new.index(name), i) for i, name in enumerate(old_channels)]
    fiber_new_idx = [want_new.index(name) for name in ("fiber_vt", "fiber_hz", "fiber_valid")]
    _log(f"{fine_path}: {len(old_channels)} channels -> {len(want_new)} ({want_new})")
    _log(f"rv block {RV_CHANNELS if do_rv else '[]'} shifts by {len(want_new) - len(want_old)}; "
         f"fibre goes to indices {fiber_new_idx}")

    cores = list(iter_cores(shape, brick))
    bshape = tuple(min(b, s) for b, s in zip(brick, shape))
    estimate_and_assert([("copy_brick_u8", bshape, np.uint8),
                         ("fiber_brick_u8 x2", (2,) + bshape, np.uint8),
                         ("ink_valid_brick_u8", bshape, np.uint8)], cfg.budget)
    if not args.force:
        check_not_building(fine_path, args.config, float(args.stale_minutes))
        check_old_complete(fine_path, old_channels, cores, origin)
    if args.dry_run:
        _log(f"dry run: {len(cores)} bricks of {bshape} would be rewritten to {new_path}")
        return 0
    if os.path.exists(prev_path):
        raise SystemExit(f"[add-fiber] {prev_path} already exists (a previous run swapped); "
                         "move or delete it before running again")

    fiber = zarr.open_array(store=fiber_path, mode="r")
    fo = tuple(int(v) for v in dict(fiber.attrs).get("origin_zyx", origin))
    fs = tuple(int(s) for s in fiber.shape[1:])
    if fo != origin or fs != shape:
        from tsm.rvfaces import SubView

        off = tuple(origin[a] - fo[a] for a in range(3))
        if any(v < 0 for v in off) or any(off[a] + shape[a] > fs[a] for a in range(3)):
            raise SystemExit(f"[add-fiber] {fiber_path} region {fo}+{fs} does not contain {origin}+{shape}")
        _log(f"fiber.zarr covers {fo}+{fs} > the config region: reading the sub-box at offset {off}")
        fiber = SubView(fiber, off, shape)

    # BrickWriter identity == the one `tsm labels` would build (brick=None), so the
    # done.json fingerprint of the swapped-in store validates for later resumes/readers.
    w = BrickWriter(new_path, want_new, shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=um, scale=float(old_attrs.get("scale", 1.0)))
    with open(os.path.join(new_path, MARKER), "w") as fh:
        json.dump({"source": fine_path, "teachers_dir": tdir, "channels": want_new,
                   "old_channels": old_channels, "brick": brick,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=1)

    todo = [c for c in cores if not w.has_brick(*(origin[a] + c[0][a] for a in range(3)))]
    _log(f"{len(cores)} bricks, {len(todo)} to do (brick {brick}, chunk {chunk})")
    t0 = time.perf_counter()
    full = (slice(None), slice(None), slice(None))
    for i, (lo, hi) in enumerate(todo):
        gz, gy, gx = (origin[a] + lo[a] for a in range(3))
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        ink_valid = None
        for new_c, old_c in remap:
            blk = np.ascontiguousarray(np.asarray(old[(old_c,) + sl]))
            if old_c == ink_valid_idx:
                ink_valid = blk
            w.write(new_c, gz, gy, gx, blk)
            del blk
        assert ink_valid is not None
        # the copy-through path of tsm labels, with no halo (a pure per-voxel copy)
        for c, blk in zip(fiber_new_idx, fiber_block(fiber, lo, hi, full, ink_valid)):
            w.write(c, gz, gy, gx, blk)
        del ink_valid
        _log(f"brick {i + 1}/{len(todo)} {lo}->{hi} done ({time.perf_counter() - t0:.0f}s, "
             f"RSS {peak_rss_mb():.0f} MiB)")

    # attrs: keep everything the old store carried (summary fields included), with the
    # writer's own identity keys overriding.
    attrs: dict[str, Any] = dict(old_attrs)
    attrs.update({"channels": want_new, "voxel_um": um, "origin_zyx": list(origin),
                  "scale": float(old_attrs.get("scale", 1.0)),
                  "fiber_appended_from": fiber_path})
    w.array.attrs.update(attrs)
    del old, w

    os.rename(fine_path, prev_path)
    os.rename(new_path, fine_path)
    try:
        os.remove(os.path.join(fine_path, MARKER))
    except OSError:
        pass
    _log(f"swapped: old store kept at {prev_path}; {fine_path} now has {len(want_new)} channels")
    chk = zarr.open_array(store=fine_path, mode="r")
    assert [str(c) for c in chk.attrs["channels"]] == want_new
    _log(f"done in {time.perf_counter() - t0:.0f}s; verify with "
         f"`uv run python dev/labels_sanity.py {args.config}`, then delete {prev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
