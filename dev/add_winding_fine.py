"""Append the native-2.4 um winding block to an existing fine label store.

    uv run python dev/add_winding_fine.py configs/paris4_faces_rvfw.json [--workers 8]

``tsm labels`` writes the fine store as
``fine_channels(faces, fiber, rv, winding_fine)`` (:func:`tsm.labels.fine_channels`), and the
``wf_*`` block is **last**, so unlike the fibre block it can simply be appended -- no existing
channel moves.  This script does that out of place and brick-wise, with the same halo,
brick grid and worker pool ``tsm labels`` uses:

1. read ``<out_dir>/labels/fine.zarr`` and check its channels are exactly
   ``fine_channels(faces, fiber, rv)`` for this config (i.e. everything but ``wf_*``);
2. create ``fine.zarr.new`` with the ``wf_*`` block appended;
3. per core brick of ``labels.brick``: copy every existing channel through unchanged and
   compute the block from the *haloed* ``sdf_in`` / ``sdf_out`` / ``faces_valid`` of the old
   store plus the coarse prior (:func:`tsm.winding_fine.winding_fine_box`, exactly the call
   ``tsm.labels._fine_brick`` makes);
4. copy the old array attrs over the new ones;
5. swap ``fine.zarr`` -> ``fine.zarr.prev`` and ``fine.zarr.new`` -> ``fine.zarr``.

The new store is a :class:`tsm.volume.BrickWriter` store with the identity ``tsm labels``
would build, and its ``done.json`` is the progress file: killing the script and re-running
resumes at the first unfinished brick.

**vs a full rebuild.**  The result is byte-identical to ``tsm labels`` with
``winding_fine`` on *except* within a few voxels of a brick boundary: the inline build reads
its halo from the brick's own in-memory faces, this one reads it back from the finished
store, and the face builders are brick-local, so the two see slightly different halo data.
Measured on the synthetic build (halo 8): the fringe is 3 voxels deep, 0.4 % of the region,
and the supervised fraction moves by 0.001
(``tests/test_winding_fine.py::test_add_winding_fine_matches_a_full_rebuild``).

**Concurrency.**  Readers are safe: nothing is modified in place, and the swap is two
``rename`` calls, so a training run keeps reading the old inode until it reopens.  A
*writer* is not: the script refuses to run while a ``tsm labels`` process holds this config
or ``done.json`` was touched less than ``--stale-minutes`` ago (``--force`` overrides).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import zarr

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from add_fiber_channels import check_not_building, check_old_complete  # noqa: E402
from tsm.config import load_config  # noqa: E402
from tsm.labels import (  # noqa: E402
    _label_opts,
    fine_channels,
    iter_cores,
    radial_field,
    read_box_padded,
)
from tsm.limits import estimate_and_assert, peak_rss_mb  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402
from tsm.winding_fine import WF_CHANNELS, upsample_coarse_prior, winding_fine_box  # noqa: E402

MARKER = "add_winding_fine.json"
_W: dict[str, Any] = {}


def _log(msg: str) -> None:
    print(f"[add-wf] {msg}", flush=True)


def _worker_init(spec: dict[str, Any]) -> None:
    """Per-process handles for :func:`_brick` (pool initializer)."""
    global _W
    from tsm.labels import load_axis

    _W = {
        "spec": spec,
        "old": zarr.open_array(store=spec["fine_path"], mode="r"),
        "coarse": zarr.open_array(store=spec["coarse_path"], mode="r"),
        "axis": load_axis(spec["axis_path"]),
    }


def _brick(job: tuple[int, tuple, tuple]) -> tuple:
    """``(i, lo, hi, the 8 wf arrays)`` for one core brick, computed on the haloed box."""
    i, lo, hi = job
    spec, old, coarse = _W["spec"], _W["old"], _W["coarse"]
    origin, co, halo, k = spec["origin"], spec["coarse_origin"], spec["halo"], spec["k"]
    idx = spec["face_idx"]
    hlo = [lo[a] - halo for a in range(3)]
    hhi = [hi[a] + halo for a in range(3)]
    glo = [origin[a] + hlo[a] for a in range(3)]
    ghi = [origin[a] + hhi[a] for a in range(3)]
    sdf_in, sdf_out, fv = (read_box_padded(old, hlo, hhi, channels=c)[0] for c in idx)
    prior = upsample_coarse_prior(coarse, [glo[a] - co[a] * k for a in range(3)],
                                  [ghi[a] - co[a] * k for a in range(3)], scale=k, normal=True)
    rad = radial_field(_W["axis"], glo, sdf_in.shape, scale=1)
    wopts = spec["wf_opts"]
    out = winding_fine_box(
        sdf_in, sdf_out, fv, prior, rad, spec["clip"],
        snap_max=(None if wopts["snap_max"] is None else float(wopts["snap_max"])),
        min_period=float(wopts["min_period"]), max_period=float(wopts["max_period"]),
        sat_margin=float(wopts["sat_margin"]), min_grad=float(wopts["min_grad"]),
        max_grad=float(wopts["max_grad"]), phase_offset=float(wopts["phase_offset"]),
        require_faces_valid=bool(wopts["require_faces_valid"]))
    core = tuple(slice(halo, halo + hi[a] - lo[a]) for a in range(3))
    return i, lo, hi, [np.ascontiguousarray(a[core]) for a in out]


def _jobs(spec: dict[str, Any], todo: list, workers: int, mp_start: str = "fork"):
    """Yield brick results in brick order, in-process or from a pool (as tsm.labels does)."""
    jobs = [(i, lo, hi) for i, (lo, hi) in enumerate(todo)]
    if workers <= 1:
        _worker_init(spec)
        for job in jobs:
            yield _brick(job)
        return
    import multiprocessing as mp

    ctx = mp.get_context(mp_start)
    pool = ctx.Pool(workers, initializer=_worker_init, initargs=(spec,))
    try:
        pending: list = []
        nxt = 0
        while nxt < len(jobs) or pending:
            while nxt < len(jobs) and len(pending) < workers + 1:
                pending.append(pool.apply_async(_brick, (jobs[nxt],)))
                nxt += 1
            yield pending.pop(0).get()
    finally:
        pool.terminate()
        pool.join()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="append the wf_* winding block to an existing fine store")
    ap.add_argument("config")
    ap.add_argument("--fine-store", default=None, help="override <out_dir>/labels/fine.zarr")
    ap.add_argument("--coarse-store", default=None, help="override <out_dir>/labels/coarse.zarr")
    ap.add_argument("--workers", type=int, default=None, help="default extra.labels.workers")
    ap.add_argument("--stale-minutes", type=float, default=5.0)
    ap.add_argument("--force", action="store_true", help="skip the in-progress / completeness checks")
    ap.add_argument("--dry-run", action="store_true", help="plan and budget only, no I/O")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    cfg = load_config(args.config)
    opts = _label_opts(cfg)
    wopts = opts["winding_fine"]
    if not wopts["enabled"]:
        raise SystemExit("[add-wf] extra.labels.winding_fine is off in this config")
    fine_path = os.path.abspath(os.path.expanduser(
        args.fine_store or (cfg.extra.get("train", {}) or {}).get("fine_store")
        or os.path.join(cfg.out_dir, "labels", "fine.zarr")))
    coarse_path = os.path.abspath(os.path.expanduser(
        args.coarse_store or (cfg.extra.get("train", {}) or {}).get("coarse_store")
        or os.path.join(cfg.out_dir, "labels", "coarse.zarr")))
    new_path, prev_path = fine_path + ".new", fine_path + ".prev"
    for p in (fine_path, coarse_path):
        if not os.path.exists(os.path.join(p, "zarr.json")):
            raise SystemExit(f"[add-wf] no store at {p}")

    old = zarr.open_array(store=fine_path, mode="r")
    old_attrs = dict(old.attrs)
    old_channels = [str(c) for c in old_attrs.get("channels", [])]
    do_faces = bool(opts["faces"]["enabled"])
    do_rv = do_faces and str(opts["faces"]["source"]) != "ct"
    do_fiber = all(c in old_channels for c in ("fiber_vt", "fiber_hz", "fiber_valid"))
    want_old = fine_channels(do_faces, do_fiber, do_rv)
    want_new = fine_channels(do_faces, do_fiber, do_rv, True)
    if old_channels == want_new:
        raise SystemExit(f"[add-wf] {fine_path} already carries {WF_CHANNELS}; nothing to do")
    if old_channels != want_old:
        raise SystemExit(f"[add-wf] {fine_path} channels {old_channels} != {want_old} expected for this "
                         f"config (faces={do_faces}, fiber={do_fiber}, rv={do_rv})")

    origin = tuple(int(v) for v in old_attrs.get("origin_zyx", cfg.region.start_zyx))
    shape = tuple(int(s) for s in old.shape[1:])
    coarse = zarr.open_array(store=coarse_path, mode="r")
    co = tuple(int(v) for v in dict(coarse.attrs).get("origin_zyx", tuple(v // 4 for v in origin)))
    k = int(round(float(dict(coarse.attrs).get("scale", 4))))
    brick = [int(v) for v in opts["brick"]]
    halo = int(opts["halo"])
    chunk = int(old.chunks[1])
    workers = max(1, int(args.workers if args.workers is not None else opts["workers"]))
    _log(f"{fine_path}: {len(old_channels)} channels -> {len(want_new)} (appending {WF_CHANNELS})")
    _log(f"region {origin}+{shape}; coarse {coarse_path} origin {co} scale {k}; "
         f"brick {brick} halo {halo} workers {workers}")
    _log(f"winding_fine opts {wopts}")

    cores = list(iter_cores(shape, brick))
    hs = tuple(min(b, s) + 2 * halo for b, s in zip(brick, shape))
    estimate_and_assert([("faces_u8 x3", (3,) + hs, np.uint8),
                         ("wf_edt_idx_i32", (3,) + hs, np.int32),
                         ("wf_f32 x8", (8,) + hs, np.float32),
                         ("prior_f32 x5", (5,) + hs, np.float32),
                         ("wf_out_u8", (8,) + hs, np.uint8),
                         ("copy_brick_u8", tuple(min(b, s) for b, s in zip(brick, shape)), np.uint8)],
                        cfg.budget)
    if not args.force:
        check_not_building(fine_path, args.config, float(args.stale_minutes))
        check_old_complete(fine_path, old_channels, cores, origin)
    if args.dry_run:
        _log(f"dry run: {len(cores)} bricks would be rewritten to {new_path}")
        return 0
    if os.path.exists(prev_path):
        raise SystemExit(f"[add-wf] {prev_path} already exists (a previous run swapped); "
                         "move or delete it before running again")

    w = BrickWriter(new_path, want_new, shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=float(old_attrs.get("voxel_um", cfg.volume.voxel_um)),
                    scale=float(old_attrs.get("scale", 1.0)))
    with open(os.path.join(new_path, MARKER), "w") as fh:
        json.dump({"source": fine_path, "coarse": coarse_path, "channels": want_new,
                   "old_channels": old_channels, "winding_fine": wopts,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=1)

    todo = [c for c in cores if not w.has_brick(*(origin[a] + c[0][a] for a in range(3)))]
    _log(f"{len(cores)} bricks, {len(todo)} to do")
    spec = {"fine_path": fine_path, "coarse_path": coarse_path, "origin": origin,
            "coarse_origin": co, "halo": halo, "k": k, "clip": float(opts["clip"]),
            "axis_path": opts["axis_path"], "wf_opts": dict(wopts),
            "face_idx": [old_channels.index(c) for c in ("sdf_in", "sdf_out", "faces_valid")]}
    n_wf = len(WF_CHANNELS)
    t0 = time.perf_counter()
    n_valid = n_vox = 0
    for i, lo, hi, wf in _jobs(spec, todo, workers, str(opts.get("mp_start") or "fork")):
        gz, gy, gx = (origin[a] + lo[a] for a in range(3))
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        for c in range(len(old_channels)):  # every old channel keeps its index
            w.write(c, gz, gy, gx, np.ascontiguousarray(np.asarray(old[(c,) + sl])))
        for j, blk in enumerate(wf):
            w.write(len(old_channels) + j, gz, gy, gx, blk)
        n_valid += int((wf[n_wf - 1] == 1).sum())
        n_vox += int(np.prod([hi[a] - lo[a] for a in range(3)]))
        _log(f"brick {i + 1}/{len(todo)} {lo}->{hi} wf_valid {(wf[n_wf - 1] == 1).mean():.3f} "
             f"({time.perf_counter() - t0:.0f}s, RSS {peak_rss_mb():.0f} MiB)")
        del wf

    attrs: dict[str, Any] = dict(old_attrs)
    attrs.update({"channels": want_new, "origin_zyx": list(origin),
                  "winding_fine_appended_from": coarse_path, "winding_fine_opts": dict(wopts)})
    w.array.attrs.update(attrs)
    del old, w

    os.rename(fine_path, prev_path)
    os.rename(new_path, fine_path)
    try:
        os.remove(os.path.join(fine_path, MARKER))
    except OSError:
        pass
    chk = zarr.open_array(store=fine_path, mode="r")
    assert [str(c) for c in chk.attrs["channels"]] == want_new
    _log(f"swapped: old store kept at {prev_path}; {fine_path} now has {len(want_new)} channels, "
         f"wf_valid {n_valid / max(n_vox, 1):.4f} of the region")
    _log(f"done in {time.perf_counter() - t0:.0f}s; delete {prev_path} when happy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
