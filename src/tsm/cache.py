"""``tsm cache``: copy the config region's CT into a local zarr so training never
depends on S3.

One array per level under ``<cache_root>/<volume-name>/L<level>.zarr`` (uint8,
128^3 chunks, no compression -- measured ~2.7x faster than zstd-1 for misaligned
128^3 crops, for 15% more disk).  The array holds the config region grown by
``volume.cache_margin`` voxels (augmentation reaches outside the crop), addressed
in absolute level coordinates through :class:`tsm.volume.LocalRegionArray`, so no
coordinate convention changes anywhere else.

Resumable: bricks already written are recorded in ``done.json`` next to the array,
and -- only when the brick is a whole multiple of the 128 storage chunk -- an existing
array whose backing chunk files are all present counts as done too.  A sub-chunk brick
shares its chunk file with its neighbours, so the presence of that file says nothing
about the neighbours (R03).

A level that finishes writes ``complete.json`` (levels, origin, size, brick, source URL,
timestamp).  :func:`tsm.volume.open_cached_reader` requires that manifest before it will
serve a cache: without it an interrupted build is indistinguishable from a finished one
and reads back as zeros (R02).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zarr

from tsm.config import RegionCfg, RunCfg
from tsm.limits import GB, MB
from tsm.volume import (
    CACHE_CHUNK,
    COMPLETE_JSON,
    VolumeReader,
    cache_array_path,
    cache_volume_name,
    write_complete,
)

CACHE_DEFAULTS = {
    "brick": 512,          # read/write box side, multiple of the 128 chunk
    "levels": 3,           # config level, +1, +2
    "compressor": "none",  # none | zstd<level>
    "probe_brick": 64,
    "disk_headroom": 1.05,
}


def _cache_opts(cfg: RunCfg) -> dict:
    raw = cfg.extra.get("cache", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.cache must be an object")
    unknown = sorted(set(raw) - set(CACHE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.cache keys: {unknown}")
    opts = dict(CACHE_DEFAULTS)
    opts.update(raw)
    if int(opts["brick"]) <= 0:
        raise ValueError("extra.cache.brick must be positive")
    if int(opts["brick"]) % CACHE_CHUNK:
        print(f"[tsm] WARNING extra.cache.brick {opts['brick']} is not a multiple of the "
              f"{CACHE_CHUNK} chunk; writes will touch partial chunks", flush=True)
    return opts


def _compressors(spec: str) -> Any:
    s = str(spec).lower()
    if s in ("none", "raw", ""):
        return None
    if s.startswith("zstd"):
        return zarr.codecs.ZstdCodec(level=int(s[4:] or 1))
    raise ValueError(f"unknown extra.cache.compressor {spec!r} (none | zstd1 ...)")


@dataclass
class LevelPlan:
    level: int
    factor: int
    voxel_um: float
    origin_zyx: tuple[int, int, int]
    size_zyx: tuple[int, int, int]
    path: str
    n_bricks: int = 0
    bytes: int = field(default=0)

    @property
    def region(self) -> RegionCfg:
        return RegionCfg(self.origin_zyx, self.size_zyx)


def padded_region(cfg: RunCfg, margin: int | None = None) -> RegionCfg:
    """Config region grown by ``volume.cache_margin`` (clipped at 0)."""
    m = int(cfg.volume.cache_margin if margin is None else margin)
    start = tuple(max(0, int(v) - m) for v in cfg.region.start_zyx)
    stop = tuple(int(v) + m for v in cfg.region.stop_zyx)
    return RegionCfg(start, tuple(b - a for a, b in zip(start, stop)))  # type: ignore[arg-type]


def plan_levels(cfg: RunCfg, opts: dict | None = None, cache_root: str | None = None) -> list[LevelPlan]:
    opts = opts or _cache_opts(cfg)
    root = cache_root or cfg.volume.cache_root
    if not root:
        raise ValueError("volume.cache_root is not set; add it to the config to use `tsm cache`")
    base = padded_region(cfg)
    brick = int(opts["brick"])
    plans: list[LevelPlan] = []
    for shift in range(int(opts["levels"])):
        f = 2 ** shift
        origin = tuple(int(v) // f for v in base.start_zyx)
        stop = tuple(-(-int(v) // f) for v in base.stop_zyx)  # ceil
        size = tuple(b - a for a, b in zip(origin, stop))
        level = cfg.volume.level + shift
        p = LevelPlan(level, f, cfg.volume.voxel_um * f, origin, size,  # type: ignore[arg-type]
                      cache_array_path(root, cfg.volume.url, level, origin))
        p.n_bricks = int(np.prod([-(-s // brick) for s in size]))
        p.bytes = int(np.prod(size))
        plans.append(p)
    return plans


def _brick_origins(size: tuple[int, int, int], brick: int) -> list[tuple[int, int, int]]:
    out = []
    for z in range(0, size[0], brick):
        for y in range(0, size[1], brick):
            for x in range(0, size[2], brick):
                out.append((z, y, x))
    return out


class _Done:
    """``done.json`` sidecar of finished brick keys (local, chunk-aligned origins)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.join(path, "done.json")
        try:
            with open(self.path) as fh:
                self.keys = set(json.load(fh))
        except (OSError, ValueError, TypeError):
            self.keys = set()

    @staticmethod
    def key(o: tuple[int, int, int]) -> str:
        return f"{o[0]},{o[1]},{o[2]}"

    def has(self, o: tuple[int, int, int]) -> bool:
        return self.key(o) in self.keys

    def add(self, o: tuple[int, int, int]) -> None:
        self.keys.add(self.key(o))
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(sorted(self.keys), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.keys = set()
        try:
            os.remove(self.path)
        except OSError:
            pass


def _chunks_present(path: str, arr: Any, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> bool:
    """True when every chunk file backing the local box [lo, hi) exists and is non-empty.

    Fallback for a cache built before/without ``done.json``.  Only sound when the brick
    grid is chunk aligned: with a sub-chunk brick several bricks share one chunk file, so
    writing the *first* of them creates the file and this test would then mark all the
    others (still zero-filled) as done -- see :func:`_chunk_fallback_ok`.
    """
    c = [int(v) for v in arr.chunks[-3:]]
    for iz in range(lo[0] // c[0], -(-hi[0] // c[0])):
        for iy in range(lo[1] // c[1], -(-hi[1] // c[1])):
            for ix in range(lo[2] // c[2], -(-hi[2] // c[2])):
                f = os.path.join(path, "c", str(iz), str(iy), str(ix))
                try:
                    if os.path.getsize(f) <= 0:
                        return False
                except OSError:
                    return False
    return True


def _chunk_fallback_ok(brick: int, arr: Any) -> bool:
    """The chunk-existence resume fallback is only valid for chunk-aligned bricks (R03)."""
    c = [int(v) for v in arr.chunks[-3:]]
    return all(x > 0 and brick % x == 0 for x in c)


def _open_source(cfg: RunCfg, plan: LevelPlan, probe: int) -> VolumeReader:
    from tsm.cli import _open_level

    return _open_level(cfg, plan.level, plan.voxel_um, plan.region, probe, use_cache=False)


def run_cache(cfg: RunCfg, dry_run: bool = False, force: bool = False,
              cache_root: str | None = None) -> dict[str, Any]:
    """Build/refresh the local CT cache for the config region.  Returns a summary dict."""
    opts = _cache_opts(cfg)
    root = os.path.abspath(os.path.expanduser(cache_root or cfg.volume.cache_root or ""))
    plans = plan_levels(cfg, opts, root)
    brick = int(opts["brick"])
    total = sum(p.bytes for p in plans)
    name = cache_volume_name(cfg.volume.url)
    print(f"[tsm] cache root {root} volume {name} margin {cfg.volume.cache_margin} "
          f"brick {brick} chunk {CACHE_CHUNK} compressor {opts['compressor']}", flush=True)
    for p in plans:
        print(f"[tsm]   L{p.level}: origin={p.origin_zyx} size={p.size_zyx} "
              f"{p.bytes / GB:.2f} GiB in {p.n_bricks} bricks -> {p.path}", flush=True)
    print(f"[tsm] cache total {total / GB:.2f} GiB ({total} bytes)", flush=True)
    if dry_run:
        print("[tsm] dry run: stopping before I/O")
        return {"bytes": total, "levels": [p.__dict__ for p in plans]}

    os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free
    need = int(total * float(opts["disk_headroom"]))
    print(f"[tsm] disk: {free / GB:.1f} GiB free, need {need / GB:.1f} GiB", flush=True)
    if free < need:
        raise RuntimeError(f"not enough free disk at {root}: {free / GB:.1f} GiB free, "
                           f"{need / GB:.1f} GiB needed")

    summary: dict[str, Any] = {"root": root, "volume": name, "levels": []}
    t_all = time.perf_counter()
    for plan in plans:
        os.makedirs(os.path.dirname(plan.path), exist_ok=True)
        attrs = {
            "source_url": cfg.volume.url,
            "alt_url": cfg.volume.alt_url,
            "level": int(plan.level),
            "origin_zyx": list(plan.origin_zyx),
            "size_zyx": list(plan.size_zyx),
            "voxel_um": float(plan.voxel_um),
            "cache_margin": int(cfg.volume.cache_margin),
            "region_start_zyx": list(cfg.region.start_zyx),
            "region_size_zyx": list(cfg.region.size_zyx),
        }
        if os.path.exists(os.path.join(plan.path, "zarr.json")):
            arr = zarr.open_array(store=plan.path, mode="r+")
            a0 = dict(arr.attrs)
            if tuple(int(v) for v in a0.get("origin_zyx", ())) != plan.origin_zyx:
                raise RuntimeError(f"{plan.path} holds a different origin than {plan.origin_zyx}")
            # never silently grow: zarr discards a wholly out-of-bounds assignment, so a
            # larger region reused to report "written" while the store stayed small (R04)
            if tuple(int(v) for v in arr.shape) != plan.size_zyx:
                raise RuntimeError(
                    f"{plan.path} has shape {tuple(int(v) for v in arr.shape)} but this run needs "
                    f"{plan.size_zyx} (region/margin changed); delete that directory or use a "
                    f"different volume.cache_root")
            if np.dtype(arr.dtype) != np.uint8:
                raise RuntimeError(f"{plan.path} has dtype {arr.dtype}, expected uint8; "
                                   "delete that directory or use a different volume.cache_root")
        else:
            arr = zarr.create_array(
                store=plan.path, shape=plan.size_zyx,
                chunks=(CACHE_CHUNK,) * 3, dtype=np.uint8, fill_value=0,
                compressors=_compressors(opts["compressor"]), attributes=attrs, overwrite=False,
            )
        done = _Done(plan.path)
        if force:
            done.clear()
        origins = _brick_origins(plan.size_zyx, brick)
        fallback = (not force) and _chunk_fallback_ok(brick, arr)
        if (not force) and not fallback and not os.path.exists(os.path.join(plan.path, "done.json")):
            print(f"[tsm] L{plan.level}: brick {brick} is not a multiple of the "
                  f"{tuple(int(v) for v in arr.chunks[-3:])} chunk, so chunk files cannot prove "
                  "completion; rebuilding every brick not listed in done.json", flush=True)
        todo = [o for o in origins
                if not (done.has(o) or (fallback and _chunks_present(
                    plan.path, arr,
                    o, tuple(min(a + brick, s) for a, s in zip(o, plan.size_zyx)))))]
        if fallback:  # record what the fallback accepted so the manifest can be published
            for o in origins:
                if o not in todo:
                    done.add(o)
        print(f"[tsm] L{plan.level}: {len(origins) - len(todo)}/{len(origins)} bricks already cached",
              flush=True)
        if not todo:
            write_complete(plan.path, plan.level, plan.origin_zyx, plan.size_zyx, brick,
                           cfg.volume.url, {"bricks": len(origins)})
            summary["levels"].append({"level": plan.level, "path": plan.path, "bricks": 0, "seconds": 0.0})
            continue
        try:  # a level under (re)construction is not a valid cache
            os.remove(os.path.join(plan.path, COMPLETE_JSON))
        except OSError:
            pass
        reader = _open_source(cfg, plan, int(opts["probe_brick"]))
        # ``full_shape_zyx`` lets LocalRegionArray clip exactly like the remote array
        arr.attrs.update({**attrs, "full_shape_zyx": [int(s) for s in reader.shape],
                          "source_used": reader.url})
        t0 = time.perf_counter()
        nbytes = 0
        for i, o in enumerate(todo, 1):
            hi = tuple(min(a + brick, s) for a, s in zip(o, plan.size_zyx))
            g0 = tuple(a + b for a, b in zip(plan.origin_zyx, o))
            g1 = tuple(a + b for a, b in zip(plan.origin_zyx, hi))
            block = reader.read(g0[0], g1[0], g0[1], g1[1], g0[2], g1[2])
            arr[o[0]:hi[0], o[1]:hi[1], o[2]:hi[2]] = block
            done.add(o)
            nbytes += int(block.size)
            del block
            dt = time.perf_counter() - t0
            if i % 10 == 0 or i == len(todo):
                rate = nbytes / MB / max(dt, 1e-9)
                eta = (len(todo) - i) * dt / i
                print(f"[tsm] L{plan.level}: {i}/{len(todo)} bricks {nbytes / GB:.2f} GiB "
                      f"{rate:.1f} MiB/s elapsed {dt / 60:.1f} min ETA {eta / 60:.1f} min", flush=True)
        dt = time.perf_counter() - t0
        missing = [o for o in origins if not done.has(o)]
        if missing:
            print(f"[tsm] L{plan.level}: {len(missing)} bricks still missing; no {COMPLETE_JSON} "
                  "written (this level is not usable as a cache yet)", flush=True)
        else:
            write_complete(plan.path, plan.level, plan.origin_zyx, plan.size_zyx, brick,
                           cfg.volume.url, {"bricks": len(origins)})
        print(f"[tsm] L{plan.level}: done in {dt / 60:.1f} min, {nbytes / GB:.2f} GiB, "
              f"{nbytes / MB / max(dt, 1e-9):.1f} MiB/s -> {plan.path}", flush=True)
        summary["levels"].append({"level": plan.level, "path": plan.path, "bricks": len(todo),
                                  "bytes": nbytes, "seconds": dt})
        del reader
    summary["seconds"] = time.perf_counter() - t_all
    with open(os.path.join(root, name, "cache.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[tsm] cache complete in {summary['seconds'] / 60:.1f} min", flush=True)
    return summary
