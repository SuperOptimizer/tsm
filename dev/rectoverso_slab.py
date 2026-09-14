"""Materialise the upstream Scroll 1 recto/verso (and hz/vt fibre) label volumes
onto our image-frame slab grid.

    uv run python dev/rectoverso_slab.py <config> --out <dir> [--z0 0 --zsize 256]

Writes ``<out>/rectoverso.zarr`` -- uint8 ``(2, Z, Y, X)``, ch0 = rectoverso
class (0 bg / 1 recto / 2 verso / 3 intersection), ch1 = hz/vt fibre class
(0 bg / 1 hz / 2 vt / 3 exclude), chunks ``(1, 128, 128, 128)`` -- plus
``<out>/coverage.json`` and ``<out>/montage_z100.png``.

The upstream labels live in the older ~7.9 um Scroll 1 frame; ``transform.json``
next to our 2.4 um volume maps between them (see :mod:`tsm.xframe`).  The label
AABB of the whole slab is only a few hundred MB, so it is fetched once up front
and every brick resamples out of RAM.  Brick-wise with resume via ``done.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm import xframe  # noqa: E402
from tsm.config import load_config  # noqa: E402

RECTOVERSO_URL = "https://dl.ash2txt.org/other/dev/meshes/s1a-rectoverso-06032025-ome.zarr"
HZVT_URL = "https://dl.ash2txt.org/other/dev/meshes/s1a-fibers-hzvt-05032025-ome.zarr"
# Scroll 4 (PHerc 1667): --rectoverso-url/--hzvt-url below
S4_RECTOVERSO_URL = "https://dl.ash2txt.org/other/dev/meshes/s4-rectoverso-13032025-ome.zarr"
S4_HZVT_URL = "https://dl.ash2txt.org/other/dev/meshes/s4-hzvt-13032025-ome.zarr"
CHANNELS = ["rectoverso", "hzvt"]
CLASS_NAMES = {
    "rectoverso": ["bg", "recto", "verso", "intersection"],
    "hzvt": ["bg", "hz", "vt", "exclude"],
}
CHUNK = 128

# module-level so forked workers share the (read-only) label slabs copy-on-write
_SLABS: list = []
_SLAB_LO: list = []
_MATRIX = None



def _transform_base(cfg) -> str:
    """The volume URL that carries ``transform.json``: the S3 open-data volume when either
    ``url`` or ``alt_url`` points there (the volcomp mirror does not ship it), else ``alt_url``
    or ``url`` as before."""
    cands = [u for u in (cfg.volume.url, cfg.volume.alt_url) if u]
    for u in cands:
        if u.startswith("s3://"):
            return u
    return cfg.volume.alt_url or cfg.volume.url

def _brick_origins(size, brick):
    return [(z, y, x)
            for z in range(0, size[0], brick[0])
            for y in range(0, size[1], brick[1])
            for x in range(0, size[2], brick[2])]


class _Done:
    def __init__(self, path):
        self.path = Path(path) / "done.json"
        self.items = set()
        if self.path.exists():
            self.items = {tuple(v) for v in json.loads(self.path.read_text())}

    def has(self, o):
        return tuple(o) in self.items

    def add(self, o):
        self.items.add(tuple(o))

    def save(self):
        self.path.write_text(json.dumps(sorted(self.items)))


def _fetch_slab(url, level, matrix, start, size, margin=2):
    """Fetch the label-frame AABB covering the image box, or None if unavailable."""
    arr = xframe.open_label_array(url, level)
    m = xframe.level_scale_matrix(level) @ matrix
    lo, hi = xframe.label_aabb_for_image_box(
        m, start, size, label_shape_zyx=tuple(int(v) for v in arr.shape[-3:]), margin=margin)
    span = tuple(b - a for a, b in zip(lo, hi))
    if min(span) <= 0:
        return None, lo, arr.shape
    data = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    return data, lo, arr.shape


def _brick(args):
    """Resample one brick; returns (origin, brick, per-z class counts)."""
    origin, brick, region_start, out_path = args
    out = zarr.open_array(store=out_path, mode="r+")
    dz, dy, dx = brick
    counts = np.zeros((len(CHANNELS), 4, dz), dtype=np.int64)
    start = tuple(int(a + b) for a, b in zip(region_start, origin))
    for c, slab in enumerate(_SLABS):
        if slab is None:
            block = np.zeros(brick, np.uint8)
        else:
            block = xframe._resample(
                _Mem(slab, _SLAB_LO[c]), _MATRIX, start, brick,
                0, (CHUNK, CHUNK, CHUNK), 1 << 62, 1)
        out[c, origin[0]:origin[0] + dz, origin[1]:origin[1] + dy,
            origin[2]:origin[2] + dx] = block
        for k in range(4):
            counts[c, k] = (block == k).sum(axis=(1, 2))
    return origin, brick, counts


class _Mem:
    """Array-like view of an in-RAM label slab, indexed in absolute label coords.

    ``shape`` reports ``lo + data.shape`` so :func:`tsm.xframe._resample` clips
    against the real upper bound; a request reaching below ``lo`` or above the
    fetched slab is zero-padded rather than wrapping.
    """

    def __init__(self, data, lo=(0, 0, 0)):
        self.data, self.lo = data, tuple(int(v) for v in lo)

    @property
    def shape(self):
        return tuple(a + b for a, b in zip(self.lo, self.data.shape))

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        want = [(s.start, s.stop) for s in key]
        have = [(max(a, self.lo[i]), min(b, self.lo[i] + self.data.shape[i]))
                for i, (a, b) in enumerate(want)]
        if any(b <= a for a, b in have):
            return np.zeros([b - a for a, b in want], self.data.dtype)
        sub = self.data[tuple(slice(a - self.lo[i], b - self.lo[i])
                              for i, (a, b) in enumerate(have))]
        if have == want:
            return sub
        out = np.zeros([b - a for a, b in want], self.data.dtype)
        out[tuple(slice(h[0] - w[0], h[1] - w[0]) for w, h in zip(want, have))] = sub
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z0", type=int, default=0)
    ap.add_argument("--zsize", type=int, default=None)
    ap.add_argument("--brick", type=int, nargs=3, default=(128, 512, 512))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--transform-url", default=None)
    ap.add_argument("--rectoverso-url", default=RECTOVERSO_URL)
    ap.add_argument("--hzvt-url", default=HZVT_URL)
    ap.add_argument("--no-montage", action="store_true")
    ap.add_argument("--montage-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = str(out_dir / "rectoverso.zarr")

    region_start = tuple(int(v) for v in cfg.region.start_zyx)
    region_size = tuple(int(v) for v in cfg.region.size_zyx)
    # the sub-slab we actually materialise (the store always spans the full region)
    z0 = int(args.z0)
    zsize = int(args.zsize) if args.zsize is not None else region_size[0] - z0

    tjson = args.transform_url or (
        _transform_base(cfg).rstrip("/") + "/transform.json")
    t0 = time.perf_counter()
    doc = xframe.read_transform(tjson)
    matrix = doc.image_to_label_zyx
    print(f"[rv] transform {tjson}\n[rv]   fixed_volume={doc.fixed_volume} "
          f"checksum={doc.checksum} ({time.perf_counter() - t0:.1f}s)", flush=True)

    if not args.montage_only:
        _build(out_dir, store, region_start, region_size, z0, zsize, matrix, doc,
               tuple(args.brick), args.workers, tjson,
               (args.rectoverso_url, args.hzvt_url), float(cfg.volume.voxel_um))
        _coverage(out_dir, store, region_start, region_size, z0, zsize)
    if not args.no_montage:
        try:
            _montage(out_dir, store, cfg, region_start)
        except Exception as e:  # montage inputs are optional extras
            print(f"[rv] montage failed: {type(e).__name__}: {e}", flush=True)


def _build(out_dir, store, region_start, region_size, z0, zsize, matrix, doc,
           brick, workers, tjson, urls=(RECTOVERSO_URL, HZVT_URL), voxel_um=2.4):
    sub_start = (region_start[0] + z0, region_start[1], region_start[2])
    sub_size = (zsize, region_size[1], region_size[2])

    slabs, los, srcs = [], [], []
    for name, url in zip(CHANNELS, urls):
        t = time.perf_counter()
        try:
            data, lo, shape = _fetch_slab(url, 0, matrix, sub_start, sub_size)
        except Exception as e:
            print(f"[rv] {name}: unavailable ({type(e).__name__}: {e}) -> zeros", flush=True)
            data, lo, shape = None, (0, 0, 0), None
        if data is not None:
            print(f"[rv] {name}: label AABB lo={lo} shape={data.shape} "
                  f"({data.nbytes / 1e6:.0f} MB, {time.perf_counter() - t:.1f}s), "
                  f"nonzero={float((data > 0).mean()):.4f}", flush=True)
        slabs.append(data)
        los.append(lo)
        srcs.append(url if data is not None else None)

    attrs = {
        "origin_zyx": list(region_start),
        "region": {"start_zyx": list(region_start), "size_zyx": list(region_size)},
        "channels": CHANNELS,
        "class_names": CLASS_NAMES,
        "sources": {"rectoverso": srcs[0], "hzvt": srcs[1], "transform_json": tjson},
        "transform_checksum": doc.checksum,
        "transform_fixed_volume": doc.fixed_volume,
        "materialised_z": [z0, z0 + zsize],
        "voxel_um": float(voxel_um),
    }
    shape = (len(CHANNELS),) + region_size
    if os.path.exists(os.path.join(store, "zarr.json")):
        arr = zarr.open_array(store=store, mode="r+")
        if tuple(int(v) for v in arr.shape) != shape:
            raise RuntimeError(f"{store} has shape {arr.shape}, need {shape}; delete it")
        a = dict(arr.attrs)
        if a.get("transform_checksum") not in (None, doc.checksum):
            raise RuntimeError(f"{store} was built with a different transform; delete it")
        mz = a.get("materialised_z", [z0, z0 + zsize])
        attrs["materialised_z"] = [min(mz[0], z0), max(mz[1], z0 + zsize)]
        arr.attrs.update(attrs)
    else:
        arr = zarr.create_array(
            store=store, shape=shape, chunks=(1, CHUNK, CHUNK, CHUNK), dtype=np.uint8,
            fill_value=0, compressors=zarr.codecs.ZstdCodec(level=1),
            attributes=attrs, overwrite=False)

    origins = [o for o in _brick_origins(region_size, brick) if z0 <= o[0] < z0 + zsize]
    done = _Done(out_dir)
    todo = [o for o in origins if not done.has(o)]
    print(f"[rv] {len(origins) - len(todo)}/{len(origins)} bricks already done", flush=True)

    counts_path = out_dir / "counts.npy"
    counts = (np.load(counts_path) if counts_path.exists()
              else np.zeros((len(CHANNELS), 4, region_size[0]), np.int64))

    jobs = []
    for o in todo:
        b = tuple(min(bb, s - oo) for bb, s, oo in zip(brick, region_size, o))
        b = (min(b[0], z0 + zsize - o[0]), b[1], b[2])
        jobs.append((o, b, region_start, store))

    # publish the label slabs as module globals *before* forking so the workers
    # inherit them copy-on-write instead of pickling ~700 MB per worker
    global _SLABS, _SLAB_LO, _MATRIX
    _SLABS, _SLAB_LO, _MATRIX = slabs, los, matrix

    t0 = time.perf_counter()
    n = 0
    if jobs:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for o, b, c in ex.map(_brick, jobs, chunksize=1):
                counts[:, :, o[0]:o[0] + b[0]] += c
                done.add(o)
                n += 1
                if n % 8 == 0 or n == len(jobs):
                    dt = time.perf_counter() - t0
                    done.save()
                    np.save(counts_path, counts)
                    print(f"[rv] {n}/{len(jobs)} bricks {dt:.0f}s "
                          f"(eta {dt / n * (len(jobs) - n):.0f}s)", flush=True)
    done.save()
    np.save(counts_path, counts)
    print(f"[rv] wrote {store} in {time.perf_counter() - t0:.0f}s", flush=True)


def _coverage(out_dir, store, region_start, region_size, z0, zsize):
    arr = zarr.open_array(store=store, mode="r")
    counts = np.load(out_dir / "counts.npy")
    per_slice = int(region_size[1]) * int(region_size[2])
    zs = slice(z0, z0 + zsize)
    tot = float(per_slice) * zsize

    rep = {
        "region": {"start_zyx": list(region_start), "size_zyx": list(region_size)},
        "materialised_z": [z0, z0 + zsize],
        "channels": {},
    }
    for c, name in enumerate(CHANNELS):
        cls = CLASS_NAMES[name]
        overall = {cls[k]: float(counts[c, k, zs].sum()) / tot for k in range(4)}
        overall["labelled"] = float(sum(overall[cls[k]] for k in (1, 2, 3)))
        rep["channels"][name] = {
            "overall_fraction": overall,
            "per_z": {
                "z_abs": [region_start[0] + z0, region_start[0] + z0 + zsize],
                "labelled": [float(counts[c, 1:, z].sum()) / per_slice
                             for z in range(z0, z0 + zsize)],
            },
        }

    # per 1024^2 tile at slab-local z=100
    zl = 100
    if z0 <= zl < z0 + zsize:
        tiles = {}
        plane = np.asarray(arr[:, zl])
        step = 1024
        for c, name in enumerate(CHANNELS):
            grid = []
            for ty in range(0, region_size[1], step):
                row = []
                for tx in range(0, region_size[2], step):
                    t = plane[c, ty:ty + step, tx:tx + step]
                    row.append(round(float((t > 0).mean()), 6))
                grid.append(row)
            tiles[name] = grid
        rep["tiles_z100"] = {"tile": step, "z_local": zl,
                             "z_abs": region_start[0] + zl, "labelled_fraction": tiles}

    (out_dir / "coverage.json").write_text(json.dumps(rep, indent=1))
    for name in CHANNELS:
        print(f"[rv] {name}: {rep['channels'][name]['overall_fraction']}", flush=True)
    print(f"[rv] wrote {out_dir / 'coverage.json'}", flush=True)


def _montage(out_dir, store, cfg, region_start, zl=100, y0=2560, x0=4096, side=1024):
    from PIL import Image, ImageDraw

    rv = zarr.open_array(store=store, mode="r")
    rvp = np.asarray(rv[0, zl, y0:y0 + side, x0:x0 + side])
    hzp = np.asarray(rv[1, zl, y0:y0 + side, x0:x0 + side])

    from tsm.volume import cache_array_path
    ct_path = cache_array_path(cfg.volume.cache_root or "/home/forrest/tsm-cache",
                               cfg.volume.url, cfg.volume.level)
    ca = zarr.open_array(store=ct_path, mode="r")
    co = [int(v) for v in ca.attrs["origin_zyx"]]
    dz, dy, dx = (region_start[i] - co[i] for i in range(3))
    ct = np.asarray(ca[dz + zl, dy + y0:dy + y0 + side, dx + x0:dx + x0 + side])
    ctp = np.clip(ct.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
    base = np.stack([ctp] * 3, -1)

    fine = Path(cfg.extra.get("train", {}).get(
        "fine_store", "/home/forrest/tsm-output/slab_faces_tta/labels/fine.zarr"))
    faces = base.copy()
    if (fine / "zarr.json").exists():
        fa = zarr.open_array(store=str(fine), mode="r")
        ch = {n: i for i, n in enumerate(fa.attrs["channels"])}
        box = (slice(zl, zl + 1), slice(y0, y0 + side), slice(x0, x0 + side))
        sin = np.asarray(fa[ch["sdf_in"], box[0], box[1], box[2]])[0]
        sout = np.asarray(fa[ch["sdf_out"], box[0], box[1], box[2]])[0]
        val = np.asarray(fa[ch["faces_valid"], box[0], box[1], box[2]])[0]
        faces[val == 2] = (faces[val == 2] * 0.4).astype(np.uint8)
        faces[(sin != 0) & (np.abs(sin.astype(np.int16) - 128) <= 3)] = (255, 40, 40)
        faces[(sout != 0) & (np.abs(sout.astype(np.int16) - 128) <= 3)] = (60, 120, 255)

    def overlay(base_rgb, lbl, colors, alpha=0.55):
        o = base_rgb.astype(np.float32).copy()
        for k, col in colors.items():
            m = lbl == k
            if m.any():
                o[m] = (1 - alpha) * o[m] + alpha * np.asarray(col, np.float32)
        return o.astype(np.uint8)

    rv_img = overlay(base, rvp, {1: (255, 40, 40), 2: (60, 120, 255), 3: (255, 240, 40)})
    hz_img = overlay(base, hzp, {1: (40, 220, 80), 2: (230, 60, 230), 3: (140, 140, 140)})

    panels = [("CT", base),
              ("faces labels (sdf_in red / sdf_out blue / ignore dim)", faces),
              ("upstream rectoverso (recto red / verso blue / both yellow)", rv_img),
              ("upstream hzvt (hz green / vt magenta)", hz_img)]
    ims = []
    for t, a in panels:
        im = Image.fromarray(a)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 7 * len(t) + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), t, fill=(255, 255, 0))
        ims.append(np.asarray(im))
    p = out_dir / "montage_z100.png"
    Image.fromarray(np.concatenate(ims, 1)).save(p)
    print(f"[rv] wrote {p} (z_abs={region_start[0] + zl} y={y0} x={x0} side={side})",
          flush=True)
    print(f"[rv] crop label fractions: rectoverso={float((rvp > 0).mean()):.4f} "
          f"hzvt={float((hzp > 0).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
