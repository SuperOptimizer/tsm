"""Fold human label corrections back into the fine label store, in place.

    uv run python dev/annot_import.py configs/paris4_faces_rv.json \
        --packets ~/tsm-output/annot/paris4_v1 [--weight 5.0] [--dry-run]

Step 2 of the correction loop (step 1 is ``dev/annot_export.py``).  Every packet directory
under ``--packets`` that carries a ``correction.u8`` (or, if the packet was exported with
``--tif``, a ``correction.tif``) is folded into ``<out_dir>/labels/fine.zarr``:

| correction class | effect |
| --- | --- |
| 0 | untouched |
| 1 | this voxel is on the **in** face (recto side) |
| 2 | this voxel is on the **out** face (verso side) |
| 3 | ignore here: ``faces_valid = 2`` (and the voxel is on neither face) |
| 4 | "wrong here": any label face voxel here is **erased** before the SDFs are recomputed |

Recomputation -- **incremental, and it has to be** (fixed 2026-09-07).  The corrected face
*sets* replace the store's own zero sets inside the crop and ``sdf_in`` / ``sdf_out`` are
recomputed from them with exactly the machinery :mod:`tsm.rvfaces` uses for the upstream bands
-- :func:`tsm.rvfaces._signed_face`, i.e. the signed EDT with the sign rule
``sign((p - q) . n_out(q))`` for ``q`` the nearest face voxel, and ``n_out`` the outward radial
direction of the umbilicus axis refined by the local band PCA
(:func:`tsm.rvfaces.local_band_normals`).  Nothing about the sign convention is re-derived
here.  The recomputation runs on the crop **plus a ``clip + 4`` halo** whose face voxels are
read from the store, so the SDF is continuous across the crop boundary; only the core is
written back.

But the recomputed field is written back **only inside the region of influence** -- the voxels
within ``clip + 1`` of a painted voxel, intersected with the crop -- and every other stored byte
is left exactly as it was.  That is not an optimisation; a full-crop recompute is **not** a
faithful round trip of what the merge builder stored, and up to 2026-09-07 every import silently
perturbed the whole crop (measured on Paris 4: 736 painted voxels moved ~10 M of the 16.7 M SDF
voxels of a 256^3 crop).  Three independent reasons, none of them fixable from the store:

1. **The PCA input band is not in the store.**  :func:`tsm.rvfaces.build_rv_face_labels` feeds
   :func:`~tsm.rvfaces.local_band_normals` the *unthinned* upstream band ``rv_class > 0`` (3-7
   voxels thick, both faces at once) and evaluates it at the *thinned* face voxels.  Re-deriving
   the band from the stored zero sets gives a different covariance in every window, hence a
   different normal (measured: 5 % of face voxels on a tilted synthetic sheet stack), hence a
   flipped *sign* over whole slabs -- ``|delta| = 228`` bytes, i.e. ``-clip`` against ``+clip``.
   Where a voxel is human- or CT-sourced there is no upstream band at all to feed it.
2. **The store is a merge of two builders.**  Wherever ``faces_source == 2`` the SDF came from
   :func:`tsm.labels.build_face_labels`, whose faces are the CT body boundary split by a local
   march and whose zero set is a different surface.  Its bytes cannot be reproduced by the
   rectoverso machinery at all, and the *union* of the two zero sets across the hand-over seam is
   a surface neither builder ever saw.
3. **The store was written brick-wise.**  Each brick saw its own ``clip + 4`` halo, so beyond
   ``clip`` the saturated ``+-clip`` bytes carry a brick-dependent sign that a packet-shaped box
   does not reproduce.

Outside the region of influence that does not matter: any face voxel the correction added or
removed is more than ``clip`` away, so the true distance there is saturated at ``+-clip`` in both
the old and the new field and only the (already brick-dependent) sign of a saturated voxel could
differ.  Inside it, the recompute uses the corrected face set over the whole haloed box -- the
untouched stored faces *and* the painted ones, i.e. the union of the old and the new zero set
minus what was erased -- so the field is continuous across the boundary of the region: at that
boundary every changed face voxel is >= ``clip + 1`` away and therefore invisible to both fields.
The PCA band is the stored ``rv_class > 0`` (the builder's own input, where the store carries it)
united with the corrected face voxels, so inside the region the reproduction is exact wherever
the geometry was rectoverso-sourced.  ``--dry-run`` reports ``sdf_max_delta_outside``, which is 0
by construction and is printed so that a regression shows up on real data.

Bookkeeping.  ``faces_source`` becomes :data:`tsm.rvfaces.SOURCE_HUMAN` (3) on every voxel the
annotator painted, and the store gains a uint8 ``faces_weight`` channel -- 1 everywhere,
``--weight`` on those voxels -- appended out of place with the same brick-wise
copy-and-rename machinery as ``dev/add_fiber_channels.py`` if it is not there yet.
:func:`tsm.train.faces_loss` reads it through the ``surface_weight`` target as a per-voxel
multiplier on every surface term, so a store without the channel trains bit-identically.

Safety.  The store is opened through :class:`tsm.volume.BrickWriter`, so its reopen validation
(shape / chunk / dtype / channel names and order / origin / voxel_um / scale) applies before a
byte is written; the writes themselves go to ``writer.array`` rather than through
``BrickWriter.write`` because a packet is not a brick and must not enter ``done.json``.  Each
packet records the sha256 of the label bricks it was cut from and the import refuses (unless
``--allow-stale``) if the store has moved since.  ``--dry-run`` reports everything and writes
nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.config import load_config  # noqa: E402
from tsm.labels import _label_opts, encode_sdf_u8, iter_cores, load_axis, radial_field  # noqa: E402
from tsm.rvfaces import SOURCE_HUMAN, local_band_normals, _signed_face  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402

FACES_WEIGHT = "faces_weight"
#: |byte - 128| of an SDF channel that still counts as "on the zero set" (same as the exporter)
ZERO_TOL = 3
C_NONE, C_IN, C_OUT, C_IGNORE, C_ERASE = 0, 1, 2, 3, 4
CLASS_NAMES = {C_IN: "in", C_OUT: "out", C_IGNORE: "ignore", C_ERASE: "erase"}


def _log(msg: str) -> None:
    print(f"[annot-import] {msg}", flush=True)


def zero_set(sdf_u8: np.ndarray, tol: int = ZERO_TOL) -> np.ndarray:
    return (sdf_u8 != 0) & (np.abs(sdf_u8.astype(np.int16) - 128) <= int(tol))


# --------------------------------------------------------------------------- #
# appending the faces_weight channel (dev/add_fiber_channels.py machinery)
# --------------------------------------------------------------------------- #
def append_faces_weight(fine_path: str, brick: Sequence[int], fill: int = 1,
                        force: bool = False) -> str:
    """Append a ``faces_weight`` channel (``fill`` everywhere) out of place, then swap it in.

    Same shape as ``dev/add_fiber_channels.py``: a new ``fine.zarr.new`` written by a
    :class:`BrickWriter` with the identity ``tsm labels`` would use, every existing channel
    copied brick-wise (so ``done.json`` is the progress file and a killed run resumes), the old
    attrs carried over, then ``fine.zarr -> fine.zarr.prev`` and ``fine.zarr.new -> fine.zarr``.
    """
    fine_path = os.path.abspath(os.path.expanduser(fine_path))
    new_path, prev_path = fine_path + ".new", fine_path + ".prev"
    old = zarr.open_array(store=fine_path, mode="r")
    attrs = dict(old.attrs)
    channels = [str(c) for c in attrs.get("channels", [])]
    if FACES_WEIGHT in channels:
        return fine_path
    if os.path.exists(prev_path) and not force:
        raise SystemExit(f"[annot-import] {prev_path} already exists (a previous append swapped); "
                         "move or delete it before running again")
    want = channels + [FACES_WEIGHT]
    shape = tuple(int(s) for s in old.shape[1:])
    origin = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))
    chunk = int(old.chunks[1])
    _log(f"appending {FACES_WEIGHT!r}: {len(channels)} -> {len(want)} channels, "
         f"copying {shape} through {new_path}")
    w = BrickWriter(new_path, want, shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=float(attrs.get("voxel_um", 2.4)),
                    scale=float(attrs.get("scale", 1.0)))
    cores = list(iter_cores(shape, [int(v) for v in brick]))
    todo = [c for c in cores if not w.has_brick(*(origin[a] + c[0][a] for a in range(3)))]
    t0 = time.perf_counter()
    for i, (lo, hi) in enumerate(todo):
        gz, gy, gx = (origin[a] + lo[a] for a in range(3))
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        for ci in range(len(channels)):
            w.write(ci, gz, gy, gx, np.ascontiguousarray(np.asarray(old[(ci,) + sl])))
        w.write(len(channels), gz, gy, gx,
                np.full(tuple(hi[a] - lo[a] for a in range(3)), np.uint8(fill)))
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            _log(f"  brick {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)")
    new_attrs = dict(attrs)
    new_attrs.update({"channels": want, "faces_weight_default": int(fill)})
    w.array.attrs.update(new_attrs)
    del old, w
    os.rename(fine_path, prev_path)
    os.rename(new_path, fine_path)
    _log(f"swapped: old store kept at {prev_path}; {fine_path} now has {len(want)} channels")
    return fine_path


# --------------------------------------------------------------------------- #
# one packet
# --------------------------------------------------------------------------- #
def read_correction(pdir: str, size: Sequence[int]) -> np.ndarray | None:
    """``correction.u8`` (preferred) or ``correction.tif``, validated against the packet size."""
    n = int(np.prod([int(v) for v in size]))
    raw = os.path.join(pdir, "correction.u8")
    if os.path.exists(raw):
        buf = np.fromfile(raw, dtype=np.uint8)
        if buf.size != n:
            raise SystemExit(f"[annot-import] {raw}: {buf.size} bytes, expected {n} "
                             f"for size {tuple(size)}")
        return buf.reshape(tuple(int(v) for v in size))
    tif = os.path.join(pdir, "correction.tif")
    if os.path.exists(tif):
        import tifffile

        a = np.asarray(tifffile.imread(tif), np.uint8)
        if a.shape != tuple(int(v) for v in size):
            raise SystemExit(f"[annot-import] {tif}: shape {a.shape}, expected {tuple(size)}")
        return a
    return None


def fold_packet(arr: zarr.Array, ch: dict[str, int], pdir: str, meta: dict[str, Any],
                corr: np.ndarray, clip: float, axis: np.ndarray, weight: float,
                faces_opts: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """Recompute the two faces of one packet's crop and write the core back into the store.

    Incremental by construction (see the module docstring): the signed EDT is evaluated over the
    haloed box from the corrected face set, but only the voxels within ``clip + 1`` of a painted
    voxel take the new value.  An empty correction therefore writes the stored bytes back
    unchanged, and a stroke can move at most its own footprint dilated by ``clip + 1``.
    """
    store_origin = tuple(int(v) for v in arr.attrs["origin_zyx"])
    shape = tuple(int(s) for s in arr.shape[1:])
    g = [int(v) for v in meta["origin_zyx"]]
    size = [int(v) for v in meta["size_zyx"]]
    lo = [g[a] - store_origin[a] for a in range(3)]
    halo = int(clip) + 4
    hlo = [max(0, lo[a] - halo) for a in range(3)]
    hhi = [min(shape[a], lo[a] + size[a] + halo) for a in range(3)]
    core = tuple(slice(lo[a] - hlo[a], lo[a] - hlo[a] + size[a]) for a in range(3))
    hshape = tuple(hhi[a] - hlo[a] for a in range(3))

    rd = lambda name: np.asarray(arr[ch[name], hlo[0]:hhi[0], hlo[1]:hhi[1], hlo[2]:hhi[2]])  # noqa: E731
    sdf_in_u8, sdf_out_u8 = rd("sdf_in"), rd("sdf_out")
    val = rd("faces_valid")
    src = rd("faces_source") if "faces_source" in ch else np.zeros(hshape, np.uint8)
    fw = rd(FACES_WEIGHT) if FACES_WEIGHT in ch else np.ones(hshape, np.uint8)

    fin = zero_set(sdf_in_u8)
    fout = zero_set(sdf_out_u8)
    before_in, before_out = int(fin[core].sum()), int(fout[core].sum())

    c = np.zeros(hshape, np.uint8)
    c[core] = corr
    touched = c > 0
    counts = {CLASS_NAMES[k]: int((corr == k).sum()) for k in CLASS_NAMES}
    # class 4 erases whatever face sat there; 3 makes it ignore (and no face); 1/2 set a face
    kill = (c == C_ERASE) | (c == C_IGNORE)
    fin &= ~kill
    fout &= ~kill
    fin = (fin | (c == C_IN)) & ~(c == C_OUT)
    fout = (fout | (c == C_OUT)) & ~(c == C_IN)

    band = fin | fout
    stats = {"packet": os.path.basename(pdir), "corrected": counts,
             "faces_in_before": before_in, "faces_out_before": before_out,
             "faces_in_after": int(fin[core].sum()), "faces_out_after": int(fout[core].sum()),
             "touched_voxels": int(touched[core].sum())}
    nodata = val == 0
    new_in, new_out = sdf_in_u8.copy(), sdf_out_u8.copy()

    # ---- region of influence -------------------------------------------------------------
    # Only voxels within `clip + 1` of a painted voxel may move: everywhere else the corrected
    # face set is more than `clip` away, so the true distance is saturated at +-clip both before
    # and after.  A full-crop recompute is NOT a faithful round trip of what the merge builder
    # stored (module docstring), so the stored bytes outside this region are authoritative and
    # are copied through untouched.
    infl = (ndi.distance_transform_edt(~touched) <= float(clip) + 1.0) if touched.any() \
        else np.zeros(hshape, bool)
    stats["influence_voxels"] = int(infl[core].sum())
    stats["influence_radius"] = float(clip) + 1.0

    if not band.any() or not infl.any():
        if not band.any():
            _log(f"{stats['packet']}: no face voxel left in the haloed box -- skipped")
            stats["skipped"] = "no faces"
            return stats
        # nothing was painted: the SDFs are the stored ones, byte for byte
        stats.update(sdf_in_changed=0, sdf_out_changed=0,
                     sdf_in_max_delta=0, sdf_out_max_delta=0, sdf_max_delta_outside=0)
        new_val, new_src = val.copy(), src.copy()
        new_fw = fw.copy()
        new_fw[new_fw == 0] = 1
    else:
        # outward direction: the store's convention -- radial from the umbilicus, refined by the
        # local PCA of the face band (tsm.rvfaces.local_band_normals), never from the CT.  The
        # builder's PCA input is the *unthinned* upstream band, so feed it the stored `rv_class`
        # where the store carries it, united with the corrected face voxels (which no builder
        # ever saw); that reproduces the builder exactly on rectoverso-sourced geometry.
        pca_band = band.copy()
        if "rv_class" in ch:
            pca_band |= rd("rv_class") > 0
        hlo_global = [hlo[a] + store_origin[a] for a in range(3)]
        radial = radial_field(axis, hlo_global, hshape, scale=1)
        pts = np.nonzero(band)
        n_pts, _ = local_band_normals(pca_band, radial, pts,
                                      int(faces_opts["rv_normal_window"]),
                                      int(faces_opts["rv_min_normal_count"]),
                                      float(faces_opts["rv_planarity_max"]))
        n_field = np.array(radial, dtype=np.float32, copy=True)
        n_field[:, pts[0], pts[1], pts[2]] = n_pts
        del radial, pts, n_pts, pca_band

        sdf_in, _ = _signed_face(fin, n_field, hshape, float(clip)) if fin.any() else (None, None)
        sdf_out, _ = _signed_face(fout, n_field, hshape, float(clip)) if fout.any() else (None, None)
        del n_field

        for sdf, new in ((sdf_in, new_in), (sdf_out, new_out)):
            if sdf is None:
                continue                       # that face vanished from the box: keep the store
            enc = encode_sdf_u8(sdf, float(clip))
            enc[nodata] = 0
            new[infl] = enc[infl]
            del enc
        del sdf_in, sdf_out

        new_val = val.copy()
        new_val[(c == C_IGNORE) & ~nodata] = 2
        new_val[((c == C_IN) | (c == C_OUT)) & ~nodata] = 1
        new_src = src.copy()
        new_src[touched & ~nodata] = SOURCE_HUMAN
        new_fw = fw.copy()
        new_fw[new_fw == 0] = 1                   # 0 = never written; the neutral weight is 1
        new_fw[touched & ~nodata] = np.uint8(min(255, max(1, int(round(float(weight))))))

        # the regression guard: nothing outside the region of influence may have moved
        out_core = ~infl[core]
        d_in = np.abs(new_in[core].astype(np.int16) - sdf_in_u8[core].astype(np.int16))
        d_out = np.abs(new_out[core].astype(np.int16) - sdf_out_u8[core].astype(np.int16))
        stats["sdf_in_max_delta"] = int(d_in.max()) if d_in.size else 0
        stats["sdf_out_max_delta"] = int(d_out.max()) if d_out.size else 0
        stats["sdf_max_delta_outside"] = int(max(d_in[out_core].max(initial=0),
                                                 d_out[out_core].max(initial=0)))
        del d_in, d_out, out_core

    stats["sdf_in_changed"] = int((new_in[core] != sdf_in_u8[core]).sum())
    stats["sdf_out_changed"] = int((new_out[core] != sdf_out_u8[core]).sum())
    stats["valid_changed"] = int((new_val[core] != val[core]).sum())
    stats["source_human"] = int((new_src[core] == SOURCE_HUMAN).sum())
    stats["weighted_voxels"] = int((new_fw[core] > 1).sum())
    if dry_run:
        return stats

    box = (slice(lo[0], lo[0] + size[0]), slice(lo[1], lo[1] + size[1]), slice(lo[2], lo[2] + size[2]))
    writes = [("sdf_in", new_in), ("sdf_out", new_out), ("faces_valid", new_val)]
    if "faces_source" in ch:
        writes.append(("faces_source", new_src))
    if FACES_WEIGHT in ch:
        writes.append((FACES_WEIGHT, new_fw))
    for name, a in writes:
        arr[(ch[name],) + box] = np.ascontiguousarray(a[core])
    return stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fold human label corrections into the fine store")
    ap.add_argument("config")
    ap.add_argument("--packets", required=True)
    ap.add_argument("--weight", type=float, default=5.0,
                    help="faces_weight on human-corrected voxels (1 = no extra weight)")
    ap.add_argument("--fine-store", default=None, help="override <out_dir>/labels/fine.zarr")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-stale", action="store_true",
                    help="import even if the store's labels moved since the packet was exported")
    ap.add_argument("--force", action="store_true", help="overwrite an existing fine.zarr.prev")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    opts = _label_opts(cfg)
    clip = float(opts["clip"])
    fine = os.path.abspath(os.path.expanduser(
        a.fine_store or os.path.join(cfg.out_dir, "labels", "fine.zarr")))
    if not os.path.exists(os.path.join(fine, "zarr.json")):
        raise SystemExit(f"[annot-import] no fine label store at {fine}")

    root = os.path.abspath(os.path.expanduser(a.packets))
    packets = []
    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)
        mpath = os.path.join(pdir, "meta.json")
        if not os.path.isdir(pdir) or not os.path.exists(mpath):
            continue
        with open(mpath) as fh:
            meta = json.load(fh)
        corr = read_correction(pdir, meta["size_zyx"])
        if corr is None:
            _log(f"{name}: no correction.u8 / correction.tif -- skipped")
            continue
        packets.append((pdir, meta, corr))
    if not packets:
        raise SystemExit(f"[annot-import] no packet under {root} carries a correction")
    _log(f"{len(packets)} packet(s) with corrections under {root}")

    if not a.dry_run:
        append_faces_weight(fine, opts["brick"], fill=1, force=a.force)

    # BrickWriter reopen validation (shape / chunk / dtype / channels / origin / voxel_um /
    # scale) before anything is written; the writes go through writer.array, since a packet
    # crop is not a brick and must never enter done.json
    a0 = zarr.open_array(store=fine, mode="r")
    attrs = dict(a0.attrs)
    channels = [str(c) for c in attrs.get("channels", [])]
    writer = BrickWriter(fine, channels, tuple(int(s) for s in a0.shape[1:]),
                         chunk=int(a0.chunks[1]),
                         origin_zyx=tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0))),
                         voxel_um=float(attrs.get("voxel_um", 2.4)),
                         scale=float(attrs.get("scale", 1.0)))
    arr = writer.array
    ch = {name: i for i, name in enumerate(channels)}
    for need in ("sdf_in", "sdf_out", "faces_valid"):
        if need not in ch:
            raise SystemExit(f"[annot-import] {fine} has no {need!r} channel; channels={channels}")
    axis = load_axis(opts["axis_path"])

    all_stats = []
    for pdir, meta, corr in packets:
        if meta.get("store") and os.path.abspath(meta["store"]) != fine:
            _log(f"WARNING {os.path.basename(pdir)}: exported from {meta['store']}, importing into {fine}")
        if meta.get("label_sha256_16") and not a.allow_stale:
            g = [int(v) for v in meta["origin_zyx"]]
            s = [int(v) for v in meta["size_zyx"]]
            o = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))
            sl = tuple(slice(g[i] - o[i], g[i] - o[i] + s[i]) for i in range(3))
            h = hashlib.sha256()
            for name in ("sdf_in", "sdf_out", "faces_valid", "faces_source", "rv_class"):
                # the exporter hashed zeros for a channel the store does not carry
                blk = np.asarray(arr[(ch[name],) + sl]) if name in ch else np.zeros(tuple(s), np.uint8)
                h.update(np.ascontiguousarray(blk).tobytes())
            if h.hexdigest()[:16] != meta["label_sha256_16"]:
                raise SystemExit(
                    f"[annot-import] {os.path.basename(pdir)}: the store's labels changed since "
                    f"the packet was exported ({h.hexdigest()[:16]} != {meta['label_sha256_16']}); "
                    "re-export, or pass --allow-stale to import anyway")
        st = fold_packet(arr, ch, pdir, meta, corr, clip, axis, a.weight, opts["faces"], a.dry_run)
        all_stats.append(st)
        _log(f"{st['packet']}: " + " ".join(f"{k}={v}" for k, v in st.items() if k != "packet"))

    tot: dict[str, int] = {}
    for st in all_stats:
        for k, v in st["corrected"].items():
            tot[k] = tot.get(k, 0) + int(v)
    worst = max((int(s.get("sdf_max_delta_outside", 0)) for s in all_stats), default=0)
    _log(f"{'DRY RUN: ' if a.dry_run else ''}{len(all_stats)} packet(s); corrected voxels per "
         f"class {tot}; face voxels moved "
         f"{sum(s.get('sdf_in_changed', 0) + s.get('sdf_out_changed', 0) for s in all_stats)} "
         f"(inside the clip+1 region of influence: "
         f"{sum(s.get('influence_voxels', 0) for s in all_stats)} voxels); "
         f"max |delta sdf| OUTSIDE it = {worst} (must be 0)")
    if worst:
        # structurally impossible (the stored bytes outside the region are copied through), so
        # this is a regression guard, not a user error
        _log(f"ERROR: {worst} bytes of SDF moved outside the region of influence -- the import "
             "is not incremental any more; do not train on this store")
        return 1
    if not a.dry_run:
        _log(f"faces_source = {SOURCE_HUMAN} and faces_weight = {a.weight:g} on the corrected voxels; "
             f"training picks the weight up automatically (data.CropDataset -> surface_weight)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
