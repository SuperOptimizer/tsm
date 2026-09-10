"""Export label-correction packets for the render3d annotation mode.

    uv run python dev/annot_export.py configs/paris4_faces_rv.json \
        --out ~/tsm-output/annot/paris4_v1 --n 12 --size 256 --select disagreement

Human-in-the-loop label correction, step 1 of 2 (step 2 is ``dev/annot_import.py``).  The
script scores every ``--size`` cube of the config's region on a stride-``--stride`` grid,
keeps the ``--n`` best **non-overlapping** ones and writes each as a self-contained *packet*
directory holding one raw ``.u8`` volume per layer plus a ``meta.json``.  A single
``packet.json`` manifest at the root lists every packet and the palette semantics; the
renderer (https://github.com/SuperOptimizer/render3d, ``.u8`` = flat uint8,
``index = (z*ny + y)*nx + x``, dims from the command line / sidecar) opens a packet, paints a
``correction`` volume and writes it back as ``correction.u8`` next to the layers.

Selection criteria (``--select``)
---------------------------------
``disagreement`` (default, no student needed)
    The brief's ideal is ``mean |delta sdf_in| + |delta sdf_out|`` between the two face
    builders -- but the *store only holds the merged result*, one SDF pair per voxel, and the
    per-source pair is thrown away at build time (only aggregate histograms survive into
    ``labels.summary.json``).  Recomputing it here would mean re-running both builders over
    the region, i.e. the whole `tsm labels` cost.  So the implemented criterion is the
    documented **proxy** the brief specifies:

        score = seam_frac + THICK_W * min(cv_thickness, 1)

    * ``seam_frac`` = the fraction of the crop's voxels that are CT-sourced
      (``faces_source == 2``) **and** within ``--near`` (8) voxels of a rectoverso-sourced
      voxel (``faces_source == 1``).  That is exactly the surface along which the two builders
      hand over, which is where they disagree by construction: the rectoverso builder gave up
      (or the CT builder was overruled) within a few voxels of there.
    * ``cv_thickness`` = the coefficient of variation (std/mean) of the ``thickness`` channel
      over the crop's voxels with ``thickness > 0``, clipped to 1.  A locally wild thickness is
      the signature of a welded sheet pair or a delaminated layer -- the two failure modes the
      builders disagree about.  ``THICK_W`` = 0.25 (a full-scale thickness blow-up is worth as
      much as a 25 % seam).
    A crop with less than ``--min-valid`` of ``faces_valid == 1`` scores ``-1`` (nothing to
    correct there).

``uncertain`` (needs ``--pred``)
    The fraction of *label* face voxels (the store's ``sdf_in`` / ``sdf_out`` zero sets, on
    ``faces_valid == 1``) whose distance to the **student's** zero crossing of the same face
    exceeds ``--pred-tol`` (3) voxels.  Both stores must cover the crop.

``random``
    Uniform, seeded by ``--seed``; the control group.

Scoring is a single streaming pass: per ``--stride`` block the pass accumulates the counts and
thickness moments above (the neighbourhood EDT is computed per tile with a ``--near`` halo, so
the block statistics are independent of the tiling), and a crop's score is the sum over the
blocks it covers.  Nothing bigger than one tile is ever resident.

Packet layout (``crop_<i>_z<z>_y<y>_x<x>/``, all volumes ``uint8``, (z, y, x) C order)
-------------------------------------------------------------------------------------
| file | meaning |
| --- | --- |
| ``ct.u8`` | the CT itself, from the config's reader (cache or S3) |
| ``faces_in.u8`` | 1 on the label's **in**-face zero set (``sdf_in`` byte 128), else 0 |
| ``faces_out.u8`` | the same for the out face |
| ``ignore.u8`` | 1 where ``faces_valid == 2`` |
| ``source.u8`` | ``faces_source`` (0 none / 1 rectoverso / 2 ct / 3 human) |
| ``rv_class.u8`` | upstream ``rv_class`` (0 bg / 1 recto / 2 verso / 3 contact) |
| ``pred_in.u8`` / ``pred_out.u8`` | with ``--pred``: the student's zero crossings |
| ``meta.json`` | origin, size, voxel_um, store path, channel semantics, per-layer sha256 |
| ``correction.u8`` | **written by the annotator**: 0 untouched / 1 in / 2 out / 3 ignore / 4 erase |

``--tif`` additionally writes a ``.tif`` copy of every layer (needs ``tifffile``, in the
``viz`` extra).  A WebKnossos variant is **not** written: WebKnossos wants a wkw/OME-Zarr
dataset with a ``datasource-properties.json`` and a working ``webknossos`` package, which is
neither trivial nor a dependency here -- the ``.u8`` volumes convert with
``webknossos convert`` upstream if anyone wants them there.
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
from tsm.labels import _label_opts  # noqa: E402
from tsm.rvfaces import SOURCE_CT, SOURCE_RECTOVERSO  # noqa: E402

#: weight of the thickness coefficient-of-variation term in the disagreement score
THICK_W = 0.25
#: |byte - 128| of an SDF channel that still counts as "on the zero set" (0.5 voxel at clip 20)
ZERO_TOL = 3
SELECTORS = ("disagreement", "uncertain", "random")
#: correction classes the annotator paints (also written into packet.json)
CORRECTION_CLASSES = {
    "0": "untouched",
    "1": "in-face (recto side, sdf_in zero set)",
    "2": "out-face (verso side, sdf_out zero set)",
    "3": "ignore here (faces_valid = 2)",
    "4": "wrong here, erase the face before recomputation",
}


def _log(msg: str) -> None:
    print(f"[annot-export] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# store helpers
# --------------------------------------------------------------------------- #
class Store:
    """Minimal channel-indexed reader for a BrickWriter store (no tsm.data dependency)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        self.arr = zarr.open_array(store=self.path, mode="r")
        attrs = dict(self.arr.attrs)
        self.channels = [str(c) for c in attrs.get("channels", [])]
        self.origin = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))
        self.shape = tuple(int(s) for s in self.arr.shape[1:])
        self.voxel_um = float(attrs.get("voxel_um", 2.4))

    def has(self, name: str) -> bool:
        return name in self.channels

    def read(self, name: str, lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
        """Local-coordinate box of one channel, zero-padded outside the store."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        out = np.zeros(tuple(h - l for l, h in zip(lo, hi)), np.uint8)
        a = [max(0, l) for l in lo]
        b = [min(s, h) for h, s in zip(hi, self.shape)]
        if any(bb <= aa for aa, bb in zip(a, b)):
            return out
        blk = np.asarray(self.arr[self.channels.index(name), a[0]:b[0], a[1]:b[1], a[2]:b[2]])
        d = [aa - l for aa, l in zip(a, lo)]
        out[d[0]:d[0] + blk.shape[0], d[1]:d[1] + blk.shape[1], d[2]:d[2] + blk.shape[2]] = blk
        return out

    def read_global(self, name: str, glo: Sequence[int], ghi: Sequence[int]) -> np.ndarray:
        return self.read(name, [g - o for g, o in zip(glo, self.origin)],
                         [g - o for g, o in zip(ghi, self.origin)])


def zero_set(sdf_u8: np.ndarray, tol: int = ZERO_TOL) -> np.ndarray:
    """The face's ~1-voxel zero set: byte 128 is sdf = 0; byte 0 is *no data*, never a face."""
    return (sdf_u8 != 0) & (np.abs(sdf_u8.astype(np.int16) - 128) <= int(tol))


# --------------------------------------------------------------------------- #
# block statistics (one streaming pass)
# --------------------------------------------------------------------------- #
def _block_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum a (Z, Y, X) array (dims multiples of ``k``) over each k**3 block -> (nz, ny, nx)."""
    z, y, x = a.shape
    return a.reshape(z // k, k, y // k, k, x // k, k).sum(axis=(1, 3, 5))


def _tiles(shape: Sequence[int], tile: int, stride: int):
    """Core boxes covering ``shape``, each a whole number of ``stride`` blocks."""
    nb = [int(s) // int(stride) for s in shape]
    step = max(1, int(tile) // int(stride))
    for bz in range(0, nb[0], step):
        for by in range(0, nb[1], step):
            for bx in range(0, nb[2], step):
                b0 = (bz, by, bx)
                b1 = tuple(min(b0[a] + step, nb[a]) for a in range(3))
                yield b0, b1


def block_stats(store: Store, stride: int, tile: int, near: int, select: str,
                pred: Store | None, pred_tol: float) -> dict[str, np.ndarray]:
    """Per-``stride`` block accumulators for the whole region (one pass over the store)."""
    nb = tuple(int(s) // int(stride) for s in store.shape)
    zeros = lambda: np.zeros(nb, np.float64)  # noqa: E731
    acc = {k: zeros() for k in ("valid1", "seam", "thick_n", "thick_s", "thick_s2",
                                "face_n", "face_far")}
    halo = int(near) if select == "disagreement" else int(np.ceil(pred_tol)) + 1
    ntile = 0
    t0 = time.perf_counter()
    for b0, b1 in _tiles(store.shape, tile, stride):
        lo = [b0[a] * stride for a in range(3)]
        hi = [b1[a] * stride for a in range(3)]
        hlo = [lo[a] - halo for a in range(3)]
        hhi = [hi[a] + halo for a in range(3)]
        core = tuple(slice(halo, halo + hi[a] - lo[a]) for a in range(3))
        bs = tuple(slice(b0[a], b1[a]) for a in range(3))

        val = store.read("faces_valid", hlo, hhi)
        acc["valid1"][bs] += _block_sum((val[core] == 1).astype(np.float64), stride)
        thick = store.read("thickness", hlo, hhi)[core].astype(np.float64)
        tpos = thick > 0
        acc["thick_n"][bs] += _block_sum(tpos.astype(np.float64), stride)
        acc["thick_s"][bs] += _block_sum(thick * tpos, stride)
        acc["thick_s2"][bs] += _block_sum(thick * thick * tpos, stride)

        if select == "disagreement":
            src = store.read("faces_source", hlo, hhi)
            rv = src == SOURCE_RECTOVERSO
            if rv.any() and (src == SOURCE_CT).any():
                d = ndi.distance_transform_edt(~rv)
                seam = (src == SOURCE_CT) & (d <= float(near))
                acc["seam"][bs] += _block_sum(seam[core].astype(np.float64), stride)
            del src, rv
        elif select == "uncertain":
            assert pred is not None
            sup = val == 1
            for lbl_ch, pred_ch in (("sdf_in", "sdf_in"), ("sdf_out", "sdf_out")):
                face = zero_set(store.read(lbl_ch, hlo, hhi)) & sup
                pz = zero_set(pred.read_global(pred_ch, [l + store.origin[a] for a, l in enumerate(hlo)],
                                               [h + store.origin[a] for a, h in enumerate(hhi)]))
                if not face.any():
                    continue
                d = ndi.distance_transform_edt(~pz) if pz.any() else np.full(face.shape, np.inf, np.float64)
                acc["face_n"][bs] += _block_sum((face & True)[core].astype(np.float64), stride)
                acc["face_far"][bs] += _block_sum((face & (d > float(pred_tol)))[core].astype(np.float64), stride)
                del face, pz, d
            del sup
        del val, thick, tpos
        ntile += 1
        if ntile % 8 == 0:
            _log(f"scored {ntile} tiles ({time.perf_counter() - t0:.0f}s)")
    _log(f"block pass done: {ntile} tiles, grid {nb} ({time.perf_counter() - t0:.0f}s)")
    return acc


def crop_scores(acc: dict[str, np.ndarray], stride: int, size: int, select: str,
                min_valid: float, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score every crop origin on the stride grid from the block accumulators."""
    k = int(size) // int(stride)
    nb = acc["valid1"].shape
    origins, rows = [], []
    rng = np.random.default_rng(int(seed))
    nvox = float(size) ** 3
    for bz in range(0, nb[0] - k + 1):
        for by in range(0, nb[1] - k + 1):
            for bx in range(0, nb[2] - k + 1):
                sl = (slice(bz, bz + k), slice(by, by + k), slice(bx, bx + k))
                g = {key: float(a[sl].sum()) for key, a in acc.items()}
                valid_frac = g["valid1"] / nvox
                if valid_frac < float(min_valid):
                    score = -1.0
                    detail: dict[str, Any] = {}
                elif select == "random":
                    score = float(rng.random())
                    detail = {}
                elif select == "uncertain":
                    score = g["face_far"] / max(g["face_n"], 1.0)
                    detail = {"face_voxels": int(g["face_n"]), "far_voxels": int(g["face_far"])}
                else:
                    seam_frac = g["seam"] / nvox
                    n = max(g["thick_n"], 1.0)
                    mean = g["thick_s"] / n
                    var = max(g["thick_s2"] / n - mean * mean, 0.0)
                    cv = float(np.sqrt(var) / mean) if mean > 1e-6 else 0.0
                    score = seam_frac + THICK_W * min(cv, 1.0)
                    detail = {"seam_frac": seam_frac, "thickness_mean": mean,
                              "thickness_cv": cv}
                origins.append((bz * stride, by * stride, bx * stride))
                rows.append({"score": float(score), "valid_frac": float(valid_frac), **detail})
    return np.asarray(origins, np.int64), rows


def pick(origins: np.ndarray, rows: list[dict[str, Any]], n: int, size: int
         ) -> list[tuple[tuple[int, int, int], dict[str, Any]]]:
    """Top-n by score with no two crops overlapping (axis-aligned box separation)."""
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["score"], tuple(origins[i])))
    out: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for i in order:
        if len(out) >= int(n):
            break
        if rows[i]["score"] < 0:
            continue
        o = tuple(int(v) for v in origins[i])
        if any(all(abs(o[a] - p[a]) < size for a in range(3)) for p, _ in out):
            continue
        out.append((o, rows[i]))  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- #
# packet writing
# --------------------------------------------------------------------------- #
def write_u8(path: str, vol: np.ndarray) -> str:
    """One raw ``.u8`` volume: uint8, (z, y, x) C order == render3d's (z*ny + y)*nx + x."""
    a = np.ascontiguousarray(np.asarray(vol, dtype=np.uint8))
    with open(path, "wb") as fh:
        fh.write(a.tobytes())
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def write_packet(out_dir: str, idx: int, store: Store, cfg: Any, origin_local: Sequence[int],
                 size: int, row: dict[str, Any], select: str, pred: Store | None,
                 ct_reader: Any, tif: bool) -> dict[str, Any]:
    lo = [int(v) for v in origin_local]
    hi = [v + int(size) for v in lo]
    g = [store.origin[a] + lo[a] for a in range(3)]
    name = f"crop_{idx:02d}_z{g[0]}_y{g[1]}_x{g[2]}"
    pdir = os.path.join(out_dir, name)
    os.makedirs(pdir, exist_ok=True)

    layers: dict[str, np.ndarray] = {}
    layers["ct"] = np.asarray(ct_reader.read(g[0], g[0] + size, g[1], g[1] + size,
                                             g[2], g[2] + size), np.uint8)
    sdf_in = store.read("sdf_in", lo, hi)
    sdf_out = store.read("sdf_out", lo, hi)
    val = store.read("faces_valid", lo, hi)
    layers["faces_in"] = zero_set(sdf_in).astype(np.uint8)
    layers["faces_out"] = zero_set(sdf_out).astype(np.uint8)
    layers["ignore"] = (val == 2).astype(np.uint8)
    layers["source"] = store.read("faces_source", lo, hi) if store.has("faces_source") \
        else np.zeros_like(val)
    layers["rv_class"] = store.read("rv_class", lo, hi) if store.has("rv_class") \
        else np.zeros_like(val)
    if pred is not None:
        for face in ("in", "out"):
            layers[f"pred_{face}"] = zero_set(pred.read_global(
                f"sdf_{face}", g, [v + size for v in g])).astype(np.uint8)

    # the checksum of the *label* bricks this packet was cut from: the importer refuses to
    # fold a correction back into a store whose labels moved under it
    label_sha = hashlib.sha256()
    for a in (sdf_in, sdf_out, val, layers["source"], layers["rv_class"]):
        label_sha.update(np.ascontiguousarray(a).tobytes())

    sha = {k: write_u8(os.path.join(pdir, f"{k}.u8"), v) for k, v in layers.items()}
    if tif:
        import tifffile

        for k, v in layers.items():
            tifffile.imwrite(os.path.join(pdir, f"{k}.tif"), np.ascontiguousarray(v, np.uint8))

    meta = {
        "type": "tsm_annot_packet",
        "version": 1,
        "name": name,
        "origin_zyx": [int(v) for v in g],
        "origin_local_zyx": lo,
        "size_zyx": [int(size)] * 3,
        "dims_zyx": [int(size)] * 3,          # required by render3d --annot (spec/annot.md)
        "dims_xyz": [int(size)] * 3,          # render3d's plain command-line order
        "done": False,                        # render3d --annot rewrites only this token
        "voxel_um": store.voxel_um,
        "store": store.path,
        "store_origin_zyx": list(store.origin),
        "store_channels": store.channels,
        "pred_store": pred.path if pred is not None else None,
        "ct_volume": cfg.volume.url,
        "select": select,
        "score": row,
        "layer_order": "zyx_c_order_uint8",
        "layers": {k: {"file": f"{k}.u8", "sha256_16": sha[k]} for k in layers},
        "channel_semantics": {
            "ct": "CT uint8 at the config volume level",
            "faces_in": "1 = label in-face zero set (sdf_in byte 128); the face toward the umbilicus",
            "faces_out": "1 = label out-face zero set (sdf_out byte 128)",
            "ignore": "1 = faces_valid == 2 (teacher uncertain, unsupervised)",
            "source": "faces_source: 0 none / 1 rectoverso / 2 ct / 3 human",
            "rv_class": "upstream rv_class: 0 bg / 1 recto / 2 verso / 3 contact",
            "pred_in": "student sdf_in zero crossing (|byte-128| <= 3)",
            "pred_out": "student sdf_out zero crossing",
        },
        "correction": {"file": "correction.u8", "classes": CORRECTION_CLASSES,
                       "written_by": "the annotator (render3d)"},
        "label_sha256_16": label_sha.hexdigest()[:16],
    }
    with open(os.path.join(pdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    _log(f"{name}: score {row['score']:.4f} valid {row['valid_frac']:.3f} -> {pdir}")
    return meta


def write_manifest(out_dir: str, metas: list[dict[str, Any]], cfg_path: str, select: str) -> str:
    path = os.path.join(out_dir, "packet.json")
    doc = {
        "type": "tsm_annot_packets",
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.abspath(cfg_path),
        "select": select,
        "voxel_order": "zyx",
        "raw_format": {"dtype": "uint8", "index": "(z*ny + y)*nx + x",
                       "note": "render3d .u8: dims come from this manifest, not from a header"},
        "palette": {
            "ct": {"style": "grey"},
            "faces_in": {"style": "mask", "rgb": [230, 40, 40], "label": "in face (red)"},
            "faces_out": {"style": "mask", "rgb": [60, 120, 255], "label": "out face (blue)"},
            "ignore": {"style": "dim", "rgb": [110, 110, 110], "label": "ignore (dim)"},
            "source": {"style": "labels", "values": {"0": "none", "1": "rectoverso", "2": "ct",
                                                     "3": "human"}},
            "rv_class": {"style": "labels", "values": {"0": "bg", "1": "recto", "2": "verso",
                                                       "3": "contact"}},
            "pred_in": {"style": "mask", "rgb": [255, 160, 40], "label": "student in face"},
            "pred_out": {"style": "mask", "rgb": [40, 220, 180], "label": "student out face"},
            "correction": {"style": "paint", "values": CORRECTION_CLASSES,
                           "rgb": {"1": [230, 40, 40], "2": [60, 120, 255],
                                   "3": [110, 110, 110], "4": [250, 250, 60]}},
        },
        "packets": [
            {"path": m["name"], "origin_zyx": m["origin_zyx"], "size_zyx": m["size_zyx"], "dims_zyx": m["size_zyx"],
             "dims_xyz": m["dims_xyz"], "voxel_um": m["voxel_um"],
             "layers": [k for k in m["layers"]], "correction": "correction.u8",
             "score": m["score"]}
            for m in metas
        ],
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="export label-correction packets for render3d")
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--select", choices=SELECTORS, default="disagreement")
    ap.add_argument("--pred", default=None, help="student pred.zarr (required by --select uncertain)")
    ap.add_argument("--pred-tol", type=float, default=3.0)
    ap.add_argument("--near", type=int, default=8, help="seam neighbourhood of the disagreement proxy")
    ap.add_argument("--min-valid", type=float, default=0.10)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fine-store", default=None, help="override <out_dir>/labels/fine.zarr")
    ap.add_argument("--tif", action="store_true", help="also write a .tif copy of every layer")
    ap.add_argument("--dry-run", action="store_true", help="score and print, write nothing")
    a = ap.parse_args(argv)

    if a.size % a.stride:
        raise SystemExit(f"[annot-export] --size {a.size} must be a multiple of --stride {a.stride}")
    cfg = load_config(a.config)
    _label_opts(cfg)  # validate the labels block early (same errors as `tsm labels`)
    fine = os.path.expanduser(a.fine_store or os.path.join(cfg.out_dir, "labels", "fine.zarr"))
    if not os.path.exists(os.path.join(fine, "zarr.json")):
        raise SystemExit(f"[annot-export] no fine label store at {fine}")
    store = Store(fine)
    for need in ("sdf_in", "sdf_out", "faces_valid", "thickness"):
        if not store.has(need):
            raise SystemExit(f"[annot-export] {fine} has no {need!r} channel "
                             f"(needs extra.labels.faces.enabled); channels={store.channels}")
    if a.select == "disagreement" and not store.has("faces_source"):
        raise SystemExit(f"[annot-export] --select disagreement needs the 'faces_source' channel "
                         f"(faces.source != 'ct'); {fine} has {store.channels}")
    pred = None
    if a.pred:
        pred = Store(a.pred)
        for need in ("sdf_in", "sdf_out"):
            if not pred.has(need):
                raise SystemExit(f"[annot-export] {pred.path} has no {need!r} channel")
    if a.select == "uncertain" and pred is None:
        raise SystemExit("[annot-export] --select uncertain needs --pred")

    _log(f"{fine}: shape {store.shape} origin {store.origin} ({len(store.channels)} channels)")
    acc = block_stats(store, a.stride, a.tile, a.near, a.select, pred, a.pred_tol)
    origins, rows = crop_scores(acc, a.stride, a.size, a.select, a.min_valid, a.seed)
    _log(f"{len(rows)} candidate crops of {a.size}^3 at stride {a.stride}; "
         f"{sum(1 for r in rows if r['score'] >= 0)} pass min-valid {a.min_valid}")
    chosen = pick(origins, rows, a.n, a.size)
    if not chosen:
        raise SystemExit("[annot-export] no crop passed the selection; lower --min-valid")
    for o, r in chosen:
        _log(f"picked {o} score {r['score']:.4f} {({k: round(v, 4) for k, v in r.items() if k != 'score'})}")
    if a.dry_run:
        return 0

    from tsm.cli import open_ct

    out_dir = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out_dir, exist_ok=True)
    reader = open_ct(cfg, level_shift=0)
    metas = [write_packet(out_dir, i, store, cfg, o, a.size, r, a.select, pred, reader, a.tif)
             for i, (o, r) in enumerate(chosen)]
    path = write_manifest(out_dir, metas, a.config, a.select)
    _log(f"wrote {len(metas)} packets and {path}")
    _log("annotate with render3d, then fold the corrections back with "
         f"`uv run python dev/annot_import.py {a.config} --packets {out_dir}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
