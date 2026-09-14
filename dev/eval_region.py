"""Compare the student prediction store with the teachers and the labels on a whole region.

usage: uv run python dev/eval_region.py configs/paris4_eval.json [--bricks 0] [--no-gallery]
                                       [--faces-labels <faces fine.zarr>]
                                       [--rectoverso <rectoverso.zarr>] [--axis <umbilicus.json>]

Reads (all under ``cfg.out_dir``, see docs/label_store.md):
  student/pred.zarr        sdf, valid, ink, sin, cos, density, nx, ny, nz, conf, spare, surface1
  teachers/{recto,ink,lasagna,m7_l2}.zarr
  labels/{fine,coarse}.zarr
  CT via tsm.cli.open_ct(cfg)

Writes ``<out_dir>/eval/metrics.json``, ``metrics.md`` and ``gallery_z*.png`` /
``gallery_yz.png`` (PIL only).  Voxel-wise statistics are accumulated brick-wise
(z slabs, coarse-aligned); the surface distances and the connectivity statistics need
a global view and allocate whole-region masks under ``tsm.limits`` budget checks.

Two-face ("faces") predictions (channels ``sdf_in``/``sdf_out``) are handled: the student
"surface" becomes the medial surface derived from the two faces (see
:func:`medial_sdf_from_faces` for the sign convention), the raw in-face-vs-recto numbers are
kept under ``surface_inface_vs_recto``, and, when a faces label store is found
(``--faces-labels``, else ``<out_dir>/labels/fine.zarr`` or ``<out_dir>_labels/labels/fine.zarr``
if it has ``sdf_in``/``sdf_out``/``faces_valid``), a ``faces`` section reports per-face
SDF MAE / zero-crossing Dice / surface distances plus the thickness MAE.

``--rectoverso`` adds the ``upstream_faces`` / ``upstream_body`` blocks -- the **only teacher-independent**
metric here: the student's two faces are scored against the upstream Scroll-1 recto/verso mesh
labels resampled onto our grid (``dev/rectoverso_slab.py``), which no part of the training
pipeline has seen.  Every other number in this report compares the student with the teachers it
was distilled from, or with labels derived from them.  It also adds ``upstream_topology``: the
same comparison counted in *sheets* rather than voxels (merges, breaks, missed and spurious
connected components), reported in its own section and never combined with the voxel-wise
``upstream_faces`` numbers.  ``upstream_body`` is the orientation-free version of the same
comparison -- the thinned union of recto/verso/contact against ``surface_side1`` (sides store),
``surface_body1`` (body store) or
``surface_in1 | surface_out1`` (two-face store), so a faces baseline and a body run are directly
comparable and a global swap of the two predicted faces cannot change the score -- and
``upstream_topology`` gains a matching ``body`` entry whose student objects are the predicted
sheet bodies.  Its reference objects are the **unthinned** upstream bands (the
thinned centre planes shatter into >100k fragments and would turn reference fragmentation into
model merges); only its distances and Dice use the thinned planes.

Metric definitions follow ``tsm.train.evaluate`` and ``tsm.equivariance.metrics``
(reused where the shapes allow it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.cli import open_ct  # noqa: E402
from tsm.config import load_config  # noqa: E402
from tsm.data import CLIP, WF_CHANNELS, body_mask_np, body_sdf, decode_density, decode_signed, face_dist  # noqa: E402
from tsm.equivariance import average_precision, surface_distances  # noqa: E402
from tsm.infer import _pool_winding, decode_pred  # noqa: E402
from tsm.labels import (  # noqa: E402
    DEFAULT_AXIS,
    _drop_small_components,
    decode_lasagna_normal,
    decode_sdf_u8,
    medial_surface,
    surface_stats,
)
from tsm.limits import check_alloc  # noqa: E402
from tsm.train import auroc  # noqa: E402

MIN_COMPONENT = 100
STATS_BOX = (256, 512, 512)  # central box for the connectivity statistics of surface1


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# stores
# --------------------------------------------------------------------------- #
class Store:
    """(C, Z, Y, X) uint8 zarr with ``origin_zyx`` / ``channels`` attrs, read in global voxels."""

    def __init__(self, path: str, scale: int = 1) -> None:
        self.path = str(path)
        self.arr = zarr.open_array(store=self.path, mode="r")
        a = dict(self.arr.attrs)
        self.channels = list(a.get("channels", [])) or [str(i) for i in range(self.arr.shape[0])]
        self.origin = [int(v) for v in a.get("origin_zyx", (0, 0, 0))]
        self.shape = [int(s) for s in self.arr.shape[1:]]
        self.voxel_um = float(a.get("voxel_um", 2.4))
        self.scale = int(scale)  # fine voxels per store voxel

    def has(self, ch: str) -> bool:
        return ch in self.channels

    def read(self, lo, hi, channels=None) -> np.ndarray:
        """(C, dz, dy, dx) block of *store* voxels at global store coordinates; 0 outside."""
        sel = None if channels is None else [self.channels.index(c) for c in channels]
        nc = self.arr.shape[0] if sel is None else len(sel)
        out = np.zeros((nc,) + tuple(int(h - l) for l, h in zip(lo, hi)), np.uint8)
        a = [max(0, int(l) - o) for l, o in zip(lo, self.origin)]
        b = [min(s, int(h) - o) for h, o, s in zip(hi, self.origin, self.shape)]
        if all(bb > aa for aa, bb in zip(a, b)):
            blk = np.asarray(self.arr[:, a[0]:b[0], a[1]:b[1], a[2]:b[2]])
            if sel is not None:
                blk = blk[sel]
            d = [aa + o - int(l) for aa, o, l in zip(a, self.origin, lo)]
            out[:, d[0]:d[0] + blk.shape[1], d[1]:d[1] + blk.shape[2], d[2]:d[2] + blk.shape[3]] = blk
        return out

    def read_fine(self, lo, hi, channels=None) -> np.ndarray:
        """Block given in *fine* (level-0) global voxels; a coarse store is read at lo//scale."""
        if self.scale == 1:
            return self.read(lo, hi, channels)
        return self.read([l // self.scale for l in lo], [-(-h // self.scale) for h in hi], channels)


def open_store(path: str, scale: int = 1) -> Store | None:
    if not os.path.exists(os.path.join(path, "zarr.json")) and not os.path.exists(os.path.join(path, ".zarray")):
        return None
    return Store(path, scale)


# --------------------------------------------------------------------------- #
# accumulators
# --------------------------------------------------------------------------- #
class Mean:
    def __init__(self) -> None:
        self.s = 0.0
        self.n = 0

    def add(self, v: np.ndarray, m: np.ndarray | None = None) -> None:
        x = v if m is None else v[m]
        if x.size:
            self.s += float(np.asarray(x, np.float64).sum())
            self.n += int(x.size)

    @property
    def value(self) -> float | None:
        return (self.s / self.n) if self.n else None


class DiceCount:
    def __init__(self) -> None:
        self.inter = self.na = self.nb = 0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        self.inter += int((a & b).sum())
        self.na += int(a.sum())
        self.nb += int(b.sum())

    @property
    def value(self) -> float | None:
        return (2.0 * self.inter / (self.na + self.nb)) if (self.na + self.nb) else None


class Reservoir:
    """Bounded, seeded simple random sample of (score, label) for AUPRC / AUROC over a whole region.

    Bottom-k reservoir: each eligible voxel is given one independent uniform key in arrival order
    and the ``cap`` smallest keys survive, so every voxel of the region has the same inclusion
    probability and the sample represents the *pooled* voxel population (prevalence included).
    An equal quota per brick would instead over-weight sparsely occupied bricks and change AP /
    AUROC with the brick size.  The result depends only on the voxel order, not on how the voxels
    are split into bricks / ``add`` calls."""

    def __init__(self, cap: int = 2_000_000, seed: int = 0) -> None:
        self.cap, self.rng = max(1, int(cap)), np.random.default_rng(seed)
        self.s = np.zeros(0, np.float32)
        self.y = np.zeros(0, bool)
        self.k = np.zeros(0, np.float64)
        self.n = 0

    def add(self, score: np.ndarray, label: np.ndarray, n_bricks: int | None = None) -> None:
        """Add eligible voxels (``n_bricks`` is accepted for call compatibility and unused)."""
        s = np.asarray(score, np.float32).ravel()
        y = np.asarray(label, bool).ravel()
        if s.size == 0:
            return
        self.n += s.size
        k = self.rng.random(s.size)
        s, y, k = np.concatenate([self.s, s]), np.concatenate([self.y, y]), np.concatenate([self.k, k])
        if k.size > self.cap:
            keep = np.argpartition(k, self.cap - 1)[: self.cap]
            s, y, k = s[keep], y[keep], k[keep]
        self.s, self.y, self.k = s, y, k

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.s, self.y

    def auprc(self) -> float | None:
        s, y = self.arrays()
        return average_precision(s, y) if s.size else None

    def auroc(self) -> float | None:
        s, y = self.arrays()
        if not s.size:
            return None
        v = auroc(s, y)
        return None if math.isnan(v) else float(v)


def _angle_deg(na: np.ndarray, nb: np.ndarray, sign_agnostic: bool = False) -> np.ndarray:
    dot = (na * nb).sum(0) / (np.linalg.norm(na, axis=0) * np.linalg.norm(nb, axis=0) + 1e-12)
    if sign_agnostic:
        dot = np.abs(dot)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _pcts(v: np.ndarray, name: str) -> dict[str, Any]:
    v = np.asarray(v, np.float64).ravel()
    if v.size == 0:
        return {f"{name}_n": 0, f"{name}_mean": None, f"{name}_median": None, f"{name}_p90": None}
    return {f"{name}_n": int(v.size), f"{name}_mean": float(v.mean()),
            f"{name}_median": float(np.median(v)), f"{name}_p90": float(np.percentile(v, 90))}


# --------------------------------------------------------------------------- #
# voxel-wise pass (brick-wise, RAM bounded)
# --------------------------------------------------------------------------- #
def voxel_pass(pred: Store, fine: Store | None, coarse: Store | None, ink_t: Store | None,
               lo: list[int], hi: list[int], clip: float, budget, brick=(64, 512, 512),
               sample_cap: int = 2_000_000, fiber_t: Store | None = None,
               wf_store: Store | None = None, flab: Store | None = None
               ) -> tuple[dict[str, Any], np.ndarray]:
    """Accumulate every voxel-wise metric and return (metrics, the student surface1 mask).

    ``wf_store`` is where the native winding target (``wf_*``) is read from: the eval fine
    store when it carries the block, else (e.g. when only the faces label store was
    rebuilt with ``extra.labels.winding_fine``) the store passed here.

    In the orientation-free **body** mode (the prediction store carries ``sdf_body``) the label
    side is not ``fine`` but the two-face store ``flab``: the reference is
    ``tsm.data.body_sdf(sdf_in, sdf_out)`` on ``faces_valid == 1`` voxels, the single definition
    the dataset and the training metrics use, so nothing here re-derives it.

    Bricks are 4-aligned (the winding comparison pools the student by 4); every decoded
    brick costs about 12 uint8 + 11 float32 planes, so ``brick`` bounds the working set."""
    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    check_alloc((dz, dy, dx), np.bool_, budget)
    surf = np.zeros((dz, dy, dx), bool)   # student surface1 over the whole region
    b = [max(4, (int(v) // 4) * 4) for v in brick]
    starts = [(z, y, x) for z in range(lo[0], hi[0], b[0])
              for y in range(lo[1], hi[1], b[1]) for x in range(lo[2], hi[2], b[2])]
    n_bricks = max(1, len(starts))
    check_alloc((pred.arr.shape[0], *b), np.uint8, budget)

    sdf_mae, valid_ap = Mean(), Reservoir(sample_cap, 1)
    ink_dice, ink_ap = DiceCount(), Reservoir(sample_cap, 2)
    # optional: the fibre head vs the eval-region fiber teacher (channels 1 = vt, 2 = hz)
    do_fiber = (fiber_t is not None and all(pred.has(c) for c in ("fiber_vt", "fiber_hz"))
                and all(fiber_t.has(c) for c in ("fiber_vt", "fiber_hz")))
    fiber_ap = {k: Reservoir(sample_cap, 10 + i) for i, k in enumerate(("vt", "hz"))}
    zc = {"hit": 0, "n_pred": 0, "n_label": 0}
    ph, nang = [], []
    dens_mae, conf_auroc = Mean(), Reservoir(sample_cap, 3)
    # the native-2.4 um winding target (tsm.winding_fine), when the fine store carries it:
    # the same three errors, at the *fine* pitch and with no pooling, restricted to
    # wf_valid == 1.  It is the only winding reference that is not aliased by the 9.6 um
    # teacher wherever the sheets are finer than the teacher can resolve.
    wf_src = next((st_ for st_ in (fine, wf_store) if st_ is not None and all(st_.has(c) for c in WF_CHANNELS)), None)
    do_wf = wf_src is not None
    ph_f, nang_f = [], []
    dens_mae_f, n_wf_valid = Mean(), 0
    med_cov = Mean()  # fraction of the predicted sheet interior where the medial surface is defined
    n_coarse_valid = 0
    st = ndi.generate_binary_structure(3, 3)  # "within 1 voxel" = Chebyshev distance 1
    surf_ch = next((c for c in ("surface1", "surface_in1", "surface_body1", "surface_side1")
                    if c in pred.channels), None)
    # in the two-face mode the label `sdf` is the MEDIAL sdf, so compare it with the student
    # medial (equidistance between the faces), not with the in face (half a sheet away)
    faces = pred.has("sdf_in") and pred.has("sdf_out")
    body = pred.has("sdf_body")
    sides = pred.has("d_face")
    if (body or sides) and flab is None:
        log("WARNING orientation-free mode without a faces label store: no label-side comparison")
    lab = flab if (body or sides) else fine   # which store the surface comparison reads
    do_lab = lab is not None
    # body mode: the body target, its zero set and the per-voxel agreement of the two bodies
    bdice, b_inter, b_union, n_pred_body, n_label_body, n_faces_valid1 = DiceCount(), 0, 0, 0, 0, 0
    # fibre exclusivity (no teacher needed): the two derived class channels must not both fire
    fib_ch = pred.has("fiber_vt") and pred.has("fiber_hz")
    i_vt = pred.channels.index("fiber_vt") if fib_ch else -1
    i_hz = pred.channels.index("fiber_hz") if fib_ch else -1
    n_fib_both = n_fib_any = n_data = 0

    for z, y, x in starts:
        blo = [z, y, x]
        bhi = [min(z + b[0], hi[0]), min(y + b[1], hi[1]), min(x + b[2], hi[2])]
        u8 = pred.read(blo, bhi)
        dec = decode_pred(u8, clip, pred.channels)
        s1 = next((dec[k] for k in ("surface1", "surface_in1", "surface_body1", "surface_side1")
                   if k in dec), None)
        sl = tuple(slice(l0 - l, h0 - l) for l0, h0, l in zip(blo, bhi, lo))
        surf[sl] = s1 if s1 is not None else np.zeros_like(dec["data"])
        pvalid = dec["data"] & (dec["valid"] > 0.5)

        if do_lab:
            # a 1-voxel halo so the zero crossing and the within-1 dilation are exact at brick edges
            hlo = [max(l, v - 1) for v, l in zip(blo, lo)]
            hhi = [min(h, v + 1) for v, h in zip(bhi, hi)]
            cs = tuple(slice(v - h, v - h + (e - v)) for v, e, h in zip(blo, bhi, hlo))
            pcore = dec["sdf_body"] if body else (dec["d_face"] if sides else dec["sdf"])
            if sides:
                # the label side is the UNSIGNED face distance + the 0/1 body of the SAME faces
                # store the dataset reads (tsm.data.face_dist / body_mask_np)
                f = lab.read(hlo, hhi, ["sdf_in", "sdf_out", "faces_valid"])
                ldata = (f[0] != 0) & (f[1] != 0)
                si, so = decode_sdf_u8(f[0], clip), decode_sdf_u8(f[1], clip)
                lsdf = face_dist(si, so) * ldata
                lbody_full = body_mask_np(si, so) & ldata
                lvalid = f[2] == 1
                lvraw = f[2]
                del si, so
            elif body:
                # the label-side body SDF is min(sdf_in, -sdf_out) of the SAME faces store the
                # dataset reads (tsm.data.body_sdf), masked by faces_valid
                f = lab.read(hlo, hhi, ["sdf_in", "sdf_out", "faces_valid"])
                ldata = (f[0] != 0) & (f[1] != 0)
                lsdf = body_sdf(decode_sdf_u8(f[0], clip), decode_sdf_u8(f[1], clip)) * ldata
                lvalid = f[2] == 1
                lvraw = f[2]
            else:
                f = lab.read(hlo, hhi, ["sdf", "sdf_valid"])
                ldata = f[0] != 0
                lsdf = decode_sdf_u8(f[0], clip) * ldata
                lvalid = f[1] == 1
                lvraw = f[1]
            lzc = (ldata & (lsdf <= 1.0) & lvalid) if sides else _thin_zc(lsdf, ldata, lvalid)
            if faces:
                u = pred.read(hlo, hhi, ["sdf_in", "sdf_out"])
                s1h = medial_zc_from_faces(u[0], u[1], clip)
                psdf, usable = medial_sdf_from_faces(u[0], u[1], clip)
                interior = usable | (face_saturated(u[0], u[1]) & (u[0] != 0) & (u[1] != 0)
                                     & (decode_sdf_u8(u[0], clip) >= 0) & (decode_sdf_u8(u[1], clip) <= 0))
                med_cov.add(usable[cs].astype(np.float32), interior[cs])
                del u, usable, interior
            else:
                s1h = (pred.read(hlo, hhi, [surf_ch])[0] > 127) if surf_ch else np.zeros_like(lzc)
                psdf = None
            zc["hit"] += int((s1h & ndi.binary_dilation(lzc, st))[cs].sum()) \
                + int((lzc & ndi.binary_dilation(s1h, st))[cs].sum())
            zc["n_pred"] += int(s1h[cs].sum())
            zc["n_label"] += int(lzc[cs].sum())
            lsdf, lvalid, lv2 = lsdf[cs], lvalid[cs], lvraw[cs]
            if sides:
                lbody = lbody_full[cs]
            band = lvalid & (np.abs(lsdf) < clip) & dec["data"]
            sdf_mae.add(np.abs((psdf[cs] if psdf is not None else pcore) - lsdf), band)
            if body or sides:
                # per-voxel agreement of the two BODIES, the orientation-free object.  body
                # mode: sdf_body > 0 on both sides; sides mode: the predicted body probability
                # vs the 0/1 label body mask
                dom = lvalid & dec["data"]
                if sides:
                    pb, lb = (dec["body"] > 0.5) & dom, lbody & dom
                else:
                    pb, lb = (pcore > 0) & dom, (lsdf > 0) & dom
                bdice.add(pb, lb)
                b_inter += int((pb & lb).sum())
                b_union += int((pb | lb).sum())
                n_pred_body += int(pb.sum())
                n_label_body += int(lb.sum())
                n_faces_valid1 += int(lvalid.sum())
                del dom, pb, lb
            # evaluation domain: drop only the ignore band (validity 2) and restrict to the data
            # domain of the *prediction* store, which is independent of the labels.  The label SDF
            # byte must NOT take part: byte 0 is exactly the validity-0 class, so requiring
            # `sdf != 0` would delete every negative and leave validity AP undefined.
            sel = (lv2 != 2) & dec["data"]
            if sel.any():
                valid_ap.add(dec["valid"][sel], lvalid[sel], n_bricks)
            del f, ldata, lvraw, lsdf, lvalid, lv2, band, lzc, s1h, psdf, pcore
            if sides:
                del lbody, lbody_full

        if ink_t is not None:
            it = ink_t.read(blo, bhi)[0]
            tgt = it >= 128
            ink_dice.add(dec["ink"] >= 0.5, tgt)
            ink_ap.add(dec["ink"], tgt, n_bricks)
            del it, tgt

        n_data += int(dec["data"].sum())
        if fib_ch:
            # exclusivity of the two derived class channels: no teacher involved, so this is a
            # property of the student alone (labels overlap on 0 % of their firing voxels)
            vt_b, hz_b = u8[i_vt] > 127, u8[i_hz] > 127
            n_fib_both += int((vt_b & hz_b).sum())
            n_fib_any += int((vt_b | hz_b).sum())
            del vt_b, hz_b

        if do_fiber:
            ft = fiber_t.read(blo, bhi, ["fiber_vt", "fiber_hz"])
            for i, k in enumerate(("vt", "hz")):
                fiber_ap[k].add(dec[f"fiber_{k}"], ft[i] >= 128, n_bricks)
            del ft

        if do_wf:
            wf = wf_src.read(blo, bhi, WF_CHANNELS)
            wm = (wf[7] == 1) & dec["data"]
            n_wf_valid += int((wf[7] == 1).sum())
            if wm.any():
                pa = np.arctan2(dec["sin"][wm], dec["cos"][wm])
                pb = np.arctan2(decode_signed(wf[0])[wm], decode_signed(wf[1])[wm])
                ph_f.append(np.degrees(np.abs(np.angle(np.exp(1j * (pa - pb))))).astype(np.float32))
                sn = np.stack([dec["nz"], dec["ny"], dec["nx"]])[:, wm]
                tn = np.stack([decode_signed(wf[5]), decode_signed(wf[4]), decode_signed(wf[3])])[:, wm]
                nang_f.append(_angle_deg(sn, tn).astype(np.float32))
                # wf_density is already in wraps per fine voxel, like the student head
                dens_mae_f.add(np.abs(dec["density"][wm] - decode_density(wf[2])[wm]))
                del pa, pb, sn, tn
            del wf, wm

        if coarse is not None and all((h - l) % 4 == 0 for l, h in zip(blo, bhi)):
            # _pool_winding wants "sdf" (the cover mask); decode_pred only aliases sdf_in
            pw = _pool_winding(dec, pvalid, 4)  # decode_pred aliases sdf_body / d_face -> sdf
            c = coarse.read([v // 4 for v in blo], [-(-v // 4) for v in bhi],
                            ["phase_sin", "phase_cos", "density", "nx", "ny", "nz", "conf", "valid"])
            sh = [min(a_, b_) for a_, b_ in zip(c.shape[1:], pw["cos"].shape)]
            c = c[:, :sh[0], :sh[1], :sh[2]]
            pw = {k: v[:sh[0], :sh[1], :sh[2]] for k, v in pw.items()}
            cv = c[7] == 1
            cover = pw["cover"] > 0
            m = cv & cover
            n_coarse_valid += int(cv.sum())
            if m.any():
                pa = np.arctan2(pw["sin"][m], pw["cos"][m])
                pb = np.arctan2(decode_signed(c[0])[m], decode_signed(c[1])[m])
                ph.append(np.degrees(np.abs(np.angle(np.exp(1j * (pa - pb))))).astype(np.float32))
                sn = np.stack([pw["nz"], pw["ny"], pw["nx"]])[:, m]
                tn = np.stack([decode_signed(c[5]), decode_signed(c[4]), decode_signed(c[3])])[:, m]
                nang.append(_angle_deg(sn, tn).astype(np.float32))
                dens_mae.add(np.abs(pw["density"][m] - decode_density(c[2])[m]))
            sel = cover & (c[7] != 2)
            if sel.any():
                conf_auroc.add(pw["conf"][sel], cv[sel], n_bricks)
            del pw, c
        del u8, dec, pvalid, s1
        log(f"brick {blo} -> {bhi} done")

    ph_a = np.concatenate(ph) if ph else np.zeros(0, np.float32)
    na_a = np.concatenate(nang) if nang else np.zeros(0, np.float32)
    m: dict[str, Any] = {
        "surface": {"student_surface": ("medial_from_faces" if faces else
                                       ("body_sdf" if body else ("d_face" if sides else "sdf"))),
                    "sdf_mae_band": sdf_mae.value, "sdf_mae_n": sdf_mae.n,
                    "valid_auprc": valid_ap.auprc(), "valid_auprc_n": valid_ap.n,
                    "zc_dice_within1": (zc["hit"] / (zc["n_pred"] + zc["n_label"])) if (zc["n_pred"] + zc["n_label"]) else None,
                    "zc_n_pred": zc["n_pred"], "zc_n_label": zc["n_label"],
                    # faces only: fraction of the predicted sheet interior where neither face SDF is
                    # doubly clipped, i.e. where the medial surface is actually defined (R21)
                    "medial_coverage_frac": med_cov.value, "medial_coverage_n": med_cov.n},
        "ink": {"dice_at_0.5": ink_dice.value, "auprc": ink_ap.auprc(),
                "n_pred": ink_dice.na, "n_teacher": ink_dice.nb, "auprc_n": ink_ap.n},
        "winding": {**_pcts(ph_a, "phase_err_deg"), **_pcts(na_a, "normal_err_deg"),
                    "density_mae": dens_mae.value, "conf_auroc": conf_auroc.auroc(),
                    "n_coarse_valid": n_coarse_valid},
    }
    if do_wf:  # only when the fine store was built with extra.labels.winding_fine
        phf_a = np.concatenate(ph_f) if ph_f else np.zeros(0, np.float32)
        naf_a = np.concatenate(nang_f) if nang_f else np.zeros(0, np.float32)
        m["winding_fine"] = {**_pcts(phf_a, "phase_err_deg"), **_pcts(naf_a, "normal_err_deg"),
                             "density_mae": dens_mae_f.value, "n_wf_valid": n_wf_valid}
    fib: dict[str, Any] = {}
    if do_fiber:  # only when both the prediction and an eval-region fiber teacher carry the channels
        fib.update({k: {"auprc": fiber_ap[k].auprc(), "auprc_n": fiber_ap[k].n} for k in ("vt", "hz")})
    if fib_ch:
        # vt and hz are two independent sigmoids; a voxel firing both is a fibre the student
        # could not orient.  The labels overlap on 0 % of their firing voxels (target <= 0.15).
        fib.update({"overlap_frac": n_fib_both / max(1, n_fib_any),
                    "firing_frac": n_fib_any / max(1, n_data),
                    "n_any": n_fib_any, "n_both": n_fib_both, "n_data": n_data})
    if fib:
        m["fiber"] = fib
    if (body or sides) and do_lab:
        m["body"] = {"labels": lab.path,
                     "target": "(sdf_in > 0) & (sdf_out < 0)" if sides else "min(sdf_in, -sdf_out)",
                     "n_faces_valid1": n_faces_valid1,
                     "body_dice": bdice.value,
                     "body_iou": (b_inter / b_union) if b_union else None,
                     "body_vol_ratio": (n_pred_body / n_label_body) if n_label_body else None,
                     "n_pred_body": n_pred_body, "n_label_body": n_label_body}
    return m, surf


def _thin_zc(sdf: np.ndarray, data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Label zero crossing, same rule as ``infer.extract_surface``."""
    from tsm.infer import _any6

    return data & (sdf <= 0.0) & _any6(data & (sdf > 0.0)) & valid


# --------------------------------------------------------------------------- #
# two-face ("faces") surface mode
# --------------------------------------------------------------------------- #
FACE_BAND = 8.0        # |label sdf| <= FACE_BAND voxels for the per-face SDF MAE
FACE_LABEL_CH = ("sdf_in", "sdf_out", "faces_valid")


def _bricks(lo, hi, brick) -> list[tuple[list[int], list[int]]]:
    b = [max(4, (int(v) // 4) * 4) for v in brick]
    return [([z, y, x], [min(z + b[0], hi[0]), min(y + b[1], hi[1]), min(x + b[2], hi[2])])
            for z in range(lo[0], hi[0], b[0])
            for y in range(lo[1], hi[1], b[1])
            for x in range(lo[2], hi[2], b[2])]


def _halo(blo, bhi, lo, hi, n: int = 1):
    """(hlo, hhi, core slices into the haloed block) for an ``n``-voxel halo clamped to the region."""
    hlo = [max(l, v - n) for v, l in zip(blo, lo)]
    hhi = [min(h, v + n) for v, h in zip(bhi, hi)]
    cs = tuple(slice(v - h, v - h + (e - v)) for v, e, h in zip(blo, bhi, hlo))
    return hlo, hhi, cs


def _dest(blo, bhi, lo):
    return tuple(slice(l0 - l, h0 - l) for l0, h0, l in zip(blo, bhi, lo))


def is_faces_store(s: Store | None) -> bool:
    return s is not None and all(s.has(c) for c in FACE_LABEL_CH)


def resolve_faces_labels(out: str, explicit: str | None) -> Store | None:
    """``--faces-labels`` if given, else ``<out>/labels/fine.zarr``, else ``<out>_labels/labels/fine.zarr``."""
    if explicit:
        s = open_store(os.path.expanduser(explicit))
        if s is None:
            raise FileNotFoundError(f"no faces label store at {explicit}")
        if not is_faces_store(s):
            raise ValueError(f"{explicit} has no {FACE_LABEL_CH} channels (has {s.channels})")
        return s
    for cand in (os.path.join(out, "labels", "fine.zarr"),
                 os.path.join(str(out).rstrip("/\\") + "_labels", "labels", "fine.zarr")):
        s = open_store(cand)
        if is_faces_store(s):
            return s
    return None


def face_saturated(sdf_in_u8: np.ndarray, sdf_out_u8: np.ndarray) -> np.ndarray:
    """Voxels where **both** face SDFs are clipped (encoded byte 1 = -clip or 255 = +clip).

    Their sum carries no distance information there (both terms are the clip value, not a distance),
    so the medial surface cannot be located: for a sheet thicker than ``2 * clip`` the sum is a flat
    plateau and its "zero crossing" lands wherever the plateau happens to end."""
    sat_i = (sdf_in_u8 == 1) | (sdf_in_u8 == 255)
    sat_o = (sdf_out_u8 == 1) | (sdf_out_u8 == 255)
    return sat_i & sat_o


def medial_sdf_from_faces(sdf_in_u8: np.ndarray, sdf_out_u8: np.ndarray, clip: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(signed distance to the medial surface, usable sheet interior mask) from the two faces.

    Sign convention (``labels._signed_side``): sdf_in / sdf_out are each positive on the *outward*
    side of their own face, so inside the sheet sdf_in > 0 > sdf_out (and thickness = sdf_in -
    sdf_out).  The equidistant (medial) surface is therefore the zero set of sdf_in + sdf_out,
    whose half is the signed distance to it (exactly so for parallel faces).

    The interior mask excludes doubly saturated voxels (:func:`face_saturated`): there the sum is
    not a distance field and the crossing would be an artefact of the clip, not geometry.  Sheets
    thinner than ``2 * clip`` are unaffected; see ``medial_coverage_frac`` in the metrics."""
    si, so = decode_sdf_u8(sdf_in_u8, clip), decode_sdf_u8(sdf_out_u8, clip)
    data = (sdf_in_u8 != 0) & (sdf_out_u8 != 0)
    inside = data & (si >= 0.0) & (so <= 0.0)
    return 0.5 * (si + so), inside & ~face_saturated(sdf_in_u8, sdf_out_u8)


def medial_zc_from_faces(sdf_in_u8: np.ndarray, sdf_out_u8: np.ndarray, clip: float) -> np.ndarray:
    """1-voxel medial surface: the zero crossing of ``sdf_in + sdf_out`` inside the sheet."""
    from tsm.infer import _any6

    g, inside = medial_sdf_from_faces(sdf_in_u8, sdf_out_u8, clip)
    return inside & (g <= 0.0) & _any6(inside & (g > 0.0))


def medial_from_faces(pred: Store, lo, hi, clip: float, budget, brick=(64, 512, 512)) -> np.ndarray:
    """Whole-region bool mask of :func:`medial_zc_from_faces` over the student prediction."""
    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    check_alloc((dz, dy, dx), np.bool_, budget)
    med = np.zeros((dz, dy, dx), bool)
    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi)
        u = pred.read(hlo, hhi, ["sdf_in", "sdf_out"])
        med[_dest(blo, bhi, lo)] = medial_zc_from_faces(u[0], u[1], clip)[cs]
        del u
    return med


def faces_pass(pred: Store, flab: Store, lo, hi, clip: float, budget, brick=(64, 512, 512),
               band: float = FACE_BAND) -> dict[str, Any]:
    """Per-face (in, out) student-vs-label metrics on ``faces_valid == 1`` voxels.

    SDF MAE in a |label sdf| <= ``band`` shell, zero-crossing Dice within 1 voxel, EDT surface
    distances both ways (the machinery of the medial comparison) and the thickness MAE."""
    from tsm.infer import _any6

    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    faces = ("in", "out")
    check_alloc((dz, dy, dx), np.bool_, budget)
    masks = {(w, f): np.zeros((dz, dy, dx), bool) for w in ("pred", "label") for f in faces}
    mae = {f: Mean() for f in faces}
    zc = {f: {"hit": 0, "n_pred": 0, "n_label": 0} for f in faces}
    th_mae, n_sup = Mean(), 0
    th_err: list[np.ndarray] = []          # bounded sample of |thickness error| for the median
    th_rng = np.random.default_rng(4)
    n_interior = n_sat = 0
    st = ndi.generate_binary_structure(3, 3)  # "within 1 voxel" = Chebyshev distance 1
    has_th = flab.has("thickness")          # the label thickness channel gates the saturated sheets
    n_bricks = max(1, len(_bricks(lo, hi, brick)))
    th_cap = max(1, 2_000_000 // n_bricks)

    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi)
        f = flab.read(hlo, hhi, ["sdf_in", "sdf_out", "faces_valid"])
        p = pred.read(hlo, hhi, ["sdf_in", "sdf_out", "surface_in1", "surface_out1"])
        sup = f[2] == 1
        dst = _dest(blo, bhi, lo)
        for i, name in enumerate(faces):
            ldata = f[i] != 0
            lsdf = decode_sdf_u8(f[i], clip) * ldata
            lzc = (ldata & (lsdf <= 0.0) & _any6(ldata & (lsdf > 0.0))) & sup
            pzc = (p[2 + i] > 127) & sup
            zc[name]["hit"] += int((pzc & ndi.binary_dilation(lzc, st))[cs].sum()) \
                + int((lzc & ndi.binary_dilation(pzc, st))[cs].sum())
            zc[name]["n_pred"] += int(pzc[cs].sum())
            zc[name]["n_label"] += int(lzc[cs].sum())
            masks[("pred", name)][dst] = pzc[cs]
            masks[("label", name)][dst] = lzc[cs]
            sel = (sup & ldata & (np.abs(lsdf) <= float(band)) & (p[i] != 0))[cs]
            mae[name].add(np.abs(decode_sdf_u8(p[i], clip) - lsdf)[cs], sel)
            del ldata, lsdf, lzc, pzc, sel
        n_sup += int(sup[cs].sum())
        if has_th:
            # Compare like with like: the gap between the two zero sets, sdf_in - sdf_out, decoded
            # from both stores with the same clip, on the label sheet interior (sdf_in >= 0 >=
            # sdf_out).  The label `thickness` channel (2*EDT at the nearest medial voxel) is
            # unclipped, so it only serves to drop sheets whose gap saturates at 2*clip.
            lsi, lso = decode_sdf_u8(f[0], clip), decode_sdf_u8(f[1], clip)
            interior = (f[0] != 0) & (f[1] != 0) & (lsi >= 0.0) & (lso <= 0.0) & sup
            lt = flab.read(hlo, hhi, ["thickness"])[0].astype(np.float32)
            unsat = interior & (lt < 2.0 * clip - 2.0)
            n_interior += int(interior[cs].sum())
            n_sat += int((interior & ~unsat)[cs].sum())
            err = np.abs((decode_sdf_u8(p[0], clip) - decode_sdf_u8(p[1], clip)) - (lsi - lso))
            m = unsat[cs] & (p[0][cs] != 0) & (p[1][cs] != 0)
            th_mae.add(err[cs], m)
            v = err[cs][m].astype(np.float32)
            if v.size > th_cap:
                v = v[th_rng.choice(v.size, th_cap, replace=False)]
            if v.size:
                th_err.append(v)
            del lsi, lso, interior, lt, unsat, err, m, v
        del f, p, sup
        log(f"faces brick {blo} -> {bhi} done")

    out: dict[str, Any] = {"labels": flab.path, "band": float(band), "n_faces_valid1": n_sup}
    for name in faces:
        z = zc[name]
        d = surface_distances(masks[("pred", name)], masks[("label", name)])
        r: dict[str, Any] = {
            "sdf_mae_band": mae[name].value, "sdf_mae_n": mae[name].n,
            "zc_dice_within1": (z["hit"] / (z["n_pred"] + z["n_label"])) if (z["n_pred"] + z["n_label"]) else None,
            "zc_n_pred": z["n_pred"], "zc_n_label": z["n_label"],
        }
        for side, key in (("a_to_b", "pred_to_label"), ("b_to_a", "label_to_pred")):
            for k in ("n", "median", "p90", "frac_gt3"):
                r[f"{key}_{k}"] = d[side].get(k)
        out[name] = r
    e = np.concatenate(th_err) if th_err else np.zeros(0, np.float32)
    out["thickness_mae"] = th_mae.value
    out["thickness_median_abs_err"] = float(np.median(e)) if e.size else None
    out["thickness_n"] = th_mae.n   # faces_valid == 1, inside the label sheet, gap not saturated
    out["thickness_frac_saturated"] = (n_sat / n_interior) if n_interior else None
    out["thickness_n_interior"] = n_interior
    return out


# --------------------------------------------------------------------------- #
# upstream (teacher-independent) fibre metrics
# --------------------------------------------------------------------------- #
def fiber_upstream_pass(pred: Store, band: Store, band_channel: str, lo, hi, clip: float, budget,
                        brick=(64, 512, 512), axis: Any = None) -> dict[str, Any]:
    """The student's fibre prediction against the **human** hz/vt bands (``hzvt_class``).

    The reference is the upstream traced fibre classes (0 bg / 1 hz / 2 vt / 3 exclude), which
    no teacher in this pipeline produced -- so, like ``upstream_faces``, a disagreement here is
    evidence about the student rather than a tautology.  Reported over the band voxels:

    * ``class_acc`` -- the derived ``fiber_vt`` / ``fiber_hz`` channels (written in both fibre
      modes) argmaxed against the human class;
    * ``angle_deg_*`` -- only for a ``fiber_mode="direction"`` store: the **axial** angle
      between the predicted direction and the human class's in-plane direction
      (``t_v`` / ``t_h`` from the predicted ``grad sdf_in`` and the scroll axis, tsm.fiber).

    ``axis`` (an umbilicus JSON path or (N, 3) control points) makes "vertical" the **local**
    axis tangent (``tsm.labels.axis_tangent_field``, built per brick at the brick's absolute
    origin), exactly as ``infer.write_fiber_class_channels`` does; without it the historical
    constant (1, 0, 0) is used and a tilted scroll is scored against the wrong basis.
    """
    import torch

    from tsm.fiber import fiber_basis, sheet_normal
    from tsm.labels import axis_tangent_field

    ax_pts = None
    if axis is not None:
        from tsm.data import load_axis_spec

        ax_pts = load_axis_spec(axis)
    sdf_ch = next((k for k in ("sdf_in", "sdf_body", "d_face", "sdf") if pred.has(k)), "sdf")
    has_dir = all(pred.has(c) for c in ("fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"))
    n_hit = n_band = 0
    angs: list[np.ndarray] = []
    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi, 1)
        c = band.read(hlo, hhi, [band_channel])[0]
        human = (c == 1) | (c == 2)
        if not human[cs].any():
            continue
        u = pred.read(hlo, hhi, [sdf_ch, "fiber_vt", "fiber_hz"])
        data = u[0] != 0
        sel = human & data
        pv, ph = u[1].astype(np.float32), u[2].astype(np.float32)
        n_hit += int((((pv > ph) == (c == 2)) & sel)[cs].sum())
        n_band += int(sel[cs].sum())
        if has_dir:
            sdf = decode_sdf_u8(u[0], clip) * data
            # any non-zero gradient defines the basis direction (as in infer.write_fiber_class_channels)
            n, _ = sheet_normal(torch.from_numpy(sdf[None].astype(np.float32)), lo=1e-6, hi=float("inf"))
            ax_fld = None
            if ax_pts is not None:
                ax_fld = torch.from_numpy(axis_tangent_field(
                    ax_pts, hlo, [h - l for l, h in zip(hlo, hhi)], scale=1))
            tv, th, ok = fiber_basis(n, ax_fld)
            d = np.stack([decode_signed(pred.read(hlo, hhi, [k])[0])
                          for k in ("fiber_dz", "fiber_dy", "fiber_dx")]).astype(np.float32)
            dt = torch.from_numpy(d)
            dt = dt / dt.norm(dim=0, keepdim=True).clamp_min(1e-6)
            tgt = torch.where(torch.from_numpy((c == 2))[None], tv, th)
            cosv = (dt * tgt).sum(0).abs().clamp(0, 1).numpy()
            m = (sel & ok[0].numpy())[cs]
            if m.any():
                angs.append(np.degrees(np.arccos(cosv[cs][m])).astype(np.float32))
        log(f"fiber upstream brick {blo} -> {bhi} done")
    out: dict[str, Any] = {"store": band.path, "channel": band_channel, "teacher_independent": True,
                           "n_band": n_band, "class_acc": (n_hit / n_band) if n_band else None,
                           "axis_tangent": ax_pts is not None,
                           "axis": str(axis) if isinstance(axis, (str, os.PathLike)) else None}
    if has_dir:
        out.update(_pcts(np.concatenate(angs) if angs else np.zeros(0, np.float32), "angle_deg"))
    return out


# --------------------------------------------------------------------------- #
# upstream (teacher-independent) face metrics
# --------------------------------------------------------------------------- #
UPSTREAM_NEAR = 8.0     # only voxels within this many voxels of any upstream band are scored
UPSTREAM_DICE_TOL = 2   # "hit" radius of the Dice, in voxels (the band registration tolerance)


def upstream_faces_pass(pred: Store, rv: Store, lo, hi, budget, brick=(64, 512, 512),
                        near: float = UPSTREAM_NEAR, tol: int = UPSTREAM_DICE_TOL,
                        min_component: int = 32) -> dict[str, Any]:
    """Score the student's two faces against the **upstream** recto/verso bands.

    This is the first *teacher-independent* face metric in the pipeline: the reference is the
    Scroll-1 recto/verso mesh labels resampled onto our grid by ``dev/rectoverso_slab.py``
    (``tsm.xframe``), not the recto/lasagna teachers the labels were distilled from, and not the
    CT-derived face builder either.  A disagreement here is evidence about the student (or about
    the registration), never a tautology.

    Upstream classes: 1 recto, 2 verso, 3 contact.  Recto is the face toward the umbilicus =
    our IN face, verso our OUT face, and a contact voxel is on both (``tsm.rvfaces``).  Each band
    is 3-7 voxels thick on our grid, so it is thinned to its EDT-ridge medial sheet
    (:func:`tsm.rvfaces.thin_band`) before the comparison; the registration is good to ~1-2
    voxels, which is why the Dice counts a hit within ``tol`` (2) voxels rather than 1.

    Everything is restricted to voxels within ``near`` (8) voxels of *any* band: elsewhere the
    upstream labels say nothing, so a student surface there is neither right nor wrong.
    Reported per face: surface distances both ways (pred->band, band->pred) and the Dice.
    """
    from tsm.rvfaces import thin_band

    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    check_alloc((dz, dy, dx), np.bool_, budget)
    faces = ("in", "out")
    masks = {(w, f): np.zeros((dz, dy, dx), bool) for w in ("pred", "band") for f in faces}
    n_near = 0
    halo = int(max(near, tol, 4)) + 4
    st = ndi.generate_binary_structure(3, 3)

    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi, halo)
        c = rv.read(hlo, hhi, ["rectoverso"])[0]
        p = pred.read(hlo, hhi, ["surface_in1", "surface_out1"])
        bands = {"in": (c == 1) | (c == 3), "out": (c == 2) | (c == 3)}
        anyband = c > 0
        nearmask = (ndi.distance_transform_edt(~anyband) <= float(near)) if anyband.any() \
            else np.zeros(c.shape, bool)
        del c, anyband
        dst = _dest(blo, bhi, lo)
        n_near += int(nearmask[cs].sum())
        for i, name in enumerate(faces):
            masks[("band", name)][dst] = thin_band(bands[name], min_component)[cs]
            masks[("pred", name)][dst] = ((p[i] > 127) & nearmask)[cs]
        del p, bands, nearmask
        log(f"upstream brick {blo} -> {bhi} done")

    out: dict[str, Any] = {"store": rv.path, "near": float(near), "dice_tol": int(tol),
                           "n_near_band": n_near, "teacher_independent": True}
    for name in faces:
        a, b = masks[("pred", name)], masks[("band", name)]
        hit = int((a & ndi.binary_dilation(b, st, iterations=int(tol))).sum()) \
            + int((b & ndi.binary_dilation(a, st, iterations=int(tol))).sum())
        d = surface_distances(a, b)
        r: dict[str, Any] = {
            f"dice_within{int(tol)}": (hit / (int(a.sum()) + int(b.sum()))) if (a.sum() + b.sum()) else None,
            "n_pred": int(a.sum()), "n_band": int(b.sum()),
        }
        for side, key in (("a_to_b", "pred_to_band"), ("b_to_a", "band_to_pred"), ("symmetric", "symmetric")):
            for k in ("n", "median", "p90", "frac_gt3", "frac_gt5", "mean"):
                r[f"{key}_{k}"] = d[side].get(k)
        out[name] = r
    return out


def body_pred_mask(pred: Store, lo, hi, clip: float, valid_min: int = 128) -> np.ndarray:
    """The predicted **sheet body** of a block, for either store kind.

    Sides store: ``body > 127`` -- the dedicated body head's probability, thresholded at 0.5.
    Body store: ``sdf_body > 0`` (``tsm.data.body_sdf`` is positive strictly inside a sheet and
    0 on both faces *and* on a labelled contact plane, so touching sheets stay distinct
    components).  Two-face store: the same set written in the old parametrisation,
    ``sdf_in > 0 > sdf_out`` -- which is what lets the existing faces baselines be scored with
    the orientation-free metrics without retraining.  Both are restricted to ``valid >= 128``
    when the store carries a ``valid`` channel."""
    if pred.has("body"):
        need = ["body"]
    elif pred.has("sdf_body"):
        need = ["sdf_body"]
    elif pred.has("sdf_in") and pred.has("sdf_out"):
        need = ["sdf_in", "sdf_out"]
    else:
        raise ValueError(f"{pred.path} has none of body / sdf_body / sdf_in+sdf_out (has {pred.channels})")
    has_valid = pred.has("valid")
    u = pred.read(lo, hi, need + (["valid"] if has_valid else []))
    if need == ["body"]:
        m = u[0] > 127            # sides mode: the body head's own probability
    elif len(need) == 1:
        m = (u[0] != 0) & (decode_sdf_u8(u[0], clip) > 0.0)
    else:
        m = ((u[0] != 0) & (u[1] != 0) & (decode_sdf_u8(u[0], clip) > 0.0)
             & (decode_sdf_u8(u[1], clip) < 0.0))
    if has_valid:
        m &= u[-1] >= int(valid_min)
    return m


def body_surface_mask(pred: Store, lo, hi) -> np.ndarray:
    """The predicted thin **sheet-side** mask, unordered: ``surface_side1`` for a sides store
    (written by :func:`tsm.infer.extract_sides`; ``--rederive-sides`` refreshes it in place
    from ``d_face`` / ``body`` / ``valid`` before scoring, for stores written by an older rule),
    ``surface_body1`` for a body store,
    ``surface_in1 | surface_out1`` for a two-face store (the union forgets which side is which,
    which is exactly what the upstream recto/verso union is scored against)."""
    if pred.has("surface_side1"):
        return pred.read(lo, hi, ["surface_side1"])[0] > 127
    if pred.has("surface_body1"):
        return pred.read(lo, hi, ["surface_body1"])[0] > 127
    if pred.has("surface_in1") and pred.has("surface_out1"):
        p = pred.read(lo, hi, ["surface_in1", "surface_out1"])
        return (p[0] > 127) | (p[1] > 127)
    raise ValueError(f"{pred.path} has no surface_side1 / surface_body1 / surface_in1+surface_out1 "
                     f"(has {pred.channels})")


def upstream_body_pass(pred: Store, rv: Store, lo, hi, budget, brick=(64, 512, 512),
                       near: float = UPSTREAM_NEAR, tol: int = UPSTREAM_DICE_TOL,
                       min_component: int = 32) -> dict[str, Any]:
    """The **orientation-free** companion of :func:`upstream_faces_pass`: one unordered score.

    The reference is the thinned union of *all* upstream classes -- recto, verso and contact --
    so it says "this voxel is a side of a sheet" and never which side.  The prediction is
    ``surface_side1`` for a sides store, ``surface_body1`` for a body store and
    ``surface_in1 | surface_out1`` for a two-face store
    (:func:`body_surface_mask`), which gives the existing faces baselines an unordered number
    without retraining and makes the two recipes directly comparable.

    Because neither side of the comparison is named, the score is **invariant under a global
    swap of the two predicted faces**, while the per-face ``upstream_faces`` Dice collapses --
    that swap is exactly the failure mode the body target removes.

    Same restriction, tolerance and keys as :func:`upstream_faces_pass` (one block, not one per
    face): ``dice_within2``, the pred->band / band->pred / symmetric distances, ``n_pred`` and
    ``n_band``.
    """
    from tsm.rvfaces import thin_band

    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    check_alloc((dz, dy, dx), np.bool_, budget)
    a = np.zeros((dz, dy, dx), bool)   # prediction, restricted to the near-band region
    b = np.zeros((dz, dy, dx), bool)   # thinned upstream band (recto U verso U contact)
    n_near = 0
    halo = int(max(near, tol, 4)) + 4
    st = ndi.generate_binary_structure(3, 3)

    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi, halo)
        c = rv.read(hlo, hhi, ["rectoverso"])[0]
        anyband = c > 0
        nearmask = (ndi.distance_transform_edt(~anyband) <= float(near)) if anyband.any() \
            else np.zeros(c.shape, bool)
        dst = _dest(blo, bhi, lo)
        n_near += int(nearmask[cs].sum())
        b[dst] = thin_band(anyband, min_component)[cs]
        a[dst] = (body_surface_mask(pred, hlo, hhi) & nearmask)[cs]
        del c, anyband, nearmask
        log(f"upstream body brick {blo} -> {bhi} done")

    hit = int((a & ndi.binary_dilation(b, st, iterations=int(tol))).sum()) \
        + int((b & ndi.binary_dilation(a, st, iterations=int(tol))).sum())
    d = surface_distances(a, b)
    out: dict[str, Any] = {
        "store": rv.path, "near": float(near), "dice_tol": int(tol), "n_near_band": n_near,
        "teacher_independent": True, "unordered": True,
        "reference": "thin_band(rectoverso > 0): recto U verso U contact, unordered",
        "prediction": ("surface_side1" if pred.has("surface_side1") else
                       ("surface_body1" if pred.has("surface_body1") else "surface_in1 | surface_out1")),
        f"dice_within{int(tol)}": (hit / (int(a.sum()) + int(b.sum()))) if (a.sum() + b.sum()) else None,
        "n_pred": int(a.sum()), "n_band": int(b.sum()),
    }
    for side, key in (("a_to_b", "pred_to_band"), ("b_to_a", "band_to_pred"), ("symmetric", "symmetric")):
        for k in ("n", "median", "p90", "frac_gt3", "frac_gt5", "mean"):
            out[f"{key}_{k}"] = d[side].get(k)
    return out


# --------------------------------------------------------------------------- #
# upstream (teacher-independent) TOPOLOGY metrics
# --------------------------------------------------------------------------- #
UPSTREAM_MARGIN = 4      # extra voxels around the near band so components are not cut artificially
TOPO_DILATE = 2          # a student component "reaches" a band voxel within this many voxels
TOPO_COVER = 0.5         # fraction of a reference object inside a dilated student component -> covered
TOPO_PARTIAL = 0.2       # the weaker fraction that still counts as "one of the pieces"
#: a reference object / student component below this many voxels is not counted on either side.
#: It has to be *large*: the upstream bands are voxelised meshes and their thinned centre planes
#: shatter into >100k fragments on the eval region (see ``n_ref_thin_fragments``), so a small
#: threshold would count reference fragmentation as model merges.  A labelled sheet crossing a
#: 1024^2 face of the region is O(10^5) voxels, a voxelisation speck O(10).
TOPO_MIN_SIZE = 2000
TOPO_FRAG_MIN = 32       # only for the reference-fragmentation sanity line


def _reference_objects(band: np.ndarray, core: np.ndarray | None, st: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(flat indices of the band voxels, object id per band voxel, object sizes, n_objects)``.

    The reference objects are the 26-components of the **unthinned** band.  When ``core`` is
    given (the band minus the label "contact" voxels, see :func:`upstream_topology_pass`) the
    components are built on ``core`` instead, so two sheets that touch only through a contact
    plane stay two objects; a contact voxel then belongs to no object (id 0) -- it is still part
    of the attribution mask, it just does not enter any object's coverage denominator -- except
    where a whole band component is contact, which becomes an object of its own.

    Everything is returned as 1-D arrays over the band voxels so the caller can free the labelled
    volumes immediately: only one int32 labelling is ever live here.
    """
    band = np.asarray(band, dtype=bool)
    idx = np.flatnonzero(band.reshape(-1))
    if idx.size == 0:
        return idx, np.zeros(0, np.int32), np.zeros(1, np.int64), 0
    a_all = None
    if core is not None:
        lab, n_all = ndi.label(band, st)
        a_all = lab.reshape(-1)[idx]           # int32 labels: widened only for the pair key
        del lab
    lab, n = ndi.label(band if core is None else core, st)
    oi = lab.reshape(-1)[idx]
    del lab
    if a_all is not None and n_all:
        has_core = np.zeros(n_all + 1, bool)
        has_core[np.unique(a_all[oi > 0])] = True
        has_core[0] = True
        pure = np.flatnonzero(~has_core)          # band components made of contact only
        if pure.size:
            newid = np.zeros(n_all + 1, oi.dtype)
            newid[pure] = n + 1 + np.arange(pure.size)
            m = oi == 0
            oi[m] = newid[a_all[m]]
            n += int(pure.size)
            del m, newid
        del has_core, pure
    del a_all
    size = np.bincount(oi[oi > 0], minlength=n + 1) if n else np.zeros(1, np.int64)
    return idx, oi, size, int(n)


def _max_label_at(lab: np.ndarray, idx: np.ndarray, dilate: int, slab: int = 32) -> np.ndarray:
    """``maximum_filter(lab, 2*dilate+1)`` sampled at the flat indices ``idx``, z slab by z slab.

    The max-label dilation is only ever read at the band voxels, and a whole second int32 volume
    is the single largest allocation of this pass, so it is never materialised: each z slab is
    filtered with a ``dilate``-voxel halo (the filter is local, so the slab result is exact for
    its core) and immediately gathered.  ``idx`` must be ascending, as ``np.flatnonzero`` returns
    it, so the slab's share of it is one ``searchsorted`` pair.
    """
    out = np.zeros(idx.size, lab.dtype)
    if idx.size == 0:
        return out
    nz, plane, d = int(lab.shape[0]), int(lab.shape[1]) * int(lab.shape[2]), int(dilate)
    for z0 in range(0, nz, int(slab)):
        z1 = min(z0 + int(slab), nz)
        a, b = np.searchsorted(idx, [z0 * plane, z1 * plane])
        if b <= a:
            continue
        h0, h1 = max(0, z0 - d), min(nz, z1 + d)
        m = ndi.maximum_filter(lab[h0:h1], size=2 * d + 1)
        out[a:b] = m.reshape(-1)[idx[a:b] - h0 * plane]
        del m
    return out


def face_topology(band: np.ndarray, stud: np.ndarray, near_mask: np.ndarray | None = None,
                  thin: np.ndarray | None = None, core: np.ndarray | None = None,
                  dilate: int = TOPO_DILATE, cover_frac: float = TOPO_COVER,
                  partial_frac: float = TOPO_PARTIAL, min_size: int = TOPO_MIN_SIZE,
                  tol: int = UPSTREAM_DICE_TOL, frag_min_size: int = TOPO_FRAG_MIN) -> dict[str, Any]:
    """Merge / break / missed / spurious counts between one upstream band and one student face.

    ``band`` is the **unthinned** reference band (the recto/verso label band as registered, 3-7
    voxels thick), ``thin`` its ~1-voxel centre planes (:func:`tsm.rvfaces.thin_band`, default:
    ``band`` itself) and ``stud`` the student's binary face, already restricted to the near-band
    region plus a margin.  The two roles are deliberately different:

    * the **objects** counted below are the 26-components of ``band`` of at least ``min_size``
      voxels (:func:`_reference_objects`, ``core`` splits them at contact planes).  The thinned
      centre planes are *not* usable as objects: thinning a voxelised mesh band shatters it --
      108561 fragments for 2323 sheet-sized ones on the first 256x1024x1024 eval region -- so
      counting them makes one honest student sheet "merge" hundreds of reference fragments.
      ``n_ref_thin_fragments`` reports that shattering next to ``n_ref_objects`` so the reader
      can see which one the counts are made of.
    * the **distances and the Dice** still use ``thin``, exactly as ``upstream_faces_pass`` does,
      so ``dice_within2`` here stays comparable with the ``upstream_faces`` section.

    Every band voxel is attributed to the student component whose ``dilate``-voxel dilation
    contains it -- implemented as a max-label dilation of the student labelling
    (``maximum_filter`` over a ``2*dilate+1`` box = ``binary_dilation`` with the 26-neighbourhood,
    ``dilate`` iterations), so a band voxel in the overlap of two dilated student components is
    attributed to one of them (the larger label) rather than to both.  That makes the per-object
    coverage fractions a partition, which is what the counts below assume; the alternative
    (dilating every component separately) costs one pass per component and cannot change a merge
    or a break unless two predictions overlap to within ``dilate`` voxels, in which case they are
    hardly two sheets.

    With ``cov(i, j)`` the fraction of object O_i attributed to student component S_j, and both
    sides restricted to components of at least ``min_size`` voxels:

    * **merge**  -- a student component with ``cov >= cover_frac`` for two or more objects: one
      predicted sheet spans several labelled sheets.  ``merged_extra`` is ``sum(k - 1)`` over
      those, i.e. how many objects too many were swallowed.
    * **break**  -- an object with ``cov >= partial_frac`` from two or more student components:
      one labelled sheet split across several predictions (``split_extra`` = sum(k - 1)).
    * **missed** -- an object no student component reaches ``partial_frac`` of.
    * **spurious** -- a student component, inside the near-band region, that no band voxel at
      all is attributed to.

    ``near_mask`` (default: everything) marks the strict near-band region: it gates the spurious
    count and the Dice, while the components themselves are built on the wider ``stud`` mask.

    Memory: only one int32 labelling of the region is ever live (the max-label dilation is done
    and gathered one z slab at a time, :func:`_max_label_at`), and ``band`` / ``core`` are dropped
    as soon as the per-band-voxel arrays exist, so a caller that builds them inside the call
    expression pays for them only until then.
    """
    st = ndi.generate_binary_structure(3, 3)
    band = np.asarray(band, dtype=bool)
    stud = np.asarray(stud, dtype=bool)
    thin = band if thin is None else np.asarray(thin, dtype=bool)
    core = None if core is None else np.asarray(core, dtype=bool)
    nearm = np.ones(band.shape, bool) if near_mask is None else np.asarray(near_mask, dtype=bool)

    # --- how fragmented is the reference itself? (sanity line, thinned centre planes) ---
    lab_t, n_thin = ndi.label(thin, st)
    t_size = np.bincount(lab_t[thin], minlength=n_thin + 1) if n_thin else np.zeros(1, np.int64)
    del lab_t
    n_thin_big = int((t_size[1:] >= int(frag_min_size)).sum()) if n_thin else 0
    del t_size

    # --- reference objects on the UNTHINNED band ---
    idx, oi, o_size, n_obj_all = _reference_objects(band, core, st)
    n_ref_vox, split_at_contact = int(idx.size), core is not None
    del band, core          # only the 1-D per-band-voxel arrays are needed from here on
    big_o = np.zeros(n_obj_all + 1, bool)
    big_o[1:] = o_size[1:] >= int(min_size)
    n_obj = int(big_o.sum())

    # --- student components, attributed to the band voxels by a max-label dilation ---
    lab_s, ns = ndi.label(stud, st)
    s_size = np.bincount(lab_s[stud], minlength=ns + 1) if ns else np.zeros(1, np.int64)
    s_near = np.zeros(ns + 1, bool)
    if ns:
        s_near[np.unique(lab_s[stud & nearm])] = True
    s_near[0] = False
    sj = _max_label_at(lab_s, idx, dilate) if ns else np.zeros(idx.size, np.int32)
    del lab_s
    big_s = np.zeros(ns + 1, bool)
    big_s[1:] = s_size[1:] >= int(min_size)

    # a student component with no band voxel at all under it is spurious -- counted over every
    # band voxel, contact voxels included, so a prediction sitting on a contact plane is not
    # called spurious just because that plane belongs to no single object
    hits_per_s = np.bincount(sj[sj > 0], minlength=ns + 1) if sj.size else np.zeros(ns + 1, np.int64)

    sel = oi > 0
    if sel.any():
        uk, cnt = np.unique(oi[sel].astype(np.int64) * (ns + 1) + sj[sel], return_counts=True)
        b_idx, s_idx = uk // (ns + 1), uk % (ns + 1)
    else:
        cnt = b_idx = s_idx = np.zeros(0, np.int64)
    del sel, oi, sj, idx
    frac = cnt / np.maximum(o_size[b_idx], 1) if cnt.size else np.zeros(0, np.float64)
    real = (s_idx > 0) & big_s[s_idx] & big_o[b_idx]   # both sides at least min_size

    k_per_s = np.bincount(s_idx[real & (frac >= float(cover_frac))], minlength=ns + 1)
    merges = int((k_per_s >= 2).sum())
    merged_extra = int((k_per_s[k_per_s >= 2] - 1).sum())
    k_per_b = np.bincount(b_idx[real & (frac >= float(partial_frac))], minlength=n_obj_all + 1)
    breaks = int((k_per_b >= 2).sum())
    split_extra = int((k_per_b[k_per_b >= 2] - 1).sum())
    missed = int((big_o & (k_per_b == 0)).sum())
    spurious = int((big_s & s_near & (hits_per_s <= 0)).sum())

    sn = stud & nearm
    n_sn, n_b = int(sn.sum()), int(thin.sum())
    hit = (int((sn & ndi.binary_dilation(thin, st, iterations=int(tol))).sum())
           + int((thin & ndi.binary_dilation(sn, st, iterations=int(tol))).sum())) if (n_sn or n_b) else 0
    per100 = (lambda v: (100.0 * v / n_obj) if n_obj else None)
    return {
        # what the reference actually is (sanity line): objects on the unthinned band, before and
        # after the min_size cut, against the fragment count of its thinned centre planes
        "n_ref_objects": n_obj,
        "n_ref_objects_all": int(n_obj_all),
        "n_ref_thin_fragments": int(n_thin),
        "n_ref_thin_fragments_big": n_thin_big,
        "contact_split": bool(split_at_contact),
        "n_student_components": int(ns),
        "n_student_components_min_size": int((big_s & s_near).sum()),
        "n_ref_voxels": n_ref_vox, "n_band_voxels": n_b, "n_student_voxels": n_sn,
        "merges": merges, "merged_extra": merged_extra,
        "breaks": breaks, "split_extra": split_extra,
        "missed": missed, "spurious": spurious,
        "merges_per_100": per100(merges),
        "breaks_per_100": per100(breaks),
        "missed_per_100": per100(missed),
        "spurious_per_100": per100(spurious),
        "missed_frac": (missed / n_obj) if n_obj else None,
        f"dice_within{int(tol)}": (hit / (n_sn + n_b)) if (n_sn + n_b) else None,
    }


def inout_cross_links(s_in: np.ndarray, s_out: np.ndarray, dilate: int = 1) -> dict[str, Any]:
    """How often one student IN-face component touches (within ``dilate``) several OUT-face ones.

    A sheet's two faces should pair up one to one, so an in-face component adjacent to two
    different out-face components means the two predicted faces disagree about where the sheets
    separate there.  Adjacency is enumerated exactly, over the ``(2*dilate+1)**3`` offsets of the
    Chebyshev ball -- one gather of the out labelling per offset at the in-face voxels, so a
    voxel adjacent to several out components contributes all of them (unlike the max-label
    attribution of :func:`face_topology`, which needs a partition of the band voxels)."""
    st = ndi.generate_binary_structure(3, 3)
    s_in = np.asarray(s_in, dtype=bool)
    s_out = np.asarray(s_out, dtype=bool)
    lab_i, ni = ndi.label(s_in, st)
    lab_o, no = ndi.label(s_out, st)
    k = np.zeros(ni + 1, np.int64)
    if ni and no:
        idx = np.nonzero(s_in)
        li = lab_i[idx].astype(np.int64)
        shape = s_in.shape
        d = int(dilate)
        pairs: list[np.ndarray] = []
        for dz in range(-d, d + 1):
            for dy in range(-d, d + 1):
                for dx in range(-d, d + 1):
                    zz, yy, xx = idx[0] + dz, idx[1] + dy, idx[2] + dx
                    ok = ((zz >= 0) & (zz < shape[0]) & (yy >= 0) & (yy < shape[1])
                          & (xx >= 0) & (xx < shape[2]))
                    lo_ = lab_o[zz[ok], yy[ok], xx[ok]].astype(np.int64)
                    sel = lo_ > 0
                    if sel.any():
                        pairs.append(np.unique(li[ok][sel] * (no + 1) + lo_[sel]))
                    del zz, yy, xx, ok, lo_, sel
        if pairs:
            uk = np.unique(np.concatenate(pairs))
            k = np.bincount(uk // (no + 1), minlength=ni + 1)
        del idx, li, pairs
    del lab_i, lab_o
    cross = int((k >= 2).sum())
    return {"n_in_components": int(ni), "n_out_components": int(no), "cross_links": cross,
            "cross_links_per_100": (100.0 * cross / ni) if ni else None}


def upstream_topology_pass(pred: Store, rv: Store, lo, hi, budget, brick=(64, 512, 512),
                           near: float = UPSTREAM_NEAR, margin: int = UPSTREAM_MARGIN,
                           tol: int = UPSTREAM_DICE_TOL, dilate: int = TOPO_DILATE,
                           cover_frac: float = TOPO_COVER, partial_frac: float = TOPO_PARTIAL,
                           min_component: int = 32, min_size: int = TOPO_MIN_SIZE,
                           split_contact: bool = True, clip: float = CLIP) -> dict[str, Any]:
    """Connected-component (Betti-0) agreement of the student's faces with the **upstream** bands.

    Companion of :func:`upstream_faces_pass`, on the same recto/verso bands and the same
    near-band restriction, but counting *sheets* instead of voxels: how many labelled sheets a
    prediction welds together (merge), how many predictions one labelled sheet falls apart into
    (break), which labelled sheets are absent and which predicted sheets have no labelled sheet
    under them.  These numbers are reported in their own ``upstream_topology`` section and are
    never mixed into the voxel-wise ``upstream_faces`` block: a good Dice with a bad merge count
    is exactly the failure mode a segmentation-quality number is meant to expose.

    **The reference objects are the unthinned bands.**  ``upstream_faces_pass`` compares the
    student with the thinned centre planes, which is right for a distance and for the Dice but
    wrong for counting sheets: thinning a voxelised mesh band shatters it into fragments (108561
    of them, 2323 of at least 32 voxels, for the in face of the first 256x1024x1024 eval region),
    so every honest student sheet "merges" hundreds of them.  The objects are therefore the
    26-components of the band *as registered*, kept above ``min_size`` (2000) voxels; the thinned
    planes are still what the distances and ``dice_within2`` are measured on, so that number stays
    comparable with the ``upstream_faces`` section.  ``n_ref_thin_fragments`` is reported next to
    ``n_ref_objects`` as the sanity line for exactly this.

    **Contact.**  Class 3 of the ``rectoverso`` channel is *contact*: the recto of one sheet
    touching the verso of the next (``tsm.rvfaces``), so it belongs to both bands and would weld
    two neighbouring sheets into one reference object.  When the region contains any class-3
    voxel the objects are built on the band minus the contact voxels (``split_contact``, on by
    default) and ``contact_split`` is reported true; the contact voxels stay in the attribution
    mask.  There is no separate ``rv_class`` channel in ``rectoverso.zarr`` -- the contact class
    is carried by the ``rectoverso`` channel itself (channels: ``rectoverso``, ``hzvt``).

    **The unordered ``body`` entry.**  Besides the per-face ``in`` / ``out`` entries (two-face
    stores only) there is always a ``body`` entry: reference objects = the 26-components of
    ``rectoverso > 0`` (recto U verso U contact, split at the contact planes exactly as the
    per-face entries are), student objects = the 26-components of the predicted sheet *body*
    (:func:`body_pred_mask`).  It is the sheet-count companion of :func:`upstream_body_pass` and
    the only topology number a body store has.

    The student mask is taken within ``near + margin`` voxels of a band so that a component is
    not cut in half by the restriction itself; the strict ``near`` region still gates which
    student components can be called spurious, and the Dice repeated here is the
    :func:`upstream_faces_pass` one (student restricted to ``near``).

    Memory: the whole region is held as five bool masks (two thinned bands, two student faces,
    the near region) plus the uint8 upstream class volume (from which the unthinned band and its
    contact-free core are rebuilt one face at a time, and dropped again inside
    :func:`face_topology` as soon as the per-band-voxel arrays exist) and, one face at a time, a
    single int32 labelling (:func:`face_topology` gathers the max-label dilation z slab by z slab
    instead of materialising a second one) -- about 3.6 GB for the 256x1024x1024 eval region, so
    no z-slab tiling of the components themselves (which would cut them at the seams and have to
    stitch them back together) is needed.
    """
    from tsm.rvfaces import RV_CONTACT, RV_RECTO, RV_VERSO, thin_band

    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    # a body store has no named faces; a two-face store gets the per-face entries AND the
    # unordered `body` one, so an existing baseline can be read on both scales at once
    faces = () if (pred.has("sdf_body") or pred.has("d_face")) else ("in", "out")
    parts = (*faces, "body")
    check_alloc((dz, dy, dx), np.bool_, budget)
    check_alloc((dz, dy, dx), np.uint8, budget)
    masks = {(w, f): np.zeros((dz, dy, dx), bool) for w in ("pred", "band") for f in parts}
    nearm = np.zeros((dz, dy, dx), bool)
    rvc = np.zeros((dz, dy, dx), np.uint8)   # the upstream class, kept for the unthinned objects
    wide = float(near) + float(margin)
    halo = int(max(wide, tol, 4)) + 4

    for blo, bhi in _bricks(lo, hi, brick):
        hlo, hhi, cs = _halo(blo, bhi, lo, hi, halo)
        c = rv.read(hlo, hhi, ["rectoverso"])[0]
        bands = {"in": (c == RV_RECTO) | (c == RV_CONTACT), "out": (c == RV_VERSO) | (c == RV_CONTACT),
                 "body": c > 0}
        anyband = c > 0
        d = ndi.distance_transform_edt(~anyband) if anyband.any() \
            else np.full(c.shape, np.inf, np.float32)
        del anyband
        dst = _dest(blo, bhi, lo)
        nearm[dst] = (d <= float(near))[cs]
        rvc[dst] = c[cs]
        if faces:
            p = pred.read(hlo, hhi, ["surface_in1", "surface_out1"])
            for i, name in enumerate(faces):
                masks[("pred", name)][dst] = ((p[i] > 127) & (d <= wide))[cs]
            del p
        for name in parts:
            masks[("band", name)][dst] = thin_band(bands[name], min_component)[cs]
        # the unordered entry counts SHEETS, so its student objects are the predicted bodies
        # (sdf_body > 0, or sdf_in > 0 > sdf_out), not the thin faces
        masks[("pred", "body")][dst] = (body_pred_mask(pred, hlo, hhi, clip) & (d <= wide))[cs]
        del c, bands, d
        log(f"upstream topology brick {blo} -> {bhi} done")

    contact = bool(split_contact) and bool((rvc == RV_CONTACT).any())
    out: dict[str, Any] = {
        "store": rv.path, "near": float(near), "margin": float(margin), "dice_tol": int(tol),
        "dilate": int(dilate), "cover_frac": float(cover_frac), "partial_frac": float(partial_frac),
        "min_size": int(min_size), "frag_min_size": int(TOPO_FRAG_MIN), "connectivity": 26,
        "reference": "unthinned upstream band (26-components >= min_size voxels)",
        "contact_split": contact,
        "teacher_independent": True,
        "caption": (f"reference objects = 26-components of the UNTHINNED upstream band, "
                    f">= {int(min_size)} voxels"
                    + (", split where the two bands meet at a label contact plane (class 3)"
                       if contact else "; no contact (class 3) voxels in this region, so no split")
                    + f"; merge = one predicted sheet (>= {int(min_size)} voxels) covers "
                    f">= {cover_frac:g} of two or more reference objects; break = one reference "
                    f"object covered (>= {partial_frac:g} each) by two or more predicted sheets; "
                    f"n_ref_thin_fragments is what the *thinned* centre planes shatter into -- "
                    f"only the distances and dice_within{int(tol)} are measured on those"),
    }
    for name in parts:
        # the unthinned band and its contact-free core are built here and referenced nowhere else,
        # so face_topology can (and does) free them as soon as it has the band-voxel arrays
        if name == "body":
            band_m = rvc > 0
            core_m = ((rvc == RV_RECTO) | (rvc == RV_VERSO)) if contact else None
        else:
            cls = RV_RECTO if name == "in" else RV_VERSO
            band_m = (rvc == cls) | (rvc == RV_CONTACT)
            core_m = (rvc == cls) if contact else None
        out[name] = face_topology(band_m, masks[("pred", name)], nearm,
                                  thin=masks[("band", name)], core=core_m,
                                  dilate=dilate, cover_frac=cover_frac, partial_frac=partial_frac,
                                  min_size=min_size, tol=tol)
        del band_m, core_m
        log(f"upstream topology {name}: {out[name]['n_ref_objects']} reference objects "
            f"({out[name]['n_ref_objects_all']} before the min_size cut, "
            f"{out[name]['n_ref_thin_fragments']} thinned fragments), "
            f"{out[name]['merges']} merges, {out[name]['breaks']} breaks, "
            f"{out[name]['missed']} missed, {out[name]['spurious']} spurious")
    del nearm, rvc
    if faces:  # a body store has no two faces to cross-link
        out["inout_cross_links"] = inout_cross_links(masks[("pred", "in")], masks[("pred", "out")])
    return out


# --------------------------------------------------------------------------- #
# gallery (PIL only)
# --------------------------------------------------------------------------- #
def _g(a: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(a, np.uint8)] * 3, -1)


def _label(im: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(np.ascontiguousarray(im))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return np.asarray(pil)


def _sdf_rgb(sdf: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Blue (negative, inside) -> white (zero) -> red (positive); black where there is no data."""
    v = np.clip(np.asarray(sdf, np.float32) / max(1e-6, float(np.abs(sdf).max())), -1, 1)
    r = np.clip(255 * (1 + np.minimum(v, 0)), 0, 255)
    b = np.clip(255 * (1 - np.maximum(v, 0)), 0, 255)
    g = np.clip(255 * (1 - np.abs(v)), 0, 255)
    rgb = np.stack([r, g, b], -1).astype(np.uint8)
    rgb[~data] = 0
    rgb[np.abs(sdf) <= 0.75] = np.array([255, 255, 0], np.uint8)  # zero line
    return rgb


def _normal_rgb(n: np.ndarray) -> np.ndarray:
    """(3, H, W) unit vector -> |components| as RGB."""
    return np.clip(np.abs(np.asarray(n, np.float32)) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def _overlay(ct: np.ndarray, layers) -> np.ndarray:
    im = _g(np.clip(ct.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)).copy()
    for mask, col in layers:
        if mask is not None and mask.any():
            im[mask] = (0.3 * im[mask] + 0.7 * np.array(col, np.float32)).astype(np.uint8)
    return im


def _grid(panels: list[np.ndarray], ncol: int) -> np.ndarray:
    rows = []
    for i in range(0, len(panels), ncol):
        row = panels[i:i + ncol]
        h = max(p.shape[0] for p in row)
        row = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0))) for p in row]
        rows.append(np.concatenate(row, 1))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(rows, 0)


def gallery(ed: str, cfg, pred: Store, stores: dict[str, Store | None], lo, hi, clip: float,
            surf: np.ndarray, n_slices: int = 3, max_px: int = 1024,
            faces_labels: Store | None = None, medial: np.ndarray | None = None) -> list[str]:
    from PIL import Image

    dz, dy, dx = (h - l for l, h in zip(lo, hi))
    step = max(1, int(math.ceil(max(dy, dx) / max_px)))
    ys, xs = slice(None, None, step), slice(None, None, step)
    try:
        ct_reader = open_ct(cfg)
    except Exception as exc:  # pragma: no cover - network
        log(f"CT unavailable ({exc}); gallery panels without CT")
        ct_reader = None
    out: list[str] = []
    recto, ink_t, las, _m7 = (stores.get(k) for k in ("recto", "ink", "lasagna", "m7_l2"))

    def panels_for(sl_z: int | None, x_cut: int | None):
        """One z slice (sl_z global) or one yz cut at global x_cut."""
        halo = 8  # the recto medial surface is a 3D operator: keep a slab around the cut plane
        if sl_z is not None:
            blo, bhi = [sl_z, lo[1], lo[2]], [sl_z + 1, hi[1], hi[2]]
            hlo = [max(lo[0], sl_z - halo), lo[1], lo[2]]
            hhi = [min(hi[0], sl_z + halo + 1), hi[1], hi[2]]
            take = lambda a: a[..., 0, ys, xs]  # noqa: E731
            take_h = lambda a: a[..., sl_z - hlo[0], ys, xs]  # noqa: E731
            ct = (ct_reader.read(sl_z, sl_z + 1, lo[1], hi[1], lo[2], hi[2])[0, ys, xs]
                  if ct_reader is not None else np.zeros((dy, dx), np.uint8)[ys, xs])
            tag = f"z={sl_z}"
        else:
            blo, bhi = [lo[0], lo[1], x_cut], [hi[0], hi[1], x_cut + 1]
            hlo = [lo[0], lo[1], max(lo[2], x_cut - halo)]
            hhi = [hi[0], hi[1], min(hi[2], x_cut + halo + 1)]
            take = lambda a: a[..., :, ys, 0]  # noqa: E731
            take_h = lambda a: a[..., :, ys, x_cut - hlo[2]]  # noqa: E731
            ct = (ct_reader.read(lo[0], hi[0], lo[1], hi[1], x_cut, x_cut + 1)[:, ys, 0]
                  if ct_reader is not None else np.zeros((dz, dy), np.uint8)[:, ys])
            tag = f"yz x={x_cut}"
        p = decode_pred(pred.read(blo, bhi), clip, pred.channels)
        p = {k: take(v) for k, v in p.items()}
        rmed = None
        if recto is not None:
            rblk = recto.read(hlo, hhi)[0]
            r = take_h(rblk)
            rmed = take_h(medial_surface(_drop_small_components(rblk >= 128, MIN_COMPONENT)))
            del rblk
        else:
            r = None
        ik = take(ink_t.read(blo, bhi)[0]) if ink_t is not None else None
        if las is not None:
            l8 = las.read([v // 4 for v in blo], [-(-v // 4) for v in bhi]).astype(np.float32) / 255.0
            l8 = np.repeat(np.repeat(np.repeat(l8, 4, 1), 4, 2), 4, 3)
            off = [v - (v // 4) * 4 for v in blo]  # the level-2 block starts at a multiple of 4
            l8 = l8[:, off[0]:off[0] + bhi[0] - blo[0], off[1]:off[1] + bhi[1] - blo[1],
                    off[2]:off[2] + bhi[2] - blo[2]]
            lcos = take(l8[0])
            lnorm = take(decode_lasagna_normal(l8))
        else:
            lcos = lnorm = None
        cut = (lambda a: a[sl_z - lo[0], ys, xs]) if sl_z is not None else (lambda a: a[:, ys, x_cut - lo[2]])
        smed = cut(medial) if medial is not None else cut(surf)
        med_tag = "medial-from-faces red / recto medial green" if medial is not None \
            else ("surface_body1 red / recto medial green" if "sdf_body" in p
                  else ("surface_side1 red / recto medial green" if "d_face" in p
                        else "surface1 red / recto medial green"))
        ps = [
            _label(_g(np.clip(ct.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)), f"CT {tag}"),
            _label(_g(r if r is not None else np.zeros_like(ct)), "recto teacher"),
            _label(_sdf_rgb(p.get("sdf", p.get("sdf_body", p.get("d_face"))), p["data"]),
                   "student sdf_body (yellow = 0)" if "sdf_body" in p
                   else ("student d_face (yellow = 0)" if "d_face" in p else "student sdf (yellow = 0)")),
            _label(_overlay(ct, [(smed, (255, 40, 40)), (rmed, (40, 255, 40))]), med_tag),
            _label(_g(ik if ik is not None else np.zeros_like(ct)), "ink teacher"),
            _label(_g((p["ink"] * 255).astype(np.uint8)), "student ink"),
            _label(_g(np.clip(lcos * 255, 0, 255).astype(np.uint8)) if lcos is not None else _g(np.zeros_like(ct)),
                   "lasagna cos (x4)"),
            _label(_g(((p["cos"] * 0.5 + 0.5) * 255).astype(np.uint8)), "student cos"),
            _label(_normal_rgb(lnorm) if lnorm is not None else _g(np.zeros_like(ct)), "lasagna normal rgb"),
            _label(_normal_rgb(np.stack([p["nz"], p["ny"], p["nx"]])), "student normal rgb"),
            _label(_g((p["valid"] * 255).astype(np.uint8)), "student valid"),
            _label(_g((p["conf"] * 255).astype(np.uint8)), "student conf"),
        ]
        if faces_labels is not None:
            from tsm.infer import _any6

            f = faces_labels.read(hlo, hhi, ["sdf_in", "sdf_out", "faces_valid"])
            sup = f[2] == 1
            lay = []
            for i, col in ((0, (40, 255, 40)), (1, (40, 120, 255))):  # label in green, out blue
                d = f[i] != 0
                sd = decode_sdf_u8(f[i], clip) * d
                lay.append((take_h(d & (sd <= 0.0) & _any6(d & (sd > 0.0)) & sup), col))
            pf = pred.read(hlo, hhi, ["surface_in1", "surface_out1"])
            lay += [(take_h(pf[0] > 127), (255, 40, 40)), (take_h(pf[1] > 127), (255, 220, 40))]
            ps.append(_label(_overlay(ct, lay), "faces: label in/out g/b, pred in/out r/y"))
            del f, sup, pf
        return ps

    for i in range(n_slices):
        z = lo[0] + int(dz * (i + 1) / (n_slices + 1))
        path = os.path.join(ed, f"gallery_z{z}.png")
        Image.fromarray(_grid(panels_for(z, None), 4)).save(path)
        out.append(path)
        log(f"gallery {path}")
    xm = lo[2] + dx // 2
    path = os.path.join(ed, "gallery_yz.png")
    Image.fromarray(_grid(panels_for(None, xm), 4)).save(path)
    out.append(path)
    log(f"gallery {path}")
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def to_markdown(m: dict[str, Any]) -> str:
    L = ["# Student vs teachers/labels on the eval region", "",
         f"region start {m['region']['start_zyx']} size {m['region']['size_zyx']} "
         f"(clip {m['clip']}, {m['seconds']:.0f}s)", ""]
    for section in ("surface", "faces", "body", "upstream_faces", "upstream_body",
                    "surface_inface_vs_recto", "ink", "fiber", "winding", "winding_fine"):
        if section not in m:
            continue
        L += [f"## {section}", "", "| metric | value |", "|---|---|"]
        for k, v in m[section].items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    L.append(f"| {k}.{k2} | {_fmt(v2)} |")
            else:
                L.append(f"| {k} | {_fmt(v)} |")
        L.append("")
    tp = m.get("upstream_topology") or {}
    # one `body` column for a body store, in / out / body for a two-face one
    cols = [c for c in ("in", "out", "body") if isinstance(tp.get(c), dict)]
    if tp and cols:
        # kept out of the section loop above on purpose: these are sheet counts, never to be
        # averaged into or read alongside the voxel-wise upstream_faces numbers
        L += ["## Topology vs upstream bands", "",
              tp.get("caption", "merge = one predicted sheet spans two labelled sheets; "
                                "break = one labelled sheet split across several predictions"), "",
              "| metric | " + " | ".join(c for c in cols) + " |",
              "|---|" + "---|" * len(cols)]
        keys = list(tp.get(cols[0], {})) if cols else []
        for k in keys:
            L.append(f"| {k} | " + " | ".join(_fmt(tp.get(c, {}).get(k)) for c in cols) + " |")
        L.append("")
        cl = tp.get("inout_cross_links") or {}
        if cl:
            L += ["| student in-vs-out | value |", "|---|---|"] + \
                 [f"| inout_cross_links.{k} | {_fmt(v)} |" for k, v in cl.items()] + [""]
    ss = m.get("surface1_stats") or {}
    L += ["## surface1 (central box) connectivity", "", "| metric | value |", "|---|---|",
          f"| n_voxels | {_fmt(ss.get('n_voxels'))} |",
          f"| thickness median | {_fmt((ss.get('thickness') or {}).get('median'))} |",
          f"| deg6_mean | {_fmt(ss.get('deg6_mean'))} |",
          f"| frac_deg6_ge3 | {_fmt(ss.get('frac_deg6_ge3'))} |",
          f"| components_26 (>= {MIN_COMPONENT}) | {_fmt((ss.get('components_26') or {}).get('n_components_min_size'))} |",
          f"| components_6 (>= {MIN_COMPONENT}) | {_fmt((ss.get('components_6') or {}).get('n_components_min_size'))} |", ""]
    if m.get("gallery"):
        L += ["## gallery", ""] + [f"- {p}" for p in m["gallery"]] + [""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def resolve_axis(cfg, explicit: str | None) -> str | None:
    """``--axis``, else ``extra.labels.axis_path``, else ``extra.train.axis_path``, else
    :data:`tsm.labels.DEFAULT_AXIS`; ``None`` when nothing exists on disk (legacy constant axis).

    ``--axis none`` forces the constant (1, 0, 0) "vertical" of every run before 2026-09-12."""
    if explicit is not None and str(explicit).lower() in ("none", "off", ""):
        return None
    cand = explicit
    if cand is None:
        extra = getattr(cfg, "extra", None) or {}
        for sect in ("labels", "train"):
            v = (extra.get(sect) or {}).get("axis_path")
            if v:
                cand = v
                break
    if cand is None:
        cand = DEFAULT_AXIS
    cand = os.path.expanduser(str(cand))
    if not os.path.exists(cand):
        log(f"WARNING axis {cand} not found; the fibre 'vertical' stays the constant (1, 0, 0)")
        return None
    return cand


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--brick", default="64,512,512", help="brick z,y,x (4-aligned)")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--faces-labels", default=None,
                    help="fine.zarr of a faces label store (default: <out_dir>/labels/fine.zarr, "
                         "else <out_dir>_labels/labels/fine.zarr, else no faces label metrics)")
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--rederive-sides", action="store_true",
                    help="recompute surface_side1 in place in the prediction store (sides mode) "
                         "from d_face/body/valid with tsm.infer.extract_sides before scoring")
    ap.add_argument("--rectoverso", default=None,
                    help="rectoverso.zarr (dev/rectoverso_slab.py) -> the teacher-independent "
                         "`upstream_faces` / `upstream_body` blocks: the student's surfaces vs "
                         "the upstream bands")
    ap.add_argument("--axis", default=None,
                    help="umbilicus JSON for the fibre 'vertical' direction (default: "
                         "extra.labels.axis_path, else extra.train.axis_path, else "
                         "tsm.labels.DEFAULT_AXIS); 'none' forces the legacy constant (1, 0, 0)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    cfg = load_config(a.config)
    out = os.path.expanduser(cfg.out_dir)
    ed = os.path.join(out, "eval")
    os.makedirs(ed, exist_ok=True)

    pred_path = os.path.join(out, "student", "pred.zarr")
    pred = open_store(pred_path)
    if pred is None:
        raise FileNotFoundError(f"no student prediction store at {pred_path}")
    summary_path = os.path.join(out, "student", "pred.summary.json")
    clip = float(json.load(open(summary_path)).get("clip", CLIP)) if os.path.exists(summary_path) else float(CLIP)
    stores = {n: open_store(os.path.join(out, "teachers", f"{n}.zarr"), 4 if n in ("lasagna", "m7_l2") else 1)
              for n in ("recto", "ink", "lasagna", "m7_l2")}
    stores["fiber"] = open_store(os.path.join(out, "teachers", "fiber.zarr"), 1)  # optional
    fine = open_store(os.path.join(out, "labels", "fine.zarr"))
    coarse = open_store(os.path.join(out, "labels", "coarse.zarr"), 4)
    faces_mode = pred.has("sdf_in") and pred.has("sdf_out")
    # orientation-free body mode: one SDF channel, tsm.data.body_sdf (no named faces)
    body_mode = pred.has("sdf_body")
    # orientation-free sides mode: the unsigned d_face + the body probability (no named faces)
    sides_mode = pred.has("d_face")
    free_mode = body_mode or sides_mode
    flab = resolve_faces_labels(out, a.faces_labels) if (faces_mode or free_mode) else None
    for n, s in [("pred", pred), ("fine", fine), ("coarse", coarse), ("faces_labels", flab)] + list(stores.items()):
        log(f"{n}: " + (f"{s.path} shape={s.shape} origin={s.origin} channels={s.channels}" if s else "missing"))
    log(f"surface mode: {'faces' if faces_mode else ('body' if body_mode else ('sides' if sides_mode else 'medial'))}")
    if a.rederive_sides:
        if not sides_mode:
            log("WARNING --rederive-sides ignored: the prediction store has no d_face channel")
        else:
            from tsm.infer import rederive_sides

            log(f"re-deriving surface_side1 in {pred.path} (extract_sides)")
            r = rederive_sides(pred_path, clip, brick=128, log=log)
            log(f"surface_side1: {r['n_voxels']} voxels ({r['frac_of_data']:.4f} of data), "
                f"params {r['surface_side1_params']}")
            pred = open_store(pred_path)          # reopen: the channel changed under us
    if a.faces_labels and not (faces_mode or free_mode):
        log("WARNING --faces-labels ignored: the prediction store has no sdf_in/sdf_out channels")

    # common region: the prediction store, clipped to the config region (and to 4-voxel alignment)
    r0 = [int(v) for v in cfg.region.start_zyx]
    r1 = [a_ + b_ for a_, b_ in zip(r0, cfg.region.size_zyx)]
    lo = [max(o, s) for o, s in zip(pred.origin, r0)]
    hi = [min(o + n, s) for o, n, s in zip(pred.origin, pred.shape, r1)]
    lo = [(v // 4) * 4 for v in lo]
    hi = [(v // 4) * 4 for v in hi]
    if any(h - l <= 0 for l, h in zip(lo, hi)):
        raise ValueError(f"empty overlap between the prediction store and the config region ({lo} {hi})")
    log(f"region {lo} -> {hi}")

    brick = tuple(int(v) for v in str(a.brick).split(","))
    vox, surf = voxel_pass(pred, fine, coarse, stores["ink"], lo, hi, clip, cfg.budget, brick,
                           fiber_t=stores["fiber"], wf_store=flab, flab=flab)

    # --- the student medial surface: the zero set of sdf in medial mode, equidistance in faces mode ---
    faces_metrics: dict[str, Any] | None = None
    if faces_mode:
        log("student medial surface from the two faces (zero crossing of sdf_in + sdf_out)")
        smed = medial_from_faces(pred, lo, hi, clip, cfg.budget, brick)
        log(f"medial-from-faces: {int(smed.sum())} voxels ({int(surf.sum())} in-face voxels)")
    else:
        smed = surf

    # --- surface distances: the student medial surface vs the recto medial surface (global) ---
    recto = stores["recto"]
    if recto is not None:
        log("recto medial surface")
        rm = recto.read(lo, hi)[0] >= 128
        rm = medial_surface(_drop_small_components(rm, MIN_COMPONENT))
        log(f"surface distances ({int(smed.sum())} student, {int(rm.sum())} recto medial voxels)")
        keys = ("n", "median", "p90", "frac_gt3", "frac_gt5", "mean")

        def _vs_recto(mask: np.ndarray, n_key: str) -> dict[str, Any]:
            d = surface_distances(mask, rm)
            o = {key: {k: d[side].get(k) for k in keys} for side, key in
                 (("a_to_b", "student_to_recto"), ("b_to_a", "recto_to_student"), ("symmetric", "symmetric"))}
            o[n_key] = int(d["n_a"])          # frac > 5 comes from _dist_stats; add the raw counts
            o["n_recto_medial"] = int(d["n_b"])
            return o

        vox["surface"].update(_vs_recto(smed, "n_student_medial" if faces_mode else "n_student_surface1"))
        if faces_mode:
            vox["surface"]["n_student_surface1"] = int(surf.sum())
            metrics_inface = _vs_recto(surf, "n_student_surface1")
        del rm

    # --- connectivity statistics of surface1 on a central box ---
    c = [(l + h) // 2 for l, h in zip(lo, hi)]
    b0 = [max(l, ci - s // 2) for l, ci, s in zip(lo, c, STATS_BOX)]
    b1 = [min(h, bi + s) for h, bi, s in zip(hi, b0, STATS_BOX)]
    box = surf[b0[0] - lo[0]:b1[0] - lo[0], b0[1] - lo[1]:b1[1] - lo[1], b0[2] - lo[2]:b1[2] - lo[2]]
    log(f"surface1 stats on {b0} -> {b1}")
    stats = surface_stats(box, MIN_COMPONENT)

    metrics: dict[str, Any] = {
        "config": os.path.abspath(a.config),
        "region": {"start_zyx": lo, "size_zyx": [h - l for l, h in zip(lo, hi)]},
        "clip": clip,
        "stores": {n: (s.path if s else None) for n, s in
                   [("pred", pred), ("fine", fine), ("coarse", coarse)] + list(stores.items())},
        **vox,
        "surface1_stats": stats,
        "surface1_stats_box": [b0, b1],
    }
    if faces_mode and recto is not None:
        metrics["surface_inface_vs_recto"] = metrics_inface
    if flab is not None and faces_mode:
        log("faces label metrics")
        metrics["faces"] = faces_pass(pred, flab, lo, hi, clip, cfg.budget, brick)
    if a.rectoverso:
        rvs = open_store(os.path.expanduser(a.rectoverso))
        if rvs is None:
            raise FileNotFoundError(f"no rectoverso store at {a.rectoverso}")
        if not rvs.has("rectoverso"):
            raise ValueError(f"{a.rectoverso} has no 'rectoverso' channel (has {rvs.channels})")
        if not (faces_mode or free_mode):
            log("WARNING --rectoverso ignored: the prediction store has no "
                "sdf_in/sdf_out, sdf_body or d_face channels")
        else:
            if faces_mode:
                log(f"upstream (teacher-independent) face metrics vs {rvs.path}")
                metrics["upstream_faces"] = upstream_faces_pass(pred, rvs, lo, hi, cfg.budget, brick)
            # the unordered score runs for BOTH store kinds: it is what makes a faces baseline
            # and a body run directly comparable
            log(f"upstream (teacher-independent) unordered body metrics vs {rvs.path}")
            metrics["upstream_body"] = upstream_body_pass(pred, rvs, lo, hi, cfg.budget, brick)
            log("upstream (teacher-independent) topology metrics")
            metrics["upstream_topology"] = upstream_topology_pass(pred, rvs, lo, hi, cfg.budget,
                                                                  brick, clip=clip)
    # --- teacher-independent fibre numbers: the human hz/vt bands ---
    axis_spec = resolve_axis(cfg, a.axis)
    metrics["axis"] = str(axis_spec) if axis_spec is not None else None
    band_store, band_ch = None, None
    if fine is not None and fine.has("hzvt_class"):
        band_store, band_ch = fine, "hzvt_class"
    elif a.rectoverso:
        rvb = open_store(os.path.expanduser(a.rectoverso))
        if rvb is not None and rvb.has("hzvt"):
            band_store, band_ch = rvb, "hzvt"
    if band_store is not None and pred.has("fiber_vt") and pred.has("fiber_hz"):
        log(f"upstream (teacher-independent) fibre metrics vs {band_store.path}:{band_ch}")
        metrics.setdefault("fiber", {})["upstream"] = fiber_upstream_pass(
            pred, band_store, band_ch, lo, hi, clip, cfg.budget, brick, axis=axis_spec)
    if not a.no_gallery:
        metrics["gallery"] = gallery(ed, cfg, pred, stores, lo, hi, clip, surf, a.slices,
                                     faces_labels=flab if faces_mode else None,
                                     medial=smed if faces_mode else None)
    metrics["seconds"] = time.perf_counter() - t0
    with open(os.path.join(ed, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    md = to_markdown(metrics)
    with open(os.path.join(ed, "metrics.md"), "w") as fh:
        fh.write(md)
    print(md)
    log(f"wrote {os.path.join(ed, 'metrics.json')} and metrics.md")
    return metrics


if __name__ == "__main__":
    main()
