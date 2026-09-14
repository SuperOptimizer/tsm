"""Propose new training slabs across a whole scroll from a coarse (level-5) survey.

Reads the coarsest usable level of the volume in z-chunks, measures per-z papyrus
occupancy / xy bounding box / umbilicus position, then proposes K level-0 slabs that
are spread over the part of the scroll that actually contains papyrus, centred on the
umbilicus in xy (shifted to cover the cross-section when it is off-centre) and kept
clear of the existing training slab and the held-out eval box.

Usage:
  taskset -c 0-5 uv run python dev/survey_regions.py \
      --out ~/tsm-output/survey_paris4 --k 8 --xy 8192 --depth 256 [--write-configs]

Network + CPU only; one z-chunk of the survey level is live at a time (peak RSS well
under 6 GB, dominated by the ~150 MB coarse xy occupancy cube).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tsm.labels import axis_yx_at, load_axis  # noqa: E402
from tsm.limits import Budget  # noqa: E402
from tsm.volume import VolumeReader, open_array  # noqa: E402

DEFAULT_URL = ("https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/"
               "volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
DEFAULT_AXIS = "~/.cache/tsm-production/axes/PHercParis4/umbilicus-full-resolution.json"
DEFAULT_BASE_CONFIG = "configs/paris4_faces_rvonly.json"

# level-0 z ranges that new slabs must stay away from
EXISTING_SLAB_Z = (34176, 34432)   # configs/paris4_faces_rvonly.json
EVAL_Z = (34432, 34688)            # held-out eval box
KEEP_CLEAR_SLABS = 2               # +/- this many slab depths around each of the above

# labels store cost: 40 GB for a 256 x 6144 x 6144 level-0 slab
LABEL_GB_PER_VOXEL = 40.0 / (256.0 * 6144.0 * 6144.0)

ALIGN = 64  # level-0 alignment of every proposed start/size (a multiple of the level-2 grid of 4)


# --------------------------------------------------------------------------- #
# survey
# --------------------------------------------------------------------------- #
@dataclass
class Survey:
    level: int
    scale: int                       # level-0 voxels per survey voxel
    shape0: tuple[int, int, int]     # level-0 shape
    shape5: tuple[int, int, int]     # survey-level shape
    occ: np.ndarray                  # (nz5,) float32, fraction of the slice above threshold
    bbox: np.ndarray                 # (nz5, 4) int32 y0,y1,x0,x1 in survey voxels; -1 if empty
    umb: np.ndarray                  # (nz5, 2) float32 umbilicus (y, x) in survey voxels
    cell: int                        # survey voxels per coarse xy cell
    counts: np.ndarray               # (nz5, cy, cx) uint8 papyrus voxels per cell
    threshold: int = 60
    sigma: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def cell0(self) -> int:
        """level-0 voxels spanned by one coarse xy cell."""
        return self.cell * self.scale


def pick_level(url: str, want: int = 5) -> int:
    """``want`` if the multiscale group has it, else the coarsest level below it."""
    for lv in range(want, -1, -1):
        try:
            open_array(url, lv)
            return lv
        except Exception as exc:  # missing level -> try the next finer one
            print(f"[survey] level {lv} unavailable ({type(exc).__name__}: {exc})", flush=True)
    raise RuntimeError(f"no usable level in {url}")


def scan(reader: VolumeReader, axis: np.ndarray, scale: int, threshold: int, sigma: float,
         zchunk: int, cell: int, log=print) -> Survey:
    nz, ny, nx = reader.shape
    cy, cx = (ny + cell - 1) // cell, (nx + cell - 1) // cell
    occ = np.zeros(nz, np.float32)
    bbox = np.full((nz, 4), -1, np.int32)
    counts = np.zeros((nz, cy, cx), np.uint8)
    area = float(ny * nx)
    t0 = time.time()
    for z0 in range(0, nz, zchunk):
        z1 = min(nz, z0 + zchunk)
        block = reader.read(z0, z1, 0, ny, 0, nx)
        for i in range(z1 - z0):
            sl = block[i]
            if sigma > 0:
                sl = ndi.gaussian_filter(sl, sigma, output=np.uint8)
            m = sl > threshold
            z = z0 + i
            occ[z] = m.sum() / area
            if occ[z] > 0:
                ys = np.flatnonzero(m.any(1))
                xs = np.flatnonzero(m.any(0))
                bbox[z] = (ys[0], ys[-1] + 1, xs[0], xs[-1] + 1)
                pad = np.zeros((cy * cell, cx * cell), np.uint8)
                pad[:ny, :nx] = m
                counts[z] = pad.reshape(cy, cell, cx, cell).sum((1, 3)).astype(np.uint8)
        del block
        log(f"[survey] z {z1}/{nz}  ({time.time() - t0:.0f}s)")
    z0s = (np.arange(nz) + 0.5) * scale
    uy, ux = axis_yx_at(axis, z0s)
    umb = np.stack([uy / scale, ux / scale], 1).astype(np.float32)
    return Survey(level=reader.level, scale=scale,
                  shape0=tuple(int(s) * scale for s in reader.shape), shape5=tuple(reader.shape),
                  occ=occ, bbox=bbox, umb=umb, cell=cell, counts=counts,
                  threshold=threshold, sigma=sigma,
                  meta={"url": reader.url, "seconds": round(time.time() - t0, 1)})


# --------------------------------------------------------------------------- #
# proposal logic (pure; unit-tested in tests/test_survey_regions.py)
# --------------------------------------------------------------------------- #
def align_down(v: int, align: int = ALIGN) -> int:
    return (int(v) // align) * align


def exclusion_intervals(depth: int, keep_clear: int = KEEP_CLEAR_SLABS,
                        zones=(EXISTING_SLAB_Z, EVAL_Z)) -> list[tuple[int, int]]:
    """level-0 z intervals a new slab must not touch (each zone grown by N slab depths)."""
    pad = int(keep_clear) * int(depth)
    raw = sorted((max(0, int(a) - pad), int(b) + pad) for a, b in zones)
    out: list[tuple[int, int]] = []
    for lo, hi in raw:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def slab_scores(occ: np.ndarray, scale: int, depth: int, align: int = ALIGN) -> tuple[np.ndarray, np.ndarray]:
    """(level-0 starts on the ``align`` grid, mean survey occupancy over each slab)."""
    nz0 = len(occ) * scale
    starts = np.arange(0, max(1, nz0 - depth + 1), align, dtype=np.int64)
    d5 = max(1, depth // scale)
    scores = np.empty(len(starts), np.float64)
    for i, s in enumerate(starts):
        a = int(s) // scale
        scores[i] = float(occ[a:a + d5].mean()) if a < len(occ) else 0.0
    return starts, scores


def choose_starts(starts: np.ndarray, scores: np.ndarray, k: int, depth: int,
                  floor: float, excl: list[tuple[int, int]],
                  min_gap: int | None = None) -> list[int]:
    """K level-0 z starts: spread over the occupied z range, best occupancy per bin.

    Candidates must clear ``floor`` occupancy, must not intersect any exclusion interval
    and must stay ``min_gap`` apart (default: half a bin, so the per-bin maxima cannot
    all pile up on the bin edges; relaxed to ``depth`` -- i.e. merely non-overlapping --
    for a bin that has nothing else left).  The allowed candidates are split into K
    contiguous bins by z and the highest-scoring usable candidate of each bin is taken.
    """
    ok = scores >= float(floor)
    for lo, hi in excl:
        ok &= ~((starts < hi) & (starts + depth > lo))
    idx = np.flatnonzero(ok)
    if len(idx) == 0 or k <= 0:
        return []
    span = int(starts[idx[-1]] - starts[idx[0]])
    if min_gap is None:
        min_gap = max(int(depth), span // (2 * max(1, int(k))))
    chosen: list[int] = []
    edges = np.linspace(0, len(idx), int(k) + 1)
    for b in range(int(k)):
        lo, hi = int(round(edges[b])), int(round(edges[b + 1]))
        if hi <= lo:
            continue
        cand = idx[lo:hi][np.argsort(-scores[idx[lo:hi]], kind="stable")]
        for gap in (int(min_gap), int(depth)):
            hit = next((int(starts[j]) for j in cand
                        if all(abs(int(starts[j]) - c) >= gap for c in chosen)), None)
            if hit is not None:
                chosen.append(hit)
                break
    return sorted(chosen)


def _window_sums(counts2d: np.ndarray, w: int) -> np.ndarray:
    """Sum of every ``w`` x ``w`` cell window; result is (cy-w+1, cx-w+1)."""
    c = np.zeros((counts2d.shape[0] + 1, counts2d.shape[1] + 1), np.float64)
    c[1:, 1:] = counts2d.cumsum(0).cumsum(1)
    return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w])


def best_xy_start(counts2d: np.ndarray | None, cell0: int, umb_yx0: tuple[float, float],
                  xy: int, shape0_yx: tuple[int, int], align: int = ALIGN,
                  max_shift_frac: float = 0.5) -> tuple[tuple[int, int], tuple[int, int], float]:
    """(start (y, x), shift from the umbilicus-centred start, papyrus sum inside the window).

    The window is centred on the umbilicus and then shifted by at most
    ``max_shift_frac * xy`` to maximise the papyrus it covers (so the umbilicus stays
    inside the window for the default 0.5).
    """
    ny, nx = shape0_yx
    base = tuple(int(np.clip(align_down(int(round(u - xy / 2))), 0, max(0, align_down(n - xy))))
                 for u, n in zip(umb_yx0, (ny, nx)))
    if counts2d is None or counts2d.size == 0:
        return base, (0, 0), 0.0
    w = max(1, int(round(xy / cell0)))
    if w > min(counts2d.shape):
        return base, (0, 0), 0.0
    sums = _window_sums(counts2d, w)
    cy0 = np.arange(sums.shape[0]) * cell0
    cx0 = np.arange(sums.shape[1]) * cell0
    lim = float(max_shift_frac) * xy
    oky = (np.abs(cy0 - base[0]) <= lim) & (cy0 + xy <= ny)
    okx = (np.abs(cx0 - base[1]) <= lim) & (cx0 + xy <= nx)
    if not oky.any() or not okx.any():
        return base, (0, 0), 0.0
    mask = np.outer(oky, okx)
    s = np.where(mask, sums, -1.0)
    flat = np.flatnonzero(s.ravel() == s.max())
    # tie-break: closest to the umbilicus-centred window
    dy = np.abs(cy0[flat // s.shape[1]] - base[0])
    dx = np.abs(cx0[flat % s.shape[1]] - base[1])
    j = flat[int(np.argmin(dy + dx))]
    iy, ix = int(j // s.shape[1]), int(j % s.shape[1])
    start = (int(cy0[iy]), int(cx0[ix]))
    start = tuple(int(np.clip(align_down(v, align), 0, max(0, align_down(n - xy, align))))
                  for v, n in zip(start, (ny, nx)))
    return start, (start[0] - base[0], start[1] - base[1]), float(sums[iy, ix])


def propose(sv: Survey, k: int, depth: int, xy: int, floor: float,
            max_shift_frac: float = 0.5, align: int = ALIGN,
            extra_excl: list[tuple[int, int]] | None = None) -> list[dict]:
    depth = align_down(depth, align) or align
    xy = align_down(xy, align) or align
    excl = exclusion_intervals(depth) + list(extra_excl or [])
    starts, scores = slab_scores(sv.occ, sv.scale, depth, align)
    zs = choose_starts(starts, scores, k, depth, floor, excl)
    out: list[dict] = []
    for i, z in enumerate(zs, 1):
        a, b = z // sv.scale, max(z // sv.scale + 1, (z + depth) // sv.scale)
        c2 = sv.counts[a:b].sum(0).astype(np.float64) if sv.counts.size else None
        nsl = max(1, b - a)
        uy = float(np.interp(z + depth / 2, np.arange(len(sv.umb)) * sv.scale,
                             sv.umb[:, 0] * sv.scale))
        ux = float(np.interp(z + depth / 2, np.arange(len(sv.umb)) * sv.scale,
                             sv.umb[:, 1] * sv.scale))
        (y0, x0), shift, _ = best_xy_start(c2, sv.cell0, (uy, ux), xy, sv.shape0[1:],
                                           align, max_shift_frac)
        w = max(1, int(round(xy / sv.cell0)))
        iy, ix = y0 // sv.cell0, x0 // sv.cell0
        win = float(c2[iy:iy + w, ix:ix + w].sum()) if c2 is not None else 0.0
        win_frac = win / max(1.0, nsl * (w * sv.cell) ** 2)
        total = float(c2.sum()) if c2 is not None else 0.0
        coverage = win / total if total > 0 else 0.0
        zocc = float(sv.occ[a:b].mean())
        bb = sv.bbox[a:b]
        bb = bb[bb[:, 0] >= 0]
        vol = float(depth) * xy * xy
        out.append({
            "k": i,
            "start_zyx": [int(z), int(y0), int(x0)],
            "size_zyx": [int(depth), int(xy), int(xy)],
            "slice_occupancy": round(zocc, 5),
            "window_occupancy": round(win_frac, 5),
            "cross_section_coverage": round(coverage, 5),
            "umbilicus_yx_l0": [round(uy, 1), round(ux, 1)],
            "shift_yx_l0": [int(shift[0]), int(shift[1])],
            "cross_section_yx_l0": ([int((bb[:, 1] - bb[:, 0]).max() * sv.scale),
                                     int((bb[:, 3] - bb[:, 2]).max() * sv.scale)]
                                    if len(bb) else [0, 0]),
            "covers_cross_section": bool(len(bb) and
                                         (bb[:, 1] - bb[:, 0]).max() * sv.scale <= xy and
                                         (bb[:, 3] - bb[:, 2]).max() * sv.scale <= xy),
            "label_gb": round(vol * LABEL_GB_PER_VOXEL, 1),
        })
    return out


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def survey_payload(sv: Survey, props: list[dict], args) -> dict:
    step = max(1, int(args.json_stride))
    zi = np.arange(0, len(sv.occ), step)
    slices = [{"z": int(i * sv.scale), "occ": round(float(sv.occ[i]), 5),
               "bbox_yx_l0": ([int(v) * sv.scale for v in sv.bbox[i]]
                              if sv.bbox[i, 0] >= 0 else None),
               "umb_yx_l0": [round(float(sv.umb[i, 0]) * sv.scale, 1),
                             round(float(sv.umb[i, 1]) * sv.scale, 1)]}
              for i in zi]
    occ_z = np.flatnonzero(sv.occ >= args.floor)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "volume": {"url": args.url, "level": sv.level, "scale": sv.scale,
                   "shape_level0_zyx": list(sv.shape0), "shape_survey_zyx": list(sv.shape5)},
        "params": {"threshold": sv.threshold, "sigma": sv.sigma, "k": args.k,
                   "depth": args.depth, "xy": args.xy, "floor": args.floor,
                   "max_shift_frac": args.max_shift_frac, "align": ALIGN,
                   "exclusions_l0": [list(e) for e in exclusion_intervals(args.depth)],
                   "json_stride": step, "cell_level0": sv.cell0},
        "extent": {
            "z_occupied_l0": ([int(occ_z[0] * sv.scale), int((occ_z[-1] + 1) * sv.scale)]
                              if len(occ_z) else None),
            "occ_max": round(float(sv.occ.max()), 5),
            "max_cross_section_yx_l0": [int((sv.bbox[sv.bbox[:, 0] >= 0][:, 1]
                                             - sv.bbox[sv.bbox[:, 0] >= 0][:, 0]).max() * sv.scale),
                                        int((sv.bbox[sv.bbox[:, 0] >= 0][:, 3]
                                             - sv.bbox[sv.bbox[:, 0] >= 0][:, 2]).max() * sv.scale)]
            if (sv.bbox[:, 0] >= 0).any() else [0, 0],
        },
        "slices": slices,
        "proposals": props,
        "meta": sv.meta,
    }


def table(props: list[dict], sv: Survey) -> str:
    hdr = (f"{'k':>2}  {'start_zyx (level 0)':>26}  {'size_zyx':>18}  {'slice%':>6}  "
           f"{'win%':>6}  {'cover%':>6}  {'shift yx':>13}  {'xsect yx':>13}  {'fits':>4}  "
           f"{'label GB':>8}")
    lines = [hdr, "-" * len(hdr)]
    for p in props:
        lines.append(
            f"{p['k']:>2}  {str(tuple(p['start_zyx'])):>26}  {str(tuple(p['size_zyx'])):>18}  "
            f"{100 * p['slice_occupancy']:>6.2f}  {100 * p['window_occupancy']:>6.2f}  "
            f"{100 * p['cross_section_coverage']:>6.2f}  "
            f"{str(tuple(p['shift_yx_l0'])):>13}  {str(tuple(p['cross_section_yx_l0'])):>13}  "
            f"{'yes' if p['covers_cross_section'] else 'NO':>4}  {p['label_gb']:>8.1f}")
    lines.append(f"total label store: {sum(p['label_gb'] for p in props):.1f} GB "
                 f"(survey level {sv.level}, {sv.scale}x)")
    return "\n".join(lines)


def render_png(path: str, sv: Survey, props: list[dict], thumbs: list[np.ndarray], args) -> None:
    from PIL import Image, ImageDraw

    W, H = 1500, 340
    tw = 300
    cols = max(1, min(5, len(props))) if props else 1
    rows = (len(props) + cols - 1) // cols if props else 0
    img = Image.new("RGB", (W, H + rows * (tw + 22) + 10), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 40
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    nz0 = sv.shape0[0]
    fx = lambda z: pad_l + pw * float(z) / nz0                    # noqa: E731
    omax = max(1e-6, float(sv.occ.max()))
    fy = lambda o: pad_t + ph * (1.0 - float(o) / omax)           # noqa: E731

    for lo, hi in exclusion_intervals(args.depth):
        d.rectangle([fx(lo), pad_t, fx(min(hi, nz0)), pad_t + ph], fill=(255, 232, 232))
    d.rectangle([fx(EXISTING_SLAB_Z[0]), pad_t, fx(EXISTING_SLAB_Z[1]), pad_t + ph], fill=(240, 170, 170))
    d.rectangle([fx(EVAL_Z[0]), pad_t, fx(EVAL_Z[1]), pad_t + ph], fill=(250, 210, 150))
    for p in props:
        z, dz = p["start_zyx"][0], p["size_zyx"][0]
        d.rectangle([fx(z), pad_t, max(fx(z + dz), fx(z) + 2), pad_t + ph], fill=(180, 220, 255))
        d.text((fx(z) + 2, pad_t + 2), f"r{p['k']}", fill=(0, 90, 160))
    d.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(0, 0, 0))
    d.line([(fx(0), fy(args.floor)), (fx(nz0), fy(args.floor))], fill=(170, 170, 170))
    d.line([(fx(i * sv.scale), fy(sv.occ[i])) for i in range(len(sv.occ))], fill=(20, 20, 20), width=1)
    for f in range(6):
        z = int(nz0 * f / 5)
        d.text((fx(z) - 18, pad_t + ph + 6), f"{z}", fill=(0, 0, 0))
    for f in range(5):
        o = omax * f / 4
        d.text((8, fy(o) - 6), f"{100 * o:.1f}%", fill=(0, 0, 0))
    d.text((pad_l, 8), f"level-{sv.level} papyrus occupancy vs level-0 z  "
                       f"(threshold {sv.threshold}, sigma {sv.sigma}; red = existing slab +/- "
                       f"{KEEP_CLEAR_SLABS} slabs, orange = eval box, blue = proposals)",
           fill=(0, 0, 0))

    for i, (p, th) in enumerate(zip(props, thumbs)):
        r, c = divmod(i, cols)
        ox, oy = 20 + c * (tw + 10), H + r * (tw + 22)
        im = Image.fromarray(th).resize((tw, tw), Image.BILINEAR).convert("RGB")
        dd = ImageDraw.Draw(im)
        sy, sx = sv.shape5[1], sv.shape5[2]
        bx = [tw * (p["start_zyx"][2] / sv.scale) / sx, tw * (p["start_zyx"][1] / sv.scale) / sy,
              tw * ((p["start_zyx"][2] + p["size_zyx"][2]) / sv.scale) / sx,
              tw * ((p["start_zyx"][1] + p["size_zyx"][1]) / sv.scale) / sy]
        dd.rectangle(bx, outline=(0, 160, 255), width=2)
        uy = tw * (p["umbilicus_yx_l0"][0] / sv.scale) / sy
        ux = tw * (p["umbilicus_yx_l0"][1] / sv.scale) / sx
        dd.line([(ux - 5, uy), (ux + 5, uy)], fill=(255, 60, 60), width=2)
        dd.line([(ux, uy - 5), (ux, uy + 5)], fill=(255, 60, 60), width=2)
        img.paste(im, (ox, oy))
        d.text((ox, oy + tw + 4), f"r{p['k']}  z={p['start_zyx'][0]}  "
                                  f"win {100 * p['window_occupancy']:.1f}%", fill=(0, 0, 0))
    img.save(path)


def thumbnails(reader: VolumeReader, sv: Survey, props: list[dict], nsl: int = 4) -> list[np.ndarray]:
    out = []
    ny, nx = sv.shape5[1], sv.shape5[2]
    for p in props:
        zc = (p["start_zyx"][0] + p["size_zyx"][0] // 2) // sv.scale
        a = int(np.clip(zc - nsl // 2, 0, max(0, sv.shape5[0] - nsl)))
        blk = reader.read(a, min(sv.shape5[0], a + nsl), 0, ny, 0, nx)
        out.append(blk.max(0))
    return out


def write_configs(base_path: str, props: list[dict], root: str, cfg_dir: str, log=print,
                  prefix: str = "paris4_r") -> list[str]:
    with open(base_path) as fh:
        base = json.load(fh)
    os.makedirs(cfg_dir, exist_ok=True)
    paths = []
    for p in props:
        cfg = json.loads(json.dumps(base))
        name = f"{prefix}{p['k']}"
        out_dir = os.path.join(root, name)
        cfg["region"] = {"start_zyx": list(p["start_zyx"]), "size_zyx": list(p["size_zyx"])}
        cfg["out_dir"] = out_dir
        labels = cfg.setdefault("extra", {}).setdefault("labels", {})
        labels["teachers_dir"] = os.path.join(out_dir, "teachers")
        labels.setdefault("faces", {})["rectoverso_store"] = os.path.join(
            out_dir, "rectoverso", "rectoverso.zarr")
        train = cfg["extra"].setdefault("train", {})
        for key, tail in (("fine_store", "fine.zarr"), ("coarse_store", "coarse.zarr")):
            if key in train:  # the base config points these at the old slab's out_dir
                train[key] = os.path.join(out_dir, "labels", tail)
        path = os.path.join(cfg_dir, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        paths.append(path)
    log(f"[survey] wrote {len(paths)} configs to {cfg_dir}")
    return paths


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--axis", default=DEFAULT_AXIS)
    ap.add_argument("--level", type=int, default=5, help="survey level (falls back to finer)")
    ap.add_argument("--out", default="/home/forrest/tsm-output/survey_paris4")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--xy", type=int, default=8192)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--threshold", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.05, help="min slice occupancy fraction")
    ap.add_argument("--zmin", type=int, default=0, help="level-0 z below which no slab is proposed")
    ap.add_argument("--zmax", type=int, default=0, help="level-0 z above which no slab is proposed (0 = none)")
    ap.add_argument("--exclude", default="", help="extra level-0 z intervals to avoid, 'z0:z1,z0:z1'")
    ap.add_argument("--name-prefix", default="paris4_r", help="config/out_dir name prefix for proposals")
    ap.add_argument("--max-shift-frac", type=float, default=0.5)
    ap.add_argument("--zchunk", type=int, default=128, help="survey z-slices per read")
    ap.add_argument("--cell", type=int, default=4, help="survey voxels per coarse xy cell")
    ap.add_argument("--json-stride", type=int, default=1, help="keep every Nth slice in survey.json")
    ap.add_argument("--ram-gb", type=float, default=6.0)
    ap.add_argument("--write-configs", action="store_true")
    ap.add_argument("--config-dir", default="configs/regions")
    ap.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    ap.add_argument("--region-root", default="/home/forrest/tsm-output/regions")
    ap.add_argument("--cache", default="", help="npz of a previous scan (skip the network read)")
    args = ap.parse_args(argv)

    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    budget = Budget(ram_bytes=int(args.ram_gb * (1 << 30)), array_bytes=1 << 30, min_avail_mb=512)
    level = pick_level(args.url, args.level)
    a0 = open_array(args.url, 0)
    reader = VolumeReader(args.url, level, 2.4, budget)
    scale = int(round(int(a0.shape[-3]) / reader.shape[0]))
    print(f"[survey] level {level} shape {reader.shape} scale {scale}x "
          f"(level 0 {tuple(int(s) for s in a0.shape[-3:])})", flush=True)
    axis = load_axis(args.axis)

    npz = args.cache or os.path.join(out, "scan.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        sv = Survey(level=level, scale=scale,
                    shape0=tuple(int(s) for s in a0.shape[-3:]), shape5=tuple(reader.shape),
                    occ=d["occ"], bbox=d["bbox"], umb=d["umb"], cell=int(d["cell"]),
                    counts=d["counts"], threshold=args.threshold, sigma=args.sigma,
                    meta={"url": args.url, "from_cache": npz})
        print(f"[survey] reusing {npz}", flush=True)
    else:
        sv = scan(reader, axis, scale, args.threshold, args.sigma, args.zchunk, args.cell)
        sv.shape0 = tuple(int(s) for s in a0.shape[-3:])
        np.savez_compressed(npz, occ=sv.occ, bbox=sv.bbox, umb=sv.umb,
                            counts=sv.counts, cell=sv.cell)

    extra: list[tuple[int, int]] = []
    if args.zmin > 0:
        extra.append((0, int(args.zmin)))
    if args.zmax > 0:
        extra.append((int(args.zmax), 10 ** 9))
    for tok in [t for t in args.exclude.split(",") if t.strip()]:
        a, b = tok.split(":")
        extra.append((int(a), int(b)))
    props = propose(sv, args.k, args.depth, args.xy, args.floor, args.max_shift_frac, extra_excl=extra)
    payload = survey_payload(sv, props, args)
    with open(os.path.join(out, "survey.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    thumbs = thumbnails(reader, sv, props)
    render_png(os.path.join(out, "survey.png"), sv, props, thumbs, args)

    print()
    print(table(props, sv))
    ext = payload["extent"]
    print(f"\npapyrus z range (occ >= {args.floor}): {ext['z_occupied_l0']} of "
          f"[0, {sv.shape0[0]}] level-0; max occupancy {100 * ext['occ_max']:.2f}%; "
          f"max cross-section (y, x) {tuple(ext['max_cross_section_yx_l0'])} level-0 voxels")
    print(f"wrote {out}/survey.json and {out}/survey.png")
    if args.write_configs:
        write_configs(args.base_config, props, args.region_root, args.config_dir, prefix=args.name_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
