"""Surface-label audit (``tsm audit``): recto vs m7 geometry at 2.4 um.

Pure numpy/scipy helpers (``sdf_from_mask``, ``thin_surface_from_sdf``,
``medial_thickness``, ``radial_vectors``, ``decode_lasagna_normal`` ...) plus
``run_audit(cfg)`` which writes ``<out_dir>/audit/audit.json`` and slice PNGs.

Conventions
-----------
* Arrays are (Z, Y, X); normals/vectors are stacked on axis 0 in (z, y, x) order.
* ``sdf_from_mask``: positive outside, negative inside (voxel units).
* ``medial_thickness``: thickness = 2*EDT - 1 at medial voxels, i.e. the number of
  foreground voxels crossed along the normal for an odd-width axis-aligned slab
  (1 for a single-voxel surface; under-reads tilted sheets by up to ~1 voxel).
* "outward" = away from the scroll axis (radial direction at that z).

The label-build half (``tsm labels``): ``build_fine_labels`` / ``build_coarse_field`` and
``run_labels`` write the stores described in docs/label_store.md (brick-wise, haloed).
``build_face_labels`` (optional, ``extra.labels.faces.enabled``) adds the two-face channels
``sdf_in, sdf_out, faces_valid, thickness`` to the fine store: the sheet body comes from the
CT, its two boundaries are split by the outward normal, and the recto teacher is used only for
the ignore band and the ``recto_is_in`` diagnostic.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

from tsm.config import RunCfg
from tsm.limits import Budget, check_alloc

__all__ = [
    "sdf_from_mask",
    "thin_surface_from_sdf",
    "medial_thickness",
    "medial_surface",
    "skeletonize3d",
    "load_axis",
    "axis_yx_at",
    "radial_vectors",
    "decode_lasagna_normal",
    "gradient_normals",
    "neighbor_count6",
    "component_stats",
    "surface_stats",
    "hist_percentiles",
    "dice_iou",
    "run_audit",
    "encode_signed_u8",
    "decode_signed_u8",
    "encode_sdf_u8",
    "decode_sdf_u8",
    "ct_mask",
    "phase_from_cos",
    "build_fine_labels",
    "build_face_labels",
    "close_ball",
    "close_normal_line",
    "FACE_CHANNELS",
    "FACE_DEFAULTS",
    "fine_channels",
    "build_coarse_field",
    "upsample_coarse_normal",
    "read_box_padded",
    "fiber_block",
    "iter_cores",
    "run_labels",
    "open_teacher",
    "merge_stats",
    "FINE_CHANNELS",
    "FIBER_CHANNELS",
    "COARSE_CHANNELS",
    "LABEL_DEFAULTS",
]

DEFAULT_AXIS = os.path.expanduser("~/.cache/tsm-production/axes/PHercParis4/umbilicus-full-resolution.json")
AUDIT_DEFAULTS: dict[str, Any] = {
    "teachers_dir": None,      # default <out_dir>/teachers
    "threshold": 0.5,
    "sigma": 2.0,              # gaussian sigma (voxels) for the recto-gradient normal
    "min_component": 100,      # components smaller than this are dropped from counts
    "near_voxels": 3,          # coverage radius
    "axis_path": DEFAULT_AXIS,
    "m7_level2": True,
    "axis_xy": "json_xy",      # 'json_xy' (verified correct against a level-5 slice, 2026-09-02) | 'swapped_xy' | 'auto'
    "skimage_skeleton": True,  # also report skimage's (curve) skeleton, for information
    "max_offset": 24,          # histogram range for medial offsets (voxels)
    "seed": 0,
}
_STRUCT26 = np.ones((3, 3, 3), dtype=bool)
_STRUCT6 = ndi.generate_binary_structure(3, 1)


# --------------------------------------------------------------------------- #
# Distance / surface primitives
# --------------------------------------------------------------------------- #
def _edt(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (float32) from each True voxel to the nearest False voxel."""
    try:
        import edt as _edt_mod

        return np.asarray(_edt_mod.edt(np.ascontiguousarray(mask, dtype=np.uint8), black_border=False),
                          dtype=np.float32)
    except ImportError:
        return ndi.distance_transform_edt(mask).astype(np.float32)


def sdf_from_mask(mask: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Signed distance in voxels: ``edt(outside) - edt(inside)`` (negative inside)."""
    mask = np.asarray(mask, dtype=bool)
    sdf = _edt(~mask)
    sdf -= _edt(mask)
    if clip is not None:
        np.clip(sdf, -float(clip), float(clip), out=sdf)
    return sdf


def _shift_any(mask: np.ndarray, pad_value: bool) -> np.ndarray:
    """True where any of the 6 face neighbours is True (outside the box = pad_value)."""
    p = np.pad(mask, 1, mode="constant", constant_values=pad_value)
    out = np.zeros_like(mask, dtype=bool)
    out |= p[:-2, 1:-1, 1:-1]
    out |= p[2:, 1:-1, 1:-1]
    out |= p[1:-1, :-2, 1:-1]
    out |= p[1:-1, 2:, 1:-1]
    out |= p[1:-1, 1:-1, :-2]
    out |= p[1:-1, 1:-1, 2:]
    return out


def thin_surface_from_sdf(sdf: np.ndarray) -> np.ndarray:
    """Inside voxels (sdf < 0) with at least one 6-neighbour outside: a 1-voxel shell.

    Voxels outside the array are treated as inside, so the shell does not include
    the box faces.  For a thick band this returns *both* faces.
    """
    inside = np.asarray(sdf) < 0
    return inside & _shift_any(~inside, pad_value=False)


def neighbor_count6(mask: np.ndarray) -> np.ndarray:
    """Number (0..6) of True 6-neighbours for every voxel (zeros outside the box)."""
    p = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    n = p[:-2, 1:-1, 1:-1].astype(np.uint8)
    n += p[2:, 1:-1, 1:-1]
    n += p[1:-1, :-2, 1:-1]
    n += p[1:-1, 2:, 1:-1]
    n += p[1:-1, 1:-1, :-2]
    n += p[1:-1, 1:-1, 2:]
    return n


def skeletonize3d(mask: np.ndarray) -> np.ndarray:
    """skimage 3D skeleton (Lee94).  NOTE: this is a *curve* thinning -- a plate collapses
    to a line -- so it is reported for information only, not used as the medial surface."""
    from skimage.morphology import skeletonize

    return np.asarray(skeletonize(np.asarray(mask, dtype=bool)), dtype=bool)


def medial_surface(mask: np.ndarray, d: np.ndarray | None = None, tol: float = 0.5) -> np.ndarray:
    """EDT ridge: foreground voxels whose EDT is within ``tol`` of the 26-neighbourhood max.

    For a slab of odd width this is exactly the centre plane; for a 1-voxel sheet it is
    the sheet itself.  Unlike a curve skeleton it keeps sheets as sheets."""
    mask = np.asarray(mask, dtype=bool)
    if d is None:
        d = _edt(mask)
    mx = ndi.maximum_filter(d, size=3, mode="constant")
    return mask & (d >= mx - float(tol))


def medial_thickness(mask: np.ndarray, medial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(medial mask, thickness samples) with thickness = 2*EDT - 1 at medial voxels."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask), np.zeros(0, dtype=np.float32)
    d = _edt(mask)
    if medial is None:
        medial = medial_surface(mask, d)
    medial = medial & mask
    return medial, (2.0 * d[medial] - 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Scroll axis / radial direction
# --------------------------------------------------------------------------- #
def load_axis(path: str = DEFAULT_AXIS) -> np.ndarray:
    """Umbilicus control points as a (N, 3) float array of (z, y, x) level-0 voxels."""
    with open(os.path.expanduser(path), "r") as fh:
        d = json.load(fh)
    pts = d["control_points"] if isinstance(d, dict) else d
    if isinstance(d, dict) and d.get("coordinate_order", "xyz") != "xyz":
        raise ValueError(f"unexpected coordinate_order {d.get('coordinate_order')!r}")
    a = np.array([[p["z"], p["y"], p["x"]] for p in pts], dtype=np.float64)
    if isinstance(d, dict) and not str(d.get("coordinate_space", "full_resolution")).startswith("full_resolution"):
        raise ValueError(f"axis file {path} is not in full-resolution voxels ({d.get('coordinate_space')!r})")
    return a[np.argsort(a[:, 0])]


def axis_yx_at(axis: np.ndarray, z: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of the axis (y, x) at level-0 z (clamped at the ends)."""
    z = np.asarray(z, dtype=np.float64)
    return np.interp(z, axis[:, 0], axis[:, 1]), np.interp(z, axis[:, 0], axis[:, 2])


def radial_at(axis: np.ndarray, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Unit outward radial vectors (3, N) in (z, y, x) order at absolute voxel coords."""
    ay, ax = axis_yx_at(axis, z)
    ry = np.asarray(y, dtype=np.float64) - ay
    rx = np.asarray(x, dtype=np.float64) - ax
    n = np.sqrt(ry * ry + rx * rx)
    n[n == 0] = 1.0
    return np.stack([np.zeros_like(ry), ry / n, rx / n]).astype(np.float32)


def radial_vectors(axis: np.ndarray, z: int, shape: Sequence[int], origin: Sequence[int]) -> np.ndarray:
    """Outward unit radial field (3, Y, X) for one absolute z slice of a box at ``origin``."""
    ny, nx = int(shape[-2]), int(shape[-1])
    y0, x0 = int(origin[-2]), int(origin[-1])
    yy, xx = np.meshgrid(np.arange(y0, y0 + ny), np.arange(x0, x0 + nx), indexing="ij")
    return radial_at(axis, np.full(yy.size, float(z)), yy.ravel(), xx.ravel()).reshape(3, ny, nx)


# --------------------------------------------------------------------------- #
# Lasagna direction decoding (ported from villa preprocess_cos_omezarr.py)
# --------------------------------------------------------------------------- #
def decode_dir_angle(dir0: np.ndarray, dir1: np.ndarray) -> np.ndarray:
    """dir0 = 0.5+0.5cos2t, dir1 = 0.5+0.5(cos2t-sin2t)/sqrt2  ->  t in (-pi/2, pi/2]."""
    cos2t = 2.0 * dir0 - 1.0
    sin2t = cos2t - np.sqrt(2.0) * (2.0 * dir1 - 1.0)
    return np.arctan2(sin2t, cos2t) * 0.5


def _enc_dir(gx: np.ndarray, gy: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    r2 = gx * gx + gy * gy + eps
    c2 = (gx * gx - gy * gy) / r2
    s2 = 2.0 * gx * gy / r2
    return 0.5 + 0.5 * c2, 0.5 + 0.5 * (c2 - s2) / np.sqrt(2.0)


def decode_lasagna_normal(ch: np.ndarray, eps: float = 1e-12, return_extra: bool = False):
    """Unit normal (3, ...) in (z, y, x) order from the 6 lasagna dir channels.

    ``ch`` is (>=8, ...) in [0, 1] ordered ``cos, grad_mag, dir0_z, dir1_z, dir0_y,
    dir1_y, dir0_x, dir1_x`` (planes (x,y), (x,z), (y,z)); channels 0..1 are ignored,
    so a 6-channel ``dir*`` stack is accepted too.  Sign is arbitrary (line field).

    With ``return_extra`` also returns ``(agreement, enc_err)``: ``agreement`` in [0, 1]
    is the magnitude of the weighted sum of the three per-plane candidates divided by
    the sum of their magnitudes (1 = the planes agree, ~0 = they cancel) and ``enc_err``
    is the weighted RMS difference between the re-encoded normal and the observed
    six channels (0 = perfectly consistent dir pairs).
    """
    ch = np.asarray(ch, dtype=np.float64)
    if ch.shape[0] == 8:
        ch = ch[2:]
    if ch.shape[0] != 6:
        raise ValueError(f"expected 6 or 8 channels, got {ch.shape[0]}")
    d0z, d1z, d0y, d1y, d0x, d1x = ch
    tz, ty, tx = decode_dir_angle(d0z, d1z), decode_dir_angle(d0y, d1y), decode_dir_angle(d0x, d1x)
    sz, cz = np.sin(tz), np.cos(tz)
    sy, cy = np.sin(ty), np.cos(ty)
    sx, cx = np.sin(tx), np.cos(tx)
    cands = [
        np.stack([cz * cy, sz * cy, cz * sy]),
        np.stack([cz * cx, sz * cx, sz * sx]),
        np.stack([cy * sx, sy * cx, sy * sx]),
    ]
    n1 = cands[0]
    for i in (1, 2):
        dot = (n1 * cands[i]).sum(0)
        cands[i] = cands[i] * np.where(dot >= 0, 1.0, -1.0)
    # pass 1: score candidates against the observed channels
    est = np.zeros_like(n1)
    for n in cands:
        ncx, ncy, ncz = n
        pz0, pz1 = _enc_dir(ncx, ncy, eps)
        py0, py1 = _enc_dir(ncx, ncz, eps)
        px0, px1 = _enc_dir(ncy, ncz, eps)
        err = ((ncx ** 2 + ncy ** 2) * ((pz0 - d0z) ** 2 + (pz1 - d1z) ** 2)
               + (ncx ** 2 + ncz ** 2) * ((py0 - d0y) ** 2 + (py1 - d1y) ** 2)
               + (ncy ** 2 + ncz ** 2) * ((px0 - d0x) ** 2 + (px1 - d1x) ** 2))
        est += n / (err + eps)
    est /= np.sqrt((est ** 2).sum(0)) + eps
    # pass 2: re-weight rows by axis reliability, sign-align, sum
    ex, ey, ez = est
    wz2 = np.sqrt(ex ** 2 + ey ** 2 + eps)
    wy2 = np.sqrt(ex ** 2 + ez ** 2 + eps)
    wx2 = np.sqrt(ey ** 2 + ez ** 2 + eps)
    rn = [cands[0] * (wz2 * wy2), cands[1] * (wz2 * wx2), cands[2] * (wy2 * wx2)]
    for i in (1, 2):
        dot = (rn[0] * rn[i]).sum(0)
        rn[i] = rn[i] * np.where(dot >= 0, 1.0, -1.0)
    nf = rn[0] + rn[1] + rn[2]
    mag = np.sqrt((nf ** 2).sum(0))
    nf /= mag + eps
    out = np.stack([nf[2], nf[1], nf[0]]).astype(np.float32)  # (x,y,z) -> (z,y,x)
    if not return_extra:
        return out
    denom = sum(np.sqrt((r ** 2).sum(0)) for r in rn) + eps
    agreement = (mag / denom).astype(np.float32)
    ncx, ncy, ncz = nf
    pz0, pz1 = _enc_dir(ncx, ncy, eps)
    py0, py1 = _enc_dir(ncx, ncz, eps)
    px0, px1 = _enc_dir(ncy, ncz, eps)
    wz, wy, wx = ncx ** 2 + ncy ** 2, ncx ** 2 + ncz ** 2, ncy ** 2 + ncz ** 2
    err = (wz * ((pz0 - d0z) ** 2 + (pz1 - d1z) ** 2) + wy * ((py0 - d0y) ** 2 + (py1 - d1y) ** 2)
           + wx * ((px0 - d0x) ** 2 + (px1 - d1x) ** 2))
    enc_err = np.sqrt(err / (2.0 * (wz + wy + wx) + eps)).astype(np.float32)
    return out, agreement, enc_err


# --------------------------------------------------------------------------- #
# Normals from a probability field
# --------------------------------------------------------------------------- #
def gradient_normals(prob: np.ndarray, sigma: float, flat_idx: np.ndarray) -> np.ndarray:
    """Unit gradient of gaussian-smoothed ``prob`` at flat indices, (3, N) in (z, y, x).

    One smoothed derivative volume is live at a time (float32)."""
    prob = np.asarray(prob, dtype=np.float32)
    g = np.zeros((3, flat_idx.size), dtype=np.float32)
    for ax in range(3):
        order = [0, 0, 0]
        order[ax] = 1
        d = ndi.gaussian_filter(prob, sigma=float(sigma), order=tuple(order), mode="nearest")
        g[ax] = d.ravel()[flat_idx]
        del d
    n = np.sqrt((g * g).sum(0))
    n[n == 0] = 1.0
    return g / n


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def _pct(v: np.ndarray, um: float | None = None) -> dict[str, Any]:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    out = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
           "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}
    if um is not None:
        out["median_um"] = float(q[2] * um)
        out["p5_um"] = float(q[0] * um)
        out["p95_um"] = float(q[4] * um)
    return out


def _hist(v: np.ndarray, lo: float, hi: float, step: float = 1.0) -> dict[str, list]:
    edges = np.arange(lo, hi + step, step)
    h, _ = np.histogram(np.asarray(v, dtype=np.float64), bins=edges)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in h]}


def dice_iou(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    na, nb = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    union = na + nb - inter
    return {"dice": (2.0 * inter / (na + nb)) if na + nb else 0.0,
            "iou": (inter / union) if union else 0.0, "n_a": na, "n_b": nb, "intersection": inter}


def component_stats(mask: np.ndarray, min_size: int, structure: np.ndarray = _STRUCT26) -> dict[str, Any]:
    lab, n = ndi.label(np.asarray(mask, dtype=bool), structure=structure)
    if n == 0:
        return {"n_components": 0, "n_components_min_size": 0, "largest": 0, "sizes_top10": []}
    sizes = np.bincount(lab.ravel())[1:]
    keep = sizes[sizes >= int(min_size)]
    top = np.sort(sizes)[::-1][:10]
    del lab
    return {"n_components": int(n), "n_components_min_size": int(keep.size), "largest": int(sizes.max()),
            "sizes_top10": [int(s) for s in top], "fg_voxels": int(sizes.sum()),
            "fg_voxels_min_size": int(keep.sum())}


def hist_percentiles(hist: np.ndarray, percentiles: Sequence[float], drop_zero: bool = True) -> list[float | None]:
    """Exact percentiles of the values counted by a 256-bin histogram, without materialising them.

    ``hist[v]`` is the number of voxels whose value is ``v``. With ``drop_zero`` the zero bin is
    ignored (background). The result matches ``np.percentile(np.repeat(np.arange(256), hist), p)``
    (linear interpolation between order statistics) but costs O(256) memory, so histogram totals
    far beyond ``2**32`` are fine. Returns ``None`` per percentile when nothing is counted."""
    h = np.asarray(hist, dtype=np.int64).copy()
    if drop_zero and h.size:
        h[0] = 0
    if h.min() < 0:
        raise ValueError("histogram counts must be non-negative")
    cum = np.cumsum(h)
    n = int(cum[-1]) if cum.size else 0
    if n == 0:
        return [None] * len(percentiles)
    out: list[float | None] = []
    for p in percentiles:
        p = float(p)
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"percentile out of range: {p}")
        x = (n - 1) * p / 100.0
        lo, hi = int(math.floor(x)), int(math.ceil(x))
        # order statistic k (0-based) = the first bin whose cumulative count exceeds k
        vlo = float(np.searchsorted(cum, lo, side="right"))
        vhi = vlo if hi == lo else float(np.searchsorted(cum, hi, side="right"))
        out.append(vlo + (x - lo) * (vhi - vlo))
    return out


def surface_stats(surf: np.ndarray, min_component: int) -> dict[str, Any]:
    """Thickness, 6-neighbour degree, 6- vs 26-components and hole proxies of a thin mask."""
    surf = np.asarray(surf, dtype=bool)
    n = int(np.count_nonzero(surf))
    if n == 0:
        return {"n_voxels": 0}
    _, thick = medial_thickness(surf)
    deg = neighbor_count6(surf)[surf]
    c26 = component_stats(surf, min_component, _STRUCT26)
    c6 = component_stats(surf, min_component, _STRUCT6)
    filled = ndi.binary_fill_holes(surf)
    holes = filled & ~surf
    del filled
    n_hole = int(np.count_nonzero(holes))
    hole_cc = int(ndi.label(holes, structure=_STRUCT6)[1]) if n_hole else 0
    del holes
    return {
        "n_voxels": n,
        "thickness": _pct(thick),
        "deg6_mean": float(deg.mean()),
        "deg6_hist": [int(c) for c in np.bincount(deg, minlength=7)],
        "frac_deg6_ge2": float((deg >= 2).mean()),
        "frac_deg6_ge3": float((deg >= 3).mean()),
        "frac_isolated": float((deg == 0).mean()),
        "components_26": c26,
        "components_6": c6,
        "enclosed_cavity_voxels": n_hole,
        "enclosed_cavities": hole_cc,
    }


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def _audit_opts(cfg: RunCfg) -> dict[str, Any]:
    raw = cfg.extra.get("audit", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.audit must be an object")
    unknown = sorted(set(raw) - set(AUDIT_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.audit keys: {unknown}")
    opts = dict(AUDIT_DEFAULTS)
    opts.update(raw)
    if not opts["teachers_dir"]:
        opts["teachers_dir"] = os.path.join(cfg.out_dir, "teachers")
    t = float(opts["threshold"])
    if not 0.0 < t < 1.0:
        raise ValueError("audit.threshold must be in (0, 1)")
    return opts


def _load_teacher_u8(teachers_dir: str, name: str, budget: Budget) -> tuple[np.ndarray, dict]:
    import zarr

    path = os.path.join(teachers_dir, f"{name}.zarr")
    arr = zarr.open_array(store=path, mode="r")
    check_alloc(arr.shape, np.uint8, budget)
    return np.asarray(arr[:]), dict(arr.attrs)


def _upsample_nearest(a: np.ndarray, k: int, shape: Sequence[int]) -> np.ndarray:
    for ax in range(a.ndim - 3, a.ndim):
        a = np.repeat(a, k, axis=ax)
    sl = (Ellipsis,) + tuple(slice(0, int(s)) for s in shape[-3:])
    return a[sl]


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.6) -> np.ndarray:
    out = rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    out[m] = (1 - alpha) * out[m] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_png(path: str, img: np.ndarray) -> str:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(img)).save(path)
    return path


def _hist_png(path: str, counts: Sequence[int], edges: Sequence[float], title: str, zero_line: bool = True) -> str:
    """Tiny PIL bar chart (no matplotlib)."""
    from PIL import Image, ImageDraw

    w, h, pad = 640, 240, 24
    img = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    n = len(counts)
    mx = max(max(counts), 1)
    bw = (w - 2 * pad) / max(n, 1)
    for i, c in enumerate(counts):
        x0 = pad + i * bw
        y0 = h - pad - (h - 2 * pad) * c / mx
        dr.rectangle([x0, y0, x0 + bw - 1, h - pad], fill=(70, 120, 200))
    if zero_line and edges[0] < 0 < edges[-1]:
        xz = pad + (0 - edges[0]) / (edges[-1] - edges[0]) * (w - 2 * pad)
        dr.line([xz, pad, xz, h - pad], fill=(200, 40, 40), width=1)
    dr.text((pad, 4), f"{title}  [{edges[0]:g} .. {edges[-1]:g}] max={mx}", fill=(0, 0, 0))
    img.save(path)
    return path


def _read_ct_slice(cfg: RunCfg, z: int) -> np.ndarray | None:
    from tsm.volume import VolumeReader, open_cached_reader

    z0, y0, x0 = cfg.region.start_zyx
    _, dy, dx = cfg.region.size_zyx
    cached = open_cached_reader(cfg.volume.cache_root, cfg.volume.url, cfg.volume.level,
                                cfg.volume.voxel_um, cfg.region, cfg.budget)
    if cached is not None:
        sl = cached.read(z, z + 1, y0, y0 + dy, x0, x0 + dx)[0]
        if sl.any():
            return sl
    for url in (cfg.volume.url, cfg.volume.alt_url):
        if not url:
            continue
        try:
            r = VolumeReader(url, cfg.volume.level, cfg.volume.voxel_um, cfg.budget)
            sl = r.read(z, z + 1, y0, y0 + dy, x0, x0 + dx)[0]
            if sl.any():
                return sl
        except Exception as exc:  # CT panel is best-effort
            print(f"[audit] CT slice from {url}: {type(exc).__name__}: {exc}")
    return None


def _log(msg: str) -> None:
    print(f"[audit] {msg}", flush=True)


def run_audit(cfg: RunCfg, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Recto-vs-m7 audit over the config region; returns the audit dict (also written)."""
    opts = _audit_opts(cfg)
    budget = cfg.budget
    um = float(cfg.volume.voxel_um)
    thr = int(round(float(opts["threshold"]) * 255))
    tdir = opts["teachers_dir"]
    adir = os.path.join(cfg.out_dir, "audit")
    shape = tuple(int(s) for s in cfg.region.size_zyx)
    origin = tuple(int(s) for s in cfg.region.start_zyx)
    n_vox = int(np.prod(shape))
    min_cc = int(opts["min_component"])
    near = float(opts["near_voxels"])
    maxoff = float(opts["max_offset"])
    rng = np.random.default_rng(int(opts["seed"]))

    # budget table (largest live arrays): sdf/edt float32, nearest-medial indices int32 x3
    from tsm.limits import estimate_and_assert

    estimate_and_assert([
        ("recto_u8", shape, np.uint8), ("m7_u8", shape, np.uint8),
        ("sdf_f32", shape, np.float32), ("edt_f32", shape, np.float32),
        ("medial_idx_i32", (3,) + shape, np.int32), ("grad_f32", shape, np.float32),
    ], budget)
    if dry_run:
        _log("dry run: stopping before I/O")
        return {}
    os.makedirs(adir, exist_ok=True)
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "region": {"start_zyx": list(origin), "size_zyx": list(shape), "voxel_um": um},
        "threshold": float(opts["threshold"]), "teachers_dir": tdir, "png": {},
    }

    recto_u8, ra = _load_teacher_u8(tdir, "recto", budget)
    m7_u8, ma = _load_teacher_u8(tdir, "m7", budget)
    for name, attrs in (("recto", ra), ("m7", ma)):
        if tuple(int(v) for v in attrs.get("origin_zyx", origin)) != origin:
            raise ValueError(f"{name}.zarr origin {attrs.get('origin_zyx')} != region {origin}")
    if recto_u8.shape[1:] != shape or m7_u8.shape[1:] != shape:
        raise ValueError(f"teacher shapes {recto_u8.shape} / {m7_u8.shape} != region {shape}")
    recto = recto_u8[0] >= thr
    m7 = m7_u8[0] >= thr
    out["fg_fraction"] = {"recto": float(recto.mean()), "m7": float(m7.mean())}
    _log(f"loaded recto/m7 {shape}: fg recto={recto.mean():.4f} m7={m7.mean():.4f}")

    # ---- 1. thickness ------------------------------------------------------
    thick: dict[str, Any] = {}
    medial: dict[str, np.ndarray] = {}
    for name, mask in (("recto", recto), ("m7", m7)):
        med, tv = medial_thickness(mask)
        medial[name] = med
        thick[name] = _pct(tv, um)
        thick[name]["hist_voxels"] = _hist(tv, 0, 60, 1)
        thick[name]["n_medial"] = int(med.sum())
        _log(f"thickness {name}: median {thick[name].get('median', 0):.1f} vox "
             f"({thick[name].get('median_um', 0):.1f} um), p5-p95 {thick[name].get('p5', 0):.1f}-{thick[name].get('p95', 0):.1f}")
    out["thickness"] = thick

    # ---- 3. agreement / components ----------------------------------------
    agree: dict[str, Any] = {"dice_iou_recto_m7": dice_iou(recto, m7)}
    agree["frac_m7_inside_recto"] = float((m7 & recto).sum() / max(m7.sum(), 1))
    agree["frac_recto_inside_m7"] = float((m7 & recto).sum() / max(recto.sum(), 1))
    d_m7 = _edt(~m7)  # distance to nearest m7 voxel
    agree["frac_recto_medial_within_near_of_m7"] = float((d_m7[medial["recto"]] <= near).mean())
    agree["dist_recto_medial_to_m7"] = _pct(d_m7[medial["recto"]], um)
    del d_m7
    d_rm = _edt(~medial["recto"])
    agree["frac_m7_within_near_of_recto_medial"] = float((d_rm[m7] <= near).mean())
    agree["dist_m7_to_recto_medial"] = _pct(d_rm[m7], um)
    agree["near_voxels"] = near
    agree["components_26"] = {"recto": component_stats(recto, min_cc), "m7": component_stats(m7, min_cc)}
    out["agreement"] = agree
    _log(f"dice recto/m7 {agree['dice_iou_recto_m7']['dice']:.3f}, m7 inside recto "
         f"{agree['frac_m7_inside_recto']:.3f}, recto-medial near m7 {agree['frac_recto_medial_within_near_of_m7']:.3f}")

    # ---- 2. geometry: m7 relative to the recto band ------------------------
    sdf_r = sdf_from_mask(recto)
    prob_r = recto_u8[0].astype(np.float32) / 255.0
    zc = origin[0] + shape[0] // 2
    # Principal normal direction of the recto sheets (axis-independent orientation reference)
    samp = np.flatnonzero(medial["recto"])
    samp = rng.choice(samp, size=min(100_000, samp.size), replace=False)
    n_s = gradient_normals(prob_r, float(opts["sigma"]), samp)
    evals, evecs = np.linalg.eigh((n_s @ n_s.T) / max(n_s.shape[1], 1))
    principal = evecs[:, -1].astype(np.float32)
    out["principal_normal_zyx"] = {"vector": [float(v) for v in principal],
                                   "eigenvalues": [float(v) for v in evals],
                                   "median_abs_dot": float(np.median(np.abs(principal @ n_s)))}
    # Scroll axis: the JSON as parsed vs. with x/y swapped; keep the one the sheet normals agree with
    axis_json = load_axis(opts["axis_path"])
    sz, sy, sx = np.unravel_index(samp, shape)
    hyps: dict[str, Any] = {}
    for hname, ax_pts in (("json_xy", axis_json), ("swapped_xy", axis_json[:, [0, 2, 1]])):
        ay, ax = axis_yx_at(ax_pts, zc)
        r_s = radial_at(ax_pts, sz + origin[0], sy + origin[1], sx + origin[2])
        inside = bool(origin[1] <= ay < origin[1] + shape[1] and origin[2] <= ax < origin[2] + shape[2])
        dist = float(np.hypot(ay - (origin[1] + shape[1] / 2), ax - (origin[2] + shape[2] / 2)))
        hyps[hname] = {"y": float(ay), "x": float(ax), "inside_roi": inside, "dist_to_roi_center_vox": dist,
                       "median_abs_normal_dot_radial": float(np.median(np.abs((n_s * r_s).sum(0)))),
                       "points": ax_pts}
    chosen = max(hyps, key=lambda k: hyps[k]["median_abs_normal_dot_radial"])
    if opts["axis_xy"] in hyps:
        chosen = opts["axis_xy"]
    axis = hyps[chosen].pop("points")
    hyps["json_xy"].pop("points", None)
    hyps["swapped_xy"].pop("points", None)
    out["axis"] = {"path": opts["axis_path"], "z_mid": zc, "n_control_points": int(axis_json.shape[0]),
                   "hypotheses": hyps, "chosen": chosen,
                   "note": ("axis lies inside the ROI (ROI is at the scroll core); the normal-vs-radial "
                            "heuristic is unreliable there because the innermost windings are flattened")
                   if hyps["json_xy"]["inside_roi"] else ""}
    _log(f"axis: {chosen} (|n.r| json {hyps['json_xy']['median_abs_normal_dot_radial']:.3f} "
         f"swapped {hyps['swapped_xy']['median_abs_normal_dot_radial']:.3f}); principal normal (z,y,x)="
         f"{np.round(principal, 3).tolist()} |n.d| {out['principal_normal_zyx']['median_abs_dot']:.3f}")
    del n_s, samp
    # nearest recto-medial voxel for every voxel, its EDT (half thickness)
    d_in = _edt(recto)
    _, idx = ndi.distance_transform_edt(~medial["recto"], return_indices=True)
    idx = idx.astype(np.int32)

    def offsets_for(sel: np.ndarray, label: str) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
        """Signed offsets of the selected voxels from the recto medial surface."""
        flat = np.flatnonzero(sel)
        if flat.size == 0:
            return {"n": 0}, (flat, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
        pz, py, px = np.unravel_index(flat, shape)
        mz, my, mx = (idx[k].ravel()[flat] for k in range(3))
        v = np.stack([pz - mz, py - my, px - mx]).astype(np.float32)
        dist = np.sqrt((v * v).sum(0))
        keep = dist <= maxoff
        flat, v, dist = flat[keep], v[:, keep], dist[keep]
        mz, my, mx = mz[keep], my[keep], mx[keep]
        pz, py, px = pz[keep], py[keep], px[keep]
        rad = radial_at(axis, pz + origin[0], py + origin[1], px + origin[2])
        nrm = gradient_normals(prob_r, float(opts["sigma"]), flat)
        n_dot_r = np.abs((nrm * rad).sum(0))
        # orientation A: recto normal oriented outward (away from the chosen axis)
        sgn = np.where((nrm * rad).sum(0) >= 0, 1.0, -1.0).astype(np.float32)
        s_normal = (v * (nrm * sgn)).sum(0)
        s_radial = (v * rad).sum(0)
        # orientation B: recto normal oriented along the ROI principal normal (axis-free)
        sgn_p = np.where((nrm * principal[:, None]).sum(0) >= 0, 1.0, -1.0).astype(np.float32)
        s_princ = (v * (nrm * sgn_p)).sum(0)
        half = d_in[mz, my, mx]  # EDT at the nearest medial voxel = half thickness
        frac = 0.5 + s_normal / np.maximum(2.0 * half, 1.0)
        sdf_here = sdf_r.ravel()[flat]
        res = {
            "n": int(flat.size), "n_dropped_far": int((~keep).sum()),
            "offset_along_recto_normal_vox": _pct(s_normal, um),
            "offset_along_radial_vox": _pct(s_radial, um),
            "hist_offset_normal": _hist(s_normal, -maxoff, maxoff, 1),
            "hist_offset_radial": _hist(s_radial, -maxoff, maxoff, 1),
            "frac_across_band": _pct(frac),
            "hist_frac_across_band": _hist(frac, -1.0, 2.0, 0.1),
            "sdf_recto_at_voxels": _pct(sdf_here, um),
            "frac_inside_recto_band": float((sdf_here < 0).mean()),
            "frac_outer_half": float((s_normal > 0.5).mean()),
            "frac_inner_half": float((s_normal < -0.5).mean()),
            "frac_middle": float((np.abs(s_normal) <= 0.5).mean()),
            "offset_along_principal_normal_vox": _pct(s_princ, um),
            "hist_offset_principal": _hist(s_princ, -maxoff, maxoff, 1),
            "frac_principal_pos": float((s_princ > 0.5).mean()),
            "frac_principal_neg": float((s_princ < -0.5).mean()),
            "abs_normal_dot_radial": _pct(n_dot_r),
            "half_thickness_at_nearest_medial": _pct(half, um),
        }
        _log(f"offset {label}: median along normal {res['offset_along_recto_normal_vox']['median']:+.2f} vox, "
             f"radial {res['offset_along_radial_vox']['median']:+.2f} vox, inside band "
             f"{res['frac_inside_recto_band']:.3f}, inner/mid/outer "
             f"{res['frac_inner_half']:.2f}/{res['frac_middle']:.2f}/{res['frac_outer_half']:.2f}; "
             f"principal-oriented median {res['offset_along_principal_normal_vox']['median']:+.2f} "
             f"neg/pos {res['frac_principal_neg']:.2f}/{res['frac_principal_pos']:.2f}")
        return res, (flat, s_normal, s_princ)

    geo: dict[str, Any] = {}
    geo["m7_vs_recto"], (m7_flat, m7_s, m7_sp) = offsets_for(m7, "m7")
    geo["m7_medial_vs_recto"], _ = offsets_for(medial["m7"], "m7-medial")
    # per recto sheet component: is the m7 offset unimodal (same face everywhere)?
    lab, nlab = ndi.label(recto, structure=_STRUCT26)
    sizes = np.bincount(lab.ravel())
    order = np.argsort(sizes[1:])[::-1][:10] + 1
    per: list[dict[str, Any]] = []
    # attribute each m7 voxel to the recto component of its nearest medial voxel
    m7_lab = np.zeros(0, dtype=lab.dtype)
    if m7_flat.size:
        mz, my, mx = (idx[k].ravel()[m7_flat] for k in range(3))
        m7_lab = lab[mz, my, mx]
    for c in order:
        selc = m7_lab == c
        if selc.sum() < min_cc:
            continue
        sv, sp = m7_s[selc], m7_sp[selc]
        per.append({"component": int(c), "recto_voxels": int(sizes[c]), "m7_voxels": int(selc.sum()),
                    "offset_normal_median": float(np.median(sv)), "offset_normal_p25": float(np.percentile(sv, 25)),
                    "offset_normal_p75": float(np.percentile(sv, 75)),
                    "frac_outer": float((sv > 0.5).mean()), "frac_inner": float((sv < -0.5).mean()),
                    "offset_principal_median": float(np.median(sp)),
                    "offset_principal_p25": float(np.percentile(sp, 25)),
                    "offset_principal_p75": float(np.percentile(sp, 75)),
                    "frac_principal_pos": float((sp > 0.5).mean()), "frac_principal_neg": float((sp < -0.5).mean()),
                    "hist_offset_principal": _hist(sp, -maxoff, maxoff, 2)})
    geo["per_recto_component"] = per
    del lab
    out["geometry"] = geo

    # ---- 4. thin extractions ----------------------------------------------
    thin: dict[str, Any] = {}
    thin_masks: dict[str, np.ndarray] = {}
    sdf_m = sdf_from_mask(m7)
    for name, sdf in (("recto", sdf_r), ("m7", sdf_m)):
        shell = thin_surface_from_sdf(sdf)
        thin_masks[f"{name}_sdf_shell"] = shell
        thin_masks[f"{name}_skeleton"] = medial[name]
        thin[name] = {"medial_ridge": surface_stats(medial[name], min_cc), "sdf_shell": surface_stats(shell, min_cc)}
        if opts["skimage_skeleton"]:
            try:
                tk = time.perf_counter()
                sk = skeletonize3d(recto if name == "recto" else m7)
                thin[name]["skimage_skeleton"] = surface_stats(sk, min_cc)
                thin[name]["skimage_skeleton"]["seconds"] = time.perf_counter() - tk
                thin_masks[f"{name}_skimage"] = sk
            except Exception as exc:
                thin[name]["skimage_skeleton"] = {"error": f"{type(exc).__name__}: {exc}"}
        for k in ("medial_ridge", "sdf_shell"):
            s = thin[name][k]
            _log(f"thin {name}/{k}: n={s.get('n_voxels')} thick median {s.get('thickness', {}).get('median')} "
                 f"deg6>=2 {s.get('frac_deg6_ge2', 0):.3f} cc26={s.get('components_26', {}).get('n_components_min_size')} "
                 f"cc6={s.get('components_6', {}).get('n_components_min_size')}")
    del sdf_m
    out["thin_surface"] = thin

    # ---- 5. m7 at level 2 ---------------------------------------------------
    l2: dict[str, Any] = {"enabled": bool(opts["m7_level2"])}
    if opts["m7_level2"]:
        try:
            from tsm.cli import run_teacher_model

            p_l2 = os.path.join(tdir, "m7_l2.zarr")
            done = os.path.exists(os.path.join(p_l2, "done.json"))
            if force or not done:
                topts = dict(cfg.extra.get("teacher", {}))
                topts["skip_existing"] = not force
                summary, _ = run_teacher_model(cfg, "m7", tdir, topts, force=force, level_shift=2, out_name="m7_l2")
                l2["summary"] = summary
            m7l2_u8, l2a = _load_teacher_u8(tdir, "m7_l2", budget)
            m7l2 = m7l2_u8[0] >= thr
            k = int(round(float(l2a.get("scale", 4.0))))
            _, tv2 = medial_thickness(m7l2)
            l2["thickness_l2_voxels"] = _pct(tv2, um * k)
            l2["fg_fraction"] = float(m7l2.mean())
            up = _upsample_nearest(m7l2, k, shape)
            l2["dice_iou_vs_recto"] = dice_iou(up, recto)
            l2["dice_iou_vs_m7_l0"] = dice_iou(up, m7)
            l2["frac_recto_medial_inside_m7_l2"] = float(up[medial["recto"]].mean())
            l2["frac_m7_l0_inside_m7_l2"] = float(up[m7].mean())
            l2["frac_m7_l2_inside_recto"] = float(recto[up].mean()) if up.any() else 0.0
            # offset of the level-2 m7 band relative to the recto medial surface (level-0 voxels)
            m7l2_up_medial = _upsample_nearest(medial_thickness(m7l2)[0], k, shape)
            l2["m7_l2_medial_vs_recto"], _ = offsets_for(m7l2_up_medial, "m7-l2-medial")
            thin_masks["m7_l2_up"] = up
            _log(f"m7 level2: thick median {l2['thickness_l2_voxels'].get('median')} l2-vox, dice vs recto "
                 f"{l2['dice_iou_vs_recto']['dice']:.3f}, vs m7-l0 {l2['dice_iou_vs_m7_l0']['dice']:.3f}")
            del m7l2_u8, m7l2_up_medial
        except Exception as exc:
            l2["error"] = f"{type(exc).__name__}: {exc}"
            _log(f"m7 level2 FAILED: {l2['error']}")
    out["m7_level2"] = l2
    del idx, d_in

    # ---- 6. lasagna cross-check --------------------------------------------
    las: dict[str, Any] = {}
    try:
        las_u8, la = _load_teacher_u8(tdir, "lasagna", budget)
        k = int(round(float(la.get("scale", 4.0))))
        cos_up = _upsample_nearest(las_u8[0].astype(np.float32) / 255.0, k, shape)
        bg = ~recto & ~m7
        bg_idx = np.flatnonzero(bg)
        bg_idx = rng.choice(bg_idx, size=min(200_000, bg_idx.size), replace=False)
        las["cos_at_recto_medial"] = _pct(cos_up[medial["recto"]])
        las["cos_at_m7"] = _pct(cos_up[m7])
        las["cos_at_m7_medial"] = _pct(cos_up[medial["m7"]])
        las["cos_at_recto_band"] = _pct(cos_up[recto])
        las["cos_at_background"] = _pct(cos_up.ravel()[bg_idx])
        las["cos_gt_0p9_frac"] = {"recto_medial": float((cos_up[medial["recto"]] > 0.9).mean()),
                                  "m7": float((cos_up[m7] > 0.9).mean()),
                                  "background": float((cos_up.ravel()[bg_idx] > 0.9).mean())}
        # normal agreement on the recto medial surface (subsampled)
        nl2 = decode_lasagna_normal(las_u8.astype(np.float32) / 255.0)  # (3, z, y, x) at level 2
        flat = np.flatnonzero(medial["recto"])
        flat = rng.choice(flat, size=min(300_000, flat.size), replace=False)
        pz, py, px = np.unravel_index(flat, shape)
        nl = nl2[:, pz // k, py // k, px // k]
        ng = gradient_normals(prob_r, float(opts["sigma"]), flat)
        dot = np.clip(np.abs((nl * ng).sum(0)), 0, 1)
        ang = np.degrees(np.arccos(dot))
        las["angle_lasagna_vs_recto_normal_deg"] = _pct(ang)
        las["angle_hist_deg"] = _hist(ang, 0, 90, 5)
        rad = radial_at(axis, pz + origin[0], py + origin[1], px + origin[2])
        las["abs_lasagna_normal_dot_radial"] = _pct(np.abs((nl * rad).sum(0)))
        las["abs_recto_normal_dot_radial"] = _pct(np.abs((ng * rad).sum(0)))
        thin_masks["lasagna_cos_up"] = cos_up
        _log(f"lasagna: cos medial {las['cos_at_recto_medial']['mean']:.3f} m7 {las['cos_at_m7']['mean']:.3f} "
             f"bg {las['cos_at_background']['mean']:.3f}; angle median {las['angle_lasagna_vs_recto_normal_deg']['median']:.1f} deg")
        del las_u8, nl2
    except Exception as exc:
        las["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"lasagna cross-check FAILED: {las['error']}")
    out["lasagna"] = las
    del sdf_r, prob_r

    # ---- PNGs ---------------------------------------------------------------
    zm = shape[0] // 2
    zabs = origin[0] + zm
    ct = _read_ct_slice(cfg, zabs)
    base = _to_rgb(ct) if ct is not None else _to_rgb(np.zeros(shape[1:], dtype=np.uint8))
    png: dict[str, str] = {}
    if ct is not None:
        png["ct"] = _save_png(os.path.join(adir, "ct.png"), ct)
    png["recto"] = _save_png(os.path.join(adir, "recto.png"), recto_u8[0, zm])
    png["m7"] = _save_png(os.path.join(adir, "m7.png"), m7_u8[0, zm])
    ov = _overlay(base, recto[zm], (255, 60, 60), 0.45)
    ov = _overlay(ov, m7[zm], (60, 255, 60), 0.7)
    png["overlay_recto_red_m7_green"] = _save_png(os.path.join(adir, "overlay_recto_m7.png"), ov)
    for name in ("recto", "m7"):
        ov = _overlay(base, thin_masks[f"{name}_sdf_shell"][zm], (255, 0, 255), 0.8)
        ov = _overlay(ov, thin_masks[f"{name}_skeleton"][zm], (0, 255, 255), 0.9)
        if f"{name}_skimage" in thin_masks:
            ov = _overlay(ov, thin_masks[f"{name}_skimage"][zm], (255, 255, 0), 1.0)
        png[f"thin_{name}_shell_magenta_medial_cyan_skimage_yellow"] = _save_png(
            os.path.join(adir, f"thin_{name}.png"), ov)
    if "m7_l2_up" in thin_masks:
        ov = _overlay(base, recto[zm], (255, 60, 60), 0.35)
        ov = _overlay(ov, thin_masks["m7_l2_up"][zm], (60, 60, 255), 0.6)
        png["overlay_recto_red_m7l2_blue"] = _save_png(os.path.join(adir, "overlay_recto_m7_l2.png"), ov)
    if "lasagna_cos_up" in thin_masks:
        png["lasagna_cos"] = _save_png(os.path.join(adir, "lasagna_cos.png"),
                                       (thin_masks["lasagna_cos_up"][zm] * 255).astype(np.uint8))
    h = geo["m7_vs_recto"].get("hist_offset_normal")
    if h:
        png["hist_m7_offset_normal"] = _hist_png(os.path.join(adir, "hist_m7_offset_normal.png"), h["counts"],
                                                 h["edges"], "m7 offset from recto medial along outward normal (vox)")
    h = geo["m7_vs_recto"].get("hist_offset_principal")
    if h:
        png["hist_m7_offset_principal"] = _hist_png(os.path.join(adir, "hist_m7_offset_principal.png"), h["counts"],
                                                    h["edges"], "m7 offset from recto medial along principal normal (vox)")
    h = geo["m7_vs_recto"].get("hist_frac_across_band")
    if h:
        png["hist_m7_frac_across_band"] = _hist_png(os.path.join(adir, "hist_m7_frac_across_band.png"), h["counts"],
                                                    h["edges"], "m7 position across recto band (0 inner, 1 outer)",
                                                    zero_line=False)
    out["png"] = png

    # ---- verdict -------------------------------------------------------------
    out["verdict"] = _verdict(out)
    out["seconds"] = time.perf_counter() - t0
    with open(os.path.join(adir, "audit.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    _log(f"wrote {os.path.join(adir, 'audit.json')} in {out['seconds']:.0f}s")
    for line in out["verdict"]["notes"]:
        _log(line)
    return out


def _verdict(a: dict[str, Any]) -> dict[str, Any]:
    """Heuristic reading of the numbers; the user reviews this, it decides nothing."""
    notes: list[str] = []
    tr = a["thickness"]["recto"].get("median", 0.0)
    tm = a["thickness"]["m7"].get("median", 0.0)
    g = a["geometry"]["m7_vs_recto"]
    med = g.get("offset_along_recto_normal_vox", {}).get("median", 0.0)
    inner, mid, outer = g.get("frac_inner_half", 0), g.get("frac_middle", 0), g.get("frac_outer_half", 0)
    half = tr / 2.0
    notes.append(f"recto median thickness {tr:.1f} vox, m7 {tm:.1f} vox")
    if tr >= 8:
        recto_kind = "band (sheet volume)"
    elif tr >= 3:
        recto_kind = "thick face"
    else:
        recto_kind = "thin face"
    if inner >= 0.3 and outer >= 0.3:
        face = "bimodal (both faces / neighbouring sheets)"
    elif abs(med) <= max(1.5, 0.2 * half) and mid >= 0.5:
        face = "middle"
    elif med > 0:
        face = "outer"
    else:
        face = "inner"
    notes.append(f"m7 sits at {med:+.1f} vox from the recto medial surface (half thickness {half:.1f}); "
                 f"inner/middle/outer fractions {inner:.2f}/{mid:.2f}/{outer:.2f} -> {face}")
    pp, pn = g.get("frac_principal_pos", 0), g.get("frac_principal_neg", 0)
    notes.append(f"axis-free (principal normal) split: neg/pos {pn:.2f}/{pp:.2f}, median "
                 f"{g.get('offset_along_principal_normal_vox', {}).get('median', 0):+.1f} vox; "
                 f"axis hypothesis used: {a.get('axis', {}).get('chosen')}")
    if a.get("axis", {}).get("warning"):
        notes.append("WARNING " + a["axis"]["warning"])
    dice = a["agreement"]["dice_iou_recto_m7"]["dice"]
    if face.startswith("bimodal"):
        rule = "recto_only (m7 is not a single-face label of the recto band; see hist_offset_normal)"
    elif face == "middle" and a["agreement"]["frac_m7_inside_recto"] > 0.9:
        rule = "recto_only" if tr > 2 * tm + 2 else "agree"
    else:
        rule = f"m7_shifted(delta_um={med * a['region']['voxel_um']:+.1f})"
    l2 = a.get("m7_level2", {})
    if "dice_iou_vs_recto" in l2:
        notes.append(f"m7 at level 2: dice vs recto {l2['dice_iou_vs_recto']['dice']:.3f} "
                     f"(level 0: {dice:.3f}); thickness {l2['thickness_l2_voxels'].get('median', 0):.1f} l2-vox")
    return {"recto_is": recto_kind, "m7_face": face, "surface_merge": rule, "notes": notes}


# =========================================================================== #
# Label build (``tsm labels``) -- see docs/label_store.md
# =========================================================================== #
FINE_CHANNELS = ["sdf", "sdf_valid", "ink", "ink_valid"]
FACE_CHANNELS = ["sdf_in", "sdf_out", "faces_valid", "thickness"]  # optional, appended (extra.labels.faces.enabled)
# optional, appended last (built when a `fiber` teacher store exists next to the others):
# the vertical / horizontal-angular fiber probabilities of the 4-class fiber teacher,
# copied through like ink, plus their validity (the CT data mask).
FIBER_CHANNELS = ["fiber_vt", "fiber_hz", "fiber_valid"]


def fine_channels(faces: bool = False, fiber: bool = False, rv: bool = False,
                  winding_fine: bool = False) -> list[str]:
    """Fine-store channel names; ``faces`` / ``fiber`` / ``rv`` / ``winding_fine`` append their
    blocks in that order (an older store is always a prefix of a newer one).  ``rv`` = the
    upstream rectoverso block (:data:`tsm.rvfaces.RV_CHANNELS`), written when
    ``faces.source != "ct"``; ``winding_fine`` = the native 2.4 um winding field derived from
    the two faces (:data:`tsm.winding_fine.WF_CHANNELS`, ``extra.labels.winding_fine``)."""
    from tsm.rvfaces import RV_CHANNELS
    from tsm.winding_fine import WF_CHANNELS

    return (list(FINE_CHANNELS) + (list(FACE_CHANNELS) if faces else [])
            + (list(FIBER_CHANNELS) if fiber else []) + (list(RV_CHANNELS) if rv else [])
            + (list(WF_CHANNELS) if winding_fine else []))


COARSE_CHANNELS = ["phase_sin", "phase_cos", "density", "nx", "ny", "nz", "conf", "valid"]
LABEL_DEFAULTS: dict[str, Any] = {
    "teachers_dir": None,            # default <out_dir>/teachers
    "clip": 20,                      # SDF clip (level-0 voxels)
    "min_component": 100,            # drop 26-components of the recto mask smaller than this
    "recto_threshold": 0.5,
    "ignore_band": [0.2, 0.8],       # recto prob in this band and > ignore_near vox from the medial -> ignore
    "ignore_near": 2,
    "ct_mask_threshold": 30,         # smoothed CT > this (dilated by ct_mask_dilate) = data
    "ct_sigma": 2.0,                 # level-0 gaussian sigma for the CT mask (level 2 uses ct_sigma/2, min 1)
    "ct_mask_dilate": 2,
    "axis_path": DEFAULT_AXIS,
    "brick": [128, 256, 256],        # fine core brick (z, y, x); halo = clip + 4
    "halo": None,
    "coarse_brick": [64, 256, 256],  # level-2 core brick
    "coarse_halo": 4,
    "phase_sigma": 1.0,              # smoothing of the cos channel before the derivative along n
    "phase_eps": 2e-3,               # |d cos/dn| below this -> sign from neighbours
    "conf_window": 5,
    "normal_agreement_min": 0.5,     # coarse valid=2 below this candidate agreement
    "dir_err_max": 0.2,              # coarse valid=2 above this re-encoding RMS error
    "probe_brick": 64,
    "core_radius_vox": 0.0,          # umbilicus core mask (level-0 voxels, 0 = off): faces_valid = 2
                                     # and the winding validity channels 0 within this in-plane
                                     # radius of the axis (tsm.labels.core_mask)
    "faces": None,                   # None/False = off; see FACE_DEFAULTS (extra.labels.faces)
    "winding_fine": None,            # None/False = off; true/dict = append the WF_CHANNELS block
                                     # (native 2.4 um winding phase from the two faces; needs
                                     # faces.enabled and a coarse store).  tsm.winding_fine
    "workers": 1,                    # fine bricks computed in parallel (fork pool); the parent
                                     # still does every write, so the store/summary are identical
    "workers_ram_bytes": None,       # ceiling for workers x per-brick RAM; default is the box's
                                     # MemAvailable - budget.min_avail_mb (budget.ram_bytes is a
                                     # *per process* limit and would be the wrong ceiling here)
    "mp_start": "fork",              # pool start method; "forkserver"/"spawn" cost a few seconds
                                     # of startup but cannot inherit a lock held by a zarr thread
}
FACE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "body_threshold": 60.0,          # smoothed CT above this is sheet body (level 0; see docs)
    "body_sigma": 2.0,
    "closing": 1,                    # 26-neighbourhood binary closing iterations (used when close_radius == 0)
    "close_radius": 6.0,             # ball (or, with close_along_normal, line) closing radius in voxels:
                                     # bridges the few-voxel air gaps *inside* a sheet bundle
    "close_along_normal": False,     # close with a line element along n_out instead of a ball
                                     # (never merges neighbouring sheets across the in-plane gap)
    "min_component": 100,            # drop 26-components of the body smaller than this
    "min_thickness": 15.0,           # component median thickness gate (voxels) -> faces_valid 2
    "max_thickness": 90.0,
    "band_reach": 8.0,               # recto band within this many voxels of a face = "band adjacent"
    "gap_min": 3.0,                  # air must appear within this many voxels beyond the out face
    "side_reach": 6.0,               # local in/out march length along n_out (voxels)
    # --- face source (tsm.rvfaces): "ct" (this builder), "rectoverso" (the upstream bands)
    #     or "merge" (rectoverso where valid, this builder elsewhere + a faces_source channel)
    "source": "ct",
    "rectoverso_store": None,        # rectoverso.zarr from dev/rectoverso_slab.py (rectoverso/merge)
    "rv_min_component": 32,          # drop 26-components of a band smaller than this before thinning
    "rv_normal_window": 7,           # local-PCA window (voxels) for the band normal
    "rv_min_normal_count": 6,        # fewer band voxels in the window -> radial direction
    "rv_planarity_max": 0.5,         # lambda0 > this * lambda1 -> not planar -> radial direction
    "rv_thin_tol": 0.5,              # EDT-ridge tolerance of the band thinning
}
_COARSE_SCALE = 4


# --------------------------------------------------------------------------- #
# Encodings
# --------------------------------------------------------------------------- #
def encode_signed_u8(v: np.ndarray) -> np.ndarray:
    """[-1, 1] -> u8 = round(127.5 + 127.5 v)."""
    return np.clip(np.rint(127.5 + 127.5 * np.asarray(v, dtype=np.float32)), 0, 255).astype(np.uint8)


def decode_signed_u8(u: np.ndarray) -> np.ndarray:
    return (np.asarray(u, dtype=np.float32) - 127.5) / 127.5


def encode_sdf_u8(sdf: np.ndarray, clip: float) -> np.ndarray:
    """Clipped signed distance -> u8 = round(128 + sdf*127/clip) in 1..255 (0 = no data)."""
    v = np.clip(np.asarray(sdf, dtype=np.float32), -float(clip), float(clip))
    return np.clip(np.rint(128.0 + v * (127.0 / float(clip))), 1, 255).astype(np.uint8)


def decode_sdf_u8(u: np.ndarray, clip: float) -> np.ndarray:
    return (np.asarray(u, dtype=np.float32) - 128.0) * (float(clip) / 127.0)


# --------------------------------------------------------------------------- #
# Masks / fields
# --------------------------------------------------------------------------- #
def ct_mask(ct_u8: np.ndarray, threshold: float = 30.0, sigma: float = 2.0, dilate: int = 2) -> np.ndarray:
    """True where gaussian-smoothed CT > threshold, grown by ``dilate`` voxels (6-connectivity)."""
    ct = np.asarray(ct_u8, dtype=np.float32)
    sm = ndi.gaussian_filter(ct, sigma=float(sigma), mode="nearest") if sigma > 0 else ct
    m = sm > float(threshold)
    del sm
    if dilate > 0 and m.any() and not m.all():
        m = ndi.binary_dilation(m, structure=_STRUCT6, iterations=int(dilate))
    return m


def phase_from_cos(cos: np.ndarray, normal_out: np.ndarray, sigma: float = 1.0, eps: float = 2e-3,
                   iters: int = 4) -> np.ndarray:
    """sin(2*pi*w) from cos(2*pi*w) and the outward normal.

    ``cos`` is in [-1, 1] (already ``2*ch - 1``), ``normal_out`` is (3, ...) in (z, y, x).
    w increases outward so d cos/dn = -2*pi*sin*|grad w|: sin = -sign(d cos/dn)*sqrt(1-cos^2),
    with the derivative as the gradient of the sigma-smoothed cos dotted with n (a 1-voxel
    central difference along n).  Where |d cos/dn| < eps (extrema of cos, flat regions) the
    sign is propagated from the 26-neighbourhood, iteratively: in-plane neighbours share the
    phase, while the two neighbours along n straddle an extremum and cancel, so the in-plane
    ones decide.  Voxels still undecided get sign +1 (their sin is ~0 at a true extremum).
    """
    c = np.asarray(cos, dtype=np.float32)
    n = np.asarray(normal_out, dtype=np.float32)
    cs = ndi.gaussian_filter(c, sigma=float(sigma), mode="nearest") if sigma > 0 else c
    d = np.zeros_like(c)
    for ax in range(3):
        g = np.gradient(cs, axis=ax) if cs.shape[ax] > 1 else np.zeros_like(cs)
        d += g.astype(np.float32) * n[ax]
        del g
    del cs
    s = np.sign(d).astype(np.float32)
    tiny = np.abs(d) < float(eps)
    del d
    s[tiny] = 0.0
    for _ in range(int(iters)):
        if not tiny.any():
            break
        f = ndi.uniform_filter(s, size=3, mode="constant")
        s = np.where(tiny, np.sign(f).astype(np.float32), s)
        tiny = tiny & (s == 0.0)
        del f
    s[s == 0.0] = 1.0
    mag = np.sqrt(np.clip(1.0 - c * c, 0.0, 1.0))
    return (-s * mag).astype(np.float32)


#: half-width (level-0 voxels) of the symmetric difference used for the umbilicus tangent
AXIS_TANGENT_DZ = 64


def axis_tangent_at(axis: np.ndarray, z: np.ndarray | float, dz: float = AXIS_TANGENT_DZ) -> np.ndarray:
    """Unit tangent (3, ...) of the umbilicus in (z, y, x) at level-0 ``z``.

    Symmetric difference over +-``dz`` level-0 voxels rather than the derivative of the
    interpolant: the control points are sparse and the knot-local derivative is a step
    function, which would make the frame discontinuous across a crop boundary."""
    z = np.asarray(z, dtype=np.float64)
    y1, x1 = axis_yx_at(axis, z + float(dz))
    y0, x0 = axis_yx_at(axis, z - float(dz))
    t = np.stack([np.full(np.shape(z), 2.0 * float(dz)), np.asarray(y1 - y0, dtype=np.float64),
                  np.asarray(x1 - x0, dtype=np.float64)])
    n = np.sqrt((t * t).sum(0))
    return (t / np.where(n == 0.0, 1.0, n)).astype(np.float32)


def axis_tangent_field(axis: np.ndarray, origin_zyx: Sequence[int], shape: Sequence[int], scale: int = 1,
                       dz: float = AXIS_TANGENT_DZ) -> np.ndarray:
    """Unit umbilicus tangent (3, Z, Y, X) for a box on the grid of ``scale`` (same convention
    as :func:`radial_field`).  The tangent depends on z only, so it is constant in each slice."""
    nz, ny, nx = (int(v) for v in shape)
    z0 = int(origin_zyx[0])
    k = float(scale)
    zs = np.arange(z0, z0 + nz, dtype=np.float64) * k + (k - 1.0) / 2.0
    t = axis_tangent_at(axis, zs, dz)  # (3, Z)
    return np.ascontiguousarray(np.broadcast_to(t[:, :, None, None], (3, nz, ny, nx)))


def radial_field(axis: np.ndarray, origin_zyx: Sequence[int], shape: Sequence[int], scale: int = 1,
                 tangent: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
    """Outward unit radial field (3, Z, Y, X) for a box whose voxel (0,0,0) sits at
    ``origin_zyx`` in the grid of ``scale`` (level-0 voxels = scale*i + (scale-1)/2).

    Without ``tangent`` the radial is the purely in-plane ``(0, y - ay, x - ax)`` -- the
    historical field, exact only where the scroll axis is exactly volume z.  With a
    ``tangent`` ``t`` ((3,) or a (3, Z, Y, X) field, :func:`axis_tangent_field`) the same
    in-plane offset is made perpendicular to the local axis, ``r = normalize(v - (v.t) t)``,
    so ``r`` and ``t`` are an orthonormal pair at every voxel."""
    dz, dy, dx = (int(v) for v in shape)
    z0, y0, x0 = (int(v) for v in origin_zyx)
    k = float(scale)
    off = (k - 1.0) / 2.0
    out = np.empty((3, dz, dy, dx), dtype=np.float32)
    ys = (np.arange(y0, y0 + dy, dtype=np.float64) * k + off)
    xs = (np.arange(x0, x0 + dx, dtype=np.float64) * k + off)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    t = None
    if tangent is not None:
        t = np.asarray(tangent, dtype=np.float64)
        if t.ndim == 1:
            t = t.reshape(3, 1, 1, 1)
        elif t.ndim != 4 or t.shape[0] != 3:
            raise ValueError(f"tangent must be (3,) or (3, Z, Y, X), got {t.shape}")
        t = np.broadcast_to(t, (3, dz, dy, dx))
    for iz in range(dz):
        z = (z0 + iz) * k + off
        ay, ax = axis_yx_at(axis, z)
        ry = yy - ay
        rx = xx - ax
        if t is None:
            rz = np.zeros_like(ry)
        else:
            tz, ty, tx = t[0, iz], t[1, iz], t[2, iz]
            d = ty * ry + tx * rx  # v . t with v_z = 0
            rz = -d * tz
            ry = ry - d * ty
            rx = rx - d * tx
        nrm = np.sqrt(rz * rz + ry * ry + rx * rx)
        nrm[nrm == 0] = 1.0
        out[0, iz] = rz / nrm
        out[1, iz] = ry / nrm
        out[2, iz] = rx / nrm
    return out


def geometry_frame(axis: np.ndarray, origin_zyx: Sequence[int], shape: Sequence[int], scale: int = 1,
                   dz: float = AXIS_TANGENT_DZ) -> dict[str, np.ndarray]:
    """``{"radial", "axis_dir"}``: the orthonormal pair (outward radial, umbilicus tangent),
    each (3, Z, Y, X) in (z, y, x), for a box on the grid of ``scale``.

    Only these two: the binormal ``a x r`` is a pseudovector and would leak ``det(L)`` into
    the net under flip augmentation, and no pseudoscalar exists in the label set."""
    a = axis_tangent_field(axis, origin_zyx, shape, scale, dz)
    r = radial_field(axis, origin_zyx, shape, scale, tangent=a)
    return {"radial": r, "axis_dir": a}


def core_mask(axis: np.ndarray, origin_zyx: Sequence[int], shape: Sequence[int], radius: float,
              scale: int = 1) -> np.ndarray:
    """Bool (Z, Y, X): voxels within ``radius`` **level-0 voxels** of the umbilicus, in plane.

    Same grid convention as :func:`radial_field` (voxel i of a box at ``origin`` in the grid of
    ``scale`` sits at level-0 coordinate ``scale * (origin + i) + (scale - 1) / 2``), and the
    distance is purely in (y, x): the axis is interpolated at each z (:func:`axis_yx_at`).

    This is the umbilicus **core** of ``extra.labels.core_radius_vox`` / ``extra.train.core_radius_vox``.
    Near the core the sheets wrap so tightly that the in/out face convention degenerates -- the
    measured ``recto_is_in`` falls to ~0.47, i.e. a coin flip -- so the labels there are noise
    with a plausible shape, which is worse for the surface head than no labels at all.  Masking
    the core (``faces_valid = 2`` = ignore, winding validity 0) removes it from supervision
    without removing the CT context around it.  ``radius <= 0`` -> all False (the term is off)."""
    dz, dy, dx = (int(v) for v in shape)
    z0, y0, x0 = (int(v) for v in origin_zyx)
    out = np.zeros((dz, dy, dx), dtype=bool)
    r = float(radius)
    if r <= 0:
        return out
    k = float(scale)
    off = (k - 1.0) / 2.0
    ys = np.arange(y0, y0 + dy, dtype=np.float64) * k + off
    xs = np.arange(x0, x0 + dx, dtype=np.float64) * k + off
    for iz in range(dz):
        ay, ax = axis_yx_at(axis, (z0 + iz) * k + off)
        d2 = (ys[:, None] - float(ay)) ** 2 + (xs[None, :] - float(ax)) ** 2
        out[iz] = d2 <= r * r
    return out


def _drop_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=_STRUCT26)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    out = keep[lab]
    del lab
    return out


# --------------------------------------------------------------------------- #
# Fine labels (level 0)
# --------------------------------------------------------------------------- #
def build_fine_labels(
    recto_prob_u8: np.ndarray,
    ink_prob_u8: np.ndarray,
    ct_u8: np.ndarray,
    coarse_normal_up: np.ndarray | None,
    radial: np.ndarray,
    clip: float = 20.0,
    *,
    min_component: int = 100,
    recto_threshold: float = 0.5,
    ignore_band: Sequence[float] = (0.2, 0.8),
    ignore_near: float = 2.0,
    ct_mask_threshold: float = 30.0,
    ct_sigma: float = 2.0,
    ct_mask_dilate: int = 2,
    normal_min_norm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(sdf_u8, sdf_valid_u8, ink_u8, ink_valid_u8) for one (haloed) box; see label_store.md.

    Sign of the SDF: sign((p - m) . n_out(m)) with m the nearest recto-medial voxel and
    n_out the upsampled coarse normal (where its norm >= ``normal_min_norm``, i.e. where the
    coarse field was valid) else the outward radial direction, both evaluated at m.
    Boxes without any medial voxel get sdf=+clip and sdf_valid=2 (sign unknowable).
    """
    recto_prob_u8 = np.asarray(recto_prob_u8)
    shape = recto_prob_u8.shape
    if ink_prob_u8.shape != shape or ct_u8.shape != shape or radial.shape != (3,) + shape:
        raise ValueError(f"shape mismatch: recto {shape} ink {ink_prob_u8.shape} ct {ct_u8.shape} radial {radial.shape}")
    if coarse_normal_up is not None and coarse_normal_up.shape != (3,) + shape:
        raise ValueError(f"coarse_normal_up {coarse_normal_up.shape} != {(3,) + shape}")
    clip = float(clip)
    thr = int(round(float(recto_threshold) * 255))
    mask = _drop_small_components(recto_prob_u8 >= thr, int(min_component))
    med = medial_surface(mask) if mask.any() else mask
    data = ct_mask(ct_u8, ct_mask_threshold, ct_sigma, ct_mask_dilate)
    lo, hi = float(ignore_band[0]), float(ignore_band[1])
    band = (recto_prob_u8 >= int(round(lo * 255))) & (recto_prob_u8 <= int(round(hi * 255)))

    if not med.any():
        sdf_u8 = np.full(shape, encode_sdf_u8(np.float32(clip), clip), dtype=np.uint8)
        sdf_u8[~data] = 0
        sdf_valid = np.where(data, np.uint8(2), np.uint8(0))
        return sdf_u8, sdf_valid, np.asarray(ink_prob_u8, dtype=np.uint8).copy(), data.astype(np.uint8)

    dist, idx = ndi.distance_transform_edt(~med, return_indices=True)
    dist = dist.astype(np.float32)
    del mask
    # outward normal at the nearest medial voxel
    mz, my, mx = idx[0], idx[1], idx[2]
    n_out = radial[:, mz, my, mx]
    # Orientation ambiguity: the coarse field marks voxels whose normal is nearly
    # tangential to the radial direction (flattened windings at the core) as invalid,
    # so wherever the upsampled coarse normal is missing while a coarse field exists,
    # the outward sign here rests on the radial fallback and is untrustworthy -> ignore.
    ambiguous = np.zeros(shape, dtype=bool)
    if coarse_normal_up is not None:
        n_c = coarse_normal_up[:, mz, my, mx]
        nrm = np.sqrt((n_c * n_c).sum(0))
        ok = nrm >= float(normal_min_norm)
        n_out = np.where(ok[None], n_c / np.maximum(nrm, 1e-6)[None], n_out)
        ambiguous = ~ok
        del n_c, nrm, ok
    grid = np.indices(shape, dtype=np.int32)
    dot = np.zeros(shape, dtype=np.float32)
    for ax, m_ax in enumerate((mz, my, mx)):
        dot += (grid[ax] - m_ax).astype(np.float32) * n_out[ax]
    del grid, idx, mz, my, mx, n_out
    sgn = np.where(dot < 0, np.float32(-1.0), np.float32(1.0))
    del dot
    sdf = sgn * np.minimum(dist, clip)
    sdf[med] = 0.0
    del sgn
    sdf_u8 = encode_sdf_u8(sdf, clip)
    del sdf
    ignore = (band & (dist > float(ignore_near))) | ambiguous
    del band, dist, med, ambiguous
    sdf_valid = np.where(data, np.where(ignore, np.uint8(2), np.uint8(1)), np.uint8(0))
    sdf_u8[~data] = 0
    ink_u8 = np.asarray(ink_prob_u8, dtype=np.uint8).copy()
    return sdf_u8, sdf_valid, ink_u8, data.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Two-face surface labels (level 0, optional) -- docs/label_store.md section
# "Two-face surface labels"
# --------------------------------------------------------------------------- #
def _normal_field(coarse_normal_up: np.ndarray | None, radial: np.ndarray,
                  normal_min_norm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(outward unit normal at every voxel, "coarse normal used" mask, fraction re-oriented).

    The upsampled coarse normal where its norm is >= ``normal_min_norm``, else the radial
    direction; the sign is then forced to the store's
    outward convention ``n . r_hat >= 0`` -- the same convention ``build_coarse_field`` applies,
    re-asserted here because a trilinear blend of coarse voxels can flip it.  The fraction of
    voxels whose sign had to be flipped is returned as a diagnostic.
    """
    n = np.array(radial, dtype=np.float32, copy=True)
    used = np.zeros(n.shape[1:], dtype=bool)
    if coarse_normal_up is not None:
        nc = np.asarray(coarse_normal_up, dtype=np.float32)
        nrm = np.sqrt((nc * nc).sum(0))
        used = nrm >= float(normal_min_norm)
        n = np.where(used[None], nc / np.maximum(nrm, 1e-6)[None], n).astype(np.float32)
    dot_r = (n * np.asarray(radial, dtype=np.float32)).sum(0)
    flip = dot_r < 0
    frac = float(flip.mean())
    if flip.any():
        n = np.where(flip[None], -n, n).astype(np.float32)
    return np.ascontiguousarray(n), used, frac


def _signed_side(idx: np.ndarray, n_out: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """sign((p - q) . n_out) for every voxel p, with q = ``idx[:, p]`` (its nearest surface voxel)."""
    grid = np.indices(tuple(int(v) for v in shape), dtype=np.int32)
    dot = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    for ax in range(3):
        dot += (grid[ax] - idx[ax]).astype(np.float32) * n_out[ax]
    del grid
    return np.where(dot < 0, np.float32(-1.0), np.float32(1.0))


def close_ball(mask: np.ndarray, radius: float) -> np.ndarray:
    """Binary closing by a ball of ``radius`` voxels, via two EDTs (exact and fast).

    dilation = {d(x, mask) <= r}, erosion of that = {d(x, complement) > r}; a closing by a ball
    of radius r fills every gap narrower than 2r."""
    r = float(radius)
    if r <= 0 or not mask.any() or mask.all():
        return mask
    pad = int(math.ceil(r))
    m = np.pad(mask, pad, mode="edge")  # edge replication, as everywhere else at box faces
    dil = _edt(~m) <= r
    del m
    out = _edt(dil) > r
    del dil
    return np.ascontiguousarray(out[pad:-pad, pad:-pad, pad:-pad])


def close_normal_line(mask: np.ndarray, n: np.ndarray, radius: float) -> np.ndarray:
    """Binary closing with a line structuring element of half-length ``radius`` oriented, per
    voxel, along ``n`` (3, Z, Y, X).

    Dilation: a voxel joins the set if ``mask`` is set at any of ``p +- t n(p)``, t = 1..r;
    erosion of that: a voxel survives only if the dilated set holds at all of those points.
    Because the element is 1-D along the sheet normal, it bridges the few-voxel air gaps
    *between the layers of one bundle* without ever merging two neighbouring sheets across the
    in-plane air gap (which lies along the same direction but farther than r).  ``n`` varies
    smoothly, so this is a closing only up to the curvature of the sheet field."""
    r = int(math.ceil(float(radius)))
    if r <= 0 or not mask.any() or mask.all():
        return mask
    mask = np.pad(mask, r, mode="edge")
    n = np.pad(n, ((0, 0),) + ((r, r),) * 3, mode="edge")
    shape = mask.shape
    grid = np.indices(shape, dtype=np.float32)

    def sample(m: np.ndarray, t: float) -> np.ndarray:
        idx = []
        for ax in range(3):
            q = np.rint(grid[ax] + t * n[ax]).astype(np.int32)
            np.clip(q, 0, shape[ax] - 1, out=q)
            idx.append(q)
        return m[idx[0], idx[1], idx[2]]

    dil = mask.copy()
    for t in range(1, r + 1):
        for sgn in (1.0, -1.0):
            dil |= sample(mask, sgn * t)
    ero = dil.copy()
    for t in range(1, r + 1):
        for sgn in (1.0, -1.0):
            ero &= sample(dil, sgn * t)
    del grid
    return np.ascontiguousarray(ero[r:-r, r:-r, r:-r])


def _local_faces(boundary: np.ndarray, body: np.ndarray, n_field: np.ndarray, reach: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``boundary`` into (face_in, face_out, ambiguous) by a purely LOCAL march.

    For a boundary voxel ``p`` with outward normal ``n_out(p)``, march ``t = 1..reach`` voxels:
    if the body is hit along ``+n_out`` the body lies *outward* of ``p``, so ``p`` is the IN face
    (the face toward the axis); if it is hit along ``-n_out`` ``p`` is the OUT face.  Hitting on
    both sides (a thin spur, a corner, or a neighbouring sheet closer than ``reach``) or on
    neither leaves the voxel ambiguous -> ``faces_valid = 2``.  No component centroid, no medial
    surface: the decision uses only the body within ``reach`` voxels of ``p`` along the normal.
    """
    fin = np.zeros(boundary.shape, dtype=bool)
    fout = np.zeros(boundary.shape, dtype=bool)
    amb = np.zeros(boundary.shape, dtype=bool)
    if not boundary.any():
        return fin, fout, amb
    bz, by, bx = np.nonzero(boundary)
    pts = np.stack([bz, by, bx], axis=1).astype(np.float32)
    n = np.ascontiguousarray(n_field[:, bz, by, bx])
    plus = _march_hits_body(pts, n, body, int(reach))
    minus = _march_hits_body(pts, -n, body, int(reach))
    del pts, n
    fin[bz, by, bx] = plus & ~minus
    fout[bz, by, bx] = minus & ~plus
    amb[bz, by, bx] = ~(plus ^ minus)
    return fin, fout, amb


def _march_hits_body(pts: np.ndarray, n: np.ndarray, body: np.ndarray, steps: int) -> np.ndarray:
    """For each point ``pts`` (N, 3) march ``steps`` voxels along ``n`` (3, N); True if any
    sampled voxel is inside ``body``, i.e. the ``steps`` voxels beyond that face are not all
    air and the next sheet starts right there (faces in contact)."""
    shape = np.array(body.shape, dtype=np.int64)
    hit = np.zeros(len(pts), dtype=bool)
    for t in range(1, int(steps) + 1):
        q = np.rint(pts + t * n.T).astype(np.int64)
        np.clip(q, 0, shape - 1, out=q)
        hit |= body[q[:, 0], q[:, 1], q[:, 2]]
        del q
    return hit


def build_face_labels(
    recto_prob_u8: np.ndarray,
    ct_u8: np.ndarray,
    coarse_normal_up: np.ndarray | None,
    radial: np.ndarray,
    clip: float = 20.0,
    *,
    body_threshold: float = 60.0,
    body_sigma: float = 2.0,
    closing: int = 1,
    close_radius: float = 6.0,
    close_along_normal: bool = False,
    min_component: int = 100,
    min_thickness: float = 15.0,
    max_thickness: float = 90.0,
    band_reach: float = 8.0,
    gap_min: float = 3.0,
    side_reach: float = 6.0,
    recto_threshold: float = 0.5,
    ignore_band: Sequence[float] = (0.2, 0.8),
    ignore_near: float = 2.0,
    ct_mask_threshold: float = 30.0,
    ct_sigma: float = 2.0,
    ct_mask_dilate: int = 2,
    normal_min_norm: float = 0.5,
    stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(sdf_in_u8, sdf_out_u8, faces_valid_u8, thickness_u8) for one (haloed) box.

    Sheet body = gaussian-smoothed CT (``body_sigma``) > ``body_threshold``, closed by a ball of
    ``close_radius`` voxels (or, with ``close_along_normal``, by a line element of that half-length
    oriented along the outward normal) so that the several thin fibre layers of one sheet *bundle*
    become one body while the wider inter-sheet air gap stays open; ``close_radius = 0`` falls back
    to ``closing`` iterations of the 26-neighbourhood structure.  26-components smaller than
    ``min_component`` are dropped.  The two faces of a sheet are the body boundary split by a
    purely LOCAL march along the outward normal ``n_out`` (upsampled coarse normal where valid,
    else radial, sign forced to ``n . r_hat >= 0``, evaluated *at the boundary voxel itself*):
    from a boundary voxel ``p``, if the body is hit within ``side_reach`` voxels along
    ``+n_out`` the body lies outward of ``p`` and ``p`` is the IN face; if it is hit along
    ``-n_out`` ``p`` is the OUT face; body on both sides or on neither -> ambiguous ->
    ``faces_valid = 2`` (:func:`_local_faces`).  Neither a component centroid nor the medial
    surface enters the decision.  The split is purely geometric -- the recto teacher is never used to name a face; ``stats`` (if given)
    receives the per-component diagnostic ``recto_is_in`` = the fraction of the
    band-adjacent boundary (recto prob >= ``recto_threshold`` within ``band_reach`` voxels)
    that landed on the IN face (~1 is expected for these scrolls).

    ``sdf_in`` / ``sdf_out`` are the signed distances to those two surfaces, positive on the
    outward side, clipped to +-``clip`` and encoded with :func:`encode_sdf_u8` (0 = no data).
    ``thickness_u8`` is the local sheet thickness (``2 * EDT`` at the nearest medial voxel)
    rounded and clipped to 255.  ``faces_valid``: 0 no data (outside the CT mask), 1 supervise,
    2 ignore -- voxels where the coarse normal was missing (the outward sense would rest on the
    radial fallback), boundary voxels the local march could not classify (and their neighbourhood),
    sheets in contact (the ``gap_min`` voxels immediately beyond the out face,
    along ``n_out``, are not all air: another sheet starts there),
    components whose median thickness is outside ``[min_thickness, max_thickness]``
    (delaminated layers / fused blobs), and the recto ignore band.
    """
    recto_prob_u8 = np.asarray(recto_prob_u8)
    shape = tuple(int(v) for v in recto_prob_u8.shape)
    if tuple(ct_u8.shape) != shape or tuple(radial.shape) != (3,) + shape:
        raise ValueError(f"shape mismatch: recto {shape} ct {ct_u8.shape} radial {radial.shape}")
    if coarse_normal_up is not None and tuple(coarse_normal_up.shape) != (3,) + shape:
        raise ValueError(f"coarse_normal_up {coarse_normal_up.shape} != {(3,) + shape}")
    clip = float(clip)
    data = ct_mask(ct_u8, ct_mask_threshold, ct_sigma, ct_mask_dilate)
    sm = ndi.gaussian_filter(np.asarray(ct_u8, dtype=np.float32), sigma=float(body_sigma), mode="nearest")
    body = sm > float(body_threshold)
    del sm
    if float(close_radius) > 0 and body.any() and not body.all():
        # A "sheet" here is a bundle of several thin fibre layers separated by air gaps of a few
        # voxels; close them into one body without merging neighbouring sheets across the larger
        # inter-sheet gap.
        if close_along_normal:
            n_field, _, _ = _normal_field(coarse_normal_up, radial, normal_min_norm)
            body = close_normal_line(body, n_field, float(close_radius))
            del n_field
        else:
            body = close_ball(body, float(close_radius))
    elif int(closing) > 0 and body.any() and not body.all():
        # pad with edge replication: scipy's closing erodes with border_value 0, which would
        # shave a shell off the box faces and turn them into spurious "boundary" voxels.
        pad = int(closing)
        b = np.pad(body, pad, mode="edge")
        b = ndi.binary_closing(b, structure=_STRUCT26, iterations=pad)
        body = np.ascontiguousarray(b[pad:-pad, pad:-pad, pad:-pad])
        del b
    body = _drop_small_components(body, int(min_component))

    no_face = np.full(shape, encode_sdf_u8(np.float32(clip), clip), dtype=np.uint8)
    if not body.any():
        no_face[~data] = 0
        if stats is not None:
            stats.setdefault("components", [])
            stats["no_body_boxes"] = stats.get("no_body_boxes", 0) + 1
        return no_face, no_face.copy(), np.where(data, np.uint8(2), np.uint8(0)), np.zeros(shape, np.uint8)

    # ---- medial surface, local thickness, component labels -----------------
    d = _edt(body)
    med = medial_surface(body, d)
    thick_med = (2.0 * d).astype(np.float32)
    del d
    lab, n_lab = ndi.label(body, structure=_STRUCT26)
    _, midx = ndi.distance_transform_edt(~med, return_indices=True)
    midx = midx.astype(np.int32)
    thickness = thick_med[midx[0], midx[1], midx[2]]
    comp = lab[midx[0], midx[1], midx[2]].astype(np.int32)  # nearest-sheet component of every voxel

    # ---- faces: split the boundary by a LOCAL march along the outward normal ----
    n_field, used_n, flip_frac = _normal_field(coarse_normal_up, radial, normal_min_norm)
    boundary = body & _shift_any(~body, False)
    face_in, face_out, face_amb = _local_faces(boundary, body, n_field, int(math.ceil(float(side_reach))))
    # diagnostic only: the old rule (side of the nearest medial voxel along n_out there)
    disagree = float("nan")
    if boundary.any():
        n_med = np.ascontiguousarray(n_field[:, midx[0], midx[1], midx[2]])
        side_c = _signed_side(midx, n_med, shape)
        del n_med
        dec = boundary & ~face_amb
        if dec.any():
            disagree = float(((side_c[dec] > 0) != face_out[dec]).mean())
        del side_c, dec
    del midx

    def signed_sdf(face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = ndi.distance_transform_edt(~face, return_indices=True)
        idx = idx.astype(np.int32)
        n_f = np.ascontiguousarray(n_field[:, idx[0], idx[1], idx[2]])
        s = _signed_side(idx, n_f, shape)
        del n_f
        sdf = s * np.minimum(dist.astype(np.float32), clip)
        sdf[face] = 0.0
        return sdf, idx

    if face_in.any():
        sdf_in, _ = signed_sdf(face_in)
    else:
        sdf_in = np.full(shape, clip, np.float32)
    if face_out.any():
        sdf_out, idx_out = signed_sdf(face_out)
    else:
        sdf_out, idx_out = np.full(shape, -clip, np.float32), None

    # ---- validity ----------------------------------------------------------
    ignore = np.zeros(shape, dtype=bool)
    # (a) component thickness gate (delaminated layers / fused multi-sheet blobs)
    comp_stats: list[dict[str, Any]] = []
    bad = np.zeros(n_lab + 1, dtype=bool)
    bad[0] = True  # background: nearest sheet unknown -> not a sheet component
    if n_lab:
        lab_med = lab[med]
        th_med = thick_med[med]
        thr_r = int(round(float(recto_threshold) * 255))
        recto_near = _edt(~(recto_prob_u8 >= thr_r)) <= float(band_reach)
        adj = boundary & recto_near
        del recto_near
        lab_adj = lab[adj]
        in_adj = face_in[adj]
        for c in range(1, n_lab + 1):
            selm = lab_med == c
            if not selm.any():
                continue
            tv = th_med[selm]
            med_t = float(np.median(tv))
            sa = lab_adj == c
            n_adj = int(sa.sum())
            rec = float(in_adj[sa].mean()) if n_adj else float("nan")
            ok = float(min_thickness) <= med_t <= float(max_thickness)
            bad[c] = not ok
            comp_stats.append({"component": int(c), "voxels": int((lab == c).sum()),
                               "thickness_median": med_t, "thickness_p10": float(np.percentile(tv, 10)),
                               "thickness_p90": float(np.percentile(tv, 90)),
                               "band_adjacent_face_voxels": n_adj, "recto_is_in": rec, "ok": bool(ok)})
        del lab_med, th_med, adj, lab_adj, in_adj
    ignore |= bad[comp]
    del comp, thick_med, med, lab
    # (b) sheets in contact: no air within gap_min voxels beyond the out face
    n_contact = 0
    if idx_out is not None and face_out.any():
        oz, oy, ox = np.nonzero(face_out)
        pts = np.stack([oz, oy, ox], axis=1).astype(np.float32)
        n_f = np.ascontiguousarray(n_field[:, oz, oy, ox])
        touch = _march_hits_body(pts, n_f, body, int(math.ceil(float(gap_min))))
        del pts, n_f, oz, oy, ox
        contact = np.zeros(shape, dtype=bool)
        contact[face_out] = touch
        n_contact = int(touch.sum())
        ignore |= contact[idx_out[0], idx_out[1], idx_out[2]] & (np.abs(sdf_out) < clip)
        del contact, touch
    face_out_n = int(face_out.sum())
    del idx_out, face_out, boundary, body
    # (c) ambiguous orientation: the nearest surface is a boundary voxel the local march could
    #     not classify (body on both sides / neither) -> the in/out naming is unknowable there
    #     ... and where the coarse field exists but its normal was missing at the voxel: the
    #     outward sense then rests on the radial fallback (same rule as build_fine_labels)
    if coarse_normal_up is not None:
        ignore |= ~used_n
    n_amb = int(face_amb.sum())
    if n_amb:
        # A voxel is ignored only when its NEAREST face voxel is an ambiguous one *and* it is
        # within `side_reach` of it.  (Until 2026-09-06 the radius was `clip`, which turned every
        # ambiguous boundary voxel into a 20-voxel ignore teardrop; on the crumpled crop 99.96 %
        # of all ignores came from this one rule.  The march itself only looks `side_reach`
        # voxels ahead, so that is the distance over which its verdict means anything.)
        d_amb = _edt(~face_amb)
        ignore |= (d_amb <= float(side_reach)) & (d_amb <= np.minimum(np.abs(sdf_in), np.abs(sdf_out)))
        del d_amb
    del face_amb
    # (d) recto ignore band
    lo, hi = float(ignore_band[0]), float(ignore_band[1])
    near = np.minimum(np.abs(sdf_in), np.abs(sdf_out)) <= float(ignore_near)
    ignore |= (recto_prob_u8 >= int(round(lo * 255))) & (recto_prob_u8 <= int(round(hi * 255))) & ~near
    del near

    faces_valid = np.where(data, np.where(ignore, np.uint8(2), np.uint8(1)), np.uint8(0))
    sdf_in_u8 = encode_sdf_u8(sdf_in, clip)
    sdf_out_u8 = encode_sdf_u8(sdf_out, clip)
    sdf_in_u8[~data] = 0
    sdf_out_u8[~data] = 0
    thickness_u8 = np.clip(np.rint(thickness), 0, 255).astype(np.uint8)
    thickness_u8[~data] = 0
    if stats is not None:
        stats.setdefault("components", []).extend(comp_stats)
        stats["face_in_voxels"] = stats.get("face_in_voxels", 0) + int(face_in.sum())
        stats["face_out_voxels"] = stats.get("face_out_voxels", 0) + face_out_n
        stats["face_ambiguous_voxels"] = stats.get("face_ambiguous_voxels", 0) + n_amb
        stats["contact_face_voxels"] = stats.get("contact_face_voxels", 0) + n_contact
        stats["coarse_normal_used"] = stats.get("coarse_normal_used", 0) + int(used_n.sum())
        stats.setdefault("normal_flipped_frac", []).append(flip_frac)
        stats.setdefault("local_vs_centroid_disagree_frac", []).append(disagree)
    return sdf_in_u8, sdf_out_u8, faces_valid, thickness_u8


# --------------------------------------------------------------------------- #
# Coarse field (level 2)
# --------------------------------------------------------------------------- #
def build_coarse_field(
    lasagna_u8: np.ndarray,
    ct_l2_u8: np.ndarray,
    radial_l2: np.ndarray,
    *,
    ct_mask_threshold: float = 30.0,
    ct_sigma: float = 1.0,
    ct_mask_dilate: int = 2,
    phase_sigma: float = 1.0,
    phase_eps: float = 2e-3,
    conf_window: int = 5,
    normal_agreement_min: float = 0.5,
    dir_err_max: float = 0.2,
    orient_min: float = 0.3,
) -> np.ndarray:
    """(8, z, y, x) uint8 coarse store channels from the 8 lasagna channels; see label_store.md."""
    las = np.asarray(lasagna_u8)
    if las.shape[0] != 8 or las.ndim != 4:
        raise ValueError(f"lasagna must be (8, z, y, x), got {las.shape}")
    shape = las.shape[1:]
    if ct_l2_u8.shape != shape or radial_l2.shape != (3,) + shape:
        raise ValueError(f"shape mismatch: lasagna {shape} ct {ct_l2_u8.shape} radial {radial_l2.shape}")
    ch = las.astype(np.float32) / 255.0
    cosw = (2.0 * ch[0] - 1.0).astype(np.float32)
    n, agree, enc_err = decode_lasagna_normal(ch, return_extra=True)
    del ch
    flip = (n * radial_l2).sum(0) < 0
    n[:, flip] *= -1.0
    del flip
    sinw = phase_from_cos(cosw, n, sigma=phase_sigma, eps=phase_eps)
    data = ct_mask(ct_l2_u8, ct_mask_threshold, ct_sigma, ct_mask_dilate)
    data &= las.any(axis=0)  # teacher wrote nothing here (empty tile)
    w = int(conf_window)
    contrast = ndi.maximum_filter(cosw, size=w, mode="nearest") - ndi.minimum_filter(cosw, size=w, mode="nearest")
    conf = np.clip(np.rint(contrast * (255.0 / 2.0)), 0, 255).astype(np.uint8)
    conf[~data] = 0
    del contrast
    bad = (agree < float(normal_agreement_min)) | (enc_err > float(dir_err_max))
    bad |= np.abs((n * radial_l2).sum(0)) < float(orient_min)  # outward orientation ambiguous
    valid = np.where(data, np.where(bad, np.uint8(2), np.uint8(1)), np.uint8(0))
    out = np.empty((8,) + shape, dtype=np.uint8)
    out[0] = encode_signed_u8(sinw)
    out[1] = encode_signed_u8(cosw)
    out[2] = las[1]
    out[3] = encode_signed_u8(n[2])  # nx
    out[4] = encode_signed_u8(n[1])  # ny
    out[5] = encode_signed_u8(n[0])  # nz
    out[6] = conf
    out[7] = valid
    return out


def upsample_coarse_normal(coarse_u8: np.ndarray, lo: Sequence[int], hi: Sequence[int],
                           scale: int = _COARSE_SCALE, valid_only: bool = True) -> np.ndarray:
    """Trilinear upsample of the coarse store's oriented normal to the fine box [lo, hi)
    (fine-grid local coordinates; may extend outside, clamped).  Returns (3, dz, dy, dx)
    float32 in (z, y, x) order, renormalised; zero where the coarse field is not valid==1
    (the trilinear weight of valid voxels decides at the border), so callers can fall back
    on the norm.  ``coarse_u8`` is the (8, ...) store array (zarr or numpy)."""
    k = int(scale)
    cshape = tuple(int(s) for s in coarse_u8.shape[1:])
    lo = [int(v) for v in lo]
    hi = [int(v) for v in hi]
    # coarse sub-block covering fine [lo, hi): fine i -> coarse (i + 0.5)/k - 0.5
    c0 = [max(0, int(np.floor((lo[a] + 0.5) / k - 0.5)) - 1) for a in range(3)]
    c1 = [min(cshape[a], int(np.ceil((hi[a] - 0.5) / k - 0.5)) + 2) for a in range(3)]
    c1 = [max(c1[a], c0[a] + 1) for a in range(3)]
    sel = (slice(3, 6), slice(c0[0], c1[0]), slice(c0[1], c1[1]), slice(c0[2], c1[2]))
    sub = np.asarray(coarse_u8[sel])
    n = decode_signed_u8(sub[::-1])  # (nz, ny, nx)
    if valid_only:
        vsel = (7,) + sel[1:]
        v = np.asarray(coarse_u8[vsel]) == 1
        n[:, ~v] = 0.0
    up = np.stack([ndi.zoom(n[a], zoom=k, order=1, mode="nearest", grid_mode=True) for a in range(3)])
    del n
    take = []
    for a in range(3):
        i = np.arange(lo[a], hi[a]) - c0[a] * k
        take.append(np.clip(i, 0, up.shape[a + 1] - 1))
    up = up[:, take[0]][:, :, take[1]][:, :, :, take[2]]
    nrm = np.sqrt((up * up).sum(0))
    ok = nrm > 1e-6
    up[:, ok] /= nrm[ok]
    if valid_only:
        up[:, nrm < 0.5] = 0.0  # mostly-invalid neighbourhood: leave zero (norm-based fallback)
    return np.ascontiguousarray(up, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Brick helpers
# --------------------------------------------------------------------------- #
def iter_cores(size: Sequence[int], brick: Sequence[int]):
    """Yield (lo, hi) local boxes tiling ``size`` with cores of ``brick`` (last ones clipped)."""
    size = [int(v) for v in size]
    brick = [max(1, int(v)) for v in brick]
    for z in range(0, size[0], brick[0]):
        for y in range(0, size[1], brick[1]):
            for x in range(0, size[2], brick[2]):
                lo = (z, y, x)
                hi = (min(z + brick[0], size[0]), min(y + brick[1], size[1]), min(x + brick[2], size[2]))
                yield lo, hi


def read_box_padded(arr: Any, lo: Sequence[int], hi: Sequence[int], channels: slice | int | None = None) -> np.ndarray:
    """Box [lo, hi) of a (C, Z, Y, X) or (Z, Y, X) array (zarr or numpy) with edge replication
    outside the array bounds (never all-zero halos at region faces)."""
    nd = arr.ndim
    shape = tuple(int(s) for s in arr.shape[nd - 3:])
    lo = [int(v) for v in lo]
    hi = [int(v) for v in hi]
    idx = [np.clip(np.arange(lo[a], hi[a]), 0, shape[a] - 1) for a in range(3)]
    sl = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
    if nd == 4:
        lead = (slice(None),) if channels is None else (channels,)
        block = np.asarray(arr[lead + sl])
        if block.ndim == 3:
            block = block[None]
        out = block[:, idx[0] - sl[0].start][:, :, idx[1] - sl[1].start][:, :, :, idx[2] - sl[2].start]
    else:
        block = np.asarray(arr[sl])
        out = block[idx[0] - sl[0].start][:, idx[1] - sl[1].start][:, :, idx[2] - sl[2].start]
    return np.ascontiguousarray(out)


def fiber_block(fiber: Any, hlo: Sequence[int], hhi: Sequence[int],
                core: tuple[slice, slice, slice], ink_valid_core: np.ndarray) -> list[np.ndarray]:
    """The :data:`FIBER_CHANNELS` block of one brick, as written by ``tsm labels``.

    Channels 1 (vertical) and 2 (horizontal/angular) of ``fiber.zarr`` are copied through
    like ink -- the haloed box ``[hlo, hhi)`` is read with edge replication and cropped to
    ``core``, keeping the teacher's uint8 encoding -- and ``fiber_valid`` is the same CT
    data mask ``build_fine_labels`` returns for ink (``ink_valid_core``, already cropped).
    """
    fb = read_box_padded(fiber, hlo, hhi, channels=slice(1, 3))
    return [np.ascontiguousarray(fb[0][core]), np.ascontiguousarray(fb[1][core]),
            np.ascontiguousarray(ink_valid_core)]


def _label_opts(cfg: RunCfg) -> dict[str, Any]:
    raw = cfg.extra.get("labels", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.labels must be an object")
    unknown = sorted(set(raw) - set(LABEL_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.labels keys: {unknown}")
    opts = dict(LABEL_DEFAULTS)
    opts.update(raw)
    if not opts["teachers_dir"]:
        opts["teachers_dir"] = os.path.join(cfg.out_dir, "teachers")
    if opts["halo"] is None:
        opts["halo"] = int(opts["clip"]) + 4
    if int(opts["halo"]) < int(opts["clip"]) + 2:
        raise ValueError("labels.halo must be >= clip + 2")
    lo, hi = opts["ignore_band"]
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("labels.ignore_band must be [lo, hi] within [0, 1]")
    if float(opts["core_radius_vox"]) < 0:
        raise ValueError("extra.labels.core_radius_vox must be >= 0")
    opts["core_radius_vox"] = float(opts["core_radius_vox"])
    opts["faces"] = _faces_opts(opts.get("faces"))
    from tsm.winding_fine import wf_opts

    opts["winding_fine"] = wf_opts(opts.get("winding_fine"))
    if opts["winding_fine"]["enabled"] and not opts["faces"]["enabled"]:
        raise ValueError("extra.labels.winding_fine needs extra.labels.faces.enabled (it is "
                         "derived from sdf_in / sdf_out)")
    if str(opts["faces"]["source"]) != "ct" and "ignore_band" not in raw:
        opts["ignore_band"] = list(MERGE_IGNORE_BAND)   # see MERGE_CT_DEFAULTS
    return opts


#: With ``faces.source != "ct"`` the CT builder is only the *fallback* where the upstream bands
#: say nothing, so it is tuned for coverage rather than for standing alone.  Measured on the
#: crumpled crop (2026-09-06 sweep, ~/tsm-output/sweep_faces): the production settings
#: (body_threshold 60, close_radius 6, side_reach 6) weld neighbouring sheets -- the body fills
#: 87 % of the crop as ONE 26-component per brick and the median labelled thickness is 48 voxels,
#: about two sheets (a visible sheet is 15-25).  body_threshold 80 + side_reach 3 splits them
#: again (median thickness 22) and doubles the supervised face voxels; min_thickness then has to
#: drop to 10 because true sheets sit at ~22.  Any key given explicitly in the config wins.
MERGE_CT_DEFAULTS: dict[str, Any] = {
    "body_threshold": 80.0,
    "close_radius": 6.0,
    "side_reach": 3.0,
    "min_thickness": 10.0,
}
#: ...and the labels-level recto ignore band is narrowed for the same reason (~14 pp less ignore
#: with no geometry change).
MERGE_IGNORE_BAND = [0.35, 0.65]


def _faces_opts(raw: Any) -> dict[str, Any]:
    """``extra.labels.faces``: None/bool/dict -> a full FACE_DEFAULTS dict."""
    if raw is None or raw is False:
        return dict(FACE_DEFAULTS)
    if raw is True:
        return {**FACE_DEFAULTS, "enabled": True}
    if not isinstance(raw, dict):
        raise ValueError("extra.labels.faces must be an object, a bool or null")
    unknown = sorted(set(raw) - set(FACE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.labels.faces keys: {unknown}")
    out = {**FACE_DEFAULTS, **raw}
    if str(out.get("source", "ct")) != "ct":
        for k, v in MERGE_CT_DEFAULTS.items():
            if k not in raw:
                out[k] = v
    if float(out["min_thickness"]) >= float(out["max_thickness"]):
        raise ValueError("labels.faces.min_thickness must be < max_thickness")
    from tsm.rvfaces import FACE_SOURCES

    if str(out["source"]) not in FACE_SOURCES:
        raise ValueError(f"labels.faces.source must be one of {list(FACE_SOURCES)}, got {out['source']!r}")
    if str(out["source"]) != "ct" and not out["rectoverso_store"]:
        raise ValueError(f"labels.faces.source = {out['source']!r} needs labels.faces.rectoverso_store")
    return out


def _fine_budget_rows(halo_shape: Sequence[int], faces: bool = False, fiber: bool = False,
                      rv: bool = False, wf: bool = False) -> list[tuple[str, Sequence[int], Any]]:
    s = tuple(int(v) for v in halo_shape)
    extra: list[tuple[str, Sequence[int], Any]] = ([("fiber_u8 x2", (2,) + s, np.uint8)] if fiber else [])
    extra += [] if not wf else [
        ("wf_edt_idx_i32", (3,) + s, np.int32),
        ("wf_f32 x8", (8,) + s, np.float32),
        ("wf_normal_f32", (3,) + s, np.float32),
        ("wf_out_u8", (8,) + s, np.uint8),
    ]
    extra += [] if not rv else [
        ("rv_class_u8 x2", (2,) + s, np.uint8),
        ("rv_moment_f64 x2", (2,) + s, np.float64),
        ("rv_face_masks_bool x4", (4,) + s, np.bool_),
        ("rv_sdf_f32 x4", (4,) + s, np.float32),
    ]
    extra += [] if not faces else [
        ("faces_edt_f64", s, np.float64),
        ("faces_idx_i32", (3,) + s, np.int32),
        ("faces_masks_bool x6", (6,) + s, np.bool_),
        ("faces_f32 x4", (4,) + s, np.float32),
    ]
    return extra + [
        ("recto/ink/ct_u8 x3", (3,) + s, np.uint8),
        ("edt_f64", s, np.float64),
        ("medial_idx_i32", (3,) + s, np.int32),
        ("grid_i32", (3,) + s, np.int32),
        ("n_out_f32", (3,) + s, np.float32),
        ("radial_f32", (3,) + s, np.float32),
        ("coarse_normal_up_f32", (3,) + s, np.float32),
        ("sdf/dot_f32 x2", (2,) + s, np.float32),
    ]


def _coarse_budget_rows(halo_shape: Sequence[int]) -> list[tuple[str, Sequence[int], Any]]:
    s = tuple(int(v) for v in halo_shape)
    return [
        ("lasagna_u8", (8,) + s, np.uint8),
        ("lasagna_f32", (8,) + s, np.float32),
        ("decode_f64_temps", (24,) + s, np.float64),
        ("normal/radial_f32", (6,) + s, np.float32),
        ("out_u8", (8,) + s, np.uint8),
    ]


# --------------------------------------------------------------------------- #
# Previews
# --------------------------------------------------------------------------- #
def _sdf_rgb(sdf_u8: np.ndarray, valid_u8: np.ndarray, clip: float) -> np.ndarray:
    v = decode_sdf_u8(sdf_u8, clip) / float(clip)  # -1..1
    t = np.clip(np.abs(v), 0, 1)
    rgb = np.empty(sdf_u8.shape + (3,), dtype=np.float32)
    pos = v >= 0
    rgb[..., 0] = np.where(pos, 255.0, 255.0 * (1 - t))
    rgb[..., 1] = 255.0 * (1 - t)
    rgb[..., 2] = np.where(pos, 255.0 * (1 - t), 255.0)
    rgb[valid_u8 == 0] = 0.0
    rgb[valid_u8 == 2] *= 0.5
    zero = (sdf_u8 == 128) & (valid_u8 > 0)
    rgb[zero] = (0.0, 0.0, 0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _write_previews(ldir: str, fine: Any, coarse: Any, clip: float, k: int) -> dict[str, str]:
    png: dict[str, str] = {}
    zm = fine.shape[1] // 2
    sdf = np.asarray(fine[0, zm])
    val = np.asarray(fine[1, zm])
    png["sdf"] = _save_png(os.path.join(ldir, "preview_sdf.png"), _sdf_rgb(sdf, val, clip))
    png["sdf_valid"] = _save_png(os.path.join(ldir, "preview_sdf_valid.png"), (val.astype(np.uint16) * 127).astype(np.uint8))
    png["ink"] = _save_png(os.path.join(ldir, "preview_ink.png"), np.asarray(fine[2, zm]))
    png["ink_valid"] = _save_png(os.path.join(ldir, "preview_ink_valid.png"), np.asarray(fine[3, zm]) * 255)
    fch = list(dict(fine.attrs).get("channels", []))
    for name in ("fiber_vt", "fiber_hz"):
        if name in fch:
            png[name] = _save_png(os.path.join(ldir, f"preview_{name}.png"), np.asarray(fine[fch.index(name), zm]))
    czm = coarse.shape[1] // 2
    c = np.asarray(coarse[:, czm])

    def up(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, k, axis=0), k, axis=1)

    for i, name in ((0, "phase_sin"), (1, "phase_cos"), (6, "conf")):
        png[f"coarse_{name}"] = _save_png(os.path.join(ldir, f"preview_coarse_{name}.png"), up(c[i]))
    png["coarse_density"] = _save_png(os.path.join(ldir, "preview_coarse_density.png"),
                                      up(np.clip(c[2].astype(np.uint16) * 8, 0, 255).astype(np.uint8)))
    png["coarse_normal_rgb"] = _save_png(os.path.join(ldir, "preview_coarse_normal.png"),
                                         up(np.ascontiguousarray(np.moveaxis(c[3:6], 0, -1))))
    png["coarse_valid"] = _save_png(os.path.join(ldir, "preview_coarse_valid.png"), up((c[7].astype(np.uint16) * 127).astype(np.uint8)))
    return png


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _llog(msg: str) -> None:
    print(f"[labels] {msg}", flush=True)


def open_teacher(tdir: str, name: str, want_shape: tuple[int, ...], want_origin: tuple[int, ...]) -> Any:
    """``<tdir>/<name>.zarr``, as a :class:`tsm.rvfaces.SubView` when it covers more than the region."""
    import zarr

    arr = zarr.open_array(store=os.path.join(tdir, f"{name}.zarr"), mode="r")
    attrs = dict(arr.attrs)
    o = tuple(int(v) for v in attrs.get("origin_zyx", want_origin))
    sh = tuple(int(s) for s in arr.shape[1:])
    if o == want_origin and sh == want_shape:
        return arr
    # A teacher store covering a *larger* region is fine and is how a crop config reuses the
    # whole-slab teachers: read the sub-box through a shifted view (tsm.rvfaces.SubView), so
    # halos are still edge-replicated at the config region's faces.
    from tsm.rvfaces import SubView

    off = tuple(want_origin[a] - o[a] for a in range(3))
    if any(v < 0 for v in off) or any(off[a] + want_shape[a] > sh[a] for a in range(3)):
        raise ValueError(f"{name}.zarr region {o}+{sh} does not contain {want_origin}+{want_shape}")
    _llog(f"{name}.zarr covers {o}+{sh} > the config region: reading the sub-box at offset {off}")
    return SubView(arr, off, want_shape)


# --------------------------------------------------------------------------- #
# Fine bricks: one pure function per brick, so ``labels.workers`` can fan them out
#
# Every worker opens its own readers once (``_fine_worker_init``) and then answers brick
# jobs with (arrays, stats deltas); the PARENT does all BrickWriter writes, done.json
# updates and stats accumulation, in brick order.  Running with workers=1 takes the same
# path in-process, and because the parent merges strictly in order the resulting store,
# done.json and summary are byte-identical whatever ``workers`` is.
# --------------------------------------------------------------------------- #
_FINE_W: dict[str, Any] = {}


def _fine_worker_init(spec: dict[str, Any]) -> None:
    """Open the per-process inputs for :func:`_fine_brick` (pool initializer)."""
    global _FINE_W
    import zarr

    from tsm.cli import open_ct

    cfg, opts = spec["cfg"], spec["opts"]
    tdir = opts["teachers_dir"]
    shape, origin = tuple(spec["shape"]), tuple(spec["origin"])
    w: dict[str, Any] = {
        "spec": spec, "opts": opts,
        "recto": open_teacher(tdir, "recto", shape, origin),
        "ink": open_teacher(tdir, "ink", shape, origin),
        "fiber": open_teacher(tdir, "fiber", shape, origin) if spec["do_fiber"] else None,
        "coarse": zarr.open_array(store=spec["coarse_path"], mode="r"),
        "ct0": open_ct(cfg, level_shift=0, probe=int(opts["probe_brick"])),
        "axis": load_axis(opts["axis_path"]),
        "rv_arr": None, "rv_origin": None,
    }
    if spec["do_rv"]:
        from tsm import rvfaces

        w["rv_arr"], w["rv_origin"] = rvfaces.open_rv_store(opts["faces"]["rectoverso_store"])
    _FINE_W = w


def _fine_brick(job: tuple[int, tuple[int, int, int], tuple[int, int, int]]) -> tuple:
    """Compute one fine brick: ``(i, lo, hi, channel arrays, rv arrays, fine delta, face delta)``.

    Pure with respect to the process: the only state it touches is the read-only handles
    :func:`_fine_worker_init` opened, and the stats it would have accumulated globally are
    returned as per-brick deltas for the parent to merge.
    """
    i, lo, hi = job
    w = _FINE_W
    spec, opts = w["spec"], w["opts"]
    faces_opts = opts["faces"]
    origin = tuple(spec["origin"])
    halo, clip, k = int(spec["halo"]), float(spec["clip"]), int(spec["k"])
    do_faces, do_rv, do_fiber = spec["do_faces"], spec["do_rv"], spec["do_fiber"]
    face_source = spec["face_source"]
    fs: dict[str, Any] = {"valid_hist": np.zeros(3, dtype=np.int64),
                          "sdf_hist": np.zeros(int(2 * clip) + 1, dtype=np.int64),
                          "n_sup": 0, "near_zero": 0, "zero": 0, "coarse_normal_used": 0,
                          "ink_valid": 0, "ink_sum": 0.0,
                          "fiber_valid": 0, "fiber_vt_sum": 0.0, "fiber_hz_sum": 0.0}
    face_stats: dict[str, Any] = {"valid_hist": np.zeros(3, dtype=np.int64),
                                  "thickness_hist": np.zeros(256, dtype=np.int64)}

    hlo = [lo[a] - halo for a in range(3)]
    hhi = [hi[a] + halo for a in range(3)]
    r_u8 = read_box_padded(w["recto"], hlo, hhi, channels=0)[0]
    i_u8 = read_box_padded(w["ink"], hlo, hhi, channels=0)[0]
    ct = w["ct0"].read(*(v for a in range(3) for v in (origin[a] + hlo[a], origin[a] + hhi[a])))
    rad = radial_field(w["axis"], [origin[a] + hlo[a] for a in range(3)], r_u8.shape, scale=1)
    n_up = upsample_coarse_normal(w["coarse"], hlo, hhi, scale=k)
    fs["coarse_normal_used"] += int((np.abs(n_up).sum(0) > 0).sum())
    sdf_u8, sdf_v, ink_u8, ink_v = build_fine_labels(
        r_u8, i_u8, ct, n_up, rad, clip, min_component=int(opts["min_component"]),
        recto_threshold=float(opts["recto_threshold"]), ignore_band=opts["ignore_band"],
        ignore_near=float(opts["ignore_near"]), ct_mask_threshold=float(opts["ct_mask_threshold"]),
        ct_sigma=float(opts["ct_sigma"]), ct_mask_dilate=int(opts["ct_mask_dilate"]))
    r_u8b, ctb, n_upb, radb = r_u8, ct, n_up, rad
    del r_u8, i_u8, ct, rad, n_up
    core = tuple(slice(halo, halo + hi[a] - lo[a]) for a in range(3))
    outs = [np.ascontiguousarray(a[core]) for a in (sdf_u8, sdf_v, ink_u8, ink_v)]
    del sdf_u8, sdf_v, ink_u8, ink_v
    rv_extra: list[np.ndarray] = []
    if do_faces:
        rvb = None
        if do_rv:
            from tsm import rvfaces

            rvb = rvfaces.read_rv_box(w["rv_arr"], w["rv_origin"],
                                      [origin[a] + hlo[a] for a in range(3)],
                                      [origin[a] + hhi[a] for a in range(3)])
            rv_fo = rvfaces.build_rv_face_labels(
                rvb[0], ctb, radb, clip,
                min_component=int(faces_opts["rv_min_component"]),
                normal_window=int(faces_opts["rv_normal_window"]),
                min_normal_count=int(faces_opts["rv_min_normal_count"]),
                planarity_max=float(faces_opts["rv_planarity_max"]),
                thin_tol=float(faces_opts["rv_thin_tol"]),
                ct_mask_threshold=float(opts["ct_mask_threshold"]), ct_sigma=float(opts["ct_sigma"]),
                ct_mask_dilate=int(opts["ct_mask_dilate"]), stats=face_stats)
        fo = None if face_source == "rectoverso" else build_face_labels(
            r_u8b, ctb, n_upb, radb, clip,
            body_threshold=float(faces_opts["body_threshold"]), body_sigma=float(faces_opts["body_sigma"]),
            closing=int(faces_opts["closing"]), close_radius=float(faces_opts["close_radius"]),
            close_along_normal=bool(faces_opts["close_along_normal"]), min_component=int(faces_opts["min_component"]),
            min_thickness=float(faces_opts["min_thickness"]), max_thickness=float(faces_opts["max_thickness"]),
            band_reach=float(faces_opts["band_reach"]), gap_min=float(faces_opts["gap_min"]),
            side_reach=float(faces_opts["side_reach"]),
            recto_threshold=float(opts["recto_threshold"]), ignore_band=opts["ignore_band"],
            ignore_near=float(opts["ignore_near"]), ct_mask_threshold=float(opts["ct_mask_threshold"]),
            ct_sigma=float(opts["ct_sigma"]), ct_mask_dilate=int(opts["ct_mask_dilate"]), stats=face_stats)
        if face_source == "rectoverso":
            from tsm import rvfaces

            fo = rv_fo
            fsrc = np.where(fo[2] == 1, np.uint8(rvfaces.SOURCE_RECTOVERSO), np.uint8(rvfaces.SOURCE_NONE))
        elif face_source == "merge":
            fo, fsrc = rvfaces.merge_face_labels(rv_fo, fo, clip, band=rvb[0] > 0, stats=face_stats)
            del rv_fo
        if do_rv:
            rv_extra = [np.ascontiguousarray(a[core]) for a in (fsrc, rvb[0], rvb[1])]
            del fsrc, rvb
        if spec["do_wf"]:
            # the native 2.4 um winding field, from the *haloed* faces (so the EDT fill that
            # recovers the sheet spacing sees a full period on both sides of the core)
            from tsm import winding_fine as wfmod

            wopts = spec["wf_opts"]
            prior = wfmod.upsample_coarse_prior(w["coarse"], hlo, hhi, scale=k)
            prior["normal"] = n_upb
            wf = wfmod.winding_fine_box(
                fo[0], fo[1], fo[2], prior, radb, clip,
                snap_max=(None if wopts["snap_max"] is None else float(wopts["snap_max"])),
                min_period=float(wopts["min_period"]),
                max_period=float(wopts["max_period"]), sat_margin=float(wopts["sat_margin"]),
                min_grad=float(wopts["min_grad"]), max_grad=float(wopts["max_grad"]),
                phase_offset=float(wopts["phase_offset"]),
                require_faces_valid=bool(wopts["require_faces_valid"]), stats=face_stats)
            del prior
            rv_extra += [np.ascontiguousarray(a[core]) for a in wf]
            del wf
        outs += [np.ascontiguousarray(a[core]) for a in fo]
        del fo
        if float(spec["core_radius_vox"]) > 0:
            # umbilicus core: the in/out face convention degenerates there (recto_is_in ~ 0.47),
            # so the faces become "ignore" (2) and the native winding block invalid.  The CT and
            # the sdf/thickness values are left alone -- only the validity masks change.
            cm = core_mask(w["axis"], [origin[a] + lo[a] for a in range(3)],
                           [hi[a] - lo[a] for a in range(3)], float(spec["core_radius_vox"]), scale=1)
            outs[6] = np.where(cm, np.uint8(2), outs[6])
            face_stats["core_masked"] = face_stats.get("core_masked", 0) + int(cm.sum())
            if spec["do_wf"]:
                rv_extra[-1] = np.where(cm, np.uint8(0), rv_extra[-1])  # wf_valid is last of WF_CHANNELS
            del cm
        face_stats["valid_hist"] += np.bincount(outs[6].ravel(), minlength=3)[:3]
        face_stats["thickness_hist"] += np.bincount(outs[7].ravel(), minlength=256)[:256]
    if do_fiber:
        outs += fiber_block(w["fiber"], hlo, hhi, core, outs[3])
    del r_u8b, ctb, n_upb, radb
    v = outs[1]
    fs["valid_hist"] += np.bincount(v.ravel(), minlength=3)[:3]
    sel = v == 1
    d = decode_sdf_u8(outs[0][sel], clip)
    fs["n_sup"] += int(sel.sum())
    fs["sdf_hist"] += np.histogram(d, bins=int(2 * clip) + 1, range=(-clip - 0.5, clip + 0.5))[0]
    fs["near_zero"] += int((np.abs(d) <= 1.0).sum())
    fs["zero"] += int((outs[0][sel] == 128).sum())
    fs["ink_valid"] += int(outs[3].sum())
    fs["ink_sum"] += float(outs[2][outs[3] == 1].sum())
    if do_fiber:
        fv = outs[-1] == 1
        fs["fiber_valid"] += int(fv.sum())
        fs["fiber_vt_sum"] += float(outs[-3][fv].sum())
        fs["fiber_hz_sum"] += float(outs[-2][fv].sum())
        del fv
    del v, sel, d
    return i, lo, hi, outs, rv_extra, fs, face_stats


def merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Accumulate one brick's stats delta into the run totals (in brick order).

    Counters and histograms add; the per-component/per-brick lists concatenate.  Adding in a
    fixed order is what keeps the float sums (and so ``labels.summary.json``) identical
    regardless of how many workers produced the bricks.
    """
    for key, val in src.items():
        if key not in dst:
            dst[key] = list(val) if isinstance(val, list) else val
        elif isinstance(val, list):
            dst[key] = list(dst[key]) + list(val)
        elif isinstance(val, np.ndarray):
            dst[key] = np.asarray(dst[key]) + val
        else:
            dst[key] = dst[key] + val
    return dst


def _fine_pool_limit(workers: int, per_brick: int, opts: dict[str, Any], budget: Budget) -> int:
    """RAM ceiling for ``workers`` concurrent bricks, and the check against it.

    ``budget.ram_bytes`` is a *per process* limit (configs set 8 GiB while the box has far
    more), so it is not the right ceiling for a pool.  The default ceiling is what the
    machine actually has free right now -- ``MemAvailable - budget.min_avail_mb`` -- and
    ``extra.labels.workers_ram_bytes`` overrides it explicitly.  Falls back to
    ``workers * budget.ram_bytes`` when /proc/meminfo is unreadable.
    """
    from tsm.limits import BudgetError, read_mem_avail_mb

    override = opts.get("workers_ram_bytes")
    if override:
        limit = int(override)
        how = "extra.labels.workers_ram_bytes"
    else:
        avail = read_mem_avail_mb()
        if avail > 0:
            limit = max(0, (avail - int(budget.min_avail_mb)) * 1024 * 1024)
            how = f"MemAvailable {avail} MiB - min_avail_mb {budget.min_avail_mb} MiB"
        else:
            limit = workers * int(budget.ram_bytes)
            how = "no /proc/meminfo: workers x ram_bytes"
    need = workers * per_brick
    _llog(f"fine workers={workers}: {need / 2**30:.2f} GiB estimated live ({per_brick / 2**30:.2f} GiB "
          f"per brick) vs {limit / 2**30:.2f} GiB allowed ({how})")
    if need > limit:
        raise BudgetError(
            f"labels.workers={workers} x {per_brick / 2**30:.2f} GiB per brick = {need / 2**30:.2f} GiB "
            f"> {limit / 2**30:.2f} GiB ({how}); lower labels.workers or set labels.workers_ram_bytes")
    return limit


def _fine_jobs(spec: dict[str, Any], todo: list, workers: int):
    """Yield ``_fine_brick`` results in brick order, in-process or from a fork pool.

    At most ``workers + 1`` bricks are outstanding, so the memory the pool holds stays
    bounded no matter how far the workers run ahead of the writer.
    """
    jobs = [(i, lo, hi) for i, (lo, hi) in enumerate(todo)]
    if workers <= 1:
        _fine_worker_init(spec)
        for job in jobs:
            yield _fine_brick(job)
        return
    import multiprocessing as mp

    start = str(spec["opts"].get("mp_start") or "fork")
    ctx = mp.get_context(start)
    _llog(f"fine: {workers} worker processes ({start}); the parent does every write")
    pool = ctx.Pool(workers, initializer=_fine_worker_init, initargs=(spec,))
    try:
        pending: list = []
        nxt = 0
        while nxt < len(jobs) or pending:
            while nxt < len(jobs) and len(pending) < workers + 1:
                pending.append(pool.apply_async(_fine_brick, (jobs[nxt],)))
                nxt += 1
            yield pending.pop(0).get()
    finally:
        pool.terminate()
        pool.join()



def run_labels(cfg: RunCfg, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    """Build ``<out_dir>/labels/{coarse,fine}.zarr`` from the teacher stores (resumable)."""
    import zarr

    from tsm.cli import open_ct
    from tsm.limits import estimate_and_assert

    opts = _label_opts(cfg)
    budget = cfg.budget
    um = float(cfg.volume.voxel_um)
    k = _COARSE_SCALE
    clip = float(opts["clip"])
    tdir = opts["teachers_dir"]
    ldir = os.path.join(cfg.out_dir, "labels")
    shape = tuple(int(s) for s in cfg.region.size_zyx)
    origin = tuple(int(s) for s in cfg.region.start_zyx)
    for v in (*shape, *origin):
        if v % k:
            raise ValueError(f"region {origin}+{shape} must be aligned to {k} (coarse store is region/{k})")
    shape_c = tuple(s // k for s in shape)
    origin_c = tuple(o // k for o in origin)
    brick = [int(v) for v in opts["brick"]]
    halo = int(opts["halo"])
    brick_c = [int(v) for v in opts["coarse_brick"]]
    halo_c = int(opts["coarse_halo"])
    hs = tuple(min(b, s) + 2 * halo for b, s in zip(brick, shape))
    hsc = tuple(min(b, s) + 2 * halo_c for b, s in zip(brick_c, shape_c))
    _llog(f"fine bricks {brick} halo {halo} (haloed {hs}); coarse bricks {brick_c} halo {halo_c} (haloed {hsc})")
    if float(opts["core_radius_vox"]) > 0:
        _llog(f"core mask: faces_valid = 2 and the winding validity channels = 0 within "
              f"{opts['core_radius_vox']:g} in-plane voxels of the umbilicus ({opts['axis_path']})")
    if bool(opts["faces"]["enabled"]):
        need = float(opts["faces"]["max_thickness"]) / 2.0 + 2.0
        _llog(f"faces enabled: {FACE_CHANNELS} appended to the fine store; opts {opts['faces']}")
        if halo < need:
            _llog(f"WARNING faces: halo {halo} < max_thickness/2 + 2 = {need:.0f}; a sheet thicker than "
                  f"{2 * (halo - 2)} voxels is cut by the brick halo and its `thickness` (and the thickness "
                  f"gate) may over-read near brick edges -- raise labels.halo if that matters")
    estimate_and_assert(_coarse_budget_rows(hsc), budget)
    has_fiber = os.path.exists(os.path.join(tdir, "fiber.zarr"))
    per_brick = estimate_and_assert(
        _fine_budget_rows(hs, bool(opts["faces"]["enabled"]), has_fiber,
                          bool(opts["faces"]["enabled"]) and str(opts["faces"]["source"]) != "ct",
                          bool(opts["winding_fine"]["enabled"])),
        budget)
    workers = max(1, int(opts["workers"]))
    if workers > 1:
        _fine_pool_limit(workers, per_brick, opts, budget)
    if dry_run:
        _llog("dry run: stopping before I/O")
        return {}
    os.makedirs(ldir, exist_ok=True)
    t0 = time.perf_counter()
    summary: dict[str, Any] = {"region": {"start_zyx": list(origin), "size_zyx": list(shape), "voxel_um": um},
                               "opts": {kk: vv for kk, vv in opts.items()}, "fine": {}, "coarse": {}}

    # recto/ink are opened here only to fail fast on a missing or mis-covering teacher store --
    # the fine stage reopens them per worker process (see _fine_worker_init).
    open_teacher(tdir, "recto", shape, origin)
    open_teacher(tdir, "ink", shape, origin)
    lasagna = open_teacher(tdir, "lasagna", shape_c, origin_c)
    # optional: the 4-class fiber teacher.  Absent -> the fiber channels are simply not built.
    if has_fiber:
        _llog(f"fiber teacher found ({os.path.join(tdir, 'fiber.zarr')}): adding {FIBER_CHANNELS}")
    else:
        _llog("no fiber teacher store: fiber channels omitted")
    axis = load_axis(opts["axis_path"])

    # ---- coarse ------------------------------------------------------------
    from tsm.volume import BrickWriter

    cw = BrickWriter(os.path.join(ldir, "coarse.zarr"), COARSE_CHANNELS, shape_c, chunk=128,
                     origin_zyx=origin_c, voxel_um=um * k, scale=float(k))
    if force:
        cw._done = {}
    cores_c = list(iter_cores(shape_c, brick_c))
    todo_c = [b for b in cores_c if not cw.has_brick(*(origin_c[a] + b[0][a] for a in range(3)))]
    _llog(f"coarse: {len(cores_c)} bricks, {len(todo_c)} to do")
    cstats = {"valid_hist": np.zeros(3, dtype=np.int64), "n_dot_r_pos": 0, "n_valid": 0, "unit_norm_sum": 0.0,
              "conf_sum": 0.0, "agreement_hist": np.zeros(10, dtype=np.int64)}
    ct_l2 = open_ct(cfg, level_shift=2, probe=int(opts["probe_brick"])) if todo_c else None
    tc = time.perf_counter()
    for i, (lo, hi) in enumerate(todo_c):
        hlo = [lo[a] - halo_c for a in range(3)]
        hhi = [hi[a] + halo_c for a in range(3)]
        las = read_box_padded(lasagna, hlo, hhi)
        ct = ct_l2.read(*(v for a in range(3) for v in (origin_c[a] + hlo[a], origin_c[a] + hhi[a])))
        rad = radial_field(axis, [origin_c[a] + hlo[a] for a in range(3)], las.shape[1:], scale=k)
        out = build_coarse_field(
            las, ct, rad, ct_mask_threshold=float(opts["ct_mask_threshold"]),
            ct_sigma=max(1.0, float(opts["ct_sigma"]) / 2.0), ct_mask_dilate=int(opts["ct_mask_dilate"]),
            phase_sigma=float(opts["phase_sigma"]), phase_eps=float(opts["phase_eps"]),
            conf_window=int(opts["conf_window"]), normal_agreement_min=float(opts["normal_agreement_min"]),
            dir_err_max=float(opts["dir_err_max"]))
        del las, ct
        core = (slice(None),) + tuple(slice(halo_c, halo_c + hi[a] - lo[a]) for a in range(3))
        out = np.ascontiguousarray(out[core])
        rad = rad[core]
        if float(opts["core_radius_vox"]) > 0:
            cm = core_mask(axis, [origin_c[a] + lo[a] for a in range(3)],
                           [hi[a] - lo[a] for a in range(3)], float(opts["core_radius_vox"]), scale=k)
            out[7] = np.where(cm, np.uint8(0), out[7])  # "valid" = the last coarse channel
            del cm
        for c in range(8):
            cw.write(c, origin_c[0] + lo[0], origin_c[1] + lo[1], origin_c[2] + lo[2], out[c])
        v = out[7]
        cstats["valid_hist"] += np.bincount(v.ravel(), minlength=3)[:3]
        sel = v == 1
        n = np.stack([decode_signed_u8(out[5]), decode_signed_u8(out[4]), decode_signed_u8(out[3])])
        cstats["n_valid"] += int(sel.sum())
        cstats["n_dot_r_pos"] += int(((n * rad).sum(0)[sel] > 0).sum())
        s, c_ = decode_signed_u8(out[0]), decode_signed_u8(out[1])
        cstats["unit_norm_sum"] += float(np.sqrt(s * s + c_ * c_)[sel].sum())
        cstats["conf_sum"] += float(out[6][sel].sum())
        del out, rad, n, s, c_, sel
        _llog(f"coarse brick {i + 1}/{len(todo_c)} {lo}->{hi} done ({time.perf_counter() - tc:.0f}s, RSS {peak_rss_mb_():.0f} MiB)")
    nv = max(cstats["n_valid"], 1)
    summary["coarse"] = {
        "bricks": len(cores_c), "computed": len(todo_c),
        "valid_fraction": {str(i): float(cstats["valid_hist"][i] / max(cstats["valid_hist"].sum(), 1)) for i in range(3)},
        "frac_n_dot_r_pos": cstats["n_dot_r_pos"] / nv, "mean_unit_norm": cstats["unit_norm_sum"] / nv,
        "mean_conf": cstats["conf_sum"] / nv, "seconds": time.perf_counter() - tc,
    }
    _llog(f"coarse: valid {summary['coarse']['valid_fraction']} n.r>0 {summary['coarse']['frac_n_dot_r_pos']:.4f} "
          f"unit-norm {summary['coarse']['mean_unit_norm']:.4f}")

    # ---- fine --------------------------------------------------------------
    coarse_arr = zarr.open_array(store=cw.path, mode="r")
    faces_opts = opts["faces"]
    do_faces = bool(faces_opts["enabled"])
    # --- face source: the CT builder, the upstream recto/verso bands, or both merged ---
    face_source = str(faces_opts["source"]) if do_faces else "ct"
    do_rv = do_faces and face_source != "ct"
    rv_arr = rv_origin = None
    if do_rv:
        from tsm import rvfaces

        rv_arr, rv_origin = rvfaces.open_rv_store(faces_opts["rectoverso_store"])
        _llog(f"faces source {face_source!r}: rectoverso store {faces_opts['rectoverso_store']} "
              f"shape {tuple(rv_arr.shape)} origin {rv_origin}; appending {rvfaces.RV_CHANNELS}")
    do_fiber = has_fiber
    do_wf = bool(opts["winding_fine"]["enabled"])
    if do_wf:
        from tsm.winding_fine import WF_CHANNELS

        _llog(f"winding_fine enabled: {WF_CHANNELS} appended to the fine store; "
              f"opts {opts['winding_fine']}")
    fchannels = fine_channels(do_faces, do_fiber, do_rv, do_wf)
    fw = BrickWriter(os.path.join(ldir, "fine.zarr"), fchannels, shape, chunk=128,
                     origin_zyx=origin, voxel_um=um, scale=1.0)
    if force:
        fw._done = {}
    cores = list(iter_cores(shape, brick))
    todo = [b for b in cores if not fw.has_brick(*(origin[a] + b[0][a] for a in range(3)))]
    _llog(f"fine: {len(cores)} bricks, {len(todo)} to do")
    fstats: dict[str, Any] = {}
    face_stats: dict[str, Any] = {}
    spec = {"cfg": cfg, "opts": opts, "shape": shape, "origin": origin, "halo": halo, "clip": clip,
            "k": k, "do_faces": do_faces, "do_rv": do_rv, "do_fiber": do_fiber,
            "do_wf": do_wf, "wf_opts": dict(opts["winding_fine"]),
            "core_radius_vox": float(opts["core_radius_vox"]),
            "face_source": face_source, "coarse_path": cw.path}
    tf = time.perf_counter()
    for i, lo, hi, outs, rv_extra, fs, fst in _fine_jobs(spec, todo, workers if todo else 1):
        for c, a in enumerate(outs):
            fw.write(c, origin[0] + lo[0], origin[1] + lo[1], origin[2] + lo[2], a)
        # the appended blocks (RV_CHANNELS then WF_CHANNELS) come after the outs, in the
        # same order fine_channels() lists them
        for j, a in enumerate(rv_extra):
            fw.write(len(outs) + j, origin[0] + lo[0], origin[1] + lo[1], origin[2] + lo[2], a)
        merge_stats(fstats, fs)
        merge_stats(face_stats, fst)
        del outs, rv_extra, fs, fst
        _llog(f"fine brick {i + 1}/{len(todo)} {lo}->{hi} done ({time.perf_counter() - tf:.0f}s, RSS {peak_rss_mb_():.0f} MiB)")
    for key, zero in (("valid_hist", np.zeros(3, dtype=np.int64)),
                      ("sdf_hist", np.zeros(int(2 * clip) + 1, dtype=np.int64)),
                      ("n_sup", 0), ("near_zero", 0), ("zero", 0), ("coarse_normal_used", 0),
                      ("ink_valid", 0), ("ink_sum", 0.0),
                      ("fiber_valid", 0), ("fiber_vt_sum", 0.0), ("fiber_hz_sum", 0.0)):
        fstats.setdefault(key, zero)
    for key, zero in (("valid_hist", np.zeros(3, dtype=np.int64)),
                      ("thickness_hist", np.zeros(256, dtype=np.int64))):
        face_stats.setdefault(key, zero)
    ns = max(fstats["n_sup"], 1)
    nvox = int(np.prod(shape))
    summary["fine"] = {
        "bricks": len(cores), "computed": len(todo),
        "sdf_valid_fraction": {str(i): float(fstats["valid_hist"][i] / max(fstats["valid_hist"].sum(), 1)) for i in range(3)},
        "sdf_hist_supervised": {"edges": [float(e) for e in np.arange(-clip - 0.5, clip + 1.0, 1.0)],
                                "counts": [int(c) for c in fstats["sdf_hist"]]},
        "frac_within_1vox_of_zero": fstats["near_zero"] / ns, "frac_zero_set": fstats["zero"] / ns,
        "frac_coarse_normal_used": fstats["coarse_normal_used"] / max(sum(int(np.prod([hh - ll + 2 * halo for ll, hh in zip(lo, hi)])) for lo, hi in todo), 1),
        "ink_valid_fraction": fstats["ink_valid"] / nvox if todo else None,
        "ink_mean_prob": (fstats["ink_sum"] / max(fstats["ink_valid"], 1) / 255.0) if todo else None,
        "seconds": time.perf_counter() - tf,
    }
    if do_fiber:
        nfv = max(fstats["fiber_valid"], 1)
        summary["fine"]["fiber"] = {
            "channels": FIBER_CHANNELS,
            "valid_fraction": fstats["fiber_valid"] / nvox if todo else None,
            "vt_mean_prob": (fstats["fiber_vt_sum"] / nfv / 255.0) if todo else None,
            "hz_mean_prob": (fstats["fiber_hz_sum"] / nfv / 255.0) if todo else None,
        }
        _llog(f"fine fiber: {summary['fine']['fiber']}")
    if do_faces:
        vh = face_stats["valid_hist"]
        th = face_stats["thickness_hist"]
        comps = face_stats.get("components", [])
        rec = np.array([c["recto_is_in"] for c in comps if c["band_adjacent_face_voxels"] > 0], dtype=np.float64)
        wts = np.array([c["band_adjacent_face_voxels"] for c in comps if c["band_adjacent_face_voxels"] > 0], dtype=np.float64)
        tq = dict(zip(("p10", "median", "p90"), hist_percentiles(th, (10, 50, 90))))
        summary["fine"]["faces"] = {
            "opts": dict(faces_opts), "channels": FACE_CHANNELS,
            "valid_fraction": {str(i): float(vh[i] / max(vh.sum(), 1)) for i in range(3)},
            "recto_is_in_mean": float((rec * wts).sum() / wts.sum()) if wts.sum() else None,
            "recto_is_in_per_component": [float(v) for v in rec[:64]],
            "n_components": len(comps), "n_components_ok": int(sum(1 for c in comps if c["ok"])),
            "core_radius_vox": float(opts["core_radius_vox"]),
            "core_masked_voxels": int(face_stats.get("core_masked", 0)),
            "contact_face_voxels": int(face_stats.get("contact_face_voxels", 0)),
            "face_in_voxels": int(face_stats.get("face_in_voxels", 0)),
            "thickness_voxels": tq,
            "components": comps[:64],
        }
        if do_rv:
            rvh = np.asarray(face_stats.get("rv_valid_hist", np.zeros(3, np.int64)))
            blk: dict[str, Any] = {
                "source": face_source,
                "store": faces_opts["rectoverso_store"],
                "channels": rvfaces.RV_CHANNELS,
                "rectoverso_valid_fraction": {str(i): float(rvh[i] / max(rvh.sum(), 1)) for i in range(3)},
                "band_voxels": int(face_stats.get("rv_band_voxels", 0)),
                "thin_in_voxels": int(face_stats.get("rv_thin_in_voxels", 0)),
                "thin_out_voxels": int(face_stats.get("rv_thin_out_voxels", 0)),
                "no_band_boxes": int(face_stats.get("rv_no_band_boxes", 0)),
                "normal_pca_used_frac": (float(face_stats.get("rv_normal_pca_used", 0))
                                         / max(int(face_stats.get("rv_normal_pts", 0)), 1)),
            }
            if face_source == "merge":
                blk.update(rvfaces.disagree_summary(face_stats))
            summary["fine"]["faces"]["rectoverso"] = blk
            _llog(f"fine faces rectoverso: {blk}")
        _llog(f"fine faces: valid {summary['fine']['faces']['valid_fraction']} "
              f"recto_is_in {summary['fine']['faces']['recto_is_in_mean']} "
              f"thickness {summary['fine']['faces']['thickness_voxels']}")
    if do_wf:
        from tsm.winding_fine import WF_CHANNELS

        wh = np.asarray(face_stats.get("wf_valid_hist", np.zeros(2, np.int64)))
        rh = np.asarray(face_stats.get("wf_resid_hist", np.zeros(50, np.int64)))
        rq = dict(zip(("p50", "p90"), hist_percentiles(rh, (50, 90), drop_zero=False)))
        rn = max(int(face_stats.get("wf_resid_n", 0)), 1)
        summary["fine"]["winding_fine"] = {
            "opts": dict(opts["winding_fine"]), "channels": WF_CHANNELS,
            "valid_fraction": float(wh[1] / max(wh.sum(), 1)),
            "gated_by_coarse_prior": opts["winding_fine"]["snap_max"] is not None,
            "no_period_boxes": int(face_stats.get("wf_no_period_boxes", 0)),
            # |resid| = |w_fine - w_coarse| in turns, over voxels the coarse prior covers
            "abs_resid_turns": {k_: (None if v_ is None else float(v_) / 100.0) for k_, v_ in rq.items()},
            "mean_signed_resid_turns": float(face_stats.get("wf_signed_resid_sum", 0.0)) / rn,
            "frac_resid_gt_0.25": float(rh[25:].sum() / max(rh.sum(), 1)),
            "mean_period_voxels": (float(face_stats.get("wf_period_sum", 0.0))
                                   / max(int(face_stats.get("wf_period_n", 0)), 1)),
        }
        _llog(f"fine winding_fine: {summary['fine']['winding_fine']}")
    _llog(f"fine: valid {summary['fine']['sdf_valid_fraction']} within 1 vox of zero "
          f"{summary['fine']['frac_within_1vox_of_zero']:.4f} zero-set {summary['fine']['frac_zero_set']:.4f}")

    try:
        summary["png"] = _write_previews(ldir, zarr.open_array(store=fw.path, mode="r"), coarse_arr, clip, k)
    except Exception as exc:  # previews are best-effort
        summary["png"] = {"error": f"{type(exc).__name__}: {exc}"}
        _llog(f"previews failed: {summary['png']['error']}")
    summary["seconds"] = time.perf_counter() - t0
    summary["peak_rss_mb"] = peak_rss_mb_()
    summary["stores"] = {"fine": fw.path, "coarse": cw.path}
    spath = os.path.join(ldir, "labels.summary.json")
    if (not todo or not todo_c) and os.path.exists(spath):  # resume: keep the stats of the run that did the work
        try:
            with open(spath, "r") as fh:
                prev = json.load(fh)
            for key, done in (("fine", not todo), ("coarse", not todo_c)):
                if done and key in prev:
                    summary[key] = prev[key]
        except (OSError, ValueError):
            pass
    with open(spath, "w") as fh:
        json.dump(summary, fh, indent=2)
    _llog(f"wrote {os.path.join(ldir, 'labels.summary.json')} in {summary['seconds']:.0f}s")
    return summary


def peak_rss_mb_() -> float:
    from tsm.limits import peak_rss_mb

    return peak_rss_mb()
