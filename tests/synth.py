"""Synthetic fine/coarse label stores + CT volumes per docs/label_store.md (tests and dev)."""

from __future__ import annotations

import os

import numpy as np
import zarr

from tsm.data import (
    CLIP,
    COARSE_CHANNELS,
    FACE_CHANNELS,
    FIBER_CHANNELS,
    FINE_CHANNELS,
    WF_CHANNELS,
    encode_density,
    encode_prob,
    encode_sdf,
    encode_signed,
)
from tsm.rvfaces import RV_CHANNELS
from tsm.volume import BrickWriter

AXIS_YX = (100.0, 100.0)  # fine-global (y, x) of the synthetic winding axis
PERIOD_FINE = 24.0        # fine voxels per wrap
SHEET_HALF = 3.0          # synthetic two-face sheets are 2*SHEET_HALF voxels thick


def synthetic_axis(z0: int = -100000, z1: int = 100000) -> np.ndarray:
    """Umbilicus control points (N, 3) in (z, y, x) for the straight axis at ``AXIS_YX``.

    Matches ``coarse_fields``: its normal (nx, ny, nz) = (dx/r, dy/r, 0) is exactly the
    outward radial direction of ``labels.radial_field`` for this axis."""
    return np.array([[z0, AXIS_YX[0], AXIS_YX[1]], [z1, AXIS_YX[0], AXIS_YX[1]]], dtype=np.float64)


def fine_fields(shape: tuple[int, int, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Wavy sheets stacked along z; outward = +z. Returns float fields + CT (u8)."""
    Z, Y, X = shape
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    period = 24.0
    phase = z - 3.0 * np.sin(x / 9.0) - 2.0 * np.cos(y / 7.0)
    # distance to the nearest sheet (sheets at phase = k*period); sign = +z side
    d = (phase + period / 2) % period - period / 2
    sdf = np.clip(d, -CLIP, CLIP).astype(np.float32)
    valid = np.ones(shape, np.uint8)
    valid[:, : Y // 8, :] = 0  # "air" strip
    valid[(np.abs(sdf) > 2.0) & (((x // 16) + (y // 16)) % 5 == 0)] = 2  # ignore patches
    rng = np.random.default_rng(seed)
    ink = np.clip(np.exp(-((y - Y * 0.6) ** 2 + (x - X * 0.4) ** 2) / (2 * (X / 6.0) ** 2)), 0, 1)
    ink = (ink * (np.abs(sdf) < 3.0)).astype(np.float32)
    ink_valid = (valid > 0).astype(np.uint8)
    ct = 40.0 + 170.0 * np.exp(-(sdf ** 2) / (2 * 1.5 ** 2)) + rng.normal(0, 6, shape)
    ct[valid == 0] = 0
    # fibre teacher probabilities: sheet-bound scalar fields, vertical in one half of x,
    # horizontal in the other, so a spatial transform is visible in the targets
    near = (np.abs(sdf) < 3.0).astype(np.float32)
    # native-2.4 um winding block (tsm.winding_fine encodings): the analytic phase of the same
    # sheets, with the integer turn on the IN face (sdf = -SHEET_HALF) and one wrap per period.
    wf_w = (phase + SHEET_HALF) / period
    wf_valid = ((valid == 1) & (x < X * 0.5)).astype(np.uint8)  # half the crop only, on purpose
    fiber_vt = (near * (x < X * 0.5) * 0.9).astype(np.float32)
    fiber_hz = (near * (x >= X * 0.5) * 0.8).astype(np.float32)
    # upstream rectoverso block (tsm.rvfaces.RV_CHANNELS): the human hz/vt bands are sparse --
    # here two stripes in y plus a small "exclude" patch, only on the sheets
    hzvt = np.zeros(shape, np.uint8)
    onsheet = np.abs(sdf) < 2.0
    hzvt[onsheet & (y < Y * 0.3)] = 1          # hz
    hzvt[onsheet & (y >= Y * 0.3) & (y < Y * 0.6)] = 2  # vt
    hzvt[onsheet & (y >= Y * 0.9)] = 3         # exclude
    # two-face view of the same sheets: a slab of half-width SHEET_HALF about each medial plane,
    # outward = +z (the sdf sign), so the IN face is at sdf = -SHEET_HALF and the OUT face at +SHEET_HALF
    return {
        "sdf": sdf, "sdf_valid": valid, "ink": ink, "ink_valid": ink_valid,
        "sdf_in": np.clip(sdf + SHEET_HALF, -CLIP, CLIP).astype(np.float32),
        "sdf_out": np.clip(sdf - SHEET_HALF, -CLIP, CLIP).astype(np.float32),
        "faces_valid": valid,
        "fiber_vt": fiber_vt, "fiber_hz": fiber_hz, "fiber_valid": ink_valid,
        "faces_source": np.ones(shape, np.uint8), "rv_class": (1 + (sdf > 0)).astype(np.uint8),
        "hzvt_class": hzvt,
        "thickness": np.full(shape, 2 * SHEET_HALF, np.float32),
        "wf_sin": np.sin(2 * np.pi * wf_w).astype(np.float32),
        "wf_cos": np.cos(2 * np.pi * wf_w).astype(np.float32),
        "wf_density": np.full(shape, 1.0 / period, np.float32),  # wraps per FINE voxel
        "wf_nx": np.zeros(shape, np.float32), "wf_ny": np.zeros(shape, np.float32),
        "wf_nz": np.ones(shape, np.float32),
        "wf_conf": np.full(shape, 0.75, np.float32), "wf_valid": wf_valid,
        "ct": np.clip(ct, 0, 255).astype(np.uint8),
    }


def coarse_fields(shape: tuple[int, int, int], seed: int = 0, origin: tuple[int, int, int] = (0, 0, 0), factor: int = 4) -> dict[str, np.ndarray]:
    """Concentric wraps around a z-axis at fine-global (y, x) = AXIS_YX: w = r_fine / PERIOD_FINE.

    ``origin`` is the store origin in coarse-level voxels; the field is evaluated at the
    coarse voxel centres in fine coordinates so a fine store at ``origin*factor`` matches.
    """
    Z, Y, X = shape
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    yf = (y + origin[1]) * factor + (factor - 1) / 2.0
    xf = (x + origin[2]) * factor + (factor - 1) / 2.0
    dy, dx = yf - AXIS_YX[0], xf - AXIS_YX[1]
    r = np.sqrt(dy ** 2 + dx ** 2) + 1e-6
    w = r / PERIOD_FINE
    nx, ny, nz = dx / r, dy / r, np.zeros_like(r)
    rng = np.random.default_rng(seed)
    conf = np.clip(0.9 - 0.3 * rng.random(shape), 0, 1).astype(np.float32)
    valid = np.ones(shape, np.uint8)
    valid[r < 3.0 * factor] = 0
    valid[(z // 2) % 4 == 3] = 2
    ct = 40.0 + 150.0 * (0.5 + 0.5 * np.cos(2 * np.pi * w)) + rng.normal(0, 6, shape)
    ct[valid == 0] = 0
    return {
        "phase_sin": np.sin(2 * np.pi * w).astype(np.float32),
        "phase_cos": np.cos(2 * np.pi * w).astype(np.float32),
        "density": np.full(shape, factor / PERIOD_FINE, np.float32),  # wraps per coarse voxel
        "nx": nx.astype(np.float32), "ny": ny.astype(np.float32), "nz": nz.astype(np.float32),
        "conf": conf, "valid": valid, "ct": np.clip(ct, 0, 255).astype(np.uint8),
    }


def encode_fine(f: dict[str, np.ndarray], faces: bool = False, fiber: bool = False,
                winding_fine: bool = False, rv: bool = False) -> dict[str, np.ndarray]:
    sdf_u8 = encode_sdf(f["sdf"])
    sdf_u8[f["sdf_valid"] == 0] = 0
    out = {"sdf": sdf_u8, "sdf_valid": f["sdf_valid"], "ink": encode_prob(f["ink"]), "ink_valid": f["ink_valid"]}
    if faces:
        for k in ("sdf_in", "sdf_out"):
            u = encode_sdf(f[k])
            u[f["faces_valid"] == 0] = 0
            out[k] = u
        out["faces_valid"] = f["faces_valid"]
        out["thickness"] = np.clip(np.rint(f["thickness"]), 0, 255).astype(np.uint8)
    if fiber:
        out["fiber_vt"] = encode_prob(f["fiber_vt"])
        out["fiber_hz"] = encode_prob(f["fiber_hz"])
        out["fiber_valid"] = f["fiber_valid"]
    if rv:
        for k in ("faces_source", "rv_class", "hzvt_class"):
            out[k] = f[k]
    if winding_fine:
        v = f["wf_valid"]
        for k in ("wf_sin", "wf_cos", "wf_nx", "wf_ny", "wf_nz"):
            u = encode_signed(f[k])
            u[v == 0] = encode_signed(np.float32(0.0))
            out[k] = u
        out["wf_density"] = np.where(v == 1, encode_density(f["wf_density"]), 0).astype(np.uint8)
        out["wf_conf"] = np.where(v == 1, encode_prob(f["wf_conf"]), 0).astype(np.uint8)
        out["wf_valid"] = v
    return out


def encode_coarse(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "phase_sin": encode_signed(f["phase_sin"]), "phase_cos": encode_signed(f["phase_cos"]),
        "density": encode_density(f["density"]), "nx": encode_signed(f["nx"]), "ny": encode_signed(f["ny"]),
        "nz": encode_signed(f["nz"]), "conf": encode_prob(f["conf"]), "valid": f["valid"],
    }


def _write_store(path: str, channels: list[str], enc: dict[str, np.ndarray], origin: tuple[int, int, int], voxel_um: float, scale: float, chunk: int) -> str:
    shape = enc[channels[0]].shape
    w = BrickWriter(path, channels, shape, chunk=chunk, origin_zyx=origin, voxel_um=voxel_um, scale=scale)
    for ci, name in enumerate(channels):
        for z0 in range(0, shape[0], chunk):
            for y0 in range(0, shape[1], chunk):
                for x0 in range(0, shape[2], chunk):
                    blk = enc[name][z0:z0 + chunk, y0:y0 + chunk, x0:x0 + chunk]
                    w.write(ci, origin[0] + z0, origin[1] + y0, origin[2] + x0, np.ascontiguousarray(blk))
    return path


def _write_ct(path: str, ct: np.ndarray, origin: tuple[int, int, int], chunk: int) -> str:
    """CT zarr v3 array covering [0, origin + shape) so global reads at ``origin`` work."""
    full = tuple(o + s for o, s in zip(origin, ct.shape))
    a = zarr.create_array(store=path, shape=full, chunks=(chunk,) * 3, dtype="uint8", fill_value=0)
    a[origin[0]:, origin[1]:, origin[2]:] = ct
    return path


def make_synthetic(
    root: str,
    fine_shape: tuple[int, int, int] = (64, 64, 64),
    fine_origin: tuple[int, int, int] = (16, 32, 32),
    factor: int = 4,
    chunk: int = 32,
    seed: int = 0,
    faces: bool = False,
    fiber: bool = False,
    winding_fine: bool = False,
    rv: bool = False,
) -> dict[str, str]:
    """Write fine.zarr, coarse.zarr (fine/factor footprint) and ct.zarr under ``root``."""
    os.makedirs(root, exist_ok=True)
    if any(v % factor for v in (*fine_shape, *fine_origin)):
        raise ValueError("fine shape/origin must be multiples of factor")
    coarse_shape = tuple(v // factor for v in fine_shape)
    coarse_origin = tuple(v // factor for v in fine_origin)
    ff = fine_fields(fine_shape, seed)
    cf = coarse_fields(coarse_shape, seed, coarse_origin, factor)
    labels = os.path.join(root, "labels")
    return {
        "fine": _write_store(os.path.join(labels, "fine.zarr"),
                             FINE_CHANNELS + (FACE_CHANNELS if faces else []) + (FIBER_CHANNELS if fiber else [])
                             + (RV_CHANNELS if rv else []) + (WF_CHANNELS if winding_fine else []),
                             encode_fine(ff, faces, fiber, winding_fine, rv), fine_origin, 2.4, 1.0, chunk),
        "coarse": _write_store(os.path.join(labels, "coarse.zarr"), COARSE_CHANNELS, encode_coarse(cf), coarse_origin, 2.4 * factor, float(factor), max(1, chunk // factor)),
        "ct": _write_ct(os.path.join(root, "ct.zarr"), ff["ct"], fine_origin, chunk),
        "labels": labels,
    }
