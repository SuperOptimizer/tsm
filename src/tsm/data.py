"""Crop dataset over the TSM label stores (docs/label_store.md) + augmentation.

Coordinates: every store is (C, Z, Y, X) in the voxel grid of its own level with
``origin_zyx`` giving the global offset.  Crop origins are kept in *local* fine
store coordinates; the CT is read at ``origin_zyx + local``.  The student is
trained at the fine (2.4 um) pitch only: winding targets are read from the
matching box of the coarse store and upsampled on the fly.

Augmentation v1 (:func:`augment`, CPU, per sample): random flips of z/y/x and a
random rot90 in the in-plane (y, x) axes.  Scalar targets (sdf, sin, cos,
density, conf, ink, valid masks) are moved with the grid unchanged; the unit
normal is transformed as a vector (component permuted with the axes, negated for
a flipped axis).  "Outward" is therefore a property of the data carried by the
normal channels, and the sdf sign / phase are left alone.  Intensity
augmentation touches the CT only.

Input radial channels (``input_radial=True``, ``extra.train.input_radial``): the crop's
outward unit radial field (``labels.radial_field`` at level 0 from the umbilicus axis,
(z, y, x) order) is appended to ``input`` as channels 2..4, so the network is *told* which
way "out" is -- the sdf sign, the winding phase sign and the normal orientation are all
defined by it and a 128^3 crop cannot infer it.  The channels follow exactly the winding
normal's rule under both augmentation paths (rotate as a vector, negate a flipped
component, resample and renormalise).  Near the axis the field is ill-defined; it is left
as is (the validity masks already ignore those voxels).

Augmentation v2 (:class:`Augment`, device, per batch): the composable,
config-driven, target-aware pipeline (flips, rot90, arbitrary 3D rotations,
scaling, elastic, and the CT-only intensity transforms) documented in
docs/label_store.md "Augmentation v2".

Fibre targets (``fiber_mode``, ``extra.train.fiber_mode``, :mod:`tsm.fiber`): ``"class"``
keeps the two axis-relative probabilities and lets a spatial transform that moves the volume
z axis by more than 45 degrees **swap** them (``fiber_swap`` on :class:`Augment`); ``"direction"``
derives an axial fibre direction + strength per crop from the same channels and the geometry,
and transforms it as a vector.  See :mod:`tsm.fiber` for the derivation and the rationale.

Scroll-axis input channels (``input_axis``, ``extra.train.input_axis``, default on): three more
input channels holding the axis direction in volume coordinates, transformed as a vector exactly
like the radial ones (``student.input_vec_slices`` gives the slices; ``Augment(input_vec=...)``).

Geometry frame (``axis_tangent``, ``extra.train.axis_tangent``, default on): the two vector
inputs are the **orthonormal** pair ``labels.geometry_frame`` builds from the umbilicus -- the
local axis tangent (a symmetric difference over +-64 level-0 voxels, so it is continuous across
crop boundaries) and the radial made perpendicular to it -- instead of a constant ``(1, 0, 0)``
axis and a purely in-plane radial.  The same tangent is the axis the direction fibre targets are
built against (``fiber.derive_direction_targets(axis=...)``).  ``axis_tangent=False`` restores
the old constant fields for the ablation.  Only those two vectors are fed: the binormal ``a x r``
is a pseudovector and would leak ``det(L)`` under flip augmentation.

Surface mode (``surface_mode``, ``extra.train.surface_mode``): "medial" (default,
unchanged: ``surface_sdf`` = the single SDF to the recto medial surface, mask
``sdf_valid``), "faces" (``surface_sdf`` = the 2-channel [sdf_in, sdf_out] of
the two-face labels, mask ``faces_valid``) or "body" (orientation-free: ``surface_sdf``
= the single channel ``min(sdf_in, -sdf_out)`` of :func:`body_sdf`, mask ``faces_valid``) or
"sides" (orientation-free with the magnitude and the sign split in two: ``surface_sdf`` = the
UNSIGNED ``min(|sdf_in|, |sdf_out|)`` of :func:`face_dist` and ``surface_body`` = the 0/1
``(sdf_in > 0) & (sdf_out < 0)`` mask of :func:`body_mask_np`, both on ``faces_valid``).
The two face channels obey exactly the same augmentation rules as the single sdf: "in" and
"out" are defined by the physical outward direction, which moves with the geometry, so no
transform ever swaps them.  The body SDF is a plain scalar SDF and rides the medial path.
In "sides" mode the face distance rides that same scalar path (it is >= 0, so the
saturation rule and the isotropic ``s_iso`` factor apply unchanged) and the body mask moves with the grid like ``ink_prob``;
neither has a sign or a side to swap, so every flip and rotation leaves them alone.

Local Shape Descriptors (``extra.train.heads.lsd``, "sides" mode only): an optional auxiliary
regression target derived from the same two face labels, :func:`lsd_targets` -- the offset to
the sheet centre plane (a vector), the windowed sheet thickness (a length) and the windowed
normal incoherence (a rotation-invariant scalar).  Carried as ``lsd`` (5) + ``lsd_valid`` (1).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

from tsm.fiber import FIBER_MODES, derive_direction_targets, axis_swap_needed, sheet_normal, swap_band_class
from tsm.limits import Budget
from tsm.student import BASE_UM, WINDING_CH, scale_channel_value
from tsm.volume import VolumeReader, VolumeReadError

CLIP = 20.0  # SDF clip in voxels (48 um at 2.4 um)

FINE_CHANNELS = ["sdf", "sdf_valid", "ink", "ink_valid"]
FACE_CHANNELS = ["sdf_in", "sdf_out", "faces_valid", "thickness"]
FIBER_CHANNELS = ["fiber_vt", "fiber_hz", "fiber_valid"]
#: the human fibre-class channel of the upstream rectoverso block (tsm.rvfaces.RV_CHANNELS);
#: optional, and used only as an override / metric, never required
BAND_CHANNEL = "hzvt_class"
#: every optional fibre target key, in the order they are carried through the pipeline
FIBER_TARGET_KEYS = ("fiber_prob", "fiber_dir", "fiber_str", "fiber_weight", "fiber_valid", "fiber_band")
#: per-voxel loss weight of the human-traced band voxels in the direction mode
FIBER_BAND_WEIGHT = 5.0
#: optional per-voxel *surface* loss weight written by ``dev/annot_import.py`` (uint8, 1 =
#: unchanged).  A store without the channel behaves exactly as before; a store with it gains
#: the ``surface_weight`` target key, which multiplies every surface-loss mask in train.py.
FACES_WEIGHT_CHANNEL = "faces_weight"
SURFACE_WEIGHT_KEY = "surface_weight"
#: the 0/1 sheet-body occupancy target of ``surface_mode="sides"`` (:func:`body_mask_np`); it
#: is a plain [0, 1] scalar field and moves with the grid exactly like ``ink_prob``
SURFACE_BODY_KEY = "surface_body"
#: the Local Shape Descriptor target and its mask (``extra.train.heads.lsd``, sides mode);
#: see :func:`lsd_targets`
LSD_KEY, LSD_VALID_KEY = "lsd", "lsd_valid"
#: target keys that are neither in ``TARGET_KEYS`` nor fibre keys (optional, mode/store-driven)
EXTRA_TARGET_KEYS = (SURFACE_WEIGHT_KEY, SURFACE_BODY_KEY, LSD_KEY, LSD_VALID_KEY)
COARSE_CHANNELS = ["phase_sin", "phase_cos", "density", "nx", "ny", "nz", "conf", "valid"]
#: the native-2.4 um winding block (tsm.winding_fine); present only in stores built with
#: ``extra.labels.winding_fine``.  Same encodings as the coarse channels above, except that
#: ``wf_density`` is already in wraps per *fine* voxel (no /factor).
WF_CHANNELS = ["wf_sin", "wf_cos", "wf_density", "wf_nx", "wf_ny", "wf_nz", "wf_conf", "wf_valid"]
#: ``extra.train.winding_source``: where the winding targets come from.
WINDING_SOURCES = ("coarse", "fine", "merge")

SURFACE_MODES = ("medial", "faces", "body", "sides")


def body_sdf(sdf_in: np.ndarray, sdf_out: np.ndarray) -> np.ndarray:
    """The orientation-free **body** SDF ``min(sdf_in, -sdf_out)`` (voxels).

    Both face SDFs are positive on the outward side of their own face, so inside a sheet
    ``sdf_in > 0 > sdf_out`` and the minimum is positive (the distance to the nearer face);
    outside, it is minus the distance to the crossed face.  It is 0 on both faces *and* on a
    labelled contact plane, so two sheets that touch stay two distinct ``body > 0`` components.
    Clipping commutes with ``min``, so the clipped face SDFs give the clipped body SDF.

    The single definition used by the dataset, the training metrics and ``dev/eval_region.py``.
    """
    return np.minimum(np.asarray(sdf_in), -np.asarray(sdf_out))


def face_dist(sdf_in: np.ndarray, sdf_out: np.ndarray) -> np.ndarray:
    """The orientation-free **unsigned** distance to the nearest face, ``min(|sdf_in|, |sdf_out|)``.

    The magnitude half of ``surface_mode="sides"``.  It is 0 on both faces *and* on a labelled
    contact plane, >= 0 everywhere, and says nothing about which side of which face a voxel is
    on -- no naming, no sign flip, and (unlike :func:`body_sdf`) no dependence on the sheet
    thickness.  Clipping commutes with ``min`` and with ``abs`` on a symmetric clip, so the
    clipped face SDFs give the clipped face distance.

    The single definition used by the dataset, the training metrics and ``dev/eval_region.py``.
    """
    return np.minimum(np.abs(np.asarray(sdf_in)), np.abs(np.asarray(sdf_out)))


def body_mask_np(sdf_in: np.ndarray, sdf_out: np.ndarray) -> np.ndarray:
    """The sheet **body** as a boolean mask: ``(sdf_in > 0) & (sdf_out < 0)``.

    The sign half of ``surface_mode="sides"`` -- exactly ``body_sdf(...) > 0``, but written as a
    plain 0/1 occupancy so the network can classify it instead of regressing a signed field
    whose interior magnitude depends on the (noisy, 20-70 voxel) sheet thickness."""
    return (np.asarray(sdf_in) > 0) & (np.asarray(sdf_out) < 0)


# --------------------------------------------------------------------------- #
# Local Shape Descriptors (sides mode, extra.train.heads.lsd)
# --------------------------------------------------------------------------- #
#: the five descriptor channels, in order (:func:`lsd_targets`)
LSD_CHANNELS = ["lsd_off_z", "lsd_off_y", "lsd_off_x", "lsd_thick", "lsd_ncov"]
#: ``extra.train.lsd_sigma``: the gaussian window of the descriptors, in voxels
LSD_SIGMA = 6.0
#: smallest gaussian sigma (voxels) the windowed descriptors are ever evaluated at: the window
#: is computed on a grid ``factor = sigma // LSD_COARSE_SIGMA`` times coarser than the crop
#: (capped at :data:`LSD_COARSE_MAX`), which costs ``factor**3`` less.  A field smoothed with
#: sigma 6 has essentially no power above the Nyquist frequency of a 4-voxel grid, so this is a
#: sampling choice, not an approximation of the window itself; only the block-replicated
#: reconstruction is coarse, and it is applied to the two *smooth* channels (thickness and
#: normal incoherence), never to the offset vector, which stays exact at full resolution.
#: The block grid is mapped onto itself by every cube symmetry of a crop whose size is a
#: multiple of the factor, so the derivation still commutes exactly with the 48 flips/rotations.
LSD_COARSE_SIGMA = 1.5
LSD_COARSE_MAX = 4
#: gaussian truncation radius, in sigmas (scipy's default is 4)
LSD_TRUNCATE = 3.0


def _block_mean(x: np.ndarray, f: int) -> np.ndarray:
    """(C, Z, Y, X) -> the mean over ``f**3`` blocks (edge-padded to a multiple of ``f``).

    One axis at a time (a strided sum per axis, shrinking the array as it goes): ~5x faster
    than the reshape-and-mean, which reduces over three strided axes of the full crop at once."""
    if f == 1:
        return x
    pad = [(0, (-int(n)) % f) for n in x.shape[1:]]
    if any(p[1] for p in pad):
        x = np.pad(x, [(0, 0)] + pad, mode="edge")
    for ax in (1, 2, 3):
        sl = [slice(None)] * 4
        sl[ax] = slice(0, None, f)
        acc = x[tuple(sl)].copy()
        for i in range(1, f):
            sl[ax] = slice(i, None, f)
            acc += x[tuple(sl)]
        x = acc
    return x * np.float32(1.0 / f ** 3)


def _block_repeat(x: np.ndarray, f: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`_block_mean`: replicate each block value, then crop to ``shape``."""
    if f == 1:
        return x
    up = np.repeat(np.repeat(np.repeat(x, f, 1), f, 2), f, 3)
    z, y, xx = (int(v) for v in shape)
    return up[:, :z, :y, :xx]


def _max_eig_sym3(s: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of the symmetric 3x3 matrices ``s`` = [aa, bb, cc, ab, ac, bc].

    Closed form (the standard trigonometric solution of the characteristic cubic), so a whole
    crop costs a handful of elementwise passes instead of a per-voxel ``eigvalsh``."""
    a00, a11, a22, a01, a02, a12 = (s[i] for i in range(6))
    tr = a00 + a11 + a22
    q = tr / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(p2, 0.0) / 6.0) + 1e-12
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    det = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
           + b02 * (b01 * b12 - b11 * b02))
    phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    return q + 2.0 * p * np.cos(phi)


def lsd_coarse_factor(sigma: float = LSD_SIGMA) -> int:
    """Grid coarsening of the windowed descriptors for a window sigma (see :data:`LSD_COARSE_SIGMA`)."""
    return int(max(1, min(LSD_COARSE_MAX, int(float(sigma) // LSD_COARSE_SIGMA))))


def lsd_targets(sdf_in: np.ndarray, sdf_out: np.ndarray, valid: np.ndarray, normal: np.ndarray,
                sigma: float = LSD_SIGMA, clip: float = CLIP,
                factor: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """**Local Shape Descriptors** for sheets (Sheridan et al., Nat Methods 2023), adapted.

    The neuron-segmentation LSDs summarise, in a gaussian window around every voxel and
    restricted to the voxels of the *same object*, where the object sits and how it is shaped;
    they are an auxiliary regression target that makes a boundary network see more than its own
    receptive field.  A sheet has a much simpler shape vocabulary than a neuron, so three
    quantities carry all of it -- all derived from the two face labels ``sdf_in`` / ``sdf_out``
    of the crop and the outward label normal ``normal`` ((z, y, x) components), on the
    label-body voxels (:func:`body_mask_np`) where ``valid == 1``:

    ``off`` (3, voxels, channels ``lsd_off_z/y/x``)
        the vector from the voxel to the **centre plane of its own sheet**,
        ``n * (d_out - d_in) / 2`` with ``d_in = |sdf_in|`` and ``d_out = |sdf_out|`` the
        distances to the two faces along the normal.  It is 0 on the medial ridge, points
        outward (along ``+n``) on the inner half of the sheet and inward on the outer half, and
        its length is at most half the sheet thickness.  Pointwise and exact -- no window --
        because the two face SDFs already carry the neighbourhood a windowed mean would
        estimate, and the sharp target is the better teacher.  A true (polar) **vector**: a
        spatial transform pushes it forward with ``L`` (which also scales it, as a length must).
    ``thick`` (1, voxels, ``lsd_thick``)
        the local sheet thickness ``d_in + d_out`` (clipped to ``2 * clip``), **averaged over
        the gaussian window of ``sigma``, restricted to body voxels**.  A length: it scales with
        an isotropic transform.
    ``ncov`` (1, [0, 1], ``lsd_ncov``)
        "normal incoherence": ``1 - lambda_max(S) / trace(S)`` of the windowed normal scatter
        ``S = G * (n n^T)`` (same-body normalised).  0 on a flat, coherent sheet; it grows
        where the normals inside the window disagree -- a crumple, a fold, or two sheets
        crossing (perpendicular sheets in equal measure give 0.5; the isotropic maximum is
        2/3).  Built from an outer product and its eigenvalues, so it is **rotation invariant**:
        no transport rule, no sign, nothing for an augmentation to get wrong.

    "Same body" is approximated by the label body mask: no per-sheet instance label exists here,
    and the window deliberately spans the gap to the neighbouring wrap -- seeing that is exactly
    what ``ncov`` is for.

    The two windowed descriptors are evaluated on a ``factor``-times coarser grid
    (``None`` = :func:`lsd_coarse_factor`, 4 at the default sigma) with the paper's same-body
    normalisation ``G*(x*body) / max(G*body, eps)``.

    Returns ``(lsd (5, Z, Y, X) = [off_z, off_y, off_x, thick, ncov], lsd_valid (1, Z, Y, X))``;
    the descriptors are 0 wherever ``lsd_valid`` is 0 (not a body voxel, not ``valid == 1``, or
    no usable normal)."""
    from scipy.ndimage import gaussian_filter

    si = np.asarray(sdf_in, np.float32)
    so = np.asarray(sdf_out, np.float32)
    nrm = np.asarray(normal, np.float32)
    if nrm.shape != (3,) + si.shape:
        raise ValueError(f"normal must be (3, Z, Y, X) matching the crop, got {nrm.shape} / {si.shape}")
    f = lsd_coarse_factor(sigma) if factor is None else max(1, int(factor))
    body = (body_mask_np(si, so) & (np.asarray(valid) == 1)
            & ((nrm * nrm).sum(0) > 0.25))  # ... and a usable (unit) outward normal
    bf = body.astype(np.float32)
    di, do = np.abs(si), np.abs(so)
    lsd = np.empty((5,) + si.shape, np.float32)
    np.multiply(nrm, (0.5 * (do - di)) * bf, out=lsd[0:3])  # the offset vector, in place
    # coarse grid: the body occupancy (the window denominator), the body-masked thickness and
    # the body-masked normal
    raw = np.empty((5,) + si.shape, np.float32)
    raw[0] = bf
    np.multiply(np.clip(di + do, 0.0, 2.0 * float(clip)), bf, out=raw[1])
    np.multiply(nrm, bf, out=raw[2:5])
    small = _block_mean(raw, f)
    del raw
    den = np.maximum(small[0], 1e-6)
    nh = small[2:5] / den
    nh /= np.maximum(np.sqrt((nh * nh).sum(0)), 1e-6)
    # mean(n n^T * body) per block: 5 of the 6 unique entries [zz, yy, zy, zx, yx].  The sixth,
    # xx, never has to be smoothed -- the trace of the scatter is ``mean(|n|^2 * body)`` = the
    # body occupancy itself, so ``xx = body - zz - yy`` after the same gaussian.
    nn = np.stack([nh[i] * nh[j] * small[0] for i, j in ((0, 0), (1, 1), (0, 1), (0, 2), (1, 2))])
    stack = np.concatenate([small[0:2], nn], 0)
    sm = np.empty_like(stack)
    sg = float(sigma) / f
    for c in range(stack.shape[0]):
        gaussian_filter(stack[c], sg, truncate=LSD_TRUNCATE, output=sm[c], mode="nearest")
    gd = np.maximum(sm[0], 1e-6)
    thick_s = sm[1] / gd
    # the windowed scatter, normalised so that its trace is exactly 1 (see above)
    scat = sm[2:7] / gd
    scat = np.stack([scat[0], scat[1], 1.0 - scat[0] - scat[1], scat[2], scat[3], scat[4]])
    ncov = np.where(sm[0] > 1e-6, 1.0 - _max_eig_sym3(scat), 0.0)
    up = _block_repeat(np.stack([thick_s, np.clip(ncov, 0.0, 1.0)]).astype(np.float32), f, si.shape)
    np.multiply(up, bf, out=lsd[3:5])
    return lsd, bf[None]


# target keys handed to train.py; every item has all of them (fixed shapes)
TARGET_KEYS = {
    "surface_sdf": 1,
    "surface_valid": 1,
    "ink_prob": 1,
    "ink_valid": 1,
    "winding": 6,  # sin, cos, density, nx, ny, nz
    "winding_conf": 1,
    "winding_valid": 1,
}


def target_keys(surface_mode: str = "medial", fiber: bool = False, fiber_mode: str = "class",
                band: bool = False, surface_weight: bool = False, lsd: bool = False) -> dict[str, int]:
    """``TARGET_KEYS`` for a surface mode. ``"faces"`` makes ``surface_sdf`` 2 channels
    ([sdf_in, sdf_out]) and ``surface_valid`` the ``faces_valid`` mask; ``"body"`` keeps the
    single ``surface_sdf`` channel (:func:`body_sdf`) with the same ``faces_valid`` mask;
    every other key, and every per-channel augmentation / pooling rule, is unchanged.

    ``fiber`` adds the fibre targets (:mod:`tsm.fiber`):

    * ``fiber_mode="class"`` (default): ``fiber_prob`` (2: [vt, hz] teacher probabilities) and
      ``fiber_valid``.  The two classes are defined relative to the **scroll axis**, so a
      transform that moves the volume z axis by more than 45 degrees swaps them
      (:func:`tsm.fiber.axis_swap_needed`); everything else follows the ink rules.
    * ``fiber_mode="direction"``: ``fiber_dir`` (3, the axial fibre direction in (z, y, x) --
      a **vector**, transformed like the winding normal), ``fiber_str`` (1), ``fiber_weight``
      (1, the per-voxel loss weight: ``fiber_band_weight`` on the human bands) and
      ``fiber_valid``.

    ``band`` adds ``fiber_band`` (1) = the human ``hzvt_class`` codes carried through for the
    teacher-independent holdout metric (0 bg / 1 hz / 2 vt / 3 exclude; the hz/vt codes are
    swapped by the same rule as the classes).

    ``surface_weight`` adds ``surface_weight`` (1) = the store's ``faces_weight`` channel, a
    per-voxel multiplier on the surface loss (1 everywhere unless a human correction was
    imported).  It is a plain scalar field: it moves with the grid like ``ink_prob``.

    ``lsd`` (``extra.train.heads.lsd``, "sides" mode only) adds the Local Shape Descriptors
    ``lsd`` (5 = [off_z, off_y, off_x, thick, ncov], :func:`lsd_targets`) and their mask
    ``lsd_valid`` (1)."""
    if surface_mode not in SURFACE_MODES:
        raise ValueError(f"surface_mode must be one of {SURFACE_MODES}, got {surface_mode!r}")
    t = dict(TARGET_KEYS)
    if surface_mode == "faces":
        t["surface_sdf"] = 2
    if surface_mode == "sides":
        # magnitude + sign split in two: surface_sdf is the UNSIGNED face distance
        # (:func:`face_dist`) and surface_body the 0/1 body mask (:func:`body_mask_np`)
        t["surface_body"] = 1
    if fiber:
        if fiber_mode not in FIBER_MODES:
            raise ValueError(f"fiber_mode must be one of {list(FIBER_MODES)}, got {fiber_mode!r}")
        t["fiber_prob"] = 2  # the teacher probabilities: the loss target in the "class" mode,
                             # carried in the "direction" mode too so that the holdout / eval
                             # AUPRC against the teacher is the same metric in both modes
        if fiber_mode == "direction":
            t["fiber_dir"], t["fiber_str"], t["fiber_weight"] = 3, 1, 1
        t["fiber_valid"] = 1
        if band:
            t["fiber_band"] = 1
    if surface_weight:
        t[SURFACE_WEIGHT_KEY] = 1
    if lsd:
        if surface_mode != "sides":
            raise ValueError(f"the lsd targets need surface_mode='sides', got {surface_mode!r}")
        t[LSD_KEY], t[LSD_VALID_KEY] = 5, 1
    return t


# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #
def decode_sdf(u8: np.ndarray, clip: float = CLIP) -> np.ndarray:
    return (u8.astype(np.float32) - 128.0) * (clip / 127.0)


def encode_sdf(sdf: np.ndarray, clip: float = CLIP) -> np.ndarray:
    v = np.clip(sdf, -clip, clip)
    return np.clip(np.round(128.0 + v * 127.0 / clip), 1, 255).astype(np.uint8)


def decode_signed(u8: np.ndarray) -> np.ndarray:
    """sin/cos/normal components: (u8 - 127.5) / 127.5 in [-1, 1]."""
    return (u8.astype(np.float32) - 127.5) / 127.5


def encode_signed(v: np.ndarray) -> np.ndarray:
    return np.clip(np.round(127.5 + 127.5 * np.clip(v, -1.0, 1.0)), 0, 255).astype(np.uint8)


def decode_density(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 1000.0


def encode_density(v: np.ndarray) -> np.ndarray:
    return np.clip(np.round(v * 1000.0), 0, 255).astype(np.uint8)


def decode_prob(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def encode_prob(v: np.ndarray) -> np.ndarray:
    return np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# label store reader
# --------------------------------------------------------------------------- #
class LabelStore:
    """Read-only view of a BrickWriter store; reopens after fork (pid check)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        self._arr: zarr.Array | None = None
        self._pid = -1
        arr = self.array
        attrs = dict(arr.attrs)
        self.channels: list[str] = list(attrs.get("channels", [])) or [str(i) for i in range(arr.shape[0])]
        self.shape_zyx: tuple[int, int, int] = tuple(int(s) for s in arr.shape[1:])  # type: ignore[assignment]
        self.origin_zyx: tuple[int, int, int] = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))  # type: ignore[assignment]
        self.voxel_um = float(attrs.get("voxel_um", BASE_UM))
        self.scale = float(attrs.get("scale", 1.0))
        self.chunk = int(arr.chunks[1])

    @property
    def array(self) -> zarr.Array:
        if self._arr is None or self._pid != os.getpid():
            self._arr = zarr.open_array(store=self.path, mode="r")
            self._pid = os.getpid()
        return self._arr

    def __getstate__(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["_arr"] = None
        d["_pid"] = -1
        return d

    @property
    def name(self) -> str:
        base = os.path.basename(self.path)
        return base[:-5] if base.endswith(".zarr") else base

    def index(self, channel: str) -> int:
        try:
            return self.channels.index(channel)
        except ValueError:
            raise KeyError(f"channel {channel!r} not in {self.path} ({self.channels})") from None

    def read(self, channel: str, z0: int, y0: int, x0: int, size: int | Sequence[int]) -> np.ndarray:
        """Local-coordinate crop of one channel, zero (= no data) padded outside."""
        sz = (size, size, size) if isinstance(size, int) else tuple(int(s) for s in size)
        out = np.zeros(sz, np.uint8)
        lo = [max(0, int(v)) for v in (z0, y0, x0)]
        hi = [min(self.shape_zyx[i], int(v) + sz[i]) for i, v in enumerate((z0, y0, x0))]
        if any(h <= l for l, h in zip(lo, hi)):
            return out
        blk = np.asarray(self.array[self.index(channel), lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        d = [lo[i] - int(v) for i, v in enumerate((z0, y0, x0))]
        out[d[0]:d[0] + blk.shape[0], d[1]:d[1] + blk.shape[1], d[2]:d[2] + blk.shape[2]] = blk
        return out

    def read_many(self, channels: Sequence[str], z0: int, y0: int, x0: int, size: int) -> np.ndarray:
        return np.stack([self.read(c, z0, y0, x0, size) for c in channels], axis=0)


# --------------------------------------------------------------------------- #
# origins
# --------------------------------------------------------------------------- #
def _origins_key(store: LabelStore, valid_channel: str, patch: int, stride: int, min_frac: float) -> dict[str, Any]:
    """Everything the sampled origins depend on: the selection criteria *and* the label store's
    generation.  Rebuilding labels in place, switching the validity channel or moving the
    valid-fraction threshold must all invalidate a cached origin list."""
    meta = os.path.join(store.path, "zarr.json")
    try:
        st = os.stat(meta)
        gen = [int(st.st_mtime_ns), int(st.st_size)]
    except OSError:
        gen = None
    try:
        attrs = json.dumps(dict(store.array.attrs), sort_keys=True, default=str)
    except Exception:
        attrs = ""
    return {"v": 1, "store": store.path, "shape": list(store.shape_zyx), "channels": list(store.channels),
            "valid_channel": str(valid_channel), "patch": int(patch), "stride": int(stride),
            "min_frac": float(min_frac), "store_meta": gen,
            "attrs_sha1": hashlib.sha1(attrs.encode()).hexdigest()}


def build_origins(
    store: LabelStore,
    valid_channel: str,
    patch: int = 128,
    stride: int = 64,
    min_frac: float = 0.05,
    cache_dir: str | None = None,
    name: str | None = None,
    force: bool = False,
    block: int = 512,
) -> np.ndarray:
    """Local (z, y, x) origins of ``patch``-cubes with >= ``min_frac`` valid==1 voxels.

    Along an axis where the store is thinner than ``patch`` (thin slabs, e.g. the
    64-deep coarse store) the window is centred on the store (negative origin);
    the overhang reads real CT but no-data labels, which every loss masks out.
    Scans the valid channel in (chunk, block, block) slabs and counts valid==1 in
    ``stride``-cubes, so only a count grid is ever held in RAM.  Cached to
    ``<store>/../origins_<name>.npy`` (int32, (N, 3)).
    """
    if patch % stride:
        raise ValueError(f"stride {stride} must divide patch {patch}")
    cache_dir = cache_dir or os.path.dirname(store.path)
    name = name or f"{store.name}_p{patch}_s{stride}_{valid_channel}_f{float(min_frac):g}"
    cache = os.path.join(cache_dir, f"origins_{name}.npy")
    manifest = cache[:-4] + ".json"
    key = _origins_key(store, valid_channel, patch, stride, min_frac)
    if not force and os.path.exists(cache):
        try:
            with open(manifest) as fh:
                old_key = json.load(fh)
        except (OSError, ValueError):
            old_key = None
        if old_key == key:
            return np.load(cache).astype(np.int32).reshape(-1, 3)
        print(f"[tsm] origins cache {os.path.basename(cache)} is stale "
              f"({'no manifest' if old_key is None else 'selection or label store changed'}); recomputing",
              flush=True)
    Z, Y, X = store.shape_zyx
    nz, ny, nx = (-(-Z // stride), -(-Y // stride), -(-X // stride))  # ceil: partial trailing cubes count too
    counts = np.zeros((nz, ny, nx), np.int64)
    ci = store.index(valid_channel)
    arr = store.array
    zstep = max(store.chunk, stride)
    zstep -= zstep % stride
    bstep = max(block, stride)
    bstep -= bstep % stride
    for z0 in range(0, Z, zstep):
        z1 = min(z0 + zstep, Z)
        for y0 in range(0, Y, bstep):
            y1 = min(y0 + bstep, Y)
            for x0 in range(0, X, bstep):
                x1 = min(x0 + bstep, X)
                v = np.asarray(arr[ci, z0:z1, y0:y1, x0:x1]) == 1
                sz, sy, sx = (-(-(z1 - z0) // stride), -(-(y1 - y0) // stride), -(-(x1 - x0) // stride))
                pad = [(0, sz * stride - (z1 - z0)), (0, sy * stride - (y1 - y0)), (0, sx * stride - (x1 - x0))]
                if any(p[1] for p in pad):
                    v = np.pad(v, pad)
                c = v.reshape(sz, stride, sy, stride, sx, stride).sum(axis=(1, 3, 5))
                counts[z0 // stride:z0 // stride + sz, y0 // stride:y0 // stride + sy, x0 // stride:x0 // stride + sx] = c
                del v
    need = min_frac * float(patch) ** 3
    cs = np.pad(counts.cumsum(0).cumsum(1).cumsum(2), ((1, 0), (1, 0), (1, 0)))

    def candidates(dim: int, n: int) -> list[tuple[int, int, int]]:
        """(origin, grid_lo, grid_hi) per axis; a store thinner than the patch gets one centred window."""
        if dim < patch:
            return [((dim - patch) // 2, 0, n)]
        return [(o, o // stride, o // stride + patch // stride) for o in range(0, dim - patch + 1, stride)]

    origins: list[tuple[int, int, int]] = []
    for oz, z0g, z1g in candidates(Z, nz):
        for oy, y0g, y1g in candidates(Y, ny):
            for ox, x0g, x1g in candidates(X, nx):
                tot = (
                    cs[z1g, y1g, x1g] - cs[z0g, y1g, x1g] - cs[z1g, y0g, x1g] - cs[z1g, y1g, x0g]
                    + cs[z0g, y0g, x1g] + cs[z0g, y1g, x0g] + cs[z1g, y0g, x0g] - cs[z0g, y0g, x0g]
                )
                if tot >= need:
                    origins.append((oz, oy, ox))
    out = np.asarray(origins, np.int32).reshape(-1, 3)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, cache)
    tmpm = manifest + ".tmp"
    with open(tmpm, "w") as fh:
        json.dump(key, fh, sort_keys=True)
    os.replace(tmpm, manifest)
    return out


_HOLD_AXES = {"z": 0, "y": 1, "x": 2}


def _n_distinct(o: np.ndarray, axis: int) -> int:
    return int(len(np.unique(o[:, axis])))


def _band_split(o: np.ndarray, patch: int, axis: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out origins with ``lo <= o[axis] < hi``; training excludes every crop that overlaps the
    held-out crop *extents* -- the last held origin covers ``[o, o + patch)``, which reaches past
    ``hi`` for an interior band, so ``o[axis] >= hi`` alone would leak up to ``patch - 1`` voxels."""
    a = o[:, axis]
    hold = (a >= lo) & (a < hi)
    if hold.any():
        lo_e, hi_e = int(a[hold].min()), int(a[hold].max()) + int(patch)
    else:
        lo_e, hi_e = int(lo), int(hi)
    train = (a + patch <= lo_e) | (a >= hi_e)
    return o[train], o[hold]


def _frac_band(o: np.ndarray, patch: int, axis: int, f: float) -> tuple[int, int]:
    """(lo, hi) of the top ``f`` fraction of the origin range along ``axis``."""
    lo_v, hi_v = int(o[:, axis].min()), int(o[:, axis].max()) + int(patch)
    return int(round(hi_v - f * (hi_v - lo_v))), hi_v


def split_holdout(origins: np.ndarray, patch: int, spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """(train_origins, holdout_origins) from ``spec``:

    * a list of ``[z, y, x]`` local origins (used as given; training drops every origin
      whose crop overlaps a holdout crop),
    * ``{"<a>_frac": f}`` for ``a`` in z/y/x: the top ``f`` fraction of that axis' origin
      range is held out,
    * ``{"<a>_range": [lo, hi]}``: origins with ``lo <= o[a] < hi`` are held out,
    * ``{"yx_frac": f}``: the corner block covering the top ``sqrt(f)`` of both the y and
      the x origin range (about ``f`` of the y/x origin area) is held out -- the mode for
      thin slabs, where no single axis is deep enough to split,
    * ``None`` / ``{}``: no holdout.

    A ``_frac`` request that leaves an empty holdout or an empty training set on its own
    axis (e.g. ``z_frac`` on a 256-deep region with 128-cubes: the z origins are only
    0/64/128, so the top 15% of z holds nothing) falls back to a y band, then an x band,
    then the ``yx_frac`` corner -- the first split that is non-empty on both sides wins,
    so a request that already worked keeps its exact old split.  ``_range`` is taken
    literally.  Training origins never overlap a holdout crop: a band leaves a ``patch``
    gap along its axis, the corner block a gap along y or x, an explicit list is tested in 3D.
    """
    o = np.asarray(origins, np.int32).reshape(-1, 3)
    if spec is None or (isinstance(spec, dict) and not spec):
        return o, np.zeros((0, 3), np.int32)
    P = int(patch)
    if isinstance(spec, dict):
        known = {f"{a}_{k}" for a in _HOLD_AXES for k in ("frac", "range")} | {"yx_frac"}
        unknown = sorted(set(spec) - known)
        if unknown:
            raise ValueError(f"unknown holdout_origins keys: {unknown}")
        if len(spec) != 1:
            raise ValueError(f"holdout_origins takes exactly one key, got {sorted(spec)}")
        ((key, val),) = spec.items()
        if len(o) == 0:
            return o, o
        if key.endswith("_range"):
            lo, hi = (int(v) for v in val)
            return _band_split(o, P, _HOLD_AXES[key[0]], lo, hi)
        f = float(val)
        if not 0.0 < f < 1.0:
            raise ValueError(f"holdout_origins.{key} must be in (0, 1)")

        def band(axis: int) -> tuple[np.ndarray, np.ndarray]:
            lo, hi = _frac_band(o, P, axis, f)
            return _band_split(o, P, axis, lo, hi)

        def corner() -> tuple[np.ndarray, np.ndarray]:
            fa = float(np.sqrt(f))
            ylo, _ = _frac_band(o, P, 1, fa)
            xlo, _ = _frac_band(o, P, 2, fa)
            hold = (o[:, 1] >= ylo) & (o[:, 2] >= xlo)
            train = (o[:, 1] + P <= ylo) | (o[:, 2] + P <= xlo)
            return o[train], o[hold]

        modes: list[tuple[str, Any]] = []
        if key == "yx_frac":
            modes = [("yx corner", corner)]
        else:
            a0 = _HOLD_AXES[key[0]]
            modes = [(f"{key[0]} band", lambda a=a0: band(a))]
            # thin axis (e.g. z on a 256-deep region: origins are only 0/64/128) -> y band, then x, then corner
            modes += [(f"{n} band", lambda a=a: band(a)) for n, a in (("y", 1), ("x", 2), ("z", 0)) if a != a0]
            modes += [("yx corner", corner)]
        for i, (name, fn) in enumerate(modes):
            train_o, hold_o = fn()
            if len(hold_o) and len(train_o):
                if i:
                    print(f"[tsm] holdout: {key} leaves no usable {modes[0][0]} split "
                          f"({_n_distinct(o, _HOLD_AXES[key[0]]) if key != 'yx_frac' else 0} distinct origins on "
                          f"that axis); holding out a {name} instead", flush=True)
                return train_o, hold_o
        return modes[0][1]()  # nothing works: return the requested split (train.py reports the empty holdout)
    h = np.asarray(spec, np.int32).reshape(-1, 3)
    if len(h) == 0:
        return o, h
    d = np.abs(o[:, None, :] - h[None, :, :])
    overlap = (d < P).all(-1).any(1)
    return o[~overlap], h


def load_axis_spec(axis: Any) -> np.ndarray:
    """Umbilicus control points from an (N, 3) array, a JSON path, or None (the default axis)."""
    from tsm.labels import DEFAULT_AXIS, load_axis

    if axis is None:
        return load_axis(DEFAULT_AXIS)
    if isinstance(axis, str):
        return load_axis(axis)
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"axis must be (N, 3) (z, y, x) control points, got {a.shape}")
    return a[np.argsort(a[:, 0])]


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
NORMAL_AXES = {"nx": 2, "ny": 1, "nz": 0}  # normal component -> spatial axis (Z=0, Y=1, X=2)


def _vector_keys(normal_key: str | Sequence[str] | None) -> list[str]:
    if normal_key is None:
        return []
    return [normal_key] if isinstance(normal_key, str) else list(normal_key)


def _orthonormalise_frame(r: torch.Tensor, a: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-orthonormalise the geometry frame ``(radial, axis)`` (each (B, 3, Z, Y, X)) after a
    resampling: ``a <- a/|a|``, ``r <- (r - (r.a) a)`` normalised.  The axis is kept as it is
    (it is the coarser, smoother field); zero vectors stay zero."""
    an = a.norm(dim=1, keepdim=True)
    a = torch.where(an > eps, a / an.clamp_min(eps), torch.zeros_like(a))
    r = r - (r * a).sum(dim=1, keepdim=True) * a
    rn = r.norm(dim=1, keepdim=True)
    r = torch.where(rn > eps, r / rn.clamp_min(eps), torch.zeros_like(r))
    return r, a


def _flip_fields(fields: dict[str, np.ndarray], axis: int, normal_key: str | Sequence[str] | None) -> None:
    for k in fields:
        fields[k] = np.flip(fields[k], axis=axis + 1)  # arrays are (C, Z, Y, X)
    for key in _vector_keys(normal_key):
        n = fields[key].copy()
        n[2 - axis] *= -1.0  # normal is stored (nx, ny, nz); spatial axis a <-> component 2-a
        fields[key] = n


def _transpose_yx(fields: dict[str, np.ndarray], normal_key: str | Sequence[str] | None) -> None:
    for k in fields:
        fields[k] = np.swapaxes(fields[k], 2, 3)
    for key in _vector_keys(normal_key):
        n = fields[key].copy()
        n[[0, 1]] = n[[1, 0]]  # swap nx <-> ny
        fields[key] = n


def spatial_transform(
    fields: dict[str, np.ndarray],
    flips: Sequence[bool],
    rot_k: int,
    normal_key: str | Sequence[str] | None = "normal",
) -> dict[str, np.ndarray]:
    """Apply flips (z, y, x) then ``rot_k`` rot90s in the (y, x) plane to (C, Z, Y, X) arrays.

    Reproduces ``np.flip`` / ``np.rot90(a, k, axes=(2, 3))`` on every field and
    transforms every field named by ``normal_key`` (one key or a sequence; channels
    ordered nx, ny, nz) as a vector.
    """
    f = dict(fields)
    for a in range(3):
        if flips[a]:
            _flip_fields(f, a, normal_key)
    k = int(rot_k) % 4
    # np.rot90(m, 1, axes=(ay, ax)) == transpose(flip(m, ax)); k=3 == flip(transpose(m), ax); k=2 == flip both
    if k == 1:
        _flip_fields(f, 2, normal_key)
        _transpose_yx(f, normal_key)
    elif k == 2:
        _flip_fields(f, 1, normal_key)
        _flip_fields(f, 2, normal_key)
    elif k == 3:
        _transpose_yx(f, normal_key)
        _flip_fields(f, 2, normal_key)
    return {kk: np.ascontiguousarray(v) for kk, v in f.items()}


def intensity_augment(ct01: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """CT in [0, 1] -> gamma / brightness+contrast / gaussian noise, each with prob p."""
    x = ct01.astype(np.float32, copy=True)
    if rng.random() < p:
        g = rng.uniform(0.7, 1.4)
        x = np.clip(x, 0.0, 1.0) ** g
    if rng.random() < p:
        c = rng.uniform(0.8, 1.2)
        b = rng.uniform(-0.1, 0.1)
        m = float(x.mean())
        x = (x - m) * c + m + b
    if rng.random() < p:
        s = rng.uniform(0.0, 0.03)
        x = x + rng.normal(0.0, s, size=x.shape).astype(np.float32)
    return x


def augment(sample: dict[str, Any], rng: np.random.Generator, spatial: bool = True, intensity: bool = True) -> dict[str, Any]:
    """Augment one dataset sample (numpy, before tensor conversion).

    ``sample["ct"]`` is (1, Z, Y, X) float in [0, 1]; targets are (C, Z, Y, X).
    """
    out = dict(sample)
    if spatial:
        flips = [bool(rng.random() < 0.5) for _ in range(3)]
        rot_k = int(rng.integers(0, 4))
        fields = {k: out[k] for k in ("ct", "radial", "axis_dir", *TARGET_KEYS, *FIBER_TARGET_KEYS,
                                      *EXTRA_TARGET_KEYS) if k in out}
        # winding = (sin, cos, density, nx, ny, nz): split the normal out for the vector transform
        w = fields.pop("winding", None)
        vkeys: list[str] = []
        if w is not None:
            fields["_wscalar"] = w[:3]
            fields["normal"] = w[3:6]
            vkeys.append("normal")
        # the LSD target: channels 0..2 are the offset VECTOR (z, y, x), 3 (thickness) and 4
        # (normal incoherence) are scalars.  Flips and rot90 are isometries, so the two lengths
        # are unchanged and only the offset components move.
        lsd = fields.pop(LSD_KEY, None)
        if lsd is not None:
            fields["_lsdvec"] = lsd[0:3][::-1].copy()  # (off_z, off_y, off_x) -> (x, y, z)
            fields["_lsdscalar"] = lsd[3:5]
            vkeys.append("_lsdvec")
        # the (z, y, x)-ordered vector fields (the radial and scroll-axis input channels, the
        # fibre direction target); the vector rule below wants (x, y, z) component order.
        # NOTE the v1 path only flips and rot90s in the (y, x) plane, so the volume z axis is
        # preserved (up to a sign, which the axial fibre classes ignore) and no fibre class
        # swap is ever needed here -- see Augment.apply_spatial for the v2 rule.
        zyx = [k for k in ("radial", "axis_dir", "fiber_dir") if k in fields]
        for k in zyx:
            fields[k] = fields[k][::-1].copy()
            vkeys.append(k)
        fields = spatial_transform(fields, flips, rot_k, normal_key=vkeys or None)
        if w is not None:
            fields["winding"] = np.concatenate([fields.pop("_wscalar"), fields.pop("normal")], axis=0)
        if lsd is not None:
            fields[LSD_KEY] = np.concatenate([np.ascontiguousarray(fields.pop("_lsdvec")[::-1]),
                                              fields.pop("_lsdscalar")], axis=0)
        for k in zyx:
            fields[k] = np.ascontiguousarray(fields[k][::-1])
        out.update(fields)
        out["aug"] = {"flips": flips, "rot_k": rot_k}
    if intensity:
        out["ct"] = intensity_augment(out["ct"], rng)
    return out


# --------------------------------------------------------------------------- #
# coarse -> fine target upsampling
# --------------------------------------------------------------------------- #
def store_factor(fine: LabelStore, coarse: LabelStore) -> int:
    """Integer pitch ratio coarse/fine from the store attrs (4 for 9.6/2.4, 2 for 4.8/2.4)."""
    r = coarse.voxel_um / fine.voxel_um
    f = int(round(r))
    if f < 1 or abs(r - f) > 1e-6:
        raise ValueError(f"coarse pitch {coarse.voxel_um} um is not an integer multiple of {fine.voxel_um} um")
    return f


def upsample_coarse_targets(box: dict[str, np.ndarray], factor: int, out_shape: Sequence[int], margin: int = 1,
                            offset: Sequence[int] = (0, 0, 0)) -> dict[str, np.ndarray]:
    """Coarse winding box (decoded floats, with ``margin`` coarse voxels on every side) -> fine targets.

    ``box`` keys: ``winding`` (6, cz, cy, cx) = sin, cos, density, nx, ny, nz (density in
    wraps per *coarse* voxel), ``conf`` (1, ...) in [0, 1], ``valid`` (1, ...) in {0, 1, 2}.
    Coarse voxel i covers fine voxels [i*f, (i+1)*f) (centre-aligned, i.e.
    ``F.interpolate(align_corners=False)``).  sin/cos, the normal and density are
    mask-aware trilinear (only valid==1 coarse voxels contribute), then sin/cos and
    the normal are renormalized and density is divided by ``factor`` (wraps per fine
    voxel); conf and valid are nearest.  Returns ``winding``, ``winding_conf``,
    ``winding_valid`` at ``out_shape``.
    """
    f = int(factor)
    oz, oy, ox = (int(v) for v in out_shape)
    w = torch.from_numpy(np.ascontiguousarray(box["winding"], dtype=np.float32))[None]
    conf = torch.from_numpy(np.ascontiguousarray(box["conf"], dtype=np.float32))[None]
    valid = torch.from_numpy(np.ascontiguousarray(box["valid"], dtype=np.float32))[None]
    m = (valid == 1).float()
    up = lambda t, mode: F.interpolate(t, scale_factor=f, mode=mode)  # noqa: E731
    cont = up(w * m, "trilinear") / up(m, "trilinear").clamp_min(1e-6)
    # ``offset`` = the fine crop origin's remainder modulo ``factor``: the coarse box was floored to
    # the coarse grid, so an unaligned fine origin starts ``offset`` fine voxels into the first
    # (non-margin) coarse voxel.  Dropping it would give X = 8 and X = 9 (factor 4) identical targets.
    off = [int(v) for v in offset]
    if len(off) != 3 or any(not 0 <= v < f for v in off):
        raise ValueError(f"offset must be three values in [0, {f}), got {tuple(offset)}")
    lo = [margin * f + v for v in off]
    need = [l + n for l, n in zip(lo, (oz, oy, ox))]
    have = [int(v) * f for v in w.shape[2:]]
    if any(nd > hv for nd, hv in zip(need, have)):
        raise ValueError(f"coarse box too small for offset {tuple(off)}: need {need} fine voxels, have {have}")
    sl = (slice(None), slice(None), slice(lo[0], lo[0] + oz), slice(lo[1], lo[1] + oy), slice(lo[2], lo[2] + ox))
    cont = cont[sl]
    sc = F.normalize(cont[:, 0:2], dim=1, eps=1e-6)
    n = F.normalize(cont[:, 3:6], dim=1, eps=1e-6)
    dens = cont[:, 2:3] / f
    out_w = torch.cat([sc, dens, n], dim=1)
    out_c = up(conf, "nearest")[sl]
    out_v = up(valid, "nearest")[sl]
    return {"winding": out_w[0].numpy(), "winding_conf": out_c[0].numpy(), "winding_valid": out_v[0].numpy()}


def fine_winding_targets(store: "LabelStore", z0: int, y0: int, x0: int, patch: int) -> dict[str, np.ndarray]:
    """The ``wf_*`` block of a fine-store crop, decoded into the winding target layout.

    Same keys and shapes as :func:`upsample_coarse_targets` -- ``winding`` (6, P, P, P) =
    sin, cos, density (wraps per *fine* voxel, already), nx, ny, nz; ``winding_conf`` and
    ``winding_valid`` (1, P, P, P).  ``wf_valid`` is 0/1 only (see tsm.winding_fine)."""
    r = lambda ch: store.read(ch, z0, y0, x0, patch)  # noqa: E731
    w = np.stack([
        decode_signed(r("wf_sin")), decode_signed(r("wf_cos")), decode_density(r("wf_density")),
        decode_signed(r("wf_nx")), decode_signed(r("wf_ny")), decode_signed(r("wf_nz")),
    ]).astype(np.float32)
    return {"winding": w, "winding_conf": decode_prob(r("wf_conf"))[None].astype(np.float32),
            "winding_valid": r("wf_valid").astype(np.float32)[None]}


def merge_winding_targets(coarse: dict[str, np.ndarray], fine: dict[str, np.ndarray],
                          source: str = "merge") -> dict[str, np.ndarray]:
    """Combine the upsampled-coarse and the native-fine winding targets.

    ``"fine"`` supervises **only** where ``wf_valid == 1`` (everything else becomes
    ``winding_valid = 0``, i.e. unsupervised); ``"merge"`` takes the fine field there and
    falls back to the coarse targets -- values, conf and validity -- everywhere else.
    """
    if source not in ("fine", "merge"):
        raise ValueError(f"winding source must be 'fine' or 'merge', got {source!r}")
    m = fine["winding_valid"] == 1
    if source == "fine":
        mf = m.astype(np.float32)
        return {"winding": fine["winding"] * mf, "winding_conf": fine["winding_conf"] * mf,
                "winding_valid": mf}
    return {"winding": np.where(m, fine["winding"], coarse["winding"]).astype(np.float32),
            "winding_conf": np.where(m, fine["winding_conf"], coarse["winding_conf"]).astype(np.float32),
            "winding_valid": np.where(m, np.float32(1.0), coarse["winding_valid"]).astype(np.float32)}


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
ReaderSpec = VolumeReader | Callable[[], VolumeReader]


class CropDataset(Dataset):
    """Random ``patch``-cubes at the fine (2.4 um) pitch.

    Surface/ink targets come from ``fine_store``; winding targets from the matching
    box of ``coarse_store`` (origin/factor, size/factor, +1 voxel margin) upsampled on
    the fly (:func:`upsample_coarse_targets`).  ``ct_reader`` may be a VolumeReader
    or a zero-arg factory (preferred with DataLoader workers: the reader is opened
    lazily per process).  ``coarse_store`` may be None (winding masks all zero).
    """

    def __init__(
        self,
        ct_reader: ReaderSpec,
        fine_store: LabelStore | str,
        coarse_store: LabelStore | str | None = None,
        patch: int = 128,
        seed: int = 0,
        length: int = 100_000,
        stride: int = 64,
        min_frac: float = 0.05,
        augment: bool = True,
        origins: np.ndarray | None = None,
        surface_mode: str = "medial",
        input_radial: bool = False,
        axis: Any = None,
        fiber: bool | None = None,
        winding_source: str = "coarse",
        fiber_mode: str = "class",
        fiber_band_weight: float = FIBER_BAND_WEIGHT,
        input_axis: bool = False,
        core_radius_vox: float = 0.0,
        axis_tangent: bool = True,
        body_ct_gate: float | None = None,
        lsd: bool = False,
        lsd_sigma: float = LSD_SIGMA,
    ) -> None:
        self.surface_mode = str(surface_mode)
        self.input_radial = bool(input_radial)
        self.input_axis = bool(input_axis)
        # True: the geometry inputs follow the real umbilicus (local tangent + perpendicular
        # radial, labels.geometry_frame).  False reproduces the pre-2026-09-12 fields -- a
        # constant (1, 0, 0) axis and the purely in-plane radial -- for the ablation.
        self.axis_tangent = bool(axis_tangent)
        # umbilicus core mask applied on the fly (extra.train.core_radius_vox), so an existing
        # store can be used without the label rebuild that extra.labels.core_radius_vox needs
        self.core_radius_vox = float(core_radius_vox or 0.0)
        if self.core_radius_vox < 0:
            raise ValueError(f"core_radius_vox must be >= 0, got {core_radius_vox!r}")
        # extra.train.body_ct_gate: CT threshold (RAW uint8 units) under which a label-body
        # voxel stops supervising the surface targets -- see :meth:`load`.  None = off.
        self.body_ct_gate = None if body_ct_gate is None else float(body_ct_gate)
        self._ct_gate_crops = 0
        # extra.train.heads.lsd: derive the Local Shape Descriptor targets in :meth:`load`
        # (sides mode only -- they are built from the two face labels of that mode)
        self.lsd = bool(lsd)
        self.lsd_sigma = float(lsd_sigma)
        if self.lsd and self.surface_mode != "sides":
            raise ValueError(f"heads.lsd needs surface_mode='sides', got {surface_mode!r}")
        if self.lsd_sigma <= 0:
            raise ValueError(f"lsd_sigma must be > 0, got {lsd_sigma!r}")
        self._axis_spec = axis
        self.axis: np.ndarray | None = None  # loaded below, once fiber/fiber_mode are known
        self.patch = int(patch)
        self.seed = int(seed)
        self.length = int(length)
        self.do_augment = bool(augment)
        self.fine = LabelStore(fine_store) if isinstance(fine_store, str) else fine_store
        self.coarse = LabelStore(coarse_store) if isinstance(coarse_store, str) else coarse_store
        self.factor = store_factor(self.fine, self.coarse) if self.coarse is not None else 1
        if self.coarse is not None and self.patch % self.factor:
            raise ValueError(f"patch {self.patch} must be divisible by the coarse factor {self.factor}")
        self._ct_spec = ct_reader
        self.read_failures = 0
        self._ct: VolumeReader | None = None
        self._ct_pid = -1
        # fiber targets: None = auto (present iff the store carries them); True forces the keys
        # (all-zero, i.e. fully masked out, when the store has no fiber channels).
        self.has_fiber_channels = all(c in self.fine.channels for c in FIBER_CHANNELS)
        self.fiber = self.has_fiber_channels if fiber is None else bool(fiber)
        if self.fiber and not self.has_fiber_channels:
            print(f"[tsm] WARNING fiber targets requested but {self.fine.path} has no {FIBER_CHANNELS} "
                  f"channels: fiber_valid is 0 everywhere (the fiber head gets no supervision)", flush=True)
        # winding targets: "coarse" (unchanged: the upsampled coarse store), "fine" (the
        # native 2.4 um wf_* block only) or "merge" (fine where wf_valid == 1, coarse elsewhere).
        # A store without the wf_* channels always behaves exactly as "coarse".
        self.winding_source = str(winding_source)
        if self.winding_source not in WINDING_SOURCES:
            raise ValueError(f"winding_source must be one of {list(WINDING_SOURCES)}, got {winding_source!r}")
        self.has_wf_channels = all(c in self.fine.channels for c in WF_CHANNELS)
        if self.winding_source != "coarse" and not self.has_wf_channels:
            print(f"[tsm] WARNING winding_source={self.winding_source!r} but {self.fine.path} has no "
                  f"{WF_CHANNELS} channels: falling back to the upsampled coarse targets", flush=True)
            self.winding_source = "coarse"
        self.fiber_mode = str(fiber_mode)
        if self.fiber_mode not in FIBER_MODES:
            raise ValueError(f"fiber_mode must be one of {list(FIBER_MODES)}, got {fiber_mode!r}")
        self.fiber_band_weight = float(fiber_band_weight)
        # the umbilicus is needed by the geometry input channels, by the core mask and -- for
        # the local tangent "vertical" is defined against -- by the direction fibre targets,
        # which are derived even when no axis channel is fed to the net
        if (self.input_radial or self.input_axis or self.core_radius_vox > 0
                or (self.fiber and self.fiber_mode == "direction")):
            self.axis = load_axis_spec(self._axis_spec)
        # the human hz/vt bands are optional: they override the teacher direction where they
        # exist and are carried through for the teacher-independent metric
        self.has_band_channel = BAND_CHANNEL in self.fine.channels
        self.fiber_band = self.fiber and self.has_band_channel
        # optional per-voxel surface loss weight (human corrections); faces / body modes only,
        # since the channel weights the two-face labels the corrections edit (the body SDF is
        # derived from exactly those two labels)
        self.has_faces_weight = FACES_WEIGHT_CHANNEL in self.fine.channels
        self.surface_weight = self.surface_mode in ("faces", "body", "sides") and self.has_faces_weight
        self.target_keys = target_keys(self.surface_mode, self.fiber, self.fiber_mode,
                                       self.fiber_band, self.surface_weight, self.lsd)
        self.valid_channel = "faces_valid" if self.surface_mode in ("faces", "body", "sides") else "sdf_valid"
        if self.valid_channel not in self.fine.channels:
            raise ValueError(f"fine store {self.fine.path} has no {self.valid_channel!r} channel "
                             f"(build it with extra.labels.faces.enabled); channels={self.fine.channels}")
        self.origins = origins if origins is not None else build_origins(self.fine, self.valid_channel, self.patch, stride, min_frac)
        if len(self.origins) == 0:
            raise ValueError("no valid crop origins in the fine store")

    def __len__(self) -> int:
        return self.length

    @property
    def reader(self) -> VolumeReader:
        if self._ct is None or self._ct_pid != os.getpid():
            spec = self._ct_spec
            self._ct = spec() if callable(spec) else spec
            self._ct_pid = os.getpid()
        return self._ct

    def _empty_targets(self) -> dict[str, np.ndarray]:
        P = self.patch
        return {k: np.zeros((c, P, P, P), np.float32) for k, c in self.target_keys.items()}

    def coarse_box(self, gz: int, gy: int, gx: int, margin: int = 1) -> dict[str, np.ndarray]:
        """Decoded coarse fields covering the fine crop at global (gz, gy, gx), plus ``margin``.

    The box is floored to the coarse grid; the discarded remainder ``g % factor`` is what the
    caller must pass to :func:`upsample_coarse_targets` as ``offset`` (a ``margin`` of 1 coarse
    voxel already supplies the extra fine voxels an unaligned origin needs)."""
        st = self.coarse
        assert st is not None
        f = self.factor
        c0 = [g // f - st.origin_zyx[i] - margin for i, g in enumerate((gz, gy, gx))]
        size = self.patch // f + 2 * margin
        r = lambda ch: st.read(ch, c0[0], c0[1], c0[2], size)  # noqa: E731
        w = np.stack([
            decode_signed(r("phase_sin")), decode_signed(r("phase_cos")), decode_density(r("density")),
            decode_signed(r("nx")), decode_signed(r("ny")), decode_signed(r("nz")),
        ])
        return {"winding": w, "conf": decode_prob(r("conf"))[None], "valid": r("valid").astype(np.float32)[None]}

    def geometry_frame(self, origin_global_zyx: Sequence[int], shape: Sequence[int]) -> dict[str, np.ndarray]:
        """``{"radial", "axis_dir"}`` (3, Z, Y, X) for a crop at an absolute level-0 origin.

        ``axis_tangent=True`` (default): the orthonormal pair of :func:`labels.geometry_frame`
        -- the local umbilicus tangent and the radial made perpendicular to it.  False: the
        legacy fields, a constant ``(1, 0, 0)`` axis and the purely in-plane radial."""
        from tsm.labels import geometry_frame, radial_field

        if self.axis_tangent:
            return geometry_frame(self.axis, origin_global_zyx, shape, scale=1)
        a = np.zeros((3,) + tuple(int(v) for v in shape), np.float32)
        a[0] = 1.0
        return {"radial": radial_field(self.axis, origin_global_zyx, shape, scale=1), "axis_dir": a}

    #: how often the CT gate prints its ``surface/ct_gated_frac`` summary (crops, per worker)
    CT_GATE_LOG_EVERY = 256

    def _ct_gate(self, ct: np.ndarray, sv: np.ndarray, t: dict[str, np.ndarray]) -> np.ndarray:
        """``faces_valid`` with the CT-dark part of the label body turned into 2 = ignore.

        ``extra.train.body_ct_gate`` is a threshold in RAW uint8 CT units (e.g. 60).  The crop's
        CT is gaussian-smoothed with sigma 2 voxels (on the *un-augmented* crop, one
        ``scipy.ndimage.gaussian_filter`` per crop, ~10 ms at patch 128) and every voxel the
        label calls papyrus body -- ``surface_body`` in "sides" mode, ``surface_sdf > 0`` in
        "body" mode -- whose smoothed CT is below the threshold is marked ``2`` (ignore), so it
        supervises neither ``surface_sdf`` nor ``surface_body`` (both loss masks are built from
        ``surface_valid``).  Air the face labels swallowed (a closed crack, a torn wrap) stops
        teaching the net that air is papyrus.  The label store is never modified.

        The gated fraction (of supervised voxels) is printed for the first crop of each worker
        and every :data:`CT_GATE_LOG_EVERY` crops after it, as ``surface/ct_gated_frac``."""
        from scipy.ndimage import gaussian_filter

        t0 = time.perf_counter()
        sm = gaussian_filter(ct.astype(np.float32), 2.0)
        body = (t["surface_body"][0] > 0) if self.surface_mode == "sides" else (t["surface_sdf"][0] > 0)
        dark = body & (sm < float(self.body_ct_gate)) & (sv > 0)
        out = np.where(dark, 2.0, sv.astype(np.float32))
        self._ct_gate_crops += 1
        if self._ct_gate_crops == 1 or self._ct_gate_crops % self.CT_GATE_LOG_EVERY == 0:
            sup = float((sv > 0).sum())
            print(f"[tsm] surface/ct_gated_frac {(float(dark.sum()) / sup if sup else 0.0):.4f} "
                  f"(body_ct_gate={self.body_ct_gate:g}, crop {self._ct_gate_crops}, "
                  f"{1e3 * (time.perf_counter() - t0):.1f} ms)", flush=True)
        return out

    def load(self, origin_local: Sequence[int]) -> dict[str, Any]:
        """Un-augmented sample at a local fine-store origin: ct (1,P,P,P) in [0,1] + targets."""
        store = self.fine
        P = self.patch
        faces_sdf: tuple[np.ndarray, np.ndarray] | None = None  # (sdf_in, sdf_out), "sides" only
        z0, y0, x0 = (int(v) for v in origin_local)
        gz, gy, gx = (store.origin_zyx[i] + v for i, v in enumerate((z0, y0, x0)))
        ct = self.reader.read(gz, gz + P, gy, gy + P, gx, gx + P)
        t = self._empty_targets()
        if self.surface_mode == "faces":
            sv = store.read("faces_valid", z0, y0, x0, P)
            for i, ch in enumerate(("sdf_in", "sdf_out")):
                t["surface_sdf"][i] = np.where(sv > 0, decode_sdf(store.read(ch, z0, y0, x0, P)), 0.0)
        elif self.surface_mode == "body":
            # orientation-free: one scalar SDF derived from the same two face labels
            sv = store.read("faces_valid", z0, y0, x0, P)
            b = body_sdf(decode_sdf(store.read("sdf_in", z0, y0, x0, P)),
                         decode_sdf(store.read("sdf_out", z0, y0, x0, P)))
            t["surface_sdf"][0] = np.where(sv > 0, b, 0.0)
        elif self.surface_mode == "sides":
            # orientation-free, magnitude and sign split: the UNSIGNED distance to the nearest
            # face (a regression that never hedges across a sign flip) plus the body occupancy
            # as a classification.  Both come from the same two face labels, no relabelling.
            sv = store.read("faces_valid", z0, y0, x0, P)
            si = decode_sdf(store.read("sdf_in", z0, y0, x0, P))
            so = decode_sdf(store.read("sdf_out", z0, y0, x0, P))
            t["surface_sdf"][0] = np.where(sv > 0, face_dist(si, so), 0.0)
            t["surface_body"][0] = np.where(sv > 0, body_mask_np(si, so), False).astype(np.float32)
            faces_sdf = (si, so)
        else:
            sv = store.read("sdf_valid", z0, y0, x0, P)
            t["surface_sdf"][0] = np.where(sv > 0, decode_sdf(store.read("sdf", z0, y0, x0, P)), 0.0)
        if self.body_ct_gate is not None and self.surface_mode in ("body", "sides"):
            sv = self._ct_gate(ct, sv, t)
        t["surface_valid"][0] = sv
        if self.surface_weight:
            # 0 is "never written" (a brick the appender did not reach), not "weight 0":
            # a missing weight must never silence supervision, so it reads as 1
            fw = store.read(FACES_WEIGHT_CHANNEL, z0, y0, x0, P).astype(np.float32)
            t[SURFACE_WEIGHT_KEY][0] = np.where(fw > 0, fw, 1.0)
        t["ink_prob"][0] = decode_prob(store.read("ink", z0, y0, x0, P))
        t["ink_valid"][0] = store.read("ink_valid", z0, y0, x0, P)
        fiber_raw: dict[str, np.ndarray] | None = None
        if self.fiber and self.has_fiber_channels:
            fiber_raw = {ch: decode_prob(store.read(ch, z0, y0, x0, P))[None] for ch in ("fiber_vt", "fiber_hz")}
            fiber_raw["fiber_valid"] = store.read("fiber_valid", z0, y0, x0, P).astype(np.float32)[None]
            t["fiber_prob"][0] = fiber_raw["fiber_vt"][0]
            t["fiber_prob"][1] = fiber_raw["fiber_hz"][0]
            t["fiber_valid"][0] = fiber_raw["fiber_valid"][0]
        if self.fiber_band:
            t["fiber_band"][0] = store.read(BAND_CHANNEL, z0, y0, x0, P)
        if self.coarse is not None:
            rem = tuple(int(g) % self.factor for g in (gz, gy, gx))
            t.update(upsample_coarse_targets(self.coarse_box(gz, gy, gx), self.factor, (P, P, P), offset=rem))
        if self.winding_source != "coarse":
            t.update(merge_winding_targets(t, fine_winding_targets(store, z0, y0, x0, P),
                                           self.winding_source))
        # the geometry frame (outward radial + local umbilicus tangent) of this crop: the
        # input channels, and the axis the direction fibre targets are built against
        frame: dict[str, np.ndarray] | None = None
        if self.axis is not None and (self.input_radial or self.input_axis
                                      or (self.fiber and self.fiber_mode == "direction")):
            frame = self.geometry_frame((gz, gy, gx), (P, P, P))
        if self.fiber and self.fiber_mode == "direction":
            # derived on the fly from the two class channels + the geometry, so no label
            # rebuild is needed (tsm.fiber); the sheet normal is grad(sdf_in) where that is a
            # unit vector and the winding normal target elsewhere
            if fiber_raw is None:
                for k in ("fiber_dir", "fiber_str", "fiber_weight", "fiber_valid"):
                    t[k][:] = 0.0
            else:
                wn = t["winding"][3:6][::-1].copy()  # (nx, ny, nz) -> (nz, ny, nx)
                wn *= (t["winding_valid"] == 1)
                d = derive_direction_targets(
                    fiber_raw["fiber_vt"], fiber_raw["fiber_hz"], t["surface_sdf"][0:1],
                    fiber_raw["fiber_valid"], wn,
                    band=t["fiber_band"] if self.fiber_band else None,
                    band_weight=self.fiber_band_weight,
                    axis=None if frame is None else frame["axis_dir"],
                    # the gradient of a *body* SDF is degenerate on the medial ridge (it flips
                    # sign there), so in that mode the winding normal is the primary sheet
                    # normal and grad(sdf) only fills its gaps
                    prefer_fallback=self.surface_mode in ("body", "sides"))
                for k, v in d.items():
                    t[k] = v.astype(np.float32)
        if self.core_radius_vox > 0:
            # the in/out convention degenerates at the umbilicus (recto_is_in ~ 0.47 near the
            # core), so drop the core from the surface *and* winding supervision: faces_valid
            # (surface_valid) 2 = ignore, winding_valid 0 = no data.  Same rule as the
            # build-time extra.labels.core_radius_vox, applied to whatever the store holds.
            from tsm.labels import core_mask

            cm = core_mask(self.axis, (gz, gy, gx), (P, P, P), self.core_radius_vox, scale=1)
            t["surface_valid"][0] = np.where(cm, 2.0, t["surface_valid"][0])
            t["winding_valid"][0] = np.where(cm, 0.0, t["winding_valid"][0])
        if self.lsd and faces_sdf is not None:
            # Local Shape Descriptors from the same two face labels (tsm.data.lsd_targets).
            # The outward normal is the winding normal target where that is a unit vector and
            # grad(sdf_in) elsewhere -- the same rule the direction fibre targets use.
            wn = t["winding"][3:6][::-1].copy()  # (nx, ny, nz) -> (nz, ny, nx)
            wn *= (t["winding_valid"] == 1)
            nrm, _ = sheet_normal(torch.from_numpy(faces_sdf[0])[None], torch.from_numpy(wn),
                                  prefer_fallback=True)
            lsd, lsd_valid = lsd_targets(faces_sdf[0], faces_sdf[1], t["surface_valid"][0],
                                         nrm.numpy(), sigma=self.lsd_sigma)
            t[LSD_KEY], t[LSD_VALID_KEY] = lsd, lsd_valid
        if self.input_radial:
            t["radial"] = frame["radial"]
        if self.input_axis:
            t["axis_dir"] = frame["axis_dir"]
        sample: dict[str, Any] = {
            "ct": (ct.astype(np.float32) / 255.0)[None],
            "voxel_um": float(store.voxel_um),
            "scale_value": scale_channel_value(store.voxel_um),
            "origin_zyx": (gz, gy, gx),
        }
        sample.update(t)
        return sample

    @staticmethod
    def to_tensors(sample: dict[str, Any]) -> dict[str, Any]:
        ct = torch.from_numpy(np.ascontiguousarray(sample["ct"], dtype=np.float32))
        parts = [ct, torch.full_like(ct, float(sample["scale_value"]))]
        for k in ("radial", "axis_dir"):
            if k in sample:
                parts.append(torch.from_numpy(np.ascontiguousarray(sample[k], dtype=np.float32)))
        x = torch.cat(parts, dim=0)
        out: dict[str, Any] = {
            # (2, 5 or 8, P, P, P): [ct01 (NOT yet z-scored), scale, (r_z, r_y, r_x), (a_z, a_y, a_x)]
            "input": x,
            "voxel_um": torch.tensor(float(sample["voxel_um"])),
            "origin_zyx": torch.tensor(sample["origin_zyx"], dtype=torch.int64),
        }
        # NOT just TARGET_KEYS: the optional fiber targets (present iff the dataset was built
        # with fiber=True) must survive the numpy -> tensor conversion too, or the fiber head
        # silently gets no supervision.
        for k in (*TARGET_KEYS, *FIBER_TARGET_KEYS, *EXTRA_TARGET_KEYS):
            if k in sample:
                out[k] = torch.from_numpy(np.ascontiguousarray(sample[k], dtype=np.float32))
        return out

    # A CT read that fails every retry (S3 outage) must not kill the run: try a few
    # other random origins before giving up.
    MAX_READ_SUBSTITUTIONS = 5

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng([self.seed, int(index)])
        for attempt in range(self.MAX_READ_SUBSTITUTIONS + 1):
            origin = self.origins[int(rng.integers(0, len(self.origins)))]
            try:
                sample = self.load(origin)
            except VolumeReadError as exc:
                self.read_failures += 1
                if attempt >= self.MAX_READ_SUBSTITUTIONS:
                    print(f"[tsm] sample {index}: CT read failed at {self.MAX_READ_SUBSTITUTIONS + 1} "
                          f"origins, giving up: {exc}", flush=True)
                    raise
                print(f"[tsm] sample {index}: CT read failed at origin {tuple(int(v) for v in origin)} "
                      f"({exc}); retrying at another random origin "
                      f"({attempt + 1}/{self.MAX_READ_SUBSTITUTIONS})", flush=True)
                self._ct = None  # drop the (possibly poisoned) store handle
                continue
            if self.do_augment:
                sample = augment(sample, rng)
            return self.to_tensors(sample)
        raise AssertionError("unreachable")


def open_reader_factory(url: str, level: int, voxel_um: float, budget: Budget | None = None) -> Callable[[], VolumeReader]:
    def make() -> VolumeReader:
        return VolumeReader(url, level, voxel_um, budget)

    return make


# --------------------------------------------------------------------------- #
# several label stores (regions / scrolls) at once
# --------------------------------------------------------------------------- #
class MultiStoreDataset(Dataset):
    """Several :class:`CropDataset` (one per label store / region / scroll) as one dataset.

    ``__getitem__`` picks a store with probability proportional to its weight (default: the
    store's number of training origins, i.e. uniform over crops) and then delegates to that
    store's ``CropDataset``, which draws the crop.  The items are exactly the ones the single
    store would produce, so nothing downstream (augmentation, losses, the model) changes.

    Every store must expose the *same* channel set (identical fine channel lists, and either
    all or none with a coarse store, again with identical channels) and the same target
    layout -- a mixed batch has to be a single supervision contract.  The one exception is the
    optional fibre block (:data:`FIBER_CHANNELS`): a store built without a ``fiber.zarr``
    teacher may lack it and then trains with ``fiber_valid = 0`` (pass ``fiber=True`` to every
    ``CropDataset``, i.e. ``extra.train.heads.fiber``, so the target layout still matches).

    Origins are carried as ``(N, 4)`` ``[store_index, z, y, x]`` (local to that store's fine
    store) in :attr:`origins` / :attr:`holdout_origins`; :meth:`load` and :meth:`to_tensors`
    accept them, so ``train.evaluate`` works unchanged.  The holdout origins are shuffled with
    a seeded permutation so that ``max_crops`` truncation stays spread over the stores.
    """

    def __init__(
        self,
        datasets: Sequence[CropDataset],
        names: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        seed: int = 0,
        length: int = 100_000,
        holdout: Sequence[np.ndarray] | None = None,
        train_origins: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("MultiStoreDataset needs at least one store")
        self.names = [str(n) for n in names] if names is not None else [d.fine.name for d in self.datasets]
        if len(self.names) != len(self.datasets):
            raise ValueError("names must have one entry per store")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"store names must be unique, got {self.names}")
        self.seed = int(seed)
        self.length = int(length)
        self._check_compatible()
        d0 = self.datasets[0]
        self.patch = d0.patch
        self.factor = d0.factor
        self.surface_mode = d0.surface_mode
        self.winding_source = d0.winding_source
        self.fiber = d0.fiber
        self.fiber_mode = d0.fiber_mode
        self.fiber_band = d0.fiber_band
        self.target_keys = d0.target_keys
        self.input_radial = d0.input_radial
        self.input_axis = d0.input_axis
        self.axis_tangent = d0.axis_tangent
        self.lsd = d0.lsd
        self.lsd_sigma = d0.lsd_sigma
        tr = ([np.asarray(o, np.int32).reshape(-1, 3) for o in train_origins] if train_origins is not None
              else [d.origins for d in self.datasets])
        if len(tr) != len(self.datasets):
            raise ValueError("train_origins must have one entry per store")
        w = [float(len(o)) for o in tr] if weights is None else [float(v) for v in weights]
        if len(w) != len(self.datasets):
            raise ValueError("weights must have one entry per store")
        if any(v < 0 for v in w):
            raise ValueError(f"store weights must be non-negative, got {w}")
        if sum(w) <= 0:
            raise ValueError("store weights sum to zero")
        self.weights = w
        self.p = np.asarray(w, np.float64) / float(sum(w))
        self._cum = np.cumsum(self.p)
        self.origins = self._tag(tr)
        hold = list(holdout) if holdout is not None else [getattr(d, "holdout_origins", np.zeros((0, 3), np.int32))
                                                          for d in self.datasets]
        h = self._tag(hold)
        self.holdout_origins = h[np.random.default_rng(self.seed).permutation(len(h))] if len(h) else h

    @staticmethod
    def _required_channels(channels: Sequence[str]) -> list[str]:
        """Channel list minus the blocks a store may legitimately lack.

        Only the fibre block is optional: :class:`CropDataset` already turns a store without it
        into ``fiber_valid = 0`` targets (fully masked), so a slab whose teachers dir had no
        ``fiber.zarr`` can train next to one that had it.  Everything else must match exactly --
        a mixed batch is one supervision contract.
        """
        return [c for c in channels if c not in FIBER_CHANNELS]

    def _check_compatible(self) -> None:
        d0 = self.datasets[0]
        for name, d in zip(self.names[1:], self.datasets[1:]):
            if self._required_channels(d.fine.channels) != self._required_channels(d0.fine.channels):
                raise ValueError(
                    f"store {name!r} has fine channels {d.fine.channels} but store {self.names[0]!r} has "
                    f"{d0.fine.channels}; every store must carry the same channel set")
            if d.fine.channels != d0.fine.channels:
                print(f"[tsm] WARNING store {name!r} channels {d.fine.channels} differ from "
                      f"{self.names[0]!r} ({d0.fine.channels}) only in {list(FIBER_CHANNELS)}; the "
                      f"store without them trains with fiber_valid = 0", flush=True)
            if (d.coarse is None) != (d0.coarse is None):
                raise ValueError(f"store {name!r} {'has no' if d.coarse is None else 'has a'} coarse store but "
                                 f"{self.names[0]!r} does{'' if d0.coarse is None else ' not'}")
            if d.coarse is not None and d0.coarse is not None and d.coarse.channels != d0.coarse.channels:
                raise ValueError(f"store {name!r} has coarse channels {d.coarse.channels} != {d0.coarse.channels}")
            for attr in ("patch", "surface_mode", "winding_source", "fiber", "fiber_mode", "input_radial", "input_axis",
                         "axis_tangent", "body_ct_gate", "lsd", "lsd_sigma"):
                if getattr(d, attr) != getattr(d0, attr):
                    hint = ("; the stores carry different fiber channels -- set extra.train.heads.fiber "
                            "to force the fibre targets on every store (one without the channels then "
                            "trains with fiber_valid = 0)" if attr == "fiber" else "")
                    raise ValueError(f"store {name!r} has {attr}={getattr(d, attr)!r} != {getattr(d0, attr)!r}{hint}")
            if d.target_keys != d0.target_keys:
                raise ValueError(f"store {name!r} has target keys {d.target_keys} != {d0.target_keys}")

    @staticmethod
    def _tag(per_store) -> np.ndarray:
        out = []
        for k, o in enumerate(per_store):
            o = np.asarray(o, np.int32).reshape(-1, 3)
            out.append(np.concatenate([np.full((len(o), 1), k, np.int32), o], axis=1))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 4), np.int32)

    def __len__(self) -> int:
        return self.length

    def store_index(self, index: int) -> int:
        """The store this sample index draws from (weighted, deterministic in ``seed``)."""
        u = float(np.random.default_rng([self.seed, int(index), 0x5700]).random())
        return int(np.searchsorted(self._cum, u, side="right").clip(0, len(self.datasets) - 1))

    def load(self, origin: Sequence[int]) -> dict[str, Any]:
        k, z0, y0, x0 = (int(v) for v in origin)
        return self.datasets[k].load((z0, y0, x0))

    @staticmethod
    def to_tensors(sample: dict[str, Any]) -> dict[str, Any]:
        return CropDataset.to_tensors(sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.datasets[self.store_index(index)][int(index)]


# --------------------------------------------------------------------------- #
# augmentation v2: composable, config-driven, target-aware, on the device per batch
# --------------------------------------------------------------------------- #
# Every transform is truly 3D and written here in torch (no albumentations / MONAI /
# TorchIO).  Spatial transforms are composed into ONE sampling field per sample
#   x_in = L^-1 x_out + e(x_out),   L = Q F R S   (rot90 . flips . small rotation . scaling)
# in crop-centred voxel coordinates (z, y, x) and applied with a single ``grid_sample``
# (trilinear for the CT and the continuous targets, nearest for the masks); ``e`` is the
# elastic displacement (coarse random grid, upsampled).  Per-channel rules (see
# docs/label_store.md "Augmentation v2"):
#   ct       trilinear, padding ``oob_fill`` (reflection by default)
#   sdf      mask-aware trilinear, x det(S)^(1/3) (distances scale), clamped to +-CLIP;
#            voxels sampled outside the crop -> valid 0; saturated labels shrunk below
#            the clip (zoom-out) -> valid 2
#   valid masks (surface/ink/winding)  nearest, 0 outside the crop
#   sin/cos  mask-aware trilinear, then renormalised (the phase is a scalar invariant)
#   density  mask-aware trilinear, x |L^-T n| (= 1/s for isotropic scaling s; the wrap
#            spacing along the sheet normal n changes), fallback det(S)^(-1/3) where n is
#            unknown
#   normal   mask-aware trilinear as a vector, then normalise(L^-T n): the exact
#            rotation part of the sampling matrix (flips negate the flipped component,
#            rot90 / arbitrary rotations rotate it, anisotropic scaling tilts it like a
#            covector)
#   conf / ink  mask-aware trilinear
# The elastic field is small; its local Jacobian is ignored for the sdf / density /
# normal rules.  Intensity transforms touch the CT only (all 3D: anisotropic blur is a
# separable 3D gaussian, cutout removes boxes, low-res resamples the volume, artefacts
# are plane waves along a random 3D direction / shells about a random 3D line).  The
# scan-domain family (window, class_contrast, aniso_blur, ring, stripe, spectral_noise; preset
# "strong_scan", all p=0 elsewhere) models what differs BETWEEN scans -- see the block above
# their dataclasses and docs/label_store.md "Scan-domain intensity family".

from dataclasses import asdict, dataclass, fields, is_dataclass


@dataclass
class FlipCfg:
    p: float = 0.5  # per axis
    axes: tuple[bool, bool, bool] = (True, True, True)  # z, y, x allowed


@dataclass
class Rot90Cfg:
    p: float = 0.75  # probability of a non-identity rot90 (k in 1..3 in-plane, or one of the 23 others)
    all_axes: bool = True  # True (default): any of the 24 cube rotations; False: about z (y-x plane) only
    # Default changed to True on 2026-09-02: two of three teachers were trained with single-plane
    # rot90 only and no 3D mirroring, which is the measured source of their orientation bias.


@dataclass
class RotateCfg:
    p: float = 0.5
    max_deg: float = 30.0
    in_plane: bool = False  # True: about the z axis only; False: about a random 3D axis
    full_p: float = 0.2  # with this probability sample a UNIFORM random rotation on SO(3) instead
                         # (random unit quaternion), i.e. any angle up to 180 deg about any axis


@dataclass
class ScaleCfg:
    p: float = 0.3
    lo: float = 0.85
    hi: float = 1.2
    anisotropic: bool = False  # True: an independent factor per axis


@dataclass
class ElasticCfg:
    p: float = 0.2
    grid: int = 4  # control points per axis
    magnitude: float = 2.0  # std of the control-point displacement, voxels


@dataclass
class GammaCfg:
    p: float = 0.3
    lo: float = 0.7
    hi: float = 1.4


@dataclass
class ContrastCfg:
    p: float = 0.3
    brightness: float = 0.1  # +- shift of the [0, 1] CT
    lo: float = 0.8
    hi: float = 1.2


@dataclass
class NoiseCfg:  # additive gaussian
    p: float = 0.3
    sigma_max: float = 0.03


@dataclass
class MultNoiseCfg:  # multiplicative gaussian: x * (1 + N(0, sigma))
    p: float = 0.2
    sigma_max: float = 0.1


@dataclass
class BlurCfg:
    p: float = 0.2
    sigma_lo: float = 0.5
    sigma_hi: float = 1.5
    axis_only_p: float = 0.3  # with this probability blur along ONE random axis only (anisotropic)


@dataclass
class SharpenCfg:  # unsharp mask: x + amount * (x - blur(x, sigma))
    p: float = 0.15
    amount_lo: float = 0.5
    amount_hi: float = 2.0
    sigma: float = 1.0


@dataclass
class ArtefactCfg:  # off by default
    p: float = 0.0
    streak_amp: float = 0.1  # low-frequency plane-wave profile along a random 3D direction
    ring_amp: float = 0.1  # sinusoidal shells about a random 3D line
    ring_period_lo: float = 4.0
    ring_period_hi: float = 16.0


@dataclass
class CutoutCfg:  # boxes filled with a random constant; targets untouched
    p: float = 0.2
    n_lo: int = 1
    n_hi: int = 3
    size_lo: float = 0.1  # fraction of each axis
    size_hi: float = 0.3


@dataclass
class LowResCfg:  # area-downsample by f in [lo, hi] then trilinear back (coarser scan)
    p: float = 0.2
    factor_lo: float = 1.0
    factor_hi: float = 2.0


# --- scan-domain intensity transforms (2026-09-07) ------------------------------------- #
# Off by default (p = 0, so the "strong" preset is unchanged); the "strong_scan" preset turns
# them on.  Ranges are calibrated on the Paris 4 vs Paris 3 comparison at 2.4 um
# (dev/scan_stats.py, ~/tsm-output/scan_stats/report.md): the two scans differ by a 8-bit
# export window of [-0.04, 0.22] vs [-0.03, 0.19] f32 (~12% of the window), a papyrus std of
# 20.1 vs 27.1 grey levels at nearly the same modes (air 42.5/41.5, papyrus 112.5/114.5, i.e.
# a contrast ratio 3.49 vs 2.70), an ACF 1/e length ratio of 0.80 (z), 0.88 (y), 0.99 (x), a
# PSD log-log slope of -4.26 vs -4.39 and a noise ratio (laplacian/local std) of 0.67 vs 0.60.
# The CT is carried as ct01 in [0, 1]; every grey-level field below is in 8-bit units (x255).
GREY = 255.0


@dataclass
class WindowCfg:  # random 8-bit export window: x' = clip((x - lo) / (hi - lo), 0, 1)
    p: float = 0.0
    lo_lo: float = -12.0  # grey levels; +-12 ~ the +-12% window jitter measured between scans
    lo_hi: float = 12.0
    hi_lo: float = 220.0
    hi_hi: float = 300.0


@dataclass
class ClassContrastCfg:  # rescale the spread of the papyrus mode only (air/papyrus contrast ratio)
    p: float = 0.0
    s_lo: float = 0.7  # 20.1 / 27.1 = 0.74 .. 27.1 / 20.1 = 1.35 measured between the two scans
    s_hi: float = 1.4
    air: float = 42.0  # grey-level mode guesses (Paris 4: 42.5 / 112.5, Paris 3: 41.5 / 114.5)
    papyrus: float = 112.0
    estimate: bool = True  # refine both modes with 2-means on a strided subsample of the crop
    band: float = 10.0  # +- grey levels of smooth (smoothstep) blending about the midpoint
    subsample: int = 4096  # voxels used by the 2-means estimate


@dataclass
class AnisoBlurCfg:  # per-axis gaussian: z resolution independent of the in-plane resolution
    p: float = 0.0
    z_lo: float = 0.0
    z_hi: float = 1.5  # ACF 1/e length z 13.3 vs 10.6 vox (ratio 0.80) between the two scans
    yx_lo: float = 0.0
    yx_hi: float = 1.0  # one draw shared by y and x (the in-plane ACF ratio is 0.88 / 0.99)


@dataclass
class RingCfg:  # multiplicative concentric rings about a centre far outside the crop, const in z
    p: float = 0.0
    amp_lo: float = 0.02
    amp_hi: float = 0.08
    n_lo: int = 1
    n_hi: int = 3
    width_lo: float = 2.0  # voxels (gaussian sigma of one ring)
    width_hi: float = 8.0
    centre_lo: float = 4.0  # ring-centre distance in units of max(H, W): >> crop, so the rings
    centre_hi: float = 20.0  # cross the crop as gently curved stripes


@dataclass
class StripeCfg:  # a detector line: one multiplicative y or x plane, constant along z
    p: float = 0.0
    amp_lo: float = 0.01
    amp_hi: float = 0.05
    n_lo: int = 1
    n_hi: int = 2
    width_lo: int = 1  # voxels
    width_hi: int = 3


@dataclass
class SpectralNoiseCfg:  # coloured noise: white noise filtered by |k|^(beta/2) in Fourier space
    p: float = 0.0
    sigma_lo: float = 2.0  # grey levels
    sigma_hi: float = 8.0
    beta_lo: float = -1.0  # beta < 0 = red (correlated), 0 = white, > 0 = blue
    beta_hi: float = 1.0


@dataclass
class AugmentConfig:
    flip: FlipCfg = None  # type: ignore[assignment]
    rot90: Rot90Cfg = None  # type: ignore[assignment]
    rotate: RotateCfg = None  # type: ignore[assignment]
    scale: ScaleCfg = None  # type: ignore[assignment]
    elastic: ElasticCfg = None  # type: ignore[assignment]
    gamma: GammaCfg = None  # type: ignore[assignment]
    contrast: ContrastCfg = None  # type: ignore[assignment]
    noise: NoiseCfg = None  # type: ignore[assignment]
    mult_noise: MultNoiseCfg = None  # type: ignore[assignment]
    blur: BlurCfg = None  # type: ignore[assignment]
    sharpen: SharpenCfg = None  # type: ignore[assignment]
    artefact: ArtefactCfg = None  # type: ignore[assignment]
    cutout: CutoutCfg = None  # type: ignore[assignment]
    lowres: LowResCfg = None  # type: ignore[assignment]
    window: WindowCfg = None  # type: ignore[assignment]
    class_contrast: ClassContrastCfg = None  # type: ignore[assignment]
    aniso_blur: AnisoBlurCfg = None  # type: ignore[assignment]
    ring: RingCfg = None  # type: ignore[assignment]
    stripe: StripeCfg = None  # type: ignore[assignment]
    spectral_noise: SpectralNoiseCfg = None  # type: ignore[assignment]
    oob_fill: str = "reflection"  # CT padding for samples outside the crop: reflection | border | zeros
    clip: float = CLIP

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            cls = _CFG_CLASSES.get(f.name)
            if cls is None:
                continue
            if v is None:
                setattr(self, f.name, cls())
            elif isinstance(v, dict):
                setattr(self, f.name, _from_dict(cls, v, f.name))
            elif not isinstance(v, cls):
                raise ValueError(f"augment.{f.name} must be an object")
        if self.oob_fill not in ("reflection", "border", "zeros"):
            raise ValueError(f"augment.oob_fill must be reflection | border | zeros, got {self.oob_fill!r}")
        for name in TRANSFORM_NAMES:
            p = float(getattr(self, name).p)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"augment.{name}.p must be in [0, 1], got {p}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AugmentConfig":
        return _from_dict(cls, d, "augment")

    @classmethod
    def preset(cls, name: str) -> "AugmentConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown augment preset {name!r}; known: {sorted(PRESETS)}")
        return cls.from_dict(PRESETS[name])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_identity(self) -> bool:
        return all(float(getattr(self, n).p) == 0.0 for n in TRANSFORM_NAMES)


_CFG_CLASSES: dict[str, type] = {
    "flip": FlipCfg, "rot90": Rot90Cfg, "rotate": RotateCfg, "scale": ScaleCfg, "elastic": ElasticCfg,
    "gamma": GammaCfg, "contrast": ContrastCfg, "noise": NoiseCfg, "mult_noise": MultNoiseCfg, "blur": BlurCfg,
    "sharpen": SharpenCfg, "artefact": ArtefactCfg, "cutout": CutoutCfg, "lowres": LowResCfg,
    "window": WindowCfg, "class_contrast": ClassContrastCfg, "aniso_blur": AnisoBlurCfg,
    "ring": RingCfg, "stripe": StripeCfg, "spectral_noise": SpectralNoiseCfg,
}
SPATIAL_NAMES = ("flip", "rot90", "rotate", "scale", "elastic")
# the scan-domain family (off unless the preset turns it on), in pipeline order
SCAN_NAMES = ("aniso_blur", "class_contrast", "spectral_noise", "ring", "stripe", "window")
INTENSITY_NAMES = ("lowres", "blur", "aniso_blur", "sharpen", "class_contrast", "gamma", "contrast",
                   "mult_noise", "noise", "spectral_noise", "ring", "stripe", "window", "artefact", "cutout")
TRANSFORM_NAMES = SPATIAL_NAMES + INTENSITY_NAMES


def _from_dict(cls: type, d: Any, path: str):
    if not isinstance(d, dict):
        raise ValueError(f"{path} must be an object")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown {path} keys: {unknown} (known: {sorted(known)})")
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        sub = _CFG_CLASSES.get(f.name) if cls is AugmentConfig else None
        if sub is not None and isinstance(v, dict):
            v = _from_dict(sub, v, f"{path}.{f.name}")
        elif f.name == "axes":
            v = tuple(bool(x) for x in v)
            if len(v) != 3:
                raise ValueError(f"{path}.axes needs 3 booleans")
        kw[f.name] = v
    return cls(**kw)


PRESETS: dict[str, dict[str, Any]] = {
    "strong": {},  # the dataclass defaults above
    "light": {  # flips/rot90 + mild intensity: the v1 recipe, on the GPU
        "rotate": {"p": 0.0}, "scale": {"p": 0.0}, "elastic": {"p": 0.0}, "mult_noise": {"p": 0.0},
        "blur": {"p": 0.0}, "sharpen": {"p": 0.0}, "cutout": {"p": 0.0}, "lowres": {"p": 0.0},
        "gamma": {"p": 0.5}, "contrast": {"p": 0.5}, "noise": {"p": 0.5},
    },
    "none": {n: {"p": 0.0} for n in TRANSFORM_NAMES},
    # strong + the scan-domain family: models the per-scan differences measured between
    # PHercParis4 and PHercParis3 at 2.4 um (see the comments above the dataclasses)
    "strong_scan": {"window": {"p": 0.5}, "class_contrast": {"p": 0.4}, "aniso_blur": {"p": 0.3},
                    "spectral_noise": {"p": 0.3}, "ring": {"p": 0.2}, "stripe": {"p": 0.05}},
}


def as_augment_config(spec: Any) -> AugmentConfig:
    """AugmentConfig | preset name | dict (a preset name under ``preset`` plus overrides) -> AugmentConfig."""
    if isinstance(spec, AugmentConfig):
        return spec
    if isinstance(spec, str):
        return AugmentConfig.preset(spec)
    if isinstance(spec, dict):
        d = dict(spec)
        base = d.pop("preset", "strong")
        merged = _deep_merge(PRESETS[base] if base in PRESETS else AugmentConfig.preset(base).to_dict(), d)
        return AugmentConfig.from_dict(merged)
    raise ValueError(f"augment spec must be a preset name, a dict or an AugmentConfig, got {type(spec).__name__}")


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(d: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``d`` with dotted-path overrides applied (``{"rotate.p": 0}``)."""
    out = json.loads(json.dumps(d))  # deep copy of plain JSON
    for path, v in overrides.items():
        keys = str(path).split(".")
        cur = out
        for k in keys[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = v
    return out


# --- spatial parameters ---------------------------------------------------------------- #
@dataclass
class SpatialParams:
    """One sample's spatial transform: x_out = Q F R S x_in (+ elastic), all in (z, y, x)."""
    flips: tuple[bool, bool, bool] = (False, False, False)
    rot90: torch.Tensor | None = None  # (3, 3) signed permutation, det +1
    rotation: torch.Tensor | None = None  # (3, 3) small rotation
    scale: torch.Tensor | None = None  # (3,) per-axis factors
    elastic: torch.Tensor | None = None  # (3, g, g, g) control displacements (voxels, zyx)

    @property
    def fired(self) -> dict[str, bool]:
        return {
            "flip": any(self.flips), "rot90": self.rot90 is not None, "rotate": self.rotation is not None,
            "scale": self.scale is not None, "elastic": self.elastic is not None,
        }

    @property
    def identity(self) -> bool:
        return not any(self.fired.values())

    def matrix(self) -> torch.Tensor:
        """L (3, 3) in (z, y, x): x_out = L x_in."""
        L = torch.eye(3, dtype=torch.float64)
        if self.scale is not None:
            L = torch.diag(self.scale.double()) @ L
        if self.rotation is not None:
            L = self.rotation.double() @ L
        L = torch.diag(torch.tensor([-1.0 if f else 1.0 for f in self.flips], dtype=torch.float64)) @ L
        if self.rot90 is not None:
            L = self.rot90.double() @ L
        return L

    def scale_iso(self) -> float:
        """det(S)^(1/3): the isotropic distance factor (1 without scaling)."""
        return float(self.scale.double().prod().abs() ** (1.0 / 3.0)) if self.scale is not None else 1.0


def rot90_matrix_inplane(k: int) -> torch.Tensor:
    """(z, y, x) matrix of ``np.rot90(a, k, axes=(y, x))`` (k=1: y' = -x, x' = y)."""
    m = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return torch.linalg.matrix_power(m, int(k) % 4)


def rotation_matrix(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rodrigues rotation about the unit ``axis`` (3,) in (z, y, x) coordinates."""
    k = axis.float() / axis.float().norm().clamp_min(1e-12)
    K = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.eye(3) * c + s * K + (1.0 - c) * torch.outer(k, k)



def uniform_rotation_matrix(gen: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random rotation on SO(3) from a random unit quaternion (Shoemake / Marsaglia)."""
    q = torch.randn(4, generator=gen)
    q = q / q.norm().clamp_min(1e-12)
    w, x, y, z = (float(v) for v in q)
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float32)

def _cube_rotation(gen: torch.Generator) -> torch.Tensor:
    """Random proper signed permutation (one of the 24 cube rotations), never the identity."""
    while True:
        perm = torch.randperm(3, generator=gen)
        signs = torch.randint(0, 2, (3,), generator=gen).float() * 2 - 1
        m = torch.zeros(3, 3)
        for i in range(3):
            m[i, perm[i]] = signs[i]
        if torch.det(m) < 0:
            m[0] = -m[0]
        if not torch.equal(m, torch.eye(3)):
            return m


class Augment:
    """Config-driven GPU augmentation of a collated batch (``CropDataset.to_tensors`` keys).

    ``aug(batch) -> (batch, fired)`` where ``fired`` counts per transform how many samples
    it touched.  Parameters are drawn from a CPU ``torch.Generator`` (``seed``) so a run
    is reproducible; the resampling runs on the batch's device.  Only ``input`` (ct01 +
    scale channel) and the target keys are transformed; other keys pass through.
    """

    def __init__(self, cfg: Any = "strong", seed: int = 0, fiber_swap: bool = True,
                 input_vec: Sequence[tuple[int, int]] | None = None) -> None:
        self.cfg = as_augment_config(cfg)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.clip = float(self.cfg.clip)
        # class mode only: swap the [vt, hz] target channels when the transform moves the
        # volume z axis more than 45 degrees (tsm.fiber.axis_swap_needed).  False reproduces
        # the pre-2026-09-07 behaviour, where the axis-relative classes were augmented as
        # invariant scalars (extra.train.fiber_swap_fix).
        self.fiber_swap = bool(fiber_swap)
        # input channel slices holding (z, y, x) vector fields (radial, scroll axis);
        # None = the historical guess "channels 2..5 when there are at least 5"
        self.input_vec = None if input_vec is None else [(int(a), int(b)) for a, b in input_vec]

    # ---- parameter sampling --------------------------------------------------------- #
    def _u(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=self.gen))

    def _fires(self, p: float) -> bool:
        return p > 0.0 and float(torch.rand((), generator=self.gen)) < p

    def sample_spatial(self) -> SpatialParams:
        c = self.cfg
        flips = tuple(bool(c.flip.axes[a]) and self._fires(c.flip.p) for a in range(3))
        r90 = None
        if self._fires(c.rot90.p):
            r90 = _cube_rotation(self.gen) if c.rot90.all_axes else rot90_matrix_inplane(int(torch.randint(1, 4, (), generator=self.gen)))
        rot = None
        if self._fires(c.rotate.p) and c.rotate.max_deg > 0:
            if not c.rotate.in_plane and self._fires(c.rotate.full_p):
                rot = uniform_rotation_matrix(self.gen)  # any angle about any axis
            else:
                axis = torch.tensor([1.0, 0.0, 0.0]) if c.rotate.in_plane else torch.randn(3, generator=self.gen)
                rot = rotation_matrix(axis, math.radians(self._u(-c.rotate.max_deg, c.rotate.max_deg)))
        sc = None
        if self._fires(c.scale.p):
            sc = torch.tensor([self._u(c.scale.lo, c.scale.hi) for _ in range(3)]) if c.scale.anisotropic else torch.full((3,), self._u(c.scale.lo, c.scale.hi))
        el = None
        if self._fires(c.elastic.p) and c.elastic.magnitude > 0:
            g = max(2, int(c.elastic.grid))
            el = torch.randn(3, g, g, g, generator=self.gen) * float(c.elastic.magnitude)
        return SpatialParams(flips, r90, rot, sc, el)  # type: ignore[arg-type]

    # ---- spatial ------------------------------------------------------------------ #
    @staticmethod
    def _grid(params: Sequence[SpatialParams], shape: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(grid (B, D, H, W, 3) for grid_sample, oob (B, 1, D, H, W) bool, L (B, 3, 3) float32)."""
        D, H, W = (int(s) for s in shape)
        axes = [torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2.0 for n in (D, H, W)]
        c = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (D, H, W, 3) zyx, centred
        Ls = torch.stack([p.matrix() for p in params]).float().to(device)  # (B, 3, 3)
        Linv = torch.linalg.inv(Ls)
        x_in = torch.einsum("bij,dhwj->bdhwi", Linv, c)
        for b, p in enumerate(params):
            if p.elastic is not None:
                e = F.interpolate(p.elastic[None].to(device), size=(D, H, W), mode="trilinear", align_corners=True)[0]
                x_in[b] = x_in[b] + e.permute(1, 2, 3, 0)
        half = torch.tensor([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0], device=device)
        oob = (x_in.abs() > half + 1e-4).any(-1).unsqueeze(1)
        n = torch.tensor([D, H, W], device=device, dtype=torch.float32)
        grid = (2.0 * x_in / n)[..., [2, 1, 0]]  # grid_sample wants (x, y, z)
        return grid, oob, Ls

    def apply_spatial(self, batch: dict[str, Any], params: Sequence[SpatialParams]) -> dict[str, Any]:
        """Resample ``input`` and the targets with the given per-sample transforms (deterministic)."""
        if all(p.identity for p in params):
            return dict(batch)
        inp = batch["input"]
        B = inp.shape[0]
        if len(params) != B:
            raise ValueError(f"{len(params)} spatial params for a batch of {B}")
        dev = inp.device
        grid, oob, L = self._grid(params, inp.shape[2:], dev)
        # two transport rules, identical for every *isotropic* transform (rotations, flips,
        # isotropic scale: inv(L)^T = L there) and different only under ScaleCfg.anisotropic:
        #   tangent directions (the radial / axis input channels, the fibre direction) push
        #     forward with L,
        #   covectors (the winding normal, which is grad of a scalar) with inv(L)^T.
        LinvT = torch.linalg.inv(L).transpose(1, 2)  # (B, 3, 3) zyx: the covector rule
        out = dict(batch)
        ct = inp[:, 0:1].float()
        ct_t = F.grid_sample(ct, grid, mode="bilinear", padding_mode=self.cfg.oob_fill, align_corners=False)
        parts = [ct_t, inp[:, 1:2]]
        # (z, y, x) vector input channels (radial, scroll axis): resample, then push forward
        # with L and renormalise
        vecs = self.input_vec if self.input_vec is not None else ([(2, 5)] if inp.shape[1] >= 5 else [])
        moved: list[torch.Tensor] = []
        for lo_c, hi_c in vecs:
            r = F.grid_sample(inp[:, lo_c:hi_c].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
            r = torch.einsum("bij,bjdhw->bidhw", L, r)
            rn = r.norm(dim=1, keepdim=True)
            moved.append(torch.where(rn > 1e-6, r / rn.clamp_min(1e-6), torch.zeros_like(r)))
        if len(moved) == 2:
            # radial first, axis second (student.input_vec_slices): trilinear resampling and an
            # elastic warp both break r . a == 0, so the frame is re-orthonormalised here
            moved[0], moved[1] = _orthonormalise_frame(moved[0], moved[1])
        cut = 2
        for (lo_c, hi_c), v in zip(vecs, moved):
            if lo_c > cut:
                parts.append(inp[:, cut:lo_c])
            parts.append(v)
            cut = hi_c
        out["input"] = torch.cat(parts + [inp[:, cut:]], dim=1)
        if "surface_valid" not in batch:
            return out
        sv, iv, wv = batch["surface_valid"].float(), batch["ink_valid"].float(), batch["winding_valid"].float()
        m1, mi, mw = (sv == 1).float(), (iv == 1).float(), (wv == 1).float()
        w = batch["winding"].float()
        sdf_in = batch["surface_sdf"].float()
        ns = int(sdf_in.shape[1])  # 1 (medial / body) or 2 (faces: [sdf_in, sdf_out]); same rules per channel
        # The body SDF (surface_mode="body") is min(sdf_in, -sdf_out) and rides the ns == 1
        # scalar path unchanged: isotropic scale multiplies it by s_iso, the saturation rule
        # turns shrunk +-clip labels into valid 2.  min() and resampling commute only up to
        # sub-voxel effects -- exactly for the cube rotations and flips (which permute voxel
        # centres), to interpolation accuracy for a continuous rotation / elastic warp -- so
        # transporting the body SDF is equivalent to transporting the faces and taking the
        # minimum afterwards.
        parts_cont = [sdf_in * m1, m1, batch["ink_prob"].float() * mi, mi, w * mw, batch["winding_conf"].float() * mw, mw]
        # fiber (optional).  "class" mode: the two probabilities move with the grid like ink
        # (mask-aware trilinear, nearest validity) and are SWAPPED per sample where the
        # transform moves the scroll axis (the classes are axis-relative, not sheet-relative).
        # "direction" mode: fiber_dir is a vector (transformed like the winding normal),
        # fiber_str / fiber_weight are scalars, and no swap is needed at all.
        fb_class, fb_dir = "fiber_prob" in batch, "fiber_dir" in batch
        fb = fb_class or fb_dir
        fb_band = "fiber_band" in batch
        n_fib = 0
        if fb:
            fv = batch["fiber_valid"].float()
            mf = (fv == 1).float()
            if fb_class:
                parts_cont.append(batch["fiber_prob"].float() * mf)
                n_fib += 2
            if fb_dir:
                parts_cont += [batch["fiber_dir"].float() * mf, batch["fiber_str"].float() * mf,
                               batch["fiber_weight"].float() * mf]
                n_fib += 5
            parts_cont.append(mf)
            n_fib += 1
        # optional per-voxel surface loss weight: a plain scalar field masked by the surface
        # validity, so it rides on the same mask-aware denominator as the SDF itself
        sw = SURFACE_WEIGHT_KEY in batch
        if sw:
            parts_cont.append(batch[SURFACE_WEIGHT_KEY].float() * m1)
        # surface_mode="sides": the 0/1 body occupancy.  A probability-like scalar in [0, 1]
        # with no sign and no side, so it moves with the grid exactly like ink_prob
        # (mask-aware trilinear on the surface validity); flips and rotations leave it alone.
        sb = "surface_body" in batch
        if sb:
            parts_cont.append(batch["surface_body"].float() * m1)
        # the Local Shape Descriptors carry their own validity (the label body), so they get
        # their own mask-aware denominator; the transport rules are applied after resampling
        ls = LSD_KEY in batch
        if ls:
            lv = batch[LSD_VALID_KEY].float()
            ml = (lv == 1).float()
            parts_cont += [batch[LSD_KEY].float() * ml, ml]
        cont = torch.cat(parts_cont, dim=1)
        s = F.grid_sample(cont, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask_in = torch.cat([sv, iv, wv] + ([fv] if fb else []) + ([batch["fiber_band"].float()] if fb_band else [])
                            + ([lv] if ls else []), dim=1)
        masks = F.grid_sample(mask_in, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        masks = torch.where(oob, torch.zeros_like(masks), masks)
        sv_t, iv_t, wv_t = masks[:, 0:1], masks[:, 1:2], masks[:, 2:3]
        div = lambda num, den: torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.zeros_like(num))  # noqa: E731
        den_s = s[:, ns:ns + 1]
        sdf = div(s[:, 0:ns], den_s.expand(-1, ns, -1, -1, -1))
        ink = div(s[:, ns + 1:ns + 2], s[:, ns + 2:ns + 3])
        den_w = s[:, ns + 10:ns + 11]
        wnd = div(s[:, ns + 3:ns + 9], den_w.expand(-1, 6, -1, -1, -1))
        conf = div(s[:, ns + 9:ns + 10], den_w)
        # per-sample scale rules
        s_iso = torch.tensor([p.scale_iso() for p in params], device=dev, dtype=torch.float32).view(B, 1, 1, 1, 1)
        sdf = (sdf * s_iso).clamp(-self.clip, self.clip)
        shrunk = (s_iso.view(B) < 1.0 - 1e-6)
        if bool(shrunk.any()):
            # labels saturated at +-clip that were shrunk below the clip are unknown -> ignore (2)
            sat = (sdf.abs() >= self.clip * s_iso * (1.0 - 1e-3)) & (sv_t == 1) & shrunk.view(B, 1, 1, 1, 1)
            sv_t = torch.where(sat.any(dim=1, keepdim=True), torch.full_like(sv_t, 2.0), sv_t)
        # (sin, cos) of the winding phase are plain SCALARS under every transform, reflections
        # included: the winding number is a scalar field, and the convention that fixes its sign
        # -- grad(w) . r > 0, "the phase increases outward" -- survives because grad(w) and the
        # radial transport by the same rule, so their inner product is invariant.  A sign flip
        # under det(L) < 0 would be a bug, not a fix.
        sc = F.normalize(wnd[:, 0:2], dim=1, eps=1e-6)
        n_in = wnd[:, 3:6]  # (nx, ny, nz)
        n_zyx = n_in.flip(1)  # (nz, ny, nx)
        n_rot = torch.einsum("bij,bjdhw->bidhw", LinvT, n_zyx)
        n_norm = n_rot.norm(dim=1, keepdim=True)
        n_out = torch.where(n_norm > 1e-6, n_rot / n_norm.clamp_min(1e-6), torch.zeros_like(n_rot)).flip(1)
        ok_n = (n_in.norm(dim=1, keepdim=True) > 0.5)
        dens = wnd[:, 2:3] * torch.where(ok_n, n_norm, 1.0 / s_iso)
        wnd_t = torch.cat([sc, dens, n_out], dim=1) * (wv_t == 1).float()
        out.update({
            "surface_sdf": sdf * (sv_t > 0).float(), "surface_valid": sv_t, "ink_prob": ink, "ink_valid": iv_t,
            "winding": wnd_t, "winding_conf": conf * (wv_t == 1).float(), "winding_valid": wv_t,
        })
        if fb:
            a0 = ns + 11
            den_f = s[:, a0 + n_fib - 1:a0 + n_fib]
            out["fiber_valid"] = masks[:, 3:4]
            swap = torch.tensor([axis_swap_needed(p.matrix()) for p in params], device=dev).view(B, 1, 1, 1, 1)
            if fb_class:
                fp = div(s[:, a0:a0 + 2], den_f.expand(-1, 2, -1, -1, -1))
                out["fiber_prob"] = torch.where(swap, fp.flip(1), fp) if self.fiber_swap else fp
                a0 += 2
            if fb_dir:
                d = div(s[:, a0:a0 + 3], den_f.expand(-1, 3, -1, -1, -1))
                d = torch.einsum("bij,bjdhw->bidhw", L, d)  # a tangent direction: push forward with L
                dn = d.norm(dim=1, keepdim=True)
                out["fiber_dir"] = torch.where(dn > 1e-6, d / dn.clamp_min(1e-6), torch.zeros_like(d))
                out["fiber_str"] = div(s[:, a0 + 3:a0 + 4], den_f)
                out["fiber_weight"] = div(s[:, a0 + 4:a0 + 5], den_f).clamp_min(0.0)
            if fb_band:
                band = masks[:, 4:5]
                # the human codes name the same two axis-relative classes, so they follow the
                # same (nearest-class) rule as the probabilities
                out["fiber_band"] = torch.where(swap, swap_band_class(band), band) if self.fiber_swap else band
        a_sw = ns + 11 + n_fib
        if sw:
            w_t = div(s[:, a_sw:a_sw + 1], den_s)
            # outside the resampled surface support the weight is undetermined -> 1 (neutral)
            out[SURFACE_WEIGHT_KEY] = torch.where(w_t > 1e-6, w_t, torch.ones_like(w_t))
        if sb:
            a_sb = a_sw + (1 if sw else 0)
            out["surface_body"] = div(s[:, a_sb:a_sb + 1], den_s).clamp(0.0, 1.0) * (sv_t > 0).float()
        if ls:
            a_ls = a_sw + (1 if sw else 0) + (1 if sb else 0)
            den_l = s[:, a_ls + 5:a_ls + 6]
            ld = div(s[:, a_ls:a_ls + 5], den_l.expand(-1, 5, -1, -1, -1))
            # off is a true (z, y, x) VECTOR and a length: L rotates it and carries the
            # isotropic scale factor with it (exactly what s_iso does to every other length).
            off = torch.einsum("bij,bjdhw->bidhw", L, ld[:, 0:3])
            lv_t = masks[:, 3 + (1 if fb else 0) + (1 if fb_band else 0):][:, 0:1]
            out[LSD_KEY] = torch.cat([off, ld[:, 3:4] * s_iso, ld[:, 4:5].clamp(0.0, 1.0)],
                                     dim=1) * (lv_t == 1).float()
            out[LSD_VALID_KEY] = lv_t
        return out

    # ---- intensity ------------------------------------------------------------------ #
    @staticmethod
    def _blur(x: torch.Tensor, sigmas: Sequence[float]) -> torch.Tensor:
        """Separable 3D gaussian on (1, 1, D, H, W); ``sigmas`` per axis (z, y, x), 0 = none."""
        for ax, sg in enumerate(sigmas):
            if sg <= 0:
                continue
            r = max(1, int(math.ceil(3.0 * sg)))
            t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
            k = torch.exp(-0.5 * (t / sg) ** 2)
            k = (k / k.sum()).view(1, 1, *[2 * r + 1 if i == ax else 1 for i in range(3)])
            pad = [0] * 6
            pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r  # F.pad pads last dim first
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k)
        return x

    def apply_intensity(self, ct: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        """(B, 1, D, H, W) CT in [0, 1] -> augmented CT; per-sample parameters."""
        c = self.cfg
        B = ct.shape[0]
        D, H, W = ct.shape[2:]
        dev = ct.device
        x = ct.float().clone()
        fired = {n: 0 for n in INTENSITY_NAMES}
        # low-resolution simulation (area down, trilinear up), 3D
        for b in range(B):
            if self._fires(c.lowres.p):
                f = self._u(c.lowres.factor_lo, c.lowres.factor_hi)
                small = tuple(max(2, int(round(n / f))) for n in (D, H, W))
                if small != (D, H, W):
                    lo = F.interpolate(x[b:b + 1], size=small, mode="area")
                    x[b:b + 1] = F.interpolate(lo, size=(D, H, W), mode="trilinear", align_corners=False)
                    fired["lowres"] += 1
        for b in range(B):
            if self._fires(c.blur.p):
                sg = self._u(c.blur.sigma_lo, c.blur.sigma_hi)
                if self._fires(c.blur.axis_only_p):
                    ax = int(torch.randint(0, 3, (), generator=self.gen))
                    sigmas = [sg if i == ax else 0.0 for i in range(3)]
                else:
                    sigmas = [sg, sg, sg]
                x[b:b + 1] = self._blur(x[b:b + 1], sigmas)
                fired["blur"] += 1
        # anisotropic blur: the z resolution of a scan is independent of its in-plane resolution
        for b in range(B):
            if self._fires(c.aniso_blur.p):
                sz = self._u(c.aniso_blur.z_lo, c.aniso_blur.z_hi)
                syx = self._u(c.aniso_blur.yx_lo, c.aniso_blur.yx_hi)
                x[b:b + 1] = self._blur(x[b:b + 1], [sz, syx, syx])
                fired["aniso_blur"] += 1
        for b in range(B):
            if self._fires(c.sharpen.p):
                a = self._u(c.sharpen.amount_lo, c.sharpen.amount_hi)
                x[b:b + 1] = x[b:b + 1] + a * (x[b:b + 1] - self._blur(x[b:b + 1], [c.sharpen.sigma] * 3))
                fired["sharpen"] += 1
        # class contrast: rescale the spread of the papyrus mode about its own centre, leaving
        # air alone (the two scans differ in papyrus std at nearly identical modes)
        for b in range(B):
            if self._fires(c.class_contrast.p):
                x[b:b + 1] = self._class_contrast(x[b:b + 1])
                fired["class_contrast"] += 1
        # elementwise ops, vectorised with identity parameters where a sample did not fire
        g = torch.ones(B)
        cc, bb = torch.ones(B), torch.zeros(B)
        ms, ns = torch.zeros(B), torch.zeros(B)
        for b in range(B):
            if self._fires(c.gamma.p):
                g[b] = self._u(c.gamma.lo, c.gamma.hi)
                fired["gamma"] += 1
            if self._fires(c.contrast.p):
                cc[b] = self._u(c.contrast.lo, c.contrast.hi)
                bb[b] = self._u(-c.contrast.brightness, c.contrast.brightness)
                fired["contrast"] += 1
            if self._fires(c.mult_noise.p):
                ms[b] = self._u(0.0, c.mult_noise.sigma_max)
                fired["mult_noise"] += 1
            if self._fires(c.noise.p):
                ns[b] = self._u(0.0, c.noise.sigma_max)
                fired["noise"] += 1
        v = lambda t: t.to(dev).view(B, 1, 1, 1, 1)  # noqa: E731
        if fired["gamma"]:
            x = x.clamp(0.0, 1.0) ** v(g)
        if fired["contrast"]:
            m = x.mean(dim=(2, 3, 4), keepdim=True)
            x = (x - m) * v(cc) + m + v(bb)
        if fired["mult_noise"]:
            x = x * (1.0 + torch.randn(x.shape, generator=self.gen).to(dev) * v(ms))
        if fired["noise"]:
            x = x + torch.randn(x.shape, generator=self.gen).to(dev) * v(ns)
        # coloured (|k|^(beta/2)-filtered) noise, on top of the white noise above
        for b in range(B):
            if self._fires(c.spectral_noise.p):
                sg = self._u(c.spectral_noise.sigma_lo, c.spectral_noise.sigma_hi) / GREY
                beta = self._u(c.spectral_noise.beta_lo, c.spectral_noise.beta_hi)
                x[b:b + 1] = (x[b:b + 1] + sg * self._spectral_noise((D, H, W), dev, beta)).clamp(0.0, 1.0)
                fired["spectral_noise"] += 1
        # ring artefacts (concentric, centre far outside the crop, constant along z)
        for b in range(B):
            if self._fires(c.ring.p):
                x[b:b + 1] = (x[b:b + 1] * self._ring((D, H, W), dev)).clamp(0.0, 1.0)
                fired["ring"] += 1
        # detector lines
        for b in range(B):
            if self._fires(c.stripe.p):
                x[b:b + 1] = (x[b:b + 1] * self._stripe((D, H, W), dev)).clamp(0.0, 1.0)
                fired["stripe"] += 1
        # 8-bit export window (last: it is the last thing the exporter does)
        wlo, whi = torch.zeros(B), torch.ones(B)
        for b in range(B):
            if self._fires(c.window.p):
                wlo[b] = self._u(c.window.lo_lo, c.window.lo_hi) / GREY
                whi[b] = self._u(c.window.hi_lo, c.window.hi_hi) / GREY
                fired["window"] += 1
        if fired["window"]:
            x = ((x - v(wlo)) / v(whi - wlo).clamp_min(1e-6)).clamp(0.0, 1.0)
        # artefacts: plane-wave streaks along a random 3D direction + shells about a random 3D line
        for b in range(B):
            if self._fires(c.artefact.p):
                x[b:b + 1] = x[b:b + 1] + self._artefact((D, H, W), dev)
                fired["artefact"] += 1
        # 3D cutout
        for b in range(B):
            if self._fires(c.cutout.p):
                n = int(torch.randint(int(c.cutout.n_lo), int(c.cutout.n_hi) + 1, (), generator=self.gen))
                for _ in range(n):
                    sz = [max(1, int(round(self._u(c.cutout.size_lo, c.cutout.size_hi) * dim))) for dim in (D, H, W)]
                    o = [int(torch.randint(0, dim - s + 1, (), generator=self.gen)) for dim, s in zip((D, H, W), sz)]
                    x[b, :, o[0]:o[0] + sz[0], o[1]:o[1] + sz[1], o[2]:o[2] + sz[2]] = self._u()
                fired["cutout"] += 1
        return x, fired

    # ---- scan-domain helpers ---------------------------------------------------------- #
    def _modes(self, x1: torch.Tensor) -> tuple[float, float]:
        """(air, papyrus) mode estimates in [0, 1] for one crop: the config guesses, optionally
        refined by 2-means (Lloyd, 10 iterations) on a strided subsample.  No RNG is consumed."""
        c = self.cfg.class_contrast
        a, p = float(c.air) / GREY, float(c.papyrus) / GREY
        if not c.estimate:
            return a, p
        v = x1.reshape(-1)
        n = max(64, int(c.subsample))
        if v.numel() > n:
            v = v[:: max(1, v.numel() // n)]
        ea, ep = a, p
        for _ in range(10):
            mid = 0.5 * (ea + ep)
            lo_m = v < mid
            hi_m = ~lo_m
            if not (bool(lo_m.any()) and bool(hi_m.any())):
                return a, p  # single-class crop: keep the guesses
            ea, ep = float(v[lo_m].mean()), float(v[hi_m].mean())
        return (ea, ep) if ep - ea > 1e-3 else (a, p)

    def _class_contrast(self, x1: torch.Tensor) -> torch.Tensor:
        """x' = m_p + (x - m_p) * s above the air/papyrus midpoint, smoothstep-blended over
        +- ``band`` grey levels, identity below."""
        c = self.cfg.class_contrast
        a, mp = self._modes(x1)
        s = self._u(c.s_lo, c.s_hi)
        mid = 0.5 * (a + mp)
        band = max(float(c.band) / GREY, 1e-6)
        w = ((x1 - (mid - band)) / (2.0 * band)).clamp(0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep: C1 at both ends of the blend
        return (x1 * (1.0 - w) + (mp + (x1 - mp) * s) * w).clamp(0.0, 1.0)

    def _spectral_noise(self, shape: tuple[int, int, int], dev: torch.device, beta: float) -> torch.Tensor:
        """Unit-std noise with PSD ~ |k|^beta: white noise filtered by |k|^(beta/2), DC removed."""
        D, H, W = shape
        w = torch.randn(1, 1, D, H, W, generator=self.gen).to(dev)
        kz = torch.fft.fftfreq(D, device=dev)
        ky = torch.fft.fftfreq(H, device=dev)
        kx = torch.fft.fftfreq(W, device=dev)
        k = (kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2).sqrt()
        f = torch.where(k > 0, k.clamp_min(1e-12) ** (0.5 * beta), torch.zeros_like(k))
        out = torch.fft.ifftn(torch.fft.fftn(w, dim=(2, 3, 4)) * f, dim=(2, 3, 4)).real
        return out / out.std().clamp_min(1e-8)

    def _ring(self, shape: tuple[int, int, int], dev: torch.device) -> torch.Tensor:
        """Multiplicative (1, 1, 1, H, W) gain: 1..3 gaussian rings about a centre ``centre_*``
        crop widths away in the y-x plane, so they cross the crop as gently curved stripes."""
        c = self.cfg.ring
        _, H, W = shape
        m = float(max(H, W))
        R = self._u(c.centre_lo, c.centre_hi) * m
        ang = self._u(0.0, 2.0 * math.pi)
        cy, cx = -R * math.cos(ang), -R * math.sin(ang)  # centre, in crop-centred coordinates
        y = torch.arange(H, device=dev, dtype=torch.float32) - (H - 1) / 2.0
        x = torch.arange(W, device=dev, dtype=torch.float32) - (W - 1) / 2.0
        d = ((y[:, None] - cy) ** 2 + (x[None, :] - cx) ** 2).sqrt()
        g = torch.ones(H, W, device=dev)
        n = int(torch.randint(int(c.n_lo), int(c.n_hi) + 1, (), generator=self.gen))
        for _ in range(n):
            r0 = R + self._u(-0.5, 0.5) * m  # a radius that actually crosses the crop
            wd = max(self._u(c.width_lo, c.width_hi), 1e-3)
            amp = self._u(c.amp_lo, c.amp_hi) * (1.0 if self._fires(0.5) else -1.0)
            g = g + amp * torch.exp(-0.5 * ((d - r0) / wd) ** 2)
        return g.view(1, 1, 1, H, W)

    def _stripe(self, shape: tuple[int, int, int], dev: torch.device) -> torch.Tensor:
        """Multiplicative (1, 1, 1, H, W) gain: a few constant y or x lines (a detector column)."""
        c = self.cfg.stripe
        _, H, W = shape
        g = torch.ones(H, W, device=dev)
        n = int(torch.randint(int(c.n_lo), int(c.n_hi) + 1, (), generator=self.gen))
        for _ in range(n):
            along_y = self._fires(0.5)
            dim = H if along_y else W
            wd = min(int(torch.randint(int(c.width_lo), int(c.width_hi) + 1, (), generator=self.gen)), dim)
            i0 = int(torch.randint(0, max(1, dim - wd + 1), (), generator=self.gen))
            amp = self._u(c.amp_lo, c.amp_hi) * (1.0 if self._fires(0.5) else -1.0)
            if along_y:
                g[i0:i0 + wd, :] = g[i0:i0 + wd, :] * (1.0 + amp)
            else:
                g[:, i0:i0 + wd] = g[:, i0:i0 + wd] * (1.0 + amp)
        return g.view(1, 1, 1, H, W)

    def _artefact(self, shape: tuple[int, int, int], dev: torch.device) -> torch.Tensor:
        c = self.cfg.artefact
        D, H, W = shape
        axes = [torch.arange(n, device=dev, dtype=torch.float32) - (n - 1) / 2.0 for n in (D, H, W)]
        z, y, x = torch.meshgrid(*axes, indexing="ij")
        out = torch.zeros((1, 1, D, H, W), device=dev)
        d = F.normalize(torch.randn(3, generator=self.gen), dim=0).to(dev)
        t = z * d[0] + y * d[1] + x * d[2]  # coordinate along the streak normal
        if c.streak_amp > 0:
            prof = torch.randn(1, 1, 8, generator=self.gen).to(dev)
            n = int(max(D, H, W) * 2)
            prof = F.interpolate(prof, size=n, mode="linear", align_corners=True)[0, 0]
            idx = ((t / (0.5 * n) + 1.0) * 0.5 * (n - 1)).clamp(0, n - 1)
            i0 = idx.floor().long()
            fr = idx - i0
            i1 = (i0 + 1).clamp(max=n - 1)
            streak = prof[i0] * (1 - fr) + prof[i1] * fr
            out = out + self._u(0.0, c.streak_amp) * streak / prof.abs().max().clamp_min(1e-6)
        if c.ring_amp > 0:
            a = F.normalize(torch.randn(3, generator=self.gen), dim=0).to(dev)  # line direction
            p0 = torch.tensor([self._u(-0.5, 0.5) * D, self._u(-0.5, 0.5) * H, self._u(-0.5, 0.5) * W], device=dev)
            q = torch.stack([z - p0[0], y - p0[1], x - p0[2]], dim=-1)
            along = (q * a).sum(-1, keepdim=True)
            r = (q - along * a).norm(dim=-1)  # distance to the line
            per = self._u(c.ring_period_lo, c.ring_period_hi)
            out = out + self._u(0.0, c.ring_amp) * torch.sin(2 * math.pi * r / per + self._u(0, 2 * math.pi))
        return out

    # ---- full pipeline ----------------------------------------------------------------- #
    def __call__(self, batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        B = int(batch["input"].shape[0])
        params = [self.sample_spatial() for _ in range(B)]
        # kept so consumers that must reproduce the transform on data outside the batch
        # (tsm.dino.DinoFeatSource reads its targets from a cache) can see what fired
        self.last_params = params
        out = self.apply_spatial(batch, params)
        fired = {n: 0 for n in TRANSFORM_NAMES}
        for p in params:
            for k, v in p.fired.items():
                fired[k] += int(v)
        ct, fi = self.apply_intensity(out["input"][:, 0:1])
        fired.update(fi)
        out["input"] = torch.cat([ct, out["input"][:, 1:]], dim=1)
        return out, fired

    def describe(self) -> str:
        c = self.cfg
        return ", ".join(f"{n}:p={float(getattr(c, n).p):g}" for n in TRANSFORM_NAMES if float(getattr(c, n).p) > 0) or "identity"



__all__ = [
    "CLIP", "FINE_CHANNELS", "FACE_CHANNELS", "FIBER_CHANNELS", "COARSE_CHANNELS", "WF_CHANNELS",
    "WINDING_SOURCES", "fine_winding_targets", "merge_winding_targets", "TARGET_KEYS", "target_keys",
    "SURFACE_MODES", "body_sdf", "face_dist", "body_mask_np", "SURFACE_BODY_KEY", "WINDING_CH", "FIBER_MODES", "FIBER_TARGET_KEYS", "FIBER_BAND_WEIGHT", "BAND_CHANNEL",
    "LabelStore", "CropDataset", "MultiStoreDataset", "build_origins", "augment", "spatial_transform", "intensity_augment",
    "Augment", "AugmentConfig", "SpatialParams", "PRESETS", "TRANSFORM_NAMES", "SPATIAL_NAMES", "INTENSITY_NAMES",
    "SCAN_NAMES", "WindowCfg", "ClassContrastCfg", "AnisoBlurCfg", "RingCfg", "StripeCfg", "SpectralNoiseCfg",
    "as_augment_config", "apply_overrides", "load_axis_spec", "rot90_matrix_inplane", "rotation_matrix", "split_holdout",
    "upsample_coarse_targets", "store_factor",
    "decode_sdf", "encode_sdf", "decode_signed", "encode_signed", "decode_density", "encode_density",
    "decode_prob", "encode_prob", "open_reader_factory",
]
