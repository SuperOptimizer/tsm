"""Fibre-orientation targets: the two class channels, and the axial direction field.

The fibre teacher labels every voxel *vertical* (along the scroll axis z) or
*horizontal / angular* (circumferential: tangent to the sheet and perpendicular to the
axis).  Both names are defined **relative to the scroll axis**, not to the sheet, so the
pair is not a scalar field: any transform that moves the volume z axis moves the class
boundary with it.  Two ways out, ``extra.train.fiber_mode``:

``"class"`` (default, the historical head)
    keep the two probabilities, but let a spatial augmentation **swap** them whenever its
    linear part sends the volume z axis more than 45 degrees away from z
    (:func:`axis_swap_needed`; the classes are *axial*, so the angle is taken modulo a sign
    reversal -- a z flip is not a swap).  For the 24 cube rotations this is exact (z maps to
    +-z, +-y or +-x, i.e. 0 or 90 degrees); for a continuous rotation it is the nearest-class
    approximation.  ``extra.train.fiber_swap_fix: false`` opts out (the pre-fix behaviour,
    kept only to reproduce the bug in an ablation).

``"direction"`` (the proper fix)
    turn the two probabilities into an **axial direction** (a sign-free unit vector) per
    voxel, which *is* a geometric field and rotates like one.  With ``n`` the local sheet
    normal and ``a`` the scroll axis (``(1, 0, 0)`` in (z, y, x)):

        t_v = normalise(a - (a.n) n)     the axis projected into the sheet plane (vertical)
        t_h = normalise(n x a)           in-plane and perpendicular to the axis (horizontal)
        d   = normalise(p_vt t_v + p_hz t_h),  s = max(p_vt, p_hz)

    ``t_v``, ``t_h`` and ``n`` are an orthonormal triad wherever ``n`` is not parallel to
    ``a`` (where it is, neither class is defined and the voxel is dropped).  The head
    predicts ``[dz, dy, dx, s]``; the loss is the axial cosine ``1 - (d^.d)^2`` (invariant
    to a sign flip of either vector) weighted by ``s``, plus a BCE on ``s``.  Nothing new is
    stored: the dataset derives ``d`` and ``s`` on the fly from ``fiber_vt`` / ``fiber_hz``
    and the gradient of the decoded ``sdf_in``, falling back to the winding normal target
    where that gradient is not a unit vector.

Where the human ``hzvt_class`` channel (upstream traced bands, sparse: 0 bg, 1 hz, 2 vt,
3 exclude) covers a voxel it overrides the teacher: ``d = t_h`` / ``t_v``, ``s = 1`` and the
loss weight becomes ``extra.train.fiber_band_weight`` (default 5); class 3 is dropped.

Everything here is torch and works on (B, C, Z, Y, X) or (C, Z, Y, X) tensors; the dataset
converts its numpy crops for the few lines it needs (:func:`derive_direction_targets`).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

__all__ = [
    "FIBER_MODES",
    "AXIS_ZYX",
    "FIBER_DIR_CHANNELS",
    "SWAP_COS",
    "normalize_vec",
    "sdf_normal",
    "sheet_normal",
    "fiber_basis",
    "direction_from_class",
    "class_from_direction",
    "axis_swap_needed",
    "swap_band_class",
    "derive_direction_targets",
]

#: ``extra.train.fiber_mode``
FIBER_MODES = ("class", "direction")
#: the scroll axis in volume (z, y, x) coordinates (the label stores are axis-aligned in z)
AXIS_ZYX = (1.0, 0.0, 0.0)
#: the direction head / prediction channels, in order
FIBER_DIR_CHANNELS = ["fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"]
#: |cos| below which a transformed z axis counts as "more than 45 degrees away from z"
SWAP_COS = math.cos(math.pi / 4.0)
#: |grad sdf| outside this range = the SDF gradient is not a unit normal there
GRAD_LO, GRAD_HI = 0.5, 1.5
EPS = 1e-6

# human hzvt_class codes (tsm.rvfaces / dev/rectoverso_slab.py)
BAND_BG, BAND_HZ, BAND_VT, BAND_EXCLUDE = 0, 1, 2, 3


def normalize_vec(v: torch.Tensor, dim: int = -4, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """(unit vector, norm) along ``dim``; the vector is exactly 0 where the norm is tiny."""
    n = v.norm(dim=dim, keepdim=True)
    return torch.where(n > eps, v / n.clamp_min(eps), torch.zeros_like(v)), n


def sdf_normal(sdf: torch.Tensor, lo: float = GRAD_LO, hi: float = GRAD_HI) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit gradient of a decoded SDF crop -> ``(n (…, 3, Z, Y, X) in (z, y, x), ok mask)``.

    ``sdf`` is (…, 1, Z, Y, X).  ``ok`` is the voxels where ``|grad|`` is in ``[lo, hi]``, i.e.
    where the field really behaves like a signed distance (the sawtooth seams of the two-face
    SDFs and the clip plateaus are excluded)."""
    x = sdf.float()
    g = torch.stack(torch.gradient(x[..., 0, :, :, :], dim=(-3, -2, -1)), dim=-4)
    n, mag = normalize_vec(g, dim=-4)
    ok = (mag >= float(lo)) & (mag <= float(hi))
    return n * ok, ok


def sheet_normal(sdf: torch.Tensor, fallback: torch.Tensor | None = None,
                 lo: float = GRAD_LO, hi: float = GRAD_HI) -> tuple[torch.Tensor, torch.Tensor]:
    """Sheet normal from ``grad sdf``, falling back to ``fallback`` (the winding normal, (z, y, x))
    where the gradient is undefined.  Returns ``(n, ok)``; ``n`` is 0 where ``ok`` is False."""
    n, ok = sdf_normal(sdf, lo, hi)
    if fallback is not None:
        fb, fmag = normalize_vec(fallback.float(), dim=-4)
        use = (~ok) & (fmag > 0.5)
        n = torch.where(use, fb, n)
        ok = ok | use
    return n * ok, ok


def _axis_like(n: torch.Tensor, axis: Sequence[float] | torch.Tensor | None) -> torch.Tensor:
    """The scroll axis as a (3, 1, 1, 1) field: vectors always carry their components at
    dim -4, so this broadcasts against both (3, Z, Y, X) and (B, 3, Z, Y, X)."""
    a = torch.as_tensor(AXIS_ZYX if axis is None else axis, dtype=n.dtype, device=n.device)
    return a.reshape(3, 1, 1, 1)


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Cross product of two (…, 3, Z, Y, X) fields, components in the given order.

    The (z, y, x) order is an odd permutation of (x, y, z), so this is minus the geometric
    cross product -- irrelevant here: the result is used only as an *axial* in-plane
    direction, and it is perpendicular to both inputs whatever the sign convention."""
    d = -4
    u0, u1, u2 = u.unbind(d)
    v0, v1, v2 = v.unbind(d)
    return torch.stack([u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0], dim=d)


def fiber_basis(n: torch.Tensor, axis: Any = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(t_v, t_h, ok)`` for a unit sheet normal field ``n`` ((…, 3, Z, Y, X), (z, y, x)).

    ``t_v`` = the scroll axis projected into the sheet plane (the *vertical* fibre direction),
    ``t_h`` = ``n x axis`` (the *horizontal / angular* direction: in the sheet, perpendicular
    to the axis).  Both are unit and mutually perpendicular; ``ok`` is False where ``n`` is
    parallel to the axis (neither class is defined) or ``n`` itself is zero."""
    a = _axis_like(n, axis)
    an = (a * n).sum(dim=-4, keepdim=True)
    tv, mv = normalize_vec(a - an * n, dim=-4)
    th, mh = normalize_vec(_cross(n, a), dim=-4)
    ok = (mv > 1e-3) & (mh > 1e-3)
    return tv * ok, th * ok, ok


def direction_from_class(p_vt: torch.Tensor, p_hz: torch.Tensor, tv: torch.Tensor,
                         th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d, s)``: the axial fibre direction ``normalise(p_vt t_v + p_hz t_h)`` and its
    strength ``max(p_vt, p_hz)``.  ``d`` is 0 where both probabilities vanish (the loss is
    weighted by ``s``, so such voxels contribute nothing to the direction term)."""
    d, _ = normalize_vec(p_vt * tv + p_hz * th, dim=-4)
    return d, torch.maximum(p_vt, p_hz)


def class_from_direction(d: torch.Tensor, s: torch.Tensor, tv: torch.Tensor,
                         th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(p_vt, p_hz)`` back out of an axial direction: ``s * |d^.t_v|`` and ``s * |d^.t_h|``.

    Sign-free (the fibre direction is an axis) and consistent with
    :func:`direction_from_class` on a pure class (``d = t_v``, ``s = 1`` -> ``(1, 0)``)."""
    dn, _ = normalize_vec(d, dim=-4)
    return s * (dn * tv).sum(dim=-4, keepdim=True).abs(), s * (dn * th).sum(dim=-4, keepdim=True).abs()


def axis_swap_needed(L: torch.Tensor | np.ndarray, axis: Any = None, max_deg: float = 45.0) -> bool:
    """True when the spatial transform ``L`` (``x_out = L x_in``, (z, y, x)) moves the scroll
    axis more than ``max_deg`` away from itself and the two *classes* must therefore be
    swapped.  The classes are axial, so the angle is measured modulo a sign flip: a z flip
    (angle 180 deg, |cos| = 1) is **not** a swap, a 90 deg rotation about y (z -> x) is."""
    m = torch.as_tensor(np.asarray(L, dtype=np.float64))
    a = torch.as_tensor(np.asarray(AXIS_ZYX if axis is None else axis, dtype=np.float64))
    v = m @ a                       # where the scroll axis points after the transform
    c = float((v @ a).abs() / (v.norm() * a.norm()).clamp_min(1e-12))
    return c < math.cos(math.radians(float(max_deg))) - 1e-9


def swap_band_class(band: torch.Tensor) -> torch.Tensor:
    """Swap the human hz/vt codes (1 <-> 2), leaving background (0) and exclude (3) alone."""
    out = band.clone()
    out = torch.where(band == BAND_HZ, torch.full_like(band, float(BAND_VT)), out)
    out = torch.where(band == BAND_VT, torch.full_like(band, float(BAND_HZ)), out)
    return out


def derive_direction_targets(
    p_vt: np.ndarray,
    p_hz: np.ndarray,
    sdf: np.ndarray,
    valid: np.ndarray,
    winding_normal_zyx: np.ndarray | None = None,
    band: np.ndarray | None = None,
    band_weight: float = 5.0,
    axis: Any = None,
) -> dict[str, np.ndarray]:
    """Dataset-side derivation (numpy, one crop) of the direction targets.

    Inputs are (1, Z, Y, X) except ``winding_normal_zyx`` (3, Z, Y, X, in (z, y, x) order):
    the teacher probabilities, the decoded ``sdf_in`` (or ``sdf`` in the medial mode), the
    ``fiber_valid`` mask (0/1) and, optionally, the human ``hzvt_class`` band.
    Returns ``fiber_dir`` (3), ``fiber_str`` (1), ``fiber_valid`` (1) and ``fiber_weight`` (1).
    """
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    fb = None if winding_normal_zyx is None else t(winding_normal_zyx)
    n, _ = sheet_normal(t(sdf), fb)
    tv, th, ok = fiber_basis(n, axis)
    d, s = direction_from_class(t(p_vt), t(p_hz), tv, th)
    v = t(valid)
    w = torch.ones_like(s)
    valid_out = ((v == 1) & ok).float()
    if band is not None:
        bt = t(band)
        is_hz, is_vt = bt == BAND_HZ, bt == BAND_VT
        human = is_hz | is_vt
        d = torch.where(human, torch.where(is_vt, tv, th), d)
        s = torch.where(human, torch.ones_like(s), s)
        w = torch.where(human, torch.full_like(w, float(band_weight)), w)
        valid_out = torch.where(bt == BAND_EXCLUDE, torch.zeros_like(valid_out),
                                torch.where(human & ok, torch.ones_like(valid_out), valid_out))
    return {"fiber_dir": (d * valid_out).numpy(), "fiber_str": (s * valid_out).numpy(),
            "fiber_valid": valid_out.numpy(), "fiber_weight": w.numpy()}
