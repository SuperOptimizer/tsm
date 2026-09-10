"""Orientation (equivariance) tools: the 48-element octahedral group, arbitrary rotations,
per-channel inverse rules for model outputs and disagreement metrics.

Conventions
-----------
* Volumes are ``(..., Z, Y, X)`` (torch or numpy); the group acts on the last three axes.
* A group element ``g`` is a 3x3 signed permutation matrix on **centred voxel coordinates
  in (z, y, x) order**: ``apply(V, g)`` returns ``V'`` with ``V'(g p) = V(p)`` (the content at
  ``p`` moves to ``g p``).  ``g[i, j] = s_i`` for ``j = perm[i]`` means output axis ``i`` is
  input axis ``perm[i]``, reversed when ``s_i = -1``.  Implemented with ``permute`` +
  ``flip`` (exact, no interpolation); ``inverse(g) = g.T``.
* A vector field stacked on the channel axis transforms as ``v' (g p) = g v(p)`` when the
  components are in (z, y, x) order; components in (x, y, z) order use ``J g J`` with ``J``
  the axis-reversal matrix (:func:`to_xyz`).
* Model outputs computed on ``apply(box, g)`` are brought back with
  :func:`invert_prediction`: scalars (probabilities, sdf, sin/cos, density, conf) get the
  inverse spatial transform only; vectors (normals) get the inverse spatial transform and
  are multiplied by ``g^{-1} = g^T``; lasagna's six encoded direction channels are decoded
  to a sign-ambiguous normal line field (:func:`tsm.labels.decode_lasagna_normal`),
  rotated as a vector and re-encoded (the double-angle encoding makes the sign irrelevant).
* :func:`random_rotation_matrix` / :func:`apply_rotation` cover small arbitrary rotations
  (``grid_sample``, trilinear for continuous fields, nearest for masks).  Every helper that
  takes ``g`` accepts either a signed permutation (exact path) or a general orthogonal
  matrix (resampling path).
"""

from __future__ import annotations

import math
from itertools import permutations, product
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from tsm.labels import _drop_small_components, _edt, _enc_dir, decode_lasagna_normal, medial_surface

__all__ = [
    "OCT48",
    "GROUPS",
    "group_elements",
    "element_name",
    "classify",
    "perm_sign",
    "is_signed_permutation",
    "inverse",
    "to_xyz",
    "apply",
    "apply_vector",
    "encode_lasagna_dirs",
    "transform_lasagna_channels",
    "invert_scalar",
    "invert_vector",
    "invert_lasagna",
    "invert_prediction",
    "STUDENT_NORMAL_CHANNELS",
    "random_rotation_matrix",
    "apply_rotation",
    "rotation_valid_mask",
    "metrics",
    "primary_metric",
    "average_precision",
]

_J = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)  # (z,y,x) <-> (x,y,z)


# --------------------------------------------------------------------------- #
# The octahedral group
# --------------------------------------------------------------------------- #
def _signed_perm(perm: Sequence[int], signs: Sequence[int]) -> np.ndarray:
    m = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        m[i, perm[i]] = signs[i]
    return m


def _build_oct48() -> list[np.ndarray]:
    out = [_signed_perm(p, s) for p in permutations(range(3)) for s in product((1, -1), repeat=3)]
    out.sort(key=lambda m: (abs(int(np.round(np.linalg.det(m))) - 1), tuple(perm_sign(m)[0]), tuple(-np.array(perm_sign(m)[1]))))
    return out


def perm_sign(g: np.ndarray) -> tuple[list[int], list[int]]:
    """``(perm, sign)`` of a signed permutation: output axis ``i`` = ``sign[i] *`` input axis ``perm[i]``."""
    g = np.asarray(g)
    if not is_signed_permutation(g):
        raise ValueError(f"not a signed permutation matrix: {g.tolist()}")
    perm = [int(np.nonzero(np.round(g[i]))[0][0]) for i in range(3)]
    sign = [int(np.sign(g[i, perm[i]])) for i in range(3)]
    return perm, sign


def is_signed_permutation(g: Any) -> bool:
    g = np.asarray(g, dtype=np.float64)
    if g.shape != (3, 3) or not np.allclose(np.abs(g).sum(0), 1) or not np.allclose(np.abs(g).sum(1), 1):
        return False
    return bool(np.allclose(np.abs(g[np.abs(g) > 0.5]), 1.0) and np.allclose(g[np.abs(g) <= 0.5], 0.0))


OCT48: list[np.ndarray] = _build_oct48()


def element_name(g: np.ndarray) -> str:
    """``'+z+y+x'`` style: output axis i = sign * input axis ``'zyx'[perm[i]]``."""
    perm, sign = perm_sign(g)
    return "".join(("+" if s > 0 else "-") + "zyx"[p] for p, s in zip(perm, sign))


def classify(g: np.ndarray) -> dict[str, Any]:
    """Structural tags of a group element (for the bias summary)."""
    perm, sign = perm_sign(g)
    det = int(round(float(np.linalg.det(np.asarray(g, dtype=np.float64)))))
    identity = perm == [0, 1, 2] and sign == [1, 1, 1]
    pure_flip = perm == [0, 1, 2] and not identity
    z_fixed = perm[0] == 0
    return {
        "name": element_name(g),
        "identity": identity,
        "det": det,  # +1 proper rotation, -1 reflection
        "pure_flip": pure_flip,  # no axis permutation
        "flips_z": sign[0] < 0 and z_fixed,
        "in_plane": z_fixed and not identity,  # z axis untouched: y/x flips, y<->x swap, rot90 about z
        "swaps_z": not z_fixed,  # z exchanged with y or x: probes anisotropy along the scan axis
        "n_flips": int(sum(1 for s in sign if s < 0)),
        "perm": perm,
        "sign": sign,
    }


def _rot90z(k: int) -> np.ndarray:
    m = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int64)  # y' = -x, x' = y (np.rot90 axes=(y,x), k=1)
    return np.linalg.matrix_power(m, k % 4).astype(np.int64)


GROUPS: dict[str, list[np.ndarray]] = {
    "oct48": OCT48,
    "rot24": [g for g in OCT48 if int(round(float(np.linalg.det(g)))) == 1],
    "flip8": [g for g in OCT48 if perm_sign(g)[0] == [0, 1, 2]],
    "rot90z4": [_rot90z(k) for k in range(4)],
    "inplane16": [g for g in OCT48 if perm_sign(g)[0][0] == 0],
}


def group_elements(name: str) -> list[np.ndarray]:
    """Elements of a named subgroup (identity first)."""
    if name not in GROUPS:
        raise ValueError(f"unknown group {name!r}; known: {sorted(GROUPS)}")
    els = list(GROUPS[name])
    els.sort(key=lambda m: 0 if np.array_equal(m, np.eye(3, dtype=np.int64)) else 1)
    return els


def inverse(g: np.ndarray) -> np.ndarray:
    """Inverse of an orthogonal matrix (signed permutation or rotation): its transpose."""
    return np.ascontiguousarray(np.asarray(g).T)


def to_xyz(g: np.ndarray) -> np.ndarray:
    """The same linear map expressed on (x, y, z) components: ``J g J``."""
    return _J @ np.asarray(g) @ _J


# --------------------------------------------------------------------------- #
# Exact spatial action
# --------------------------------------------------------------------------- #
def _as_tensor(v: Any) -> tuple[torch.Tensor, bool]:
    if isinstance(v, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(v)), True
    return v, False


def apply(volume: Any, g: np.ndarray) -> Any:
    """``V'(g p) = V(p)`` on the last three axes via permute + flip (exact).

    The output shape is the input shape with the spatial axes permuted (identical for cubes).
    Numpy in -> numpy out; torch in -> torch out.
    """
    perm, sign = perm_sign(g)
    v, was_np = _as_tensor(volume)
    if v.ndim < 3:
        raise ValueError("volume needs at least 3 dims")
    lead = v.ndim - 3
    order = list(range(lead)) + [lead + perm[i] for i in range(3)]
    out = v.permute(order)
    flips = [lead + i for i in range(3) if sign[i] < 0]
    if flips:
        out = torch.flip(out, flips)
    out = out.contiguous()
    return out.numpy() if was_np else out


def _mix(field: Any, m: np.ndarray) -> Any:
    """Channel mix ``out_i = sum_j m[i, j] field_j`` on axis ``-4`` (3 components)."""
    v, was_np = _as_tensor(field)
    if v.shape[-4] != 3:
        raise ValueError(f"vector field needs 3 components on axis -4, got shape {tuple(v.shape)}")
    mt = torch.as_tensor(np.asarray(m, dtype=np.float64), dtype=v.dtype if v.is_floating_point() else torch.float32, device=v.device)
    v = v.to(mt.dtype)
    out = torch.einsum("ij,...jzyx->...izyx", mt, v)
    return out.numpy() if was_np else out


def apply_vector(field: Any, g: np.ndarray, order: str = "zyx") -> Any:
    """Transform a vector field ``(..., 3, Z, Y, X)``: spatial action, then ``v -> g v``.

    ``order`` names the component order on the channel axis (``'zyx'`` or ``'xyz'``).
    """
    m = np.asarray(g, dtype=np.float64)
    if order == "xyz":
        m = to_xyz(m)
    elif order != "zyx":
        raise ValueError("order must be 'zyx' or 'xyz'")
    return _mix(_spatial(field, g), m)


def _spatial(volume: Any, g: np.ndarray, mode: str = "bilinear") -> Any:
    """Spatial action for a signed permutation (exact) or any rotation (grid_sample)."""
    if is_signed_permutation(g):
        return apply(volume, np.round(np.asarray(g)).astype(np.int64))
    return apply_rotation(volume, np.asarray(g, dtype=np.float64), mode=mode)


# --------------------------------------------------------------------------- #
# Lasagna direction encoding
# --------------------------------------------------------------------------- #
def encode_lasagna_dirs(n_zyx: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(3, ...) normal in (z, y, x) -> the 6 lasagna channels ``dir0_z, dir1_z, dir0_y, dir1_y,
    dir0_x, dir1_x`` (planes (x,y), (x,z), (y,z); docs/label_store.md formulas).  Sign-free."""
    n = np.asarray(n_zyx, dtype=np.float64)
    nz, ny, nx = n[0], n[1], n[2]
    z0, z1 = _enc_dir(nx, ny, eps)
    y0, y1 = _enc_dir(nx, nz, eps)
    x0, x1 = _enc_dir(ny, nz, eps)
    return np.stack([z0, z1, y0, y1, x0, x1]).astype(np.float32)


def transform_lasagna_channels(ch: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Re-encode the 8 (or 6) lasagna channels for normals ``n -> m n`` ((z, y, x) components):
    decode (line field), rotate, encode.  Channels ``cos, grad_mag`` are copied.  No spatial action."""
    ch = np.asarray(ch, dtype=np.float32)
    head = ch[:2] if ch.shape[0] == 8 else None
    n = decode_lasagna_normal(ch)  # (3, ...) in (z, y, x), sign arbitrary
    n2 = np.einsum("ij,j...->i...", np.asarray(m, dtype=np.float64), n)
    enc = encode_lasagna_dirs(n2)
    return np.concatenate([head, enc]) if head is not None else enc


# --------------------------------------------------------------------------- #
# Inverse rules for model outputs computed on apply(box, g)
# --------------------------------------------------------------------------- #
def invert_scalar(pred: Any, g: np.ndarray) -> Any:
    """Scalar channels: inverse spatial transform only."""
    return _spatial(pred, inverse(g))


def invert_vector(pred: Any, g: np.ndarray, order: str = "zyx") -> Any:
    """Vector channels ``(..., 3, Z, Y, X)``: inverse spatial transform, then ``v -> g^T v``."""
    return apply_vector(pred, inverse(g), order=order)


def invert_lasagna(pred: Any, g: np.ndarray) -> Any:
    """Lasagna 8-channel output: inverse spatial transform of all channels, then decode the six
    direction channels to a normal line field, rotate by ``g^T`` and re-encode."""
    v, was_np = _as_tensor(pred)
    if v.shape[-4] != 8:
        raise ValueError(f"lasagna prediction needs 8 channels on axis -4, got {tuple(v.shape)}")
    back = _spatial(v, inverse(g)).float().cpu().numpy()
    lead = back.shape[:-4]
    flat = back.reshape((-1, 8) + back.shape[-3:])
    out = np.stack([transform_lasagna_channels(f, inverse(g)) for f in flat]).reshape(lead + (8,) + back.shape[-3:])
    return out if was_np else torch.from_numpy(out).to(v.device)


STUDENT_NORMAL_CHANNELS = (6, 7, 8)  # nx, ny, nz of the 11 activated student heads


def invert_prediction(pred: Any, g: np.ndarray, kind: str, normal_channels: Sequence[int] = STUDENT_NORMAL_CHANNELS) -> Any:
    """Bring a prediction computed on ``apply(box, g)`` back into the box frame.

    ``kind``: ``'scalar'`` (recto / m7 / ink probabilities, any scalar stack), ``'lasagna'``
    (8 channels), ``'student'`` (11 activated heads: everything scalar except the
    ``normal_channels`` = (nx, ny, nz) in (x, y, z) order, transformed as a vector) or
    ``'vector_zyx'`` / ``'vector_xyz'`` (a bare 3-component field).
    """
    if kind == "scalar":
        return invert_scalar(pred, g)
    if kind == "lasagna":
        return invert_lasagna(pred, g)
    if kind in ("vector_zyx", "vector_xyz"):
        return invert_vector(pred, g, order=kind[-3:])
    if kind == "student":
        v, was_np = _as_tensor(pred)
        back = invert_scalar(v, g)
        idx = list(normal_channels)
        n = invert_vector(v[..., idx, :, :, :], g, order="xyz").to(back.dtype)
        back = back.clone()
        back[..., idx, :, :, :] = n
        return back.numpy() if was_np else back
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------- #
# Arbitrary rotations
# --------------------------------------------------------------------------- #
def random_rotation_matrix(rng: np.random.Generator | int | None = None, max_deg: float | None = None) -> np.ndarray:
    """Random rotation in (z, y, x) coordinates.

    ``max_deg=None``: uniform on SO(3) (random unit quaternion).  Otherwise a uniformly random
    axis with an angle uniform in ``[-max_deg, max_deg]`` (small perturbations)."""
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if max_deg is None:
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
    axis = q[1:] / max(np.linalg.norm(q[1:]), 1e-12)
    ang = math.radians(rng.uniform(-float(max_deg), float(max_deg)))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _rotation_grid(R: np.ndarray, shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sampling grid for ``V'(p) = V(R^T p)`` (centred voxel coords, align_corners=True)."""
    n = np.asarray(shape, dtype=np.float64)
    Rt = np.asarray(R, dtype=np.float64).T
    A = Rt * (n[None, :] - 1) / (n[:, None] - 1)  # normalised (z,y,x) coords in -> out
    A_xyz = _J @ A @ _J  # grid_sample wants (x, y, z)
    theta = torch.zeros(1, 3, 4, dtype=torch.float64)
    theta[0, :, :3] = torch.from_numpy(A_xyz)
    grid = F.affine_grid(theta, [1, 1, *[int(s) for s in shape]], align_corners=True)
    return grid.to(device=device, dtype=dtype)


def apply_rotation(volume: Any, R: np.ndarray, mode: str = "bilinear") -> Any:
    """``V'(p) = V(R^{-1} p)`` about the box centre via ``grid_sample`` (zeros outside).

    ``mode`` ``'bilinear'`` (trilinear) for continuous fields, ``'nearest'`` for masks / labels.
    Works on ``(..., Z, Y, X)``; a general rotation only makes sense on the sampled
    coordinates, so the output has the input shape.
    """
    v, was_np = _as_tensor(volume)
    shape = tuple(int(s) for s in v.shape[-3:])
    orig_dtype = v.dtype
    x = v.reshape(1, -1, *shape).float()
    grid = _rotation_grid(R, shape, x.device, x.dtype)
    out = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)
    out = out.reshape(v.shape)
    if not v.is_floating_point():
        out = out.round().to(orig_dtype) if mode == "nearest" else out
    return out.numpy() if was_np else out


def rotation_valid_mask(shape: Sequence[int], R: np.ndarray, margin: int = 2) -> np.ndarray:
    """Voxels whose forward-then-inverse rotation never sampled outside the box (bool ``(Z,Y,X)``),
    eroded by ``margin`` voxels for the trilinear footprint."""
    ones = np.ones(tuple(int(s) for s in shape), dtype=np.float32)
    fwd = apply_rotation(ones, R)
    back = apply_rotation(fwd, inverse(R))
    m = back > 0.999
    if margin > 0:
        m = ndi.binary_erosion(m, iterations=int(margin), border_value=0)
    return m


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _np(v: Any) -> np.ndarray:
    return v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)


def _dist_stats(d: np.ndarray) -> dict[str, float | int | None]:
    d = np.asarray(d, dtype=np.float64).ravel()
    if d.size == 0:
        return {"n": 0, "median": None, "p90": None, "frac_gt3": None, "frac_gt5": None, "mean": None}
    return {"n": int(d.size), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "frac_gt3": float((d > 3).mean()), "frac_gt5": float((d > 5).mean()), "mean": float(d.mean())}


def surface_distances(ma: np.ndarray, mb: np.ndarray) -> dict[str, Any]:
    """Distances (voxels) between two thin surfaces via EDT, both directions + symmetric."""
    ma, mb = np.asarray(ma, dtype=bool), np.asarray(mb, dtype=bool)
    out: dict[str, Any] = {"n_a": int(ma.sum()), "n_b": int(mb.sum())}
    dab = _edt(~mb)[ma] if mb.any() else np.full(int(ma.sum()), np.inf)
    dba = _edt(~ma)[mb] if ma.any() else np.full(int(mb.sum()), np.inf)
    out["a_to_b"] = _dist_stats(dab)
    out["b_to_a"] = _dist_stats(dba)
    out["symmetric"] = _dist_stats(np.concatenate([dab, dba]))
    return out


def _thin_zero_crossing(sdf: np.ndarray, data: np.ndarray) -> np.ndarray:
    """1-voxel surface: data voxels with sdf <= 0 that have a 6-neighbour with sdf > 0."""
    neg = (sdf <= 0) & data
    pos = (sdf > 0) & data
    any6 = ndi.binary_dilation(pos, structure=ndi.generate_binary_structure(3, 1))
    return neg & any6


def average_precision(score: np.ndarray, target: np.ndarray) -> float | None:
    """Area under the precision-recall curve (step-wise, like sklearn's ``average_precision_score``).

    This is *the* average-precision implementation of the project; :func:`tsm.train.average_precision`
    is a thin wrapper returning ``nan`` instead of ``None``.  Equal scores form **one** threshold (a
    model cannot separate voxels it scored identically), so the result never depends on the order in
    which voxels are traversed: four tied scores with two positives give ``0.5`` for every label
    permutation.

    Degenerate inputs return ``None`` and never raise: empty input, no positive, or no negative --
    average precision is undefined without both classes.  ``score`` and ``target`` must have the same
    number of elements.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=bool).ravel()
    if s.size != t.size:
        raise ValueError(f"score and target must have the same size, got {s.size} and {t.size}")
    n_pos = int(t.sum())
    if t.size == 0 or n_pos == 0 or n_pos == t.size:
        return None
    order = np.argsort(-s, kind="stable")
    s, t = s[order], t[order]
    last = np.flatnonzero(np.concatenate([np.diff(s) != 0, [True]]))  # last index of each tie group
    tp = np.cumsum(t)[last]
    precision = tp / (last + 1.0)
    recall = tp / n_pos
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def _angle_deg(na: np.ndarray, nb: np.ndarray, sign_agnostic: bool) -> np.ndarray:
    dot = (na * nb).sum(0) / (np.linalg.norm(na, axis=0) * np.linalg.norm(nb, axis=0) + 1e-12)
    if sign_agnostic:
        dot = np.abs(dot)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _ang_stats(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "frac_gt10": None, "frac_gt30": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "frac_gt10": float((a > 10).mean()), "frac_gt30": float((a > 30).mean())}


def _mask_of(mask: Any, shape: Sequence[int]) -> np.ndarray:
    return np.ones(tuple(shape), dtype=bool) if mask is None else np.asarray(_np(mask) > 0.5, dtype=bool).reshape(tuple(shape))


def surface_metrics(pa: Any, pb: Any, mask: Any = None, thr: float = 0.5, min_component: int = 100) -> dict[str, Any]:
    pa, pb = _np(pa).reshape(_np(pa).shape[-3:]), _np(pb).reshape(_np(pb).shape[-3:])
    m = _mask_of(mask, pa.shape)
    a, b = (pa >= thr) & m, (pb >= thr) & m
    inter = int((a & b).sum())
    na, nb = int(a.sum()), int(b.sum())
    out: dict[str, Any] = {
        "dice": (2.0 * inter / (na + nb)) if na + nb else 1.0,
        "mean_abs": float(np.abs(pa - pb)[m].mean()) if m.any() else 0.0,
        "max_abs": float(np.abs(pa - pb)[m].max()) if m.any() else 0.0,
        "n_a": na, "n_b": nb,
    }
    ma = medial_surface(_drop_small_components(a, min_component)) & m
    mb = medial_surface(_drop_small_components(b, min_component)) & m
    out["medial"] = surface_distances(ma, mb)
    return out


def ink_metrics(pa: Any, pb: Any, mask: Any = None, thr: float = 0.5) -> dict[str, Any]:
    pa, pb = _np(pa).reshape(_np(pa).shape[-3:]), _np(pb).reshape(_np(pb).shape[-3:])
    m = _mask_of(mask, pa.shape)
    a, b = (pa >= thr) & m, (pb >= thr) & m
    inter = int((a & b).sum())
    na, nb = int(a.sum()), int(b.sum())
    ap_ab = average_precision(pb[m], a[m])
    ap_ba = average_precision(pa[m], b[m])
    aps = [v for v in (ap_ab, ap_ba) if v is not None]
    return {
        "dice": (2.0 * inter / (na + nb)) if na + nb else 1.0,
        "mean_abs": float(np.abs(pa - pb)[m].mean()) if m.any() else 0.0,
        "n_a": na, "n_b": nb,
        "ap_b_vs_a": ap_ab, "ap_a_vs_b": ap_ba, "auprc": (float(np.mean(aps)) if aps else None),
    }


def lasagna_metrics(a: Any, b: Any, mask: Any = None, cos_min: float = 0.9) -> dict[str, Any]:
    a, b = _np(a), _np(b)
    a, b = a.reshape((8,) + a.shape[-3:]), b.reshape((8,) + b.shape[-3:])
    m = _mask_of(mask, a.shape[-3:])
    out: dict[str, Any] = {
        "cos_mae": float(np.abs(a[0] - b[0])[m].mean()) if m.any() else 0.0,
        "grad_mag_mae": float(np.abs(a[1] - b[1])[m].mean()) if m.any() else 0.0,
        "dir_mae": float(np.abs(a[2:] - b[2:])[:, m].mean()) if m.any() else 0.0,
    }
    sel = m & (a[0] > cos_min) & (b[0] > cos_min)
    if sel.any():
        na = decode_lasagna_normal(a[:, sel][:, :, None, None])[:, :, 0, 0]
        nb = decode_lasagna_normal(b[:, sel][:, :, None, None])[:, :, 0, 0]
        out["normal_angle"] = _ang_stats(_angle_deg(na, nb, sign_agnostic=True))
    else:
        out["normal_angle"] = _ang_stats(np.zeros(0))
    out["normal_angle"]["cos_min"] = cos_min
    return out


def student_metrics(a: Any, b: Any, mask: Any = None, clip: float = 20.0) -> dict[str, Any]:
    """Activated student heads (11, Z, Y, X): sdf, valid, ink, sin, cos, density, nx, ny, nz, conf, spare."""
    a, b = _np(a), _np(b)
    a, b = a.reshape((-1,) + a.shape[-3:]), b.reshape((-1,) + b.shape[-3:])
    m = _mask_of(mask, a.shape[-3:])
    band = m & (np.abs(a[0]) < clip) & (np.abs(b[0]) < clip)
    valid = band & (a[1] > 0.5) & (b[1] > 0.5)
    out: dict[str, Any] = {"n_band": int(band.sum()), "n_valid": int(valid.sum())}
    out["sdf_mae_band"] = float(np.abs(a[0] - b[0])[band].mean()) if band.any() else 0.0
    out["valid_mean_abs"] = float(np.abs(a[1] - b[1])[m].mean()) if m.any() else 0.0
    ia, ib = (a[2] >= 0.5) & m, (b[2] >= 0.5) & m
    out["ink_dice"] = (2.0 * float((ia & ib).sum()) / float(ia.sum() + ib.sum())) if (ia.sum() + ib.sum()) else 1.0
    sa = _thin_zero_crossing(a[0], m & (a[1] > 0.5))
    sb = _thin_zero_crossing(b[0], m & (b[1] > 0.5))
    out["zero_crossing"] = surface_distances(sa, sb)
    if valid.any():
        pha = np.arctan2(a[3], a[4])[valid]
        phb = np.arctan2(b[3], b[4])[valid]
        d = np.abs(np.angle(np.exp(1j * (pha - phb))))
        out["phase_err_deg"] = {"mean": float(np.degrees(d).mean()), "median": float(np.degrees(np.median(d)))}
        out["density_mae"] = float(np.abs(a[5] - b[5])[valid].mean())
        out["normal_angle"] = _ang_stats(_angle_deg(a[6:9][:, valid], b[6:9][:, valid], sign_agnostic=False))
        out["conf_mean_abs"] = float(np.abs(a[9] - b[9])[valid].mean())
    else:
        out["phase_err_deg"] = {"mean": None, "median": None}
        out["density_mae"] = None
        out["normal_angle"] = _ang_stats(np.zeros(0))
        out["conf_mean_abs"] = None
    return out


def metrics(pred_identity: Any, pred_g_inverted: Any, kind: str, mask: Any = None, **kw: Any) -> dict[str, Any]:
    """Disagreement between the identity prediction and a transformed-and-inverted one.

    ``kind``: ``'surface'`` (recto / m7 probability), ``'ink'``, ``'lasagna'`` (8 ch) or
    ``'student'`` (11 activated heads, physical units).  ``mask`` restricts every statistic
    (e.g. the valid region of an arbitrary rotation).  Identical inputs give zero
    disagreement (dice 1, distances 0, angles 0).
    """
    if kind == "surface":
        return surface_metrics(pred_identity, pred_g_inverted, mask, **kw)
    if kind == "ink":
        return ink_metrics(pred_identity, pred_g_inverted, mask, **kw)
    if kind == "lasagna":
        return lasagna_metrics(pred_identity, pred_g_inverted, mask, **kw)
    if kind == "student":
        return student_metrics(pred_identity, pred_g_inverted, mask, **kw)
    raise ValueError(f"unknown kind {kind!r}")


def primary_metric(kind: str, m: dict[str, Any]) -> tuple[str, float]:
    """One headline disagreement number per kind (0 = perfect agreement)."""
    if kind in ("surface", "ink"):
        return "1-dice", 1.0 - float(m["dice"])
    if kind == "lasagna":
        v = m["normal_angle"]["mean"]
        return "normal_angle_mean_deg", float(v if v is not None else 0.0)
    if kind == "student":
        return "sdf_mae_band", float(m["sdf_mae_band"])
    raise ValueError(kind)
