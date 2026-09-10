"""Cross-frame resampling: upstream label volumes onto our image grid.

The registration tooling ships a ``transform.json`` (schema 1.0.0) next to the
*moving* image volume.  Its ``transformation_matrix`` is a 3x4 homogeneous
matrix in **XYZ** order (point = ``[x, y, z, 1]``, both sides in **level-0
voxel indices**, not micrometres) that maps

    p_fixed = M @ p_moving

``fixed_volume`` names the frame the labels were authored in (for Paris 4 that
is ``PHercParis4-20230205180739_masked``, the older ~7.91 um scan), and the
volume the file sits next to is the moving frame (our 2.4 um scan).  So the
labels live in the fixed frame and

    label_xyz = M @ image_xyz          (forward, used to resample labels)
    image_xyz = inv(M) @ label_xyz     (inverse)

The scale is baked into ``M`` -- its linear block has singular values ~0.3066,
i.e. 2.4/7.83, so no separate voxel-size factor is applied anywhere.  Our zarr
arrays are indexed ZYX, so every matrix exposed here is conjugated by the
axis-reversal permutation ``S`` (``S @ M @ S.T``, ``S`` self-inverse).  Voxel
index ``i`` denotes the centre of voxel ``i`` in both frames, so mapping a
voxel centre is just applying the matrix to the integer index -- there is no
half-voxel offset (the convention villa's ``scipy.ndimage.map_coordinates``
call also assumes).

Reimplemented here so nothing imports villa at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = "1.0.0"

#: Homogeneous permutation swapping XYZ <-> ZYX ordering (self-inverse).
SWAP = np.array(
    [[0, 0, 1, 0],
     [0, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]], dtype=np.float64,
)


@dataclass(frozen=True)
class Transform:
    """Parsed ``transform.json``."""

    matrix_xyz: np.ndarray          # 4x4, label_xyz = M @ image_xyz
    fixed_volume: str
    fixed_landmarks: list
    moving_landmarks: list
    checksum: str

    @property
    def image_to_label_zyx(self) -> np.ndarray:
        return image_to_label_matrix_zyx(self.matrix_xyz)

    @property
    def label_to_image_zyx(self) -> np.ndarray:
        return label_to_image_matrix_zyx(self.matrix_xyz)


def _read_bytes(path_or_url: str) -> bytes:
    text = str(path_or_url)
    if text.startswith(("http://", "https://", "s3://")):
        import fsspec

        opts: dict[str, Any] = {"anon": True} if text.startswith("s3://") else {}
        with fsspec.open(text, mode="rb", **opts) as f:
            return f.read()
    return Path(text).read_bytes()


def matrix_checksum(matrix: np.ndarray) -> str:
    """Short stable digest of a matrix, for provenance in output attrs."""
    return hashlib.sha1(
        np.ascontiguousarray(matrix, dtype=np.float64).tobytes()).hexdigest()[:16]


def read_transform(url_or_path: str) -> Transform:
    """Load a ``transform.json`` from a local path, http(s) URL or s3 URI."""
    data = json.loads(_read_bytes(url_or_path).decode("utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported transform.json schema_version {version!r}, "
                         f"expected {SCHEMA_VERSION!r}")
    for key in ("fixed_volume", "transformation_matrix"):
        if key not in data:
            raise ValueError(f"transform.json missing required key {key!r}")
    m34 = np.asarray(data["transformation_matrix"], dtype=np.float64)
    if m34.shape != (3, 4):
        raise ValueError(f"transformation_matrix must be 3x4, got {m34.shape}")
    m44 = np.vstack([m34, [0.0, 0.0, 0.0, 1.0]])
    return Transform(
        matrix_xyz=m44,
        fixed_volume=str(data["fixed_volume"]),
        fixed_landmarks=[list(map(float, p)) for p in data.get("fixed_landmarks", [])],
        moving_landmarks=[list(map(float, p)) for p in data.get("moving_landmarks", [])],
        checksum=matrix_checksum(m44),
    )


def swap_xyz_zyx(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous affine between XYZ and ZYX point ordering."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {m.shape}")
    return SWAP @ m @ SWAP.T


def image_to_label_matrix_zyx(matrix_xyz: np.ndarray) -> np.ndarray:
    """ZYX affine mapping image voxel indices to label voxel indices."""
    return swap_xyz_zyx(matrix_xyz)


def label_to_image_matrix_zyx(matrix_xyz: np.ndarray) -> np.ndarray:
    """ZYX affine mapping label voxel indices to image voxel indices."""
    m = np.asarray(matrix_xyz, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {m.shape}")
    return swap_xyz_zyx(np.linalg.inv(m))


def apply_affine_zyx(matrix_zyx: np.ndarray, points_zyx: np.ndarray) -> np.ndarray:
    """Apply a 4x4 ZYX affine to an ``(N, 3)`` array of ZYX points."""
    m = np.asarray(matrix_zyx, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {m.shape}")
    pts = np.asarray(points_zyx, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    return pts @ m[:3, :3].T + m[:3, 3]


def level_scale_matrix(level: int) -> np.ndarray:
    """ZYX affine from level-0 voxel indices to level-``level`` indices.

    Pixel centres: ``c_l = (c_0 + 0.5) / 2**l - 0.5``.
    """
    s = 1.0 / float(2 ** int(level))
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = m[1, 1] = m[2, 2] = s
    m[:3, 3] = 0.5 * s - 0.5
    return m


def label_aabb_for_image_box(
    matrix_image_to_label_zyx: np.ndarray,
    start_zyx: Sequence[int],
    size_zyx: Sequence[int],
    label_shape_zyx: Sequence[int] | None = None,
    margin: int = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Label-frame AABB (start, stop) enclosing an image-frame box.

    ``stop`` is exclusive.  The box spans image voxel *centres* ``start`` ..
    ``start + size - 1``; ``margin`` voxels of slack are added on each side so
    rounding never reaches outside the fetched slab.  Clipped to
    ``[0, label_shape_zyx)`` when that is given.
    """
    p = np.asarray(start_zyx, dtype=np.float64)
    d = np.asarray(size_zyx, dtype=np.float64) - 1.0
    if np.any(d < 0):
        raise ValueError(f"size_zyx must be positive, got {tuple(size_zyx)}")
    offsets = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)],
                       dtype=np.float64)
    corners = p + offsets * d
    mapped = apply_affine_zyx(matrix_image_to_label_zyx, corners)
    lo = np.floor(mapped.min(axis=0)).astype(np.int64) - int(margin)
    hi = np.ceil(mapped.max(axis=0)).astype(np.int64) + 1 + int(margin)
    if label_shape_zyx is not None:
        shape = np.asarray(label_shape_zyx, dtype=np.int64)
        lo = np.clip(lo, 0, shape)
        hi = np.clip(hi, 0, shape)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]


def open_label_array(url_or_path: str, level: int = 0):
    """Open level ``level`` of an OME-Zarr (or a plain zarr array) read-only."""
    import zarr

    text = str(url_or_path).rstrip("/")
    if text.startswith("s3://"):
        import s3fs

        store: Any = s3fs.S3Map(root=text, s3=s3fs.S3FileSystem(anon=True), check=False)
        root = zarr.open_group(store=store, mode="r")
        return root[str(level)]
    try:
        root = zarr.open_group(store=text, mode="r")
    except Exception:
        return zarr.open_array(store=text, mode="r")
    if str(level) in root:
        return root[str(level)]
    return zarr.open_array(store=text, mode="r")


def _nearest_gather(
    slab: np.ndarray,
    slab_start: Sequence[int],
    matrix: np.ndarray,
    start_zyx: Sequence[int],
    size_zyx: Sequence[int],
    fill: int,
) -> np.ndarray:
    """Nearest-neighbour gather of ``slab`` onto an image-grid box.

    ``matrix`` maps image voxel indices to label voxel indices in the same
    (level) frame ``slab`` is indexed in; ``slab`` covers label voxels
    ``slab_start ... slab_start + slab.shape``.
    """
    a = np.asarray(matrix, dtype=np.float64)
    lin, t = a[:3, :3], a[:3, 3]
    p0 = np.asarray(start_zyx, dtype=np.float64)
    dz, dy, dx = (int(v) for v in size_zyx)
    base = lin @ p0 + t - np.asarray(slab_start, dtype=np.float64)

    az = np.arange(dz, dtype=np.float64)
    ay = np.arange(dy, dtype=np.float64)
    ax = np.arange(dx, dtype=np.float64)

    sh = slab.shape
    idx = None
    ok = None
    strides = (sh[1] * sh[2], sh[2], 1)
    for k in range(3):
        c = (base[k]
             + lin[k, 0] * az[:, None, None]
             + lin[k, 1] * ay[None, :, None]
             + lin[k, 2] * ax[None, None, :])
        ck = np.rint(c).astype(np.int32, copy=False)
        del c
        good = (ck >= 0) & (ck < sh[k])
        np.clip(ck, 0, sh[k] - 1, out=ck)
        contrib = ck.astype(np.int64) * strides[k]
        if idx is None:
            idx, ok = contrib, good
        else:
            idx += contrib
            ok &= good
        del ck, good
    out = slab.reshape(-1)[idx]
    if not ok.all():
        out = np.where(ok, out, np.asarray(fill, dtype=slab.dtype))
    return np.ascontiguousarray(out.reshape(dz, dy, dx))


def resample_labels_to_image_box(
    label_zarr_url,
    level: int,
    matrix_image_to_label_zyx: np.ndarray,
    start_zyx: Sequence[int],
    size_zyx: Sequence[int],
    *,
    fill: int = 0,
    sub_box: Sequence[int] = (128, 128, 128),
    max_read_bytes: int = 1 << 30,
    margin: int = 1,
) -> np.ndarray:
    """Nearest-neighbour resample of a label volume onto an image-grid box.

    ``label_zarr_url`` may be a URL/path (OME-Zarr group or plain array) or an
    already-open array-like.  ``matrix_image_to_label_zyx`` is the *level-0*
    image->label affine; the level-``level`` scaling is composed in here.

    Reads the label AABB of the whole box in one request when it fits inside
    ``max_read_bytes``, otherwise splits the image box along its longest axis
    and recurses.  The gather itself runs over ``sub_box``-sized tiles so the
    coordinate temporaries stay small.  Returns the label dtype (uint8).
    """
    arr = (open_label_array(label_zarr_url, level)
           if isinstance(label_zarr_url, (str, Path)) else label_zarr_url)
    matrix = level_scale_matrix(level) @ np.asarray(matrix_image_to_label_zyx, dtype=np.float64)
    return _resample(arr, matrix, tuple(int(v) for v in start_zyx),
                     tuple(int(v) for v in size_zyx), fill, tuple(int(v) for v in sub_box),
                     int(max_read_bytes), int(margin))


def _resample(arr, matrix, start, size, fill, sub_box, max_read_bytes, margin) -> np.ndarray:
    shape = tuple(int(v) for v in arr.shape[-3:])
    lo, hi = label_aabb_for_image_box(matrix, start, size, label_shape_zyx=shape, margin=margin)
    span = [max(0, b - a) for a, b in zip(lo, hi)]
    nbytes = int(np.prod(span)) * int(np.dtype(arr.dtype).itemsize)

    if nbytes > max_read_bytes and int(np.prod(size)) > int(np.prod(sub_box)):
        axis = int(np.argmax(size))
        half = size[axis] // 2
        if half >= 1:
            s1 = list(size); s1[axis] = half
            s2 = list(size); s2[axis] = size[axis] - half
            p2 = list(start); p2[axis] += half
            a = _resample(arr, matrix, start, tuple(s1), fill, sub_box, max_read_bytes, margin)
            b = _resample(arr, matrix, tuple(p2), tuple(s2), fill, sub_box, max_read_bytes, margin)
            return np.concatenate([a, b], axis=axis)

    dtype = np.dtype(arr.dtype)
    if min(span) <= 0:  # box maps entirely outside the label volume
        return np.full(size, fill, dtype=dtype)
    slab = np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])

    out = np.empty(size, dtype=dtype)
    for oz in range(0, size[0], sub_box[0]):
        for oy in range(0, size[1], sub_box[1]):
            for ox in range(0, size[2], sub_box[2]):
                off = (oz, oy, ox)
                sz = tuple(min(sb, s - o) for sb, s, o in zip(sub_box, size, off))
                tile = _nearest_gather(slab, lo, matrix,
                                       tuple(s + o for s, o in zip(start, off)), sz, fill)
                out[oz:oz + sz[0], oy:oy + sz[1], ox:ox + sz[2]] = tile
    return out


__all__ = [
    "SCHEMA_VERSION",
    "SWAP",
    "Transform",
    "read_transform",
    "matrix_checksum",
    "swap_xyz_zyx",
    "image_to_label_matrix_zyx",
    "label_to_image_matrix_zyx",
    "apply_affine_zyx",
    "level_scale_matrix",
    "label_aabb_for_image_box",
    "open_label_array",
    "resample_labels_to_image_box",
]
