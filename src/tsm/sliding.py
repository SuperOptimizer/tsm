"""Sliding-window brick inference: gaussian-blended windows over output tiles.

Per output tile (a near-equal core of at most ``out_tile``, expanded by ``halo`` on every side into an
input box that is read once, zero padded) the box is normalised on the GPU,
covered by an evenly spaced grid of ``patch`` windows (last window aligned to
the box end, nnU-Net style), and the logits are accumulated with a gaussian
weight into ``acc`` / ``wsum`` (fp16 on the device, asserted below a byte
limit -- the tile shrinks otherwise).  After division the core is cropped, the
teacher activation applied, quantised to uint8 and handed to a ``BrickWriter``.
Tiles already present in the writer are skipped (resume).

Coordinates (region, writer) are in the voxel grid of the zarr level being read.

Throughput knobs (``WindowSpec``): windows whose input is all air
(``max <= empty_window_max``) are skipped; the net and window inputs run in
``channels_last_3d`` with ``cudnn.benchmark`` (shapes are fixed per teacher);
optional ``torch.compile`` (last chunk padded to the fixed batch); the next
tile's box is prefetched in a background thread so the S3 read overlaps the GPU;
``backend="trt"`` runs a :class:`tsm.trt.TRTTeacher` / :class:`tsm.trt.TRTStudent`
built by the caller.

``weight="uniform"`` switches the gaussian blend for the receptive-field halo tiling:
with ``step = patch - 2*halo`` the window cores tile the box, every core voxel comes
from exactly one window (weight 1) and the halos (weight 0) only exist to give those
core voxels their full receptive field.

Nets exposing ``submit(x) -> handle`` / ``wait(handle) -> out`` (the TRT wrapper on
its own CUDA stream) are driven software-pipelined: pass ``j+1`` is sliced,
normalised and submitted before pass ``j`` is waited for and blended, so the
engine overlaps the torch-side work.

Test-time augmentation (``tta``): ``"none"``, ``"flip8"`` (the 8 axis flips),
``"flip4rot2"`` (8 passes: single-axis flips x in-plane rot90 k in {0,1}) and
``"flip8_rot4"`` (flips x rot90 about z; the 32 combinations contain every element
of the 16-element symmetry group twice, so the 16 distinct ones are run).  Passes
are averaged in pre-activation space.  ``tta_field="lasagna"`` transforms the 6
encoded direction channels back into the original frame (see
:func:`transform_lasagna_dirs`).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tsm.config import RegionCfg
from tsm.limits import Budget, check_alloc, free_cuda, nbytes
from tsm.teachers import apply_activation
from tsm.volume import BrickWriter, VolumeReader

MB = 1 << 20
ACC_LIMIT_BYTES = 1536 * 1024 * 1024  # acc+wsum on GPU; 16 GB card, teachers peak ~9 GB
MIN_OUT_TILE = 32

__all__ = [
    "WindowSpec",
    "TTA_MODES",
    "Xform",
    "tta_transforms",
    "transform_lasagna_dirs",
    "TilePlan",
    "gaussian_weight",
    "uniform_core_weight",
    "window_weight",
    "window_starts",
    "plan_tiles",
    "plan_bytes",
    "prepare_net",
    "predict_box",
    "run_sliding",
]


@dataclass
class WindowSpec:
    patch: int
    step: int | None = None  # default patch // 2
    out_tile: int = 192
    halo: int | None = None  # default patch - step
    batch: int = 1
    tta: str = "none"  # "none" | "flip8" | "flip4rot2" | "flip8_rot4" (bool: False -> none, True -> flip8)
    tta_field: str = "scalar"  # "scalar" (per-channel invariant) | "lasagna" (8-ch cos/grad/dir encoding)
    dtype: Any = torch.bfloat16
    empty_max: int = 0  # input box with max <= empty_max is written as zeros
    norm_scope: str = "window"  # "window" (per patch, as trained) | "box" (once per input box)
    # window-level air skipping: a window whose input max <= empty_window_max is
    # not run and contributes nothing (voxels covered only by skipped windows are
    # 0).  Safe on masked volumes (0 outside the scroll); -1 disables.
    empty_window_max: int = 0
    # GPU throughput knobs (no effect on the CPU path / non-Module nets)
    channels_last: bool = False  # net + window inputs in channels_last_3d (measured slower on RTX 5080: off)
    cudnn_benchmark: bool = True  # shapes are fixed per teacher
    compile: str | None = None  # None | "default" | "max-autotune-no-cudagraphs" | "reduce-overhead"
    prefetch: bool = True  # read the next tile's box in a background thread
    backend: str = "torch"  # "torch" | "trt" (net is then a tsm.trt.TRTTeacher built by the caller)
    # window blending weight: "gaussian" (nnU-Net importance map, for overlapping windows)
    # or "uniform" (1 inside the window core = the patch shrunk by ``halo`` on every side,
    # 0 in the halo).  "uniform" is what the receptive-field halo tiling wants: windows
    # only overlap in their halos, every core voxel is written by exactly one window, and
    # the blend is an unweighted mean where cores do overlap (2*halo < patch required).
    weight: str = "gaussian"

    def __post_init__(self) -> None:
        self.patch = int(self.patch)
        if self.patch <= 0:
            raise ValueError("patch must be positive")
        self.step = int(self.step) if self.step is not None else max(1, self.patch // 2)
        if not 0 < self.step <= self.patch:
            raise ValueError(f"step {self.step} must be in (0, patch={self.patch}]")
        self.halo = int(self.halo) if self.halo is not None else self.patch - self.step
        if self.halo < 0:
            raise ValueError("halo must be non-negative")
        self.out_tile = int(self.out_tile)
        if self.out_tile <= 0:
            raise ValueError("out_tile must be positive")
        self.batch = max(1, int(self.batch))
        self.empty_max = int(self.empty_max)
        self.empty_window_max = int(self.empty_window_max)
        if self.compile in ("", "none", "off", False, 0):
            self.compile = None
        if self.compile is not None:
            self.compile = str(self.compile)
            if self.compile not in ("default", "max-autotune-no-cudagraphs", "reduce-overhead", "max-autotune"):
                raise ValueError(f"unknown compile mode {self.compile!r}")
        if self.backend not in ("torch", "trt"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.weight not in ("gaussian", "uniform"):
            raise ValueError(f"unknown weight {self.weight!r}; known: ('gaussian', 'uniform')")
        if self.weight == "uniform":
            if 2 * self.halo >= self.patch:
                raise ValueError(f"weight 'uniform' needs 2*halo ({2 * self.halo}) < patch ({self.patch})")
            if self.step > self.patch - 2 * self.halo:
                # uniform weights are 0 in the halo, so consecutive window *cores* must touch or
                # overlap; a larger step leaves output voxels with zero accumulated weight.
                raise ValueError(
                    f"weight 'uniform' needs step <= patch - 2*halo: step {self.step} > "
                    f"{self.patch - 2 * self.halo} (patch {self.patch}, halo {self.halo}) leaves "
                    f"uncovered output voxels between the window cores")
        if isinstance(self.tta, bool) or self.tta is None:
            self.tta = "flip8" if self.tta else "none"
        self.tta = str(self.tta)
        if self.tta not in TTA_MODES:
            raise ValueError(f"unknown tta {self.tta!r}; known: {TTA_MODES}")
        if self.tta_field not in ("scalar", "lasagna"):
            raise ValueError(f"unknown tta_field {self.tta_field!r}")

    @property
    def n_passes(self) -> int:
        return len(tta_transforms(self.tta))

    @property
    def pad_batch(self) -> bool:
        """Compiled / TRT nets see one fixed [batch,1,p,p,p] shape: pad the last chunk."""
        return self.compile is not None or self.backend == "trt"


# --------------------------------------------------------------------------- #
# Gaussian importance map
# --------------------------------------------------------------------------- #
_GAUSS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def gaussian_weight(
    patch: int,
    sigma_scale: float = 1.0 / 8,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """nnU-Net style gaussian importance map [1,1,p,p,p]; max 1, min-clamped to 1e-3 of max."""
    dev = torch.device(device)
    key = (int(patch), float(sigma_scale), str(dev), str(dtype))
    w = _GAUSS_CACHE.get(key)
    if w is not None:
        return w
    p = int(patch)
    sigma = p * sigma_scale
    c = (p - 1) / 2.0
    ax = torch.arange(p, dtype=torch.float64)
    g1 = torch.exp(-0.5 * ((ax - c) / sigma) ** 2)
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    g = g / g.max()
    g = torch.clamp(g, min=1e-3)
    w = g.to(dtype=dtype, device=dev).reshape(1, 1, p, p, p).contiguous()
    _GAUSS_CACHE[key] = w
    return w


def uniform_core_weight(patch: int, halo: int, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[1,1,p,p,p] mask: 1 on the window core ``[halo, patch-halo)`` per axis, 0 in the halo.

    With ``step = patch - 2*halo`` the cores tile the box exactly, so every voxel is
    predicted by the one window that holds it at distance >= halo from its border (the
    receptive-field halo); where the last window's alignment makes two cores overlap the
    division by ``wsum`` turns the accumulator into their unweighted mean."""
    p, h = int(patch), int(halo)
    if 2 * h >= p:
        raise ValueError(f"uniform core weight needs 2*halo ({2 * h}) < patch ({p})")
    key = ("uniform", p, h, str(torch.device(device)), str(dtype))
    w = _GAUSS_CACHE.get(key)
    if w is not None:
        return w
    w = torch.zeros((1, 1, p, p, p), dtype=dtype, device=torch.device(device))
    w[:, :, h : p - h, h : p - h, h : p - h] = 1.0
    _GAUSS_CACHE[key] = w
    return w


def window_weight(spec: "WindowSpec", device: str | torch.device = "cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Blending weight for one window: gaussian importance map or the uniform core mask."""
    if spec.weight == "uniform":
        return uniform_core_weight(spec.patch, spec.halo, device=device, dtype=dtype)
    return gaussian_weight(spec.patch, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def window_starts(length: int, patch: int, step: int) -> list[int]:
    """Evenly spaced window starts covering [0, length); first at 0, last at length-patch."""
    if length < patch:
        raise ValueError(f"length {length} < patch {patch}")
    if length == patch:
        return [0]
    n = int(math.ceil((length - patch) / step)) + 1
    actual = (length - patch) / (n - 1)
    starts = [int(round(i * actual)) for i in range(n)]
    starts[-1] = length - patch
    return sorted(set(starts))


@dataclass(frozen=True)
class TilePlan:
    core0: tuple[int, int, int]  # absolute start of the core (level grid)
    core_shape: tuple[int, int, int]
    box0: tuple[int, int, int]  # absolute start of the input box
    box_shape: tuple[int, int, int]

    @property
    def core_offset(self) -> tuple[int, int, int]:
        return tuple(c - b for c, b in zip(self.core0, self.box0))  # type: ignore[return-value]

    def n_windows(self, patch: int, step: int) -> int:
        n = 1
        for L in self.box_shape:
            n *= len(window_starts(L, patch, step))
        return n


def _even_edges(size: int, out_tile: int) -> list[int]:
    """Boundaries of ``ceil(size/out_tile)`` near-equal cores covering ``[0, size)``.

    Every core is within one voxel of ``size / count``, hence never smaller than
    ``out_tile / 2`` (unless ``size < out_tile``, which gives a single core): the
    clipped-last-tile scheme's slivers -- cores covered by a single window, with no
    blending and a visible seam -- cannot occur.
    """
    count = max(1, (size + out_tile - 1) // out_tile)
    return [round(i * size / count) for i in range(count + 1)]


def plan_tiles(region: RegionCfg, spec: WindowSpec) -> list[TilePlan]:
    """Partition the region into near-equal cores of at most ``out_tile`` per axis; each
    box is core + halo, widened to at least ``patch`` on every axis (centred on the core).

    Per axis the count is ``ceil(size / out_tile)`` and the boundaries are
    ``round(i * size / count)``, so the cores tile the region exactly and their sizes
    differ by at most one voxel (no ``out_tile``-aligned starts, no small last tile)."""
    start = tuple(int(v) for v in region.start_zyx)
    size = tuple(int(v) for v in region.size_zyx)
    edges = [_even_edges(size[i], spec.out_tile) for i in range(3)]
    counts = [len(e) - 1 for e in edges]
    tiles: list[TilePlan] = []
    for iz in range(counts[0]):
        for iy in range(counts[1]):
            for ix in range(counts[2]):
                idx = (iz, iy, ix)
                core0 = [start[i] + edges[i][idx[i]] for i in range(3)]
                core_shape = [edges[i][idx[i] + 1] - edges[i][idx[i]] for i in range(3)]
                box0, box_shape = [], []
                for i in range(3):
                    b0 = core0[i] - spec.halo
                    L = core_shape[i] + 2 * spec.halo
                    if L < spec.patch:
                        extra = spec.patch - L
                        b0 -= extra // 2
                        L = spec.patch
                    box0.append(b0)
                    box_shape.append(L)
                tiles.append(TilePlan(tuple(core0), tuple(core_shape), tuple(box0), tuple(box_shape)))  # type: ignore[arg-type]
    return tiles


def _acc_channels(channels_out: int, activation: str, fg_channel: int | None) -> int:
    # 2-class softmax fg == sigmoid(l_fg - l_other): accumulating the logit
    # difference is exact and halves the accumulator.
    if activation == "softmax" and channels_out == 2 and fg_channel is not None:
        return 1
    return channels_out


def plan_bytes(
    spec: WindowSpec, region: RegionCfg, channels_out: int, activation: str = "none",
    fg_channel: int | None = None,
) -> tuple[list[tuple[str, tuple[int, ...], Any]], list[TilePlan]]:
    """Byte-table rows (name, shape, dtype) for the largest tile plus the tile list."""
    tiles = plan_tiles(region, spec)
    big = max(tiles, key=lambda tp: int(np.prod(tp.box_shape)))
    bz, by, bx = big.box_shape
    c_acc = _acc_channels(channels_out, activation, fg_channel)
    p = spec.patch
    rows = [
        ("input_box_u8 (host)", (bz, by, bx), np.uint8),
        ("input_box_f32 (gpu)", (1, 1, bz, by, bx), np.float32),
        ("acc_f16 (gpu)", (c_acc, bz, by, bx), np.float16),
        ("wsum_f16 (gpu)", (1, bz, by, bx), np.float16),
        ("window_in (gpu)", (spec.batch, 1, p, p, p), np.float32),
        ("window_out (gpu)", (spec.batch, channels_out, p, p, p), np.float32),
        ("core_out_u8 (host)", (channels_out,) + tuple(big.core_shape), np.uint8),
    ]
    return rows, tiles


def _acc_bytes(spec: WindowSpec, region: RegionCfg, c_acc: int) -> int:
    tiles = plan_tiles(region, spec)
    big = max(tiles, key=lambda tp: int(np.prod(tp.box_shape)))
    n = int(np.prod(big.box_shape))
    return (c_acc + 1) * n * 2


def fit_out_tile(spec: WindowSpec, region: RegionCfg, c_acc: int, limit: int = ACC_LIMIT_BYTES) -> WindowSpec:
    """Shrink ``out_tile`` (in steps of 32) until acc+wsum fit in ``limit`` bytes."""
    while _acc_bytes(spec, region, c_acc) > limit:
        nt = spec.out_tile - 32
        if nt < MIN_OUT_TILE:
            raise MemoryError(
                f"acc+wsum {_acc_bytes(spec, region, c_acc) / MB:.0f} MiB exceed {limit / MB:.0f} MiB "
                f"even at out_tile {spec.out_tile} (patch {spec.patch}, halo {spec.halo})"
            )
        spec.out_tile = nt
    return spec


# --------------------------------------------------------------------------- #
# Inference on one box
# --------------------------------------------------------------------------- #
TTA_MODES = ("none", "flip8", "flip4rot2", "flip8_rot4")
_ALL_FLIPS: list[tuple[int, ...]] = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
# geometric action of torch.rot90(x, 1, dims=(3, 4)) ((y, x) plane, "from y towards x")
# on a vector's (x, y, z) components: a +y vector becomes +x, a +x vector becomes -y.
_ROT1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.int64)


@dataclass(frozen=True)
class Xform:
    """One TTA pass: ``x -> rot90(flip(x, flips), k, dims=(3, 4))`` on [B, C, Z, Y, X]."""

    flips: tuple[int, ...] = ()
    k: int = 0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.flips:
            x = torch.flip(x, self.flips)
        if self.k:
            x = torch.rot90(x, self.k, dims=(3, 4))
        return x

    def invert(self, y: torch.Tensor) -> torch.Tensor:
        if self.k:
            y = torch.rot90(y, -self.k, dims=(3, 4))
        if self.flips:
            y = torch.flip(y, self.flips)
        return y

    def matrix(self) -> np.ndarray:
        """Signed permutation on (x, y, z) vector components induced by ``apply``."""
        f = np.diag([-1 if 4 in self.flips else 1, -1 if 3 in self.flips else 1, -1 if 2 in self.flips else 1])
        m = f
        for _ in range(self.k % 4):
            m = _ROT1 @ m
        return m

    @property
    def label(self) -> str:
        ax = "".join("zyx"[d - 2] for d in self.flips) or "-"
        return f"f{ax}r{self.k}"


def _xform_signature(xf: Xform) -> tuple[int, ...]:
    probe = torch.arange(2 * 4 * 4, dtype=torch.int64).reshape(1, 1, 2, 4, 4)
    return tuple(int(v) for v in xf.apply(probe).reshape(-1).tolist())


_TTA_CACHE: dict[str, list[Xform]] = {}


def tta_transforms(mode: str | bool | None) -> list[Xform]:
    """Distinct passes of a TTA mode (identity first)."""
    if isinstance(mode, bool) or mode is None:
        mode = "flip8" if mode else "none"
    if mode not in TTA_MODES:
        raise ValueError(f"unknown tta {mode!r}; known: {TTA_MODES}")
    if mode in _TTA_CACHE:
        return _TTA_CACHE[mode]
    if mode == "none":
        cands = [Xform()]
    elif mode == "flip8":
        cands = [Xform(f, 0) for f in _ALL_FLIPS]
    elif mode == "flip4rot2":
        cands = [Xform(f, k) for k in (0, 1) for f in [(), (2,), (3,), (4,)]]
    else:  # flip8_rot4: 8 flips x 4 rotations = the 16-element group, each element twice
        cands = [Xform(f, k) for k in range(4) for f in _ALL_FLIPS]
    seen: set[tuple[int, ...]] = set()
    out: list[Xform] = []
    for xf in cands:
        sig = _xform_signature(xf)
        if sig not in seen:
            seen.add(sig)
            out.append(xf)
    _TTA_CACHE[mode] = out
    return out


# lasagna direction planes: channel pair index -> (first axis, second axis) in (x, y, z) = (0, 1, 2)
# ordering cos, grad_mag, dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x  (tsm.labels.decode_lasagna_normal)
_LASAGNA_PLANES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
_SQRT2 = math.sqrt(2.0)


def transform_lasagna_dirs(y: torch.Tensor, m: np.ndarray) -> torch.Tensor:
    """Re-encode lasagna's 6 direction channels of ``y`` [B, 8, ...] for normals ``n = m @ n'``.

    ``m`` is a signed permutation of (x, y, z) (flips / rot90).  Each plane's channels
    ``dir0 = 0.5 + 0.5 cos 2t``, ``dir1 = 0.5 + 0.5 (cos 2t - sin 2t) / sqrt 2`` encode
    the line angle ``t = atan2(n_b, n_a)`` of plane ``(a, b)``; a signed permutation
    maps every plane onto a plane with ``cos 2t -> +-cos 2t`` (negated when the two
    axes swap order) and ``sin 2t -> s_a s_b sin 2t``.  Channels 0..1 are copied.
    Works on raw (pre-clamp) outputs: the map is affine per pair.
    """
    if y.shape[1] != 8:
        raise ValueError(f"expected 8 lasagna channels, got {y.shape[1]}")
    m = np.asarray(m, dtype=np.int64)
    if m.shape != (3, 3) or not np.array_equal(np.abs(m).sum(0), [1, 1, 1]) or not np.array_equal(np.abs(m).sum(1), [1, 1, 1]):
        raise ValueError(f"not a signed permutation matrix: {m.tolist()}")
    perm = [int(np.nonzero(m[a])[0][0]) for a in range(3)]  # n_a = sign_a * n'_perm[a]
    sign = [int(m[a, perm[a]]) for a in range(3)]
    src_index = {pl: i for i, pl in enumerate(_LASAGNA_PLANES)}
    cs = []
    for (a, b) in _LASAGNA_PLANES:
        pa, pb = perm[a], perm[b]
        swapped = pa > pb
        i = src_index[(pb, pa) if swapped else (pa, pb)]
        d0, d1 = y[:, 2 + 2 * i], y[:, 3 + 2 * i]
        c = 2.0 * d0 - 1.0
        s = c - _SQRT2 * (2.0 * d1 - 1.0)
        if swapped:
            c = -c
        if sign[a] * sign[b] < 0:
            s = -s
        cs.append((c, s))
    parts = [y[:, 0:2]]
    for c, s in cs:
        parts.append(torch.stack([0.5 + 0.5 * c, 0.5 + 0.5 * (c - s) / _SQRT2], dim=1))
    return torch.cat(parts, dim=1)


def _invert_field(y: torch.Tensor, xf: Xform, field: str) -> torch.Tensor:
    """Bring one pass's output back into the original frame (spatially and component-wise)."""
    y = xf.invert(y)
    if field == "lasagna" and (xf.flips or xf.k):
        y = transform_lasagna_dirs(y, xf.matrix().T)  # n = M^T n' for a signed permutation
    return y


def _window_maxes(x: torch.Tensor, starts: list[tuple[int, int, int]], p: int) -> list[int]:
    """Per-window input max (one device sync for all windows)."""
    m = torch.stack([x[:, :, z : z + p, y : y + p, xx : xx + p].amax() for (z, y, xx) in starts])
    return [int(v) for v in m.tolist()]


def _assert_core_covered(spec: "WindowSpec", box_shape: Sequence[int], core_offset: Sequence[int],
                         core_shape: Sequence[int]) -> None:
    """Geometric check (data independent): every output core voxel is inside some window's weight
    support.  Only ``weight="uniform"`` can leave holes -- the gaussian map is positive everywhere."""
    if spec.weight != "uniform":
        return
    p, s, h = spec.patch, spec.step, spec.halo
    for a, (dim, off, n) in enumerate(zip(box_shape, core_offset, core_shape)):
        cov = np.zeros(int(dim), bool)
        for st in window_starts(int(dim), p, s):
            cov[st + h : st + p - h] = True
        miss = int((~cov[int(off) : int(off) + int(n)]).sum())
        if miss:
            raise ValueError(
                f"weight 'uniform' leaves {miss} uncovered voxels on axis {a} of the output core "
                f"(box {tuple(box_shape)}, core {tuple(core_shape)} at {tuple(core_offset)}, "
                f"patch {p}, step {s}, halo {h})")


def predict_box(
    box_u8: np.ndarray,
    net: Callable[[torch.Tensor], Any],
    normalizer: Callable[[torch.Tensor], torch.Tensor],
    spec: WindowSpec,
    channels_out: int,
    activation: str,
    fg_channel: int | None,
    device: torch.device,
    select: Callable[[Any], torch.Tensor] | None = None,
    core_offset: Sequence[int] = (0, 0, 0),
    core_shape: Sequence[int] | None = None,
    box_origin_zyx: Sequence[int] = (0, 0, 0),
) -> tuple[np.ndarray, int]:
    """Blend windows over ``box_u8`` (Z,Y,X); return (uint8 [C_out, core], n_windows).

    ``C_out`` is 1 (fg channel) for softmax/sigmoid single-target teachers and
    ``channels_out`` otherwise.  Windows whose input max is <= ``spec.empty_window_max``
    are skipped (not counted in ``n_windows``); voxels no window covers are 0.

    A net that sets ``needs_box_origin = True`` (``infer.StudentNet`` with the radial
    input channels) is called as ``net(xin, box_origin_zyx=[(z, y, x), ...])`` with the
    absolute origin of every window in the batch, derived from ``box_origin_zyx`` (this
    box's absolute origin in the level grid).  Such a net is incompatible with sliding's
    own TTA, which would move the grid under it.
    """
    select = select or (lambda o: o)
    bz, by, bx = box_u8.shape
    p, s = spec.patch, spec.step
    if min(box_u8.shape) < p:
        raise ValueError(f"box {box_u8.shape} smaller than patch {p}")
    core_shape = tuple(core_shape) if core_shape is not None else box_u8.shape
    cz, cy, cx = (int(v) for v in core_offset)
    c_acc = _acc_channels(channels_out, activation, fg_channel)
    acc_dtype = torch.float16 if device.type == "cuda" else torch.float32
    cl = bool(spec.channels_last) and device.type == "cuda"

    x = torch.from_numpy(box_u8).to(device)
    starts = [(z, y, xx) for z in window_starts(bz, p, s) for y in window_starts(by, p, s)
              for xx in window_starts(bx, p, s)]
    _assert_core_covered(spec, box_u8.shape, (cz, cy, cx), core_shape)
    n_all = len(starts)
    if spec.empty_window_max >= 0:
        maxes = _window_maxes(x[None, None], starts, p)
        starts = [st for st, m in zip(starts, maxes) if m > spec.empty_window_max]
    n_win_all = len(starts)
    if spec.norm_scope == "box":
        x = normalizer(x)[None, None]  # [1,1,Z,Y,X] float32 on device
    else:  # "window": normalise each patch on its own, as the teachers were trained (villa instance norm)
        x = x[None, None]
    acc = torch.zeros((c_acc, bz, by, bx), dtype=acc_dtype, device=device)
    wsum = torch.zeros((1, bz, by, bx), dtype=acc_dtype, device=device)
    w = window_weight(spec, device=device, dtype=torch.float32)
    w_acc = w[0].to(acc_dtype)

    use_amp = device.type == "cuda" and spec.dtype in (torch.bfloat16, torch.float16)
    xforms = tta_transforms(spec.tta)
    n_pass = len(xforms)
    is_async = hasattr(net, "submit") and hasattr(net, "wait")
    wants_origin = bool(getattr(net, "needs_box_origin", False))
    bo = tuple(int(v) for v in box_origin_zyx)
    chunks = [starts[i : i + spec.batch] for i in range(0, len(starts), spec.batch)]
    jobs = [(ci, xf) for ci in range(len(chunks)) for xf in xforms]  # chunk-major: window sums complete in order

    def _prep(chunk: list[tuple[int, int, int]], xf: Xform) -> torch.Tensor:
        xin = torch.cat([x[:, :, z : z + p, y : y + p, xx : xx + p] for (z, y, xx) in chunk], dim=0)
        if spec.norm_scope != "box":
            xin = torch.stack([normalizer(xin[b, 0])[None] for b in range(xin.shape[0])], dim=0)
        if spec.pad_batch and xin.shape[0] < spec.batch:  # fixed shape for compiled / TRT nets
            pad = xin[-1:].expand(spec.batch - xin.shape[0], -1, -1, -1, -1)
            xin = torch.cat([xin, pad], dim=0)
        xin = xf.apply(xin)
        if cl:
            xin = xin.contiguous(memory_format=torch.channels_last_3d)
        return xin

    def _post(out: Any, xf: Xform) -> torch.Tensor:
        out = select(out).float()  # [B, C, p, p, p]
        if out.shape[1] != channels_out:
            raise ValueError(f"net produced {out.shape[1]} channels, expected {channels_out}")
        if c_acc == 1 and channels_out == 2:
            out = (out[:, fg_channel] - out[:, 1 - fg_channel])[:, None]
        return _invert_field(out, xf, spec.tta_field)

    if wants_origin and len(xforms) > 1:
        raise ValueError("a net that needs box origins (radial input channels) cannot use sliding's TTA; "
                         "set spec.tta=False and let the net do its own TTA")
    n_win = 0
    win_sum: torch.Tensor | None = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=spec.dtype, enabled=use_amp):
        pending: tuple[int, Xform, Any] | None = None
        for j in range(len(jobs) + 1):
            if j < len(jobs):
                ci, xf = jobs[j]
                xin = _prep(chunks[ci], xf)
                kw: dict[str, Any] = {}
                if wants_origin:
                    org = [(bo[0] + z, bo[1] + y, bo[2] + xx) for (z, y, xx) in chunks[ci]]
                    org += [org[-1]] * (int(xin.shape[0]) - len(org))  # spec.pad_batch repeats the last window
                    kw["box_origin_zyx"] = org
                handle = net.submit(xin, **kw) if is_async else net(xin, **kw)
                del xin
                nxt = (ci, xf, handle)
            else:
                nxt = None
            if pending is not None:
                ci, xf, handle = pending
                out = _post(net.wait(handle) if is_async else handle, xf)
                win_sum = out if win_sum is None else win_sum + out
                del out
                if xf is xforms[-1]:  # last pass of this chunk: blend into the accumulator
                    chunk = chunks[ci]
                    if n_pass > 1:
                        win_sum = win_sum / n_pass
                    win_sum = win_sum * w
                    for b, (z, y, xx) in enumerate(chunk):
                        acc[:, z : z + p, y : y + p, xx : xx + p] += win_sum[b].to(acc_dtype)
                        wsum[:, z : z + p, y : y + p, xx : xx + p] += w_acc
                    win_sum = None
                    n_win += len(chunk)
            pending = nxt
    del x
    # crop core, divide (uncovered voxels -> 0), activate, quantise
    core = acc[:, cz : cz + core_shape[0], cy : cy + core_shape[1], cx : cx + core_shape[2]].float()
    wcore = wsum[:, cz : cz + core_shape[0], cy : cy + core_shape[1], cx : cx + core_shape[2]].float()
    del acc, wsum
    covered = wcore > 0
    if n_all == n_win_all and not bool(covered.all()):  # no window was skipped: a hole is a tiling bug
        raise ValueError(
            f"window tiling leaves {int((~covered).sum())} of {covered.numel()} output core voxels with "
            f"zero accumulated weight (patch {p}, step {s}, halo {spec.halo}, weight {spec.weight!r}, "
            f"box {tuple(box_u8.shape)}, core {tuple(core_shape)} at {(cz, cy, cx)})")
    core /= torch.where(covered, wcore, torch.ones_like(wcore))
    del wcore
    if c_acc == 1 and channels_out == 2:
        prob = torch.sigmoid(core)
    else:
        prob = apply_activation(core[None], activation)[0]
        if activation in ("softmax", "sigmoid") and fg_channel is not None:
            prob = prob[fg_channel : fg_channel + 1]
    del core
    prob = prob * covered
    q = torch.round(prob.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    del prob, covered
    return q, n_win


def prepare_net(net: Callable[[torch.Tensor], Any], spec: WindowSpec, device: torch.device) -> Callable[[torch.Tensor], Any]:
    """Apply the GPU throughput knobs (channels_last, cudnn.benchmark, torch.compile) to a
    torch net.  Non-Module callables (TRT wrappers, test doubles) and the CPU path are
    returned untouched apart from cudnn.benchmark."""
    if device.type != "cuda":
        return net
    if spec.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if spec.backend != "torch" or not isinstance(net, torch.nn.Module):
        return net
    if spec.channels_last:
        net = net.to(memory_format=torch.channels_last_3d)
    if spec.compile is not None:
        net = torch.compile(net, mode=spec.compile, dynamic=False)
    return net


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_sliding(
    reader: VolumeReader,
    net: Callable[[torch.Tensor], Any],
    normalizer: Callable[[torch.Tensor], torch.Tensor],
    spec: WindowSpec,
    region: RegionCfg,
    writer: BrickWriter,
    channels_out: int,
    activation: str,
    budget: Budget,
    device: str | torch.device = "cuda",
    voxel_scale: float = 1.0,
    fg_channel: int | None = None,
    select: Callable[[Any], torch.Tensor] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the engine over ``region`` (level grid), writing every tile into ``writer``.

    With ``spec.prefetch`` the next tile's input box is read in a background thread
    (one box in flight) so the S3 read overlaps the GPU compute.
    Returns a summary dict (tiles, windows, seconds, voxels ...)."""
    device = torch.device(device)
    log = log or (lambda m: print(m, flush=True))
    c_acc = _acc_channels(channels_out, activation, fg_channel)
    spec = fit_out_tile(spec, region, c_acc)
    tiles = plan_tiles(region, spec)
    n_out = 1 if (fg_channel is not None and activation in ("softmax", "sigmoid")) else channels_out
    if n_out != len(writer.channels):
        raise ValueError(f"writer has {len(writer.channels)} channels, engine produces {n_out}")
    boxes_in_flight = 2 if spec.prefetch else 1
    for tp in tiles:
        check_alloc(tp.box_shape, np.uint8, budget)
        check_alloc((n_out,) + tp.core_shape, np.uint8, budget)
        if boxes_in_flight * nbytes(tp.box_shape, np.uint8) > budget.ram_bytes:
            raise MemoryError(f"{boxes_in_flight} input boxes {tp.box_shape} exceed ram_bytes")
    net = prepare_net(net, spec, device)

    t0 = time.perf_counter()
    done = empty = 0
    windows = 0
    voxels = 0
    read_wait = 0.0
    n_tiles = len(tiles)
    todo = [(i, tp) for i, tp in enumerate(tiles) if not writer.has_brick(*tp.core0)]
    skipped = n_tiles - len(todo)
    log(f"[sliding] {n_tiles} tiles out_tile={spec.out_tile} patch={spec.patch} step={spec.step} "
        f"halo={spec.halo} batch={spec.batch} tta={spec.tta} ({spec.n_passes} passes) device={device} scale={voxel_scale:g} "
        f"backend={spec.backend} channels_last={spec.channels_last} compile={spec.compile} "
        f"prefetch={spec.prefetch} empty_window_max={spec.empty_window_max}")

    def _read(tp: TilePlan) -> np.ndarray:
        b0, b1 = tp.box0, tuple(a + b for a, b in zip(tp.box0, tp.box_shape))
        return reader.read(b0[0], b1[0], b0[1], b1[1], b0[2], b1[2])

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsm-prefetch") if spec.prefetch else None
    fut: Future[np.ndarray] | None = None
    try:
        for k, (i, tp) in enumerate(todo):
            z0, y0, x0 = tp.core0
            tt = time.perf_counter()
            if pool is not None:
                if fut is None:
                    fut = pool.submit(_read, tp)
                box = fut.result()
                fut = pool.submit(_read, todo[k + 1][1]) if k + 1 < len(todo) else None
            else:
                box = _read(tp)
            tr = time.perf_counter() - tt
            read_wait += tr
            if int(box.max()) <= spec.empty_max:
                out = np.zeros((n_out,) + tp.core_shape, dtype=np.uint8)
                n_win = 0
                empty += 1
            else:
                out, n_win = predict_box(
                    box, net, normalizer, spec, channels_out, activation, fg_channel, device,
                    select=select, core_offset=tp.core_offset, core_shape=tp.core_shape,
                    box_origin_zyx=tp.box0,
                )
            del box
            for c in range(n_out):
                writer.write(c, z0, y0, x0, out[c])
            del out
            free_cuda()
            done += 1
            windows += n_win
            voxels += int(np.prod(tp.core_shape))
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0.0
            remaining = n_tiles - i - 1
            eta = remaining / rate if rate > 0 else float("nan")
            log(f"[sliding] tile {i + 1}/{n_tiles} core={tp.core0}+{tp.core_shape} box={tp.box_shape} "
                f"win={n_win} {time.perf_counter() - tt:.1f}s (read wait {tr:.1f}s) | {rate:.3f} tiles/s "
                f"ETA {eta / 60:.1f} min")
    finally:
        if pool is not None:
            if fut is not None:
                fut.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    seconds = time.perf_counter() - t0
    summary = {
        "tiles": n_tiles, "done": done, "skipped": skipped, "empty": empty, "windows": windows,
        "seconds": seconds, "tiles_per_s": done / seconds if seconds > 0 else 0.0,
        "windows_per_s": windows / seconds if seconds > 0 else 0.0, "read_wait_s": read_wait,
        "voxels": voxels, "voxels_per_s": voxels / seconds if seconds > 0 else 0.0,
        "out_tile": spec.out_tile, "patch": spec.patch, "step": spec.step, "halo": spec.halo,
        "tta": spec.tta, "tta_passes": spec.n_passes, "tta_field": spec.tta_field, "batch": spec.batch,
        "voxel_scale": voxel_scale,
        "backend": spec.backend, "channels_last": spec.channels_last, "compile": spec.compile,
        "prefetch": spec.prefetch, "empty_window_max": spec.empty_window_max,
    }
    return summary
