"""Offline cache of the conv-teacher **encoder** features (``tsm feats``).

Motivation (``docs/student_v2_plan.md`` F.2): ``feat_distill`` currently runs the
frozen recto / ink encoders live inside the train loop (+40 % per step, so it is
applied every 2nd step).  Those encoders see the *clean* volume, so their stage
features are a function of the region alone and can be computed once per region
and read per crop, exactly like the DINO token cache (:mod:`tsm.dino`).

Layout under ``<out_dir>/feats/``::

    <teacher>_s<stage>.zarr    float16 (C, D/2**k, H/2**k, W/2**k), chunks (C, 8, 32, 32)
    pca_<teacher>_s<stage>.npz mean / components / explained / n_samples (absent when pca_dim is null)
    progress.json              completed ``<teacher>/s<stage>/<block>`` keys plus the run
                               manifest (window, border, pca_dim, region, dtype) -- the stage is
                               resumable, and a changed manifest rebuilds the stage arrays

``stage`` is the encoder stage index, which for the published ResEnc U-Nets is
also the downsampling power: ``encoder_strides`` are ``[1, 2, 4, 8, 16, ...]`` and
``encoder_channels`` ``[32, 64, 128, 256, 320, 320, 320]``, i.e. stage 3 = /8 with
256 ch and stage 4 = /16 with 320 ch.  Both are verified against the loaded
encoder before anything is written.

Windowing (the encoder's own halo)
----------------------------------
``sliding.predict_box`` blends overlapping *decoder* windows with a ramp; that is
wrong for features, whose values near a window face are contaminated by the zero
padding of a receptive field that is hundreds of voxels wide at /8.  Here we
instead run ``window``-sized cubes (256^3) and keep only the **central core** of
each window's feature map, discarding a ``border`` of 32 level-0 voxels (= 4
feature cells at /8, 2 at /16) on every face, so consecutive windows step by
``window - 2 * border`` = 192 voxels and the retained cores tile the region
exactly once -- no blending, no seam weights, one value per cell.  At the outer
faces of the region there is no neighbouring window to prefer, so the border is
kept there (:func:`core_intervals` returns the exact partition).  32 voxels is
4 cells at /8: enough that the padding influence has decayed through the 3x3x3
stacks of the last two stages, and cheap (192^3 / 256^3 = 42 % of each forward is
discarded).  ``tests/test_feats.py::test_window_cores_agree_with_a_single_pass``
measures the residual disagreement on a smooth synthetic input.

Normalisation is per window with the teacher's own :class:`tsm.teachers.Normalizer`
(``zscore_instance`` for recto, ``percentile_minmax`` for ink) -- the same
``norm_scope="window"`` rule ``tsm teacher`` / ``sliding.predict_box`` use at
inference, so the cached features *are* the inference-time features.  It also
means a window-sized intensity offset is a real (teacher-intrinsic) seam source,
independent of the receptive-field halo.

Reduction: with ``pca_dim`` (default 64) a PCA fitted on features sampled from
evenly spaced windows reduces 256/320 -> 64 channels before storage (2.4 GB per
teacher per /8 stage over the 256x6144x6144 slab instead of 9.7 GB).  The stored
target is then the linear sketch ``(x - mean) @ V.T`` and the distillation cosine
is taken in that space; ``pca_dim: null`` stores the raw features and makes the
cached term numerically identical to the live one.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from tsm.dino import DinoTokenCache, apply_pca, fit_pca, load_pca, save_pca, window_starts

__all__ = [
    "FEATS_DEFAULTS",
    "RESENC_CHANNELS",
    "feats_opts",
    "core_intervals",
    "window_plan",
    "feats_path",
    "pca_path",
    "FeatCache",
    "CachedFeatSource",
    "run_feats",
]

# Published vesuvius ResEnc U-Net encoder widths per stage (stage k is at /2**k).
# Verified against ``teachers.encoder_channels`` / ``encoder_strides`` at run time;
# only used for the ``--dry-run`` accounting, which must not load a checkpoint.
RESENC_CHANNELS: dict[int, int] = {0: 32, 1: 64, 2: 128, 3: 256, 4: 320, 5: 320, 6: 320}

FEATS_DEFAULTS: dict[str, Any] = {
    "teachers": ["recto", "ink"],
    "stages": [3, 4],            # encoder stage index == downsampling power (/8, /16)
    "window": 256,               # level-0 voxels per encoder window (the teachers' train patch)
    "border": 32,                # voxels discarded on every face of each window (= 4 cells at /8)
    "pca_dim": 64,               # null -> store the raw stage channels
    "pca_sample_windows": 16,    # windows sampled (evenly spaced) to fit each PCA
    "pca_vectors_per_window": 4096,
    "dtype": "float16",
    "block_windows": 4,          # windows per axis per I/O block (bounds the CT read)
    "backend": "torch",          # bf16 autocast on cuda
    "models_dir": None,          # default teachers.DEFAULT_MODELS_DIR
    "device": None,              # default cuda when available
    "limit_blocks": 0,           # >0: stop after N blocks per teacher (smoke runs)
}


def feats_opts(cfg: Any) -> dict[str, Any]:
    raw = (cfg.extra or {}).get("feats", {}) if hasattr(cfg, "extra") else {}
    if not isinstance(raw, dict):
        raise ValueError("extra.feats must be an object")
    unknown = sorted(set(raw) - set(FEATS_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.feats keys: {unknown}")
    o = dict(FEATS_DEFAULTS)
    o.update(raw)
    o["teachers"] = [str(t) for t in o["teachers"]]
    o["stages"] = [int(s) for s in o["stages"]]
    if not o["teachers"] or not o["stages"]:
        raise ValueError("extra.feats needs at least one teacher and one stage")
    if str(o["backend"]) != "torch":
        raise ValueError(f"unknown extra.feats.backend {o['backend']!r} (only 'torch')")
    unit = 1 << max(o["stages"])
    win, bor = int(o["window"]), int(o["border"])
    if win % unit or bor % unit:
        raise ValueError(f"extra.feats.window/border must be divisible by {unit} (the coarsest stage stride)")
    if 2 * bor >= win:
        raise ValueError(f"extra.feats.border {bor} leaves no core in a {win} window")
    if int(o["block_windows"]) < 1:
        raise ValueError("extra.feats.block_windows must be >= 1")
    o["window"], o["border"] = win, bor
    return o


# --------------------------------------------------------------------------- #
# window tiling: cores partition the region
# --------------------------------------------------------------------------- #
def core_intervals(extent: int, window: int, border: int) -> list[tuple[int, int, int]]:
    """``[(window_start, core_lo, core_hi), ...]`` along one axis, in level-0 voxels.

    The ``core_lo``/``core_hi`` half-open intervals partition ``[0, extent)``
    exactly: each window keeps everything but a ``border`` margin on the faces
    that have a neighbour, and the whole window where it touches the region
    boundary.  The final window is clamped to ``extent - window`` (upstream's
    ``window_starts`` rule), and its core simply starts where the previous one
    ended, so the clamped overlap costs nothing and never duplicates a cell.
    """
    extent, window, border = int(extent), int(window), int(border)
    w = min(window, extent)
    b = 0 if w >= extent else border
    starts = window_starts(extent, w, 2 * b)
    out: list[tuple[int, int, int]] = []
    prev = 0
    for i, s in enumerate(starts):
        hi = extent if i == len(starts) - 1 else s + w - b
        if not (s <= prev <= hi <= s + w):
            raise AssertionError(f"bad core {(s, prev, hi)} for extent={extent} window={window} border={border}")
        out.append((int(s), int(prev), int(hi)))
        prev = hi
    return out


def window_plan(size_zyx: Sequence[int], window: int, border: int) -> list[list[tuple[int, int, int]]]:
    """Per-axis :func:`core_intervals` for a region size."""
    return [core_intervals(int(n), window, border) for n in size_zyx]


def _blocks(plan: Sequence[Sequence[tuple[int, int, int]]], per: int) -> Iterator[tuple[int, tuple[list[int], list[int], list[int]]]]:
    """Group the per-axis window indices into ``per``-sized chunks; yield (block index, indices)."""
    chunks = [[list(range(i, min(i + per, len(ax)))) for i in range(0, len(ax), per)] for ax in plan]
    n = 0
    for cz in chunks[0]:
        for cy in chunks[1]:
            for cx in chunks[2]:
                yield n, (cz, cy, cx)
                n += 1


def feats_path(out_dir: str, teacher: str, stage: int) -> str:
    return os.path.join(out_dir, f"{teacher}_s{int(stage)}.zarr")


def pca_path(out_dir: str, teacher: str, stage: int) -> str:
    return os.path.join(out_dir, f"pca_{teacher}_s{int(stage)}.npz")


# --------------------------------------------------------------------------- #
# forward helper
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def window_features(enc: Any, normalizer: Any, vol_u8: np.ndarray, stages: Sequence[int],
                    device: str | torch.device = "cpu") -> dict[int, torch.Tensor]:
    """``{stage: (C, d, h, w) float32}`` for one uint8 window, normalised the teacher's way."""
    from tsm.teachers import encoder_features

    x = normalizer(np.ascontiguousarray(vol_u8))[None, None].to(device)
    dev = torch.device(device).type
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        f = encoder_features(enc, x)
    return {int(k): f[int(k)][0].float().cpu() for k in stages}


# --------------------------------------------------------------------------- #
# cached store + the feat_distill "cached" source
# --------------------------------------------------------------------------- #
class FeatCache(DinoTokenCache):
    """Read-only view of ``<out_dir>/feats/<teacher>_s<stage>.zarr``.

    Same reader contract as :class:`tsm.dino.DinoTokenCache` (``origin_zyx``,
    ``scale`` = the stage stride, ``read_crop(origin_zyx, patch)`` -> ``(C, n, n, n)``
    float32 or None when the crop is unaligned / outside), plus the teacher and
    stage identity from the attrs.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        a = dict(self.array.attrs)
        self.teacher = str(a.get("teacher", ""))
        self.stage = int(a.get("stage", int(math.log2(self.scale))))
        self.stride = int(a.get("stride", self.scale))
        self.pca = a.get("pca")
        self.raw_channels = int(a.get("raw_channels", self.channels))


class CachedFeatSource(torch.nn.Module):
    """``feat_distill`` term against a cached teacher-encoder feature grid.

    Mirrors :class:`tsm.dino.DinoFeatSource`: no network runs in the train loop,
    the target for a crop is read at ``(origin - cache.origin) / stride`` and the
    student stage is mapped to the cached channel count by the usual 1x1x1
    :class:`tsm.student.FeatureProjector`.

    Augmentation follows the same **signed-permutation-only** rule as the DINO
    source.  A conv encoder is translation- but not rotation-equivariant, and the
    cached grid is a stack of scalar fields, so it resamples with the crop only
    under flips / rot90 (a relabelling of the axes).  Samples whose spatial
    transform contains a small rotation, an anisotropic scale or an elastic field
    are dropped from this term and the kept fraction is logged as
    ``feat_<teacher>_frac``.  (Intensity augmentation is likewise not matched by a
    cached target -- that is the documented caveat of F.2, measured by
    ``configs/ablate_feats.json``.)
    """

    def __init__(self, name: str, caches: Mapping[int, Any], pairs: Sequence[Sequence[int]],
                 student_ch: Mapping[int, int] | Sequence[int], kind: str = "cosine",
                 groups: int = 8, norm: bool = True) -> None:
        super().__init__()
        from tsm.student import FeatureProjector

        self.name = str(name)
        self.kind = str(kind)
        self.pairs = [(int(a), int(b)) for a, b in pairs]
        self.caches = {int(k): v for k, v in caches.items()}
        missing = sorted({tk for _, tk in self.pairs} - set(self.caches))
        if missing:
            raise ValueError(f"feat_distill cached teacher {name!r}: no cache for stage(s) {missing}")
        teacher_ch = {tk: int(self.caches[tk].channels) for _, tk in self.pairs}
        self.proj = FeatureProjector(self.pairs, student_ch, teacher_ch, groups=groups, norm=norm)

    @property
    def channels(self) -> dict[int, int]:
        return {k: int(c.channels) for k, c in self.caches.items()}

    def targets(self, origin_zyx: Sequence[int], patch: int, spatial: Any = None) -> dict[int, torch.Tensor] | None:
        """Cached features for one crop at every paired stage, transformed like the
        crop, or None when this sample must be skipped."""
        from tsm.dino import apply_signed_permutation, signed_permutation

        perm, flips = [0, 1, 2], [False, False, False]
        if spatial is not None and not getattr(spatial, "identity", False):
            if spatial.elastic is not None or spatial.scale is not None or spatial.rotation is not None:
                return None
            sp = signed_permutation(spatial.matrix())
            if sp is None:
                return None
            perm, flips = sp
        out: dict[int, torch.Tensor] = {}
        for tk in sorted({t for _, t in self.pairs}):
            a = self.caches[tk].read_crop(origin_zyx, patch)
            if a is None:
                return None
            t = torch.from_numpy(np.asarray(a, dtype=np.float32))
            if perm != [0, 1, 2] or any(flips):
                t = apply_signed_permutation(t, perm, flips)
            out[tk] = t.contiguous()
        return out

    def forward(self, feats: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                spatial: Sequence[Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        import torch.nn.functional as F

        ref = feats[f"enc{self.pairs[0][0]}"]
        B = int(ref.shape[0])
        patch = int(batch["input"].shape[-1])
        origins = batch["origin_zyx"].tolist()
        keep: list[int] = []
        got: list[dict[int, torch.Tensor]] = []
        for i in range(B):
            t = self.targets(origins[i], patch, spatial[i] if spatial is not None else None)
            if t is not None:
                keep.append(i)
                got.append(t)
        frac = len(keep) / max(1, B)
        if not keep:
            return ref.sum() * 0.0, {f"feat_{self.name}": 0.0, f"feat_{self.name}_frac": 0.0}
        proj = self.proj({k: v[keep] for k, v in feats.items()})
        terms = []
        for (sk, tk), s in proj.items():
            t = torch.stack([g[tk] for g in got]).to(device=s.device, dtype=torch.float32)
            s = s.float()
            if t.shape[2:] != s.shape[2:]:
                t = F.interpolate(t, size=s.shape[2:], mode="trilinear", align_corners=False)
            if self.kind == "cosine":
                terms.append((1.0 - F.cosine_similarity(s, t, dim=1, eps=1e-6)).mean())
            else:
                terms.append(F.mse_loss(s, t))
        loss = torch.stack(terms).mean()
        return loss, {f"feat_{self.name}": float(loss.detach()), f"feat_{self.name}_frac": frac}


def open_caches(feats_dir: str, teacher: str, stages: Sequence[int]) -> dict[int, FeatCache]:
    """``{stage: FeatCache}`` for one teacher (raises with the expected path when absent)."""
    out = {}
    for k in sorted({int(s) for s in stages}):
        p = feats_path(os.path.expanduser(feats_dir), teacher, k)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} (run `tsm feats <config>` first, or set feat_distill.feats_dir)")
        out[k] = FeatCache(p)
    return out


# --------------------------------------------------------------------------- #
# tsm feats: build the cache over the config region
# --------------------------------------------------------------------------- #
def _open_out(zarr: Any, path: str, shape: Sequence[int], chunks: Sequence[int], dtype: Any,
              attrs: Mapping[str, Any], force: bool) -> tuple[Any, bool]:
    """``(array, created)``.  An existing array whose shape/dtype disagrees with the
    request (a changed pca_dim, stage grid or region) is rebuilt rather than reused."""
    shape = tuple(int(s) for s in shape)
    if os.path.exists(os.path.join(path, "zarr.json")) and not force:
        arr = zarr.open_array(store=path, mode="r+")
        if tuple(int(v) for v in arr.shape) == shape and np.dtype(arr.dtype) == np.dtype(dtype):
            return arr, False
        print(f"[tsm] feats {path}: existing array {tuple(int(v) for v in arr.shape)}/{arr.dtype} "
              f"does not match the requested {shape}/{np.dtype(dtype).name}; rebuilding it", flush=True)
    return zarr.create_array(store=path, shape=shape, chunks=tuple(int(c) for c in chunks),
                             dtype=np.dtype(dtype), fill_value=0, attributes=dict(attrs),
                             overwrite=True), True


# Fields of the run manifest that invalidate *every* cached block when they change.
# ``stages`` / ``teachers`` are deliberately absent: completion is keyed per
# (teacher, stage, block), so adding a stage or a teacher only builds the new arrays (R10).
_MANIFEST_KEYS = ("window", "border", "pca_dim", "region_start_zyx", "region_size_zyx",
                  "dtype", "voxel_um", "level")


def _manifest(cfg: Any, o: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "window": int(o["window"]), "border": int(o["border"]),
        "pca_dim": (None if o["pca_dim"] in (None, 0) else int(o["pca_dim"])),
        "region_start_zyx": [int(v) for v in cfg.region.start_zyx],
        "region_size_zyx": [int(v) for v in cfg.region.size_zyx],
        "dtype": str(o["dtype"]), "voxel_um": float(cfg.volume.voxel_um),
        "level": int(cfg.volume.level),
        "stages": [int(k) for k in o["stages"]], "teachers": list(o["teachers"]),
    }


def _sample_windows(plan: Sequence[Sequence[tuple[int, int, int]]], n: int) -> list[tuple[int, int, int]]:
    all_w = [(iz, iy, ix) for iz in range(len(plan[0])) for iy in range(len(plan[1])) for ix in range(len(plan[2]))]
    n = min(int(n), len(all_w))
    idx = np.linspace(0, len(all_w) - 1, n).round().astype(int)
    return [all_w[int(i)] for i in idx]


def run_feats(cfg: Any, dry_run: bool = False, force: bool = False, loader: Any = None) -> dict[str, Any]:
    """Run the frozen teacher encoders over ``cfg.region`` and cache their stage features.

    One pass per teacher (the encoders are loaded one at a time to bound VRAM);
    inside a pass the region is read in blocks of ``block_windows**3`` windows and
    each window's core is written straight to the stage arrays.  Completed
    ``(teacher, stage, block)`` triples are recorded in ``progress.json`` next to the run
    manifest, so an interrupted run continues where it stopped, adding a stage to an
    existing cache only builds that stage, and a changed manifest (window, border,
    pca_dim, region, dtype) rebuilds the affected arrays (``--force`` restarts and
    rewrites everything).
    """
    import zarr

    from tsm.limits import format_table, read_rss_bytes
    from tsm.teachers import DEFAULT_MODELS_DIR, encoder_channels, encoder_of, encoder_strides, load_teacher
    from tsm.volume import VolumeReader

    o = feats_opts(cfg)
    stages = o["stages"]
    win, bor = o["window"], o["border"]
    pdim = None if o["pca_dim"] in (None, 0) else int(o["pca_dim"])
    out_dir = os.path.join(os.path.expanduser(cfg.out_dir), "feats")
    start = tuple(int(v) for v in cfg.region.start_zyx)
    size = tuple(int(v) for v in cfg.region.size_zyx)
    unit = 1 << max(stages)
    if any(s % unit for s in size):
        raise ValueError(f"region.size_zyx {size} must be divisible by {unit}")
    plan = window_plan(size, win, bor)
    n_windows = int(np.prod([len(a) for a in plan]))
    grids = {k: tuple(s >> k for s in size) for k in stages}
    dtype_np = np.dtype(str(o["dtype"]))
    device = o["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    per = int(o["block_windows"])
    span = tuple(max(a[i:i + per][-1][0] - a[i:i + per][0][0] + min(win, n) for i in range(0, len(a), per))
                 for a, n in zip(plan, size))

    def out_channels(stage: int) -> int:
        return pdim if pdim else RESENC_CHANNELS.get(int(stage), 0)

    rows: list[Sequence[object]] = [["array", "shape", "bytes", "GiB"]]
    total = 0
    for name in o["teachers"]:
        for k in stages:
            c = out_channels(k)
            n = c * int(np.prod(grids[k])) * dtype_np.itemsize
            total += n
            rows.append([f"{name}_s{k}.zarr", f"{(c, *grids[k])}", n, f"{n / (1 << 30):.3f}"])
    rows.append(["TOTAL", "", total, f"{total / (1 << 30):.3f}"])
    print(f"[tsm] feats region {start} size {size} stages {stages} -> grids "
          + ", ".join(f"/{1 << k}:{grids[k]}" for k in stages))
    print(f"[tsm] feats windows {win}^3 core {win - 2 * bor}^3 (border {bor}): "
          f"{n_windows} per teacher ({[len(a) for a in plan]}) on {device}, pca_dim={pdim}")
    print(f"[tsm] feats block {per}^3 windows -> CT read {span} = {int(np.prod(span)) / (1 << 20):.0f} MiB")
    print(format_table(rows))
    if dry_run:
        print("[tsm] dry run: stopping before I/O")
        return {"windows": n_windows, "windows_per_axis": [len(a) for a in plan], "grids": grids,
                "bytes": {f"{n}_s{k}": out_channels(k) * int(np.prod(grids[k])) * dtype_np.itemsize
                          for n in o["teachers"] for k in stages},
                "total_bytes": total, "block_read_bytes": int(np.prod(span))}

    os.makedirs(out_dir, exist_ok=True)
    reader = __import__('tsm.cli', fromlist=['open_ct']).open_ct(cfg, 0)
    loader = loader or load_teacher
    models_dir = o["models_dir"] or DEFAULT_MODELS_DIR
    prog_file = os.path.join(out_dir, "progress.json")
    manifest = _manifest(cfg, o)
    done: set[str] = set()
    stale = False
    if os.path.exists(prog_file) and not force:
        try:
            prev = json.load(open(prog_file))
        except (OSError, ValueError):
            prev = {}
        if not isinstance(prev, dict):
            prev = {}
        changed = [k for k in _MANIFEST_KEYS
                   if k in prev.get("manifest", {}) and prev["manifest"][k] != manifest[k]]
        if "manifest" not in prev:
            stale = True
            print("[tsm] feats: progress.json predates the run manifest; rebuilding every block",
                  flush=True)
        elif changed:
            stale = True
            print("[tsm] feats: the run manifest changed ("
                  + ", ".join(f"{k}: {prev['manifest'][k]!r} -> {manifest[k]!r}" for k in changed)
                  + "); rebuilding every stage array", flush=True)
        else:
            done = {str(k) for k in prev.get("blocks", [])}
            if done:
                print(f"[tsm] feats resuming: {len(done)} (teacher, stage, block) keys already done")
    force_arrays = bool(force or stale)

    def _save_progress() -> None:
        with open(prog_file, "w") as fh:
            json.dump({"blocks": sorted(done), "manifest": manifest,
                       "window": manifest["window"], "border": manifest["border"],
                       "stages": manifest["stages"]}, fh)

    written: dict[str, str] = {}
    n_blocks = 0
    for name in o["teachers"]:
        model, spec = loader(name, models_dir, device=device, dtype=torch.float32)
        enc = encoder_of(model).eval()
        enc.requires_grad_(False)
        del model
        ch, st = encoder_channels(enc), encoder_strides(enc)
        for k in stages:
            if k >= len(ch):
                raise ValueError(f"{name}: stage {k} outside the encoder ({len(ch)} stages)")
            if st[k] != (1 << k):
                raise ValueError(f"{name}: stage {k} has stride {st[k]}, expected {1 << k} "
                                 "(the stage index is used as the downsampling power)")
        print(f"[tsm] feats {name}: encoder channels {ch} strides {st}; "
              + ", ".join(f"stage {k} = {ch[k]} ch at /{st[k]}" for k in stages), flush=True)

        pcas: dict[int, dict[str, np.ndarray] | None] = {k: None for k in stages}
        if pdim:
            need = [k for k in stages
                    if not (os.path.exists(pca_path(out_dir, name, k)) and not force_arrays)]
            for k in stages:
                if k not in need:
                    pcas[k] = load_pca(pca_path(out_dir, name, k))
                    print(f"[tsm] feats {name} s{k}: PCA reused from {pca_path(out_dir, name, k)}")
            if need:
                fitted = _fit_pcas(reader, enc, spec, start, plan, need, o, device, pdim, name)
                for k, p in fitted.items():
                    save_pca(pca_path(out_dir, name, k), p)
                    pcas[k] = p
                    print(f"[tsm] feats {name} s{k}: PCA {p['components'].shape} on {int(p['n_samples'])} vectors "
                          f"(explained {float(p['explained'].sum()):.3f}) -> {pca_path(out_dir, name, k)}", flush=True)

        arrays = {}
        for k in stages:
            c = pdim if pdim else int(ch[k])
            attrs = {"origin_zyx": list(start), "stride": 1 << k, "scale": 1 << k, "channels": int(c),
                     "raw_channels": int(ch[k]), "pca": (int(pdim) if pdim else None), "teacher": name,
                     "stage": int(k), "voxel_um": float(cfg.volume.voxel_um), "window": win, "border": bor,
                     "normalizer": spec.normalizer.mode, "l2_normalised": False, "dtype": dtype_np.name}
            arrays[k], created = _open_out(zarr, feats_path(out_dir, name, k), (c, *grids[k]),
                                           (c, 8, 32, 32), dtype_np, attrs, force_arrays)
            written[f"{name}_s{k}"] = feats_path(out_dir, name, k)
            if created:  # a fresh (or rebuilt) array has no valid completion keys
                done -= {d for d in done if d.startswith(f"{name}/s{k}/")}
        _save_progress()

        for bi, sel in _blocks(plan, per):
            # completion is per (teacher, stage, block): a block completed for stage 3 must
            # still run when stage 4 is added to the same out_dir (R10)
            keys = [f"{name}/s{k}/{bi}" for k in stages]
            if all(key in done for key in keys):
                continue
            lo = [plan[i][sel[i][0]][0] for i in range(3)]
            hi = [plan[i][sel[i][-1]][0] + min(win, size[i]) for i in range(3)]
            vol = reader.read(start[0] + lo[0], start[0] + hi[0], start[1] + lo[1], start[1] + hi[1],
                              start[2] + lo[2], start[2] + hi[2])
            for iz in sel[0]:
                for iy in sel[1]:
                    for ix in sel[2]:
                        wz, wy, wx = (plan[a][i] for a, i in ((0, iz), (1, iy), (2, ix)))
                        sub = vol[wz[0] - lo[0]:wz[0] - lo[0] + min(win, size[0]),
                                  wy[0] - lo[1]:wy[0] - lo[1] + min(win, size[1]),
                                  wx[0] - lo[2]:wx[0] - lo[2] + min(win, size[2])]
                        f = window_features(enc, spec.normalizer, sub, stages, device=device)
                        for k in stages:
                            a = f[k]
                            cut = tuple(slice((c[1] - c[0]) >> k, (c[2] - c[0]) >> k) for c in (wz, wy, wx))
                            core = a[(slice(None), *cut)]
                            arr = np.moveaxis(core.numpy(), 0, -1)
                            if pcas[k] is not None:
                                arr = apply_pca(arr, pcas[k], normalize=False)
                            dst = tuple(slice(c[1] >> k, c[2] >> k) for c in (wz, wy, wx))
                            arrays[k][(slice(None), *dst)] = np.moveaxis(arr, -1, 0).astype(dtype_np)
                        del f
            del vol
            done.update(keys)
            n_blocks += 1
            _save_progress()
            print(f"[tsm] feats {name} block {bi} voxels {tuple(lo)}..{tuple(hi)} "
                  f"({len(sel[0]) * len(sel[1]) * len(sel[2])} windows) rss={read_rss_bytes() / (1 << 20):.0f} MiB",
                  flush=True)
            if int(o["limit_blocks"]) and n_blocks >= int(o["limit_blocks"]):
                print(f"[tsm] feats stopping after {n_blocks} blocks (limit_blocks)")
                break
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[tsm] feats wrote {len(written)} arrays under {out_dir} ({total / (1 << 30):.2f} GiB)")
    return {"windows": n_windows, "grids": grids, "blocks": n_blocks, "out_dir": out_dir,
            "arrays": written, "total_bytes": total}


def _fit_pcas(reader: Any, enc: Any, spec: Any, start: Sequence[int],
              plan: Sequence[Sequence[tuple[int, int, int]]], stages: Sequence[int],
              o: Mapping[str, Any], device: Any, pdim: int, name: str) -> dict[int, dict[str, np.ndarray]]:
    """Fit one PCA per stage on feature vectors sampled from evenly spaced window cores."""
    win = int(o["window"])
    rng = np.random.default_rng(0)
    per_win = int(o["pca_vectors_per_window"])
    chunks: dict[int, list[np.ndarray]] = {k: [] for k in stages}
    picks = _sample_windows(plan, int(o["pca_sample_windows"]))
    lens = [min(win, ax[-1][2]) for ax in plan]  # the last core_hi is the region extent on that axis
    for c, (iz, iy, ix) in enumerate(picks):
        wz, wy, wx = plan[0][iz], plan[1][iy], plan[2][ix]
        vol = reader.read(start[0] + wz[0], start[0] + wz[0] + lens[0],
                          start[1] + wy[0], start[1] + wy[0] + lens[1],
                          start[2] + wx[0], start[2] + wx[0] + lens[2])
        f = window_features(enc, spec.normalizer, vol, stages, device=device)
        for k in stages:
            cut = tuple(slice((w[1] - w[0]) >> k, (w[2] - w[0]) >> k) for w in (wz, wy, wx))
            a = np.moveaxis(f[k][(slice(None), *cut)].numpy(), 0, -1).reshape(-1, f[k].shape[0])
            n = min(per_win, a.shape[0])
            chunks[k].append(a[rng.choice(a.shape[0], size=n, replace=False)])
        del f, vol
        print(f"[tsm] feats {name} PCA sample {c + 1}/{len(picks)} at window {(iz, iy, ix)}", flush=True)
    return {k: fit_pca(np.concatenate(chunks[k], 0), dim=pdim) for k in stages}
