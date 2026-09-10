"""Pure-PyTorch, forward-only reimplementation of villa's ``dinovol_2`` 3D ViT
(``dinovol_v2_ps8_with_paris4_352500``) plus the offline token-cache stage.

Nothing here imports ``dinovol_2`` / ``timm`` / ``einops``; the architecture is
inferred from the checkpoint's own tensor shapes exactly like
:mod:`tsm.teachers` does for the U-Nets, so a shape-only inventory
(``~/.cache/tsm-production/models/dinovol_v2_*/*/*.tsm-model/model.json``) is
enough to build a net with exact state-dict key parity.  See
``dev/parity_dino.py`` for the numerical check against the reference code.

Architecture (all constants derived from the inventory, values for the released
checkpoint in brackets)::

    down_projection.proj   Conv3d(1, D, k=P, stride=P)          [864, patch 8]
    cls_token [1,1,D], reg_token [1,R,D], mask_token [1,1,D]    [R = 4]
    blocks.{i}                                                  [depth 24]
      norm1                LayerNorm(D)
      attn.qkv             Linear(D, 3D, bias=False)            [heads 16, head_dim 54]
      attn.q_bias/v_bias   (D,)   -- k has no bias (zeros)
      attn.proj            Linear(D, D)
      rope_embed           mixed RoPE: periods [F], mix_frequencies [H, D/2H... ]
                                                            [periods 9, mix (16, 27, 3)]
      norm2                LayerNorm(D)
      mlp                  SwiGLU: fc1_g, fc1_x, norm(LayerNorm on the hidden), fc2
                                                            [hidden 2304, SiLU gate]
    norm                   LayerNorm(D)

No absolute position embedding: position enters only through the per-block
*mixed* RoPE, which rotates q and k of the **patch tokens only** (the 1 + 4
prefix tokens are left untouched).  Coordinates are normalised per axis
(``normalize_coords="separate"``): ``(arange(0.5, n) / n) * 2 - 1`` per axis,
meshgrid'd in (z, y, x) order and flattened in the same ``(d h w)`` order as the
patch grid.  The train-time coordinate augmentations (``shift_coords``,
``jitter_coords``, ``rescale_coords``) are training-only upstream and are not
implemented here (inference mode).  ``periods`` only seeds ``mix_frequencies``
at init upstream and is unused in the forward pass; it is kept as a buffer for
state-dict parity.

Input normalisation: the released config carries no ``normalization_scheme`` and
no nnU-Net intensity properties, so dinovol's default applies --
``RobustNormalization`` (``dinovol_2/dataset/normalization.py``): clip to the
instance's [p1, p99], then ``(x - median) / (1.4826 * MAD)`` with the upstream
fallbacks when the MAD degenerates.  Upstream normalises *per crop* (the SSL
dataset normalises each view in ``_finalize_crop``), so :func:`normalize_robust`
is applied per window here, and that is the default of
:func:`patch_token_grid`.

Token cache (``tsm dino``): the ViT is run in ``window``-sized cubes with
``overlap_tokens`` tokens of overlap, each window's patch tokens are
L2-normalised, blended with the same linear ramp weights upstream's
``compute_patch_embedding_grid`` uses, and the stitched grid is L2-normalised
again.  The 864-dim grid is then reduced to 64 dims with a PCA fitted on a
sample of tokens (``pca.npz``) and stored fp16 at the /8 token grid; the
ink-likeness map (cosine similarity to the expert reference embedding) is
computed on the *full* 864-dim tokens before the reduction and stored uint8 at
the same /8 grid (``u8 = round((cos + 1) * 127.5)``).
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tsm.teachers import DEFAULT_MODELS_DIR, shapes_of, strip_prefix

__all__ = [
    "ViTSpec",
    "DinoVolViT",
    "MixedRope3d",
    "infer_vit_spec",
    "build_dinovol",
    "load_dinovol",
    "normalize_robust",
    "rope_coords",
    "apply_rope",
    "patch_token_grid",
    "fit_pca",
    "apply_pca",
    "save_pca",
    "load_pca",
    "ink_reference_embedding",
    "signed_permutation",
    "apply_signed_permutation",
    "DinoTokenCache",
    "DinoFeatSource",
    "DINO_DEFAULTS",
    "run_dino",
    "DEFAULT_DINO_FILE",
]

DEFAULT_DINO_FILE = "dinovol.pt"
DEFAULT_AUX_GLOB = "~/.cache/tsm-production/models/auxiliary/*/avg_ref_embedding.npy"
DEFAULT_INVENTORY_GLOB = "~/.cache/tsm-production/models/dinovol_v2_*/*/*.tsm-model/model.json"


# --------------------------------------------------------------------------- #
# architecture spec (inferred from tensor shapes)
# --------------------------------------------------------------------------- #
@dataclass
class ViTSpec:
    in_channels: int
    embed_dim: int
    depth: int
    num_heads: int
    patch: int
    mlp_hidden: int
    num_reg_tokens: int
    class_token: bool = True
    mask_token: bool = True
    rope_periods: int = 9  # F = head_dim // (2 * ndim)
    ndim: int = 3

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def num_prefix_tokens(self) -> int:
        return int(self.class_token) + self.num_reg_tokens


def infer_vit_spec(sd: Mapping[str, Any]) -> ViTSpec:
    """Build a :class:`ViTSpec` from a state dict / shape-only inventory.

    ``backbone.`` (and the usual ``module.`` / ``_orig_mod.``) prefixes are
    stripped first, so both the raw checkpoint and the ``.tsm-model`` inventory
    work.
    """
    shapes = shapes_of(strip_prefix(sd, ("module.", "_orig_mod.", "backbone.")))
    w = shapes.get("down_projection.proj.weight")
    if w is None:
        raise ValueError("no down_projection.proj.weight; not a dinovol backbone")
    embed_dim, in_channels, patch = int(w[0]), int(w[1]), int(w[2])
    if tuple(w[2:]) != (patch, patch, patch):
        raise ValueError(f"non-cubic patch {tuple(w[2:])}")
    depth = 0
    while f"blocks.{depth}.norm1.weight" in shapes:
        depth += 1
    if depth == 0:
        raise ValueError("no transformer blocks")
    mix = shapes.get("blocks.0.rope_embed.mix_frequencies")
    if mix is None:
        raise ValueError("no blocks.0.rope_embed.mix_frequencies (only mixed RoPE is supported)")
    num_heads, num_pairs, ndim = (int(v) for v in mix)
    if num_pairs * 2 != embed_dim // num_heads:
        raise ValueError(f"mix_frequencies {tuple(mix)} inconsistent with embed_dim {embed_dim}")
    mlp_hidden = int(shapes["blocks.0.mlp.fc1_g.weight"][0])
    reg = shapes.get("reg_token")
    return ViTSpec(
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch=patch,
        mlp_hidden=mlp_hidden,
        num_reg_tokens=int(reg[1]) if reg is not None else 0,
        class_token="cls_token" in shapes,
        mask_token="mask_token" in shapes,
        rope_periods=int(shapes["blocks.0.rope_embed.periods"][0]),
        ndim=ndim,
    )


# --------------------------------------------------------------------------- #
# mixed RoPE (port of dinovol_2/model/rope.py, inference path only)
# --------------------------------------------------------------------------- #
def rope_coords(grid: Sequence[int], device: Any = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(T, 3) coordinates in [-1, 1] for a (d, h, w) token grid, ``normalize_coords='separate'``.

    Token ``t = ((iz * h) + iy) * w + ix`` -- the same flattening as the patch grid.
    """
    axes = [(torch.arange(0.5, int(n), device=device, dtype=dtype) / int(n)) for n in grid]
    c = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(grid))
    return 2.0 * c - 1.0


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, prefix_tokens: int = 0) -> torch.Tensor:
    """Rotate the last ``T`` tokens of ``x`` (..., N, head_dim); the first
    ``prefix_tokens`` (cls + registers) pass through untouched."""
    if x.shape[-2] - prefix_tokens != sin.shape[-2]:
        raise ValueError(f"rope tokens {sin.shape[-2]} != {x.shape[-2]} - {prefix_tokens}")
    head = x[..., :prefix_tokens, :]
    tail = x[..., prefix_tokens:, :].to(sin.dtype)
    tail = (tail * cos + rope_rotate_half(tail) * sin).to(x.dtype)
    return tail if prefix_tokens == 0 else torch.cat((head, tail), dim=-2)


class MixedRope3d(nn.Module):
    """Per-block mixed RoPE: a learned frequency vector per (head, pair) in 3D.

    ``angles = 2*pi * einsum('td,hpd->htp', coords, mix_frequencies)``, tiled to
    the full head dim; ``periods`` (the geometric ``base**(2i/axis_dim)`` ladder)
    only seeds ``mix_frequencies`` upstream and is kept for key parity.
    """

    def __init__(self, head_dim: int, num_heads: int, num_periods: int, ndim: int = 3, base: float = 100.0) -> None:
        super().__init__()
        self.head_dim, self.num_heads, self.ndim, self.base = int(head_dim), int(num_heads), int(ndim), float(base)
        self.num_pairs = self.head_dim // 2
        self.register_buffer("periods", self._build_periods(int(num_periods)), persistent=True)
        self.mix_frequencies = nn.Parameter(torch.zeros(self.num_heads, self.num_pairs, self.ndim))

    def _build_periods(self, count: int) -> torch.Tensor:
        axis_dim = self.head_dim // self.ndim
        return self.base ** (2 * torch.arange(count, dtype=torch.float32) / axis_dim)

    def embed(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(sin, cos), each (num_heads, T, head_dim), from (T, 3) coordinates."""
        mix = self.mix_frequencies.to(coords.dtype)
        angles = 2 * math.pi * torch.einsum("td,hpd->htp", coords, mix)
        angles = angles.tile(2)
        return torch.sin(angles), torch.cos(angles)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed(coords)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
class PatchEmbed3d(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EvaAttention(nn.Module):
    """EVA-02 fused-qkv attention with q/v biases only (k bias is a constant zero)."""

    def __init__(self, dim: int, num_heads: int, num_prefix_tokens: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("k_bias", torch.zeros(dim), persistent=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        B, N, C = x.shape
        bias = torch.cat((self.q_bias, self.k_bias.to(self.q_bias.dtype), self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=bias.to(x.dtype))
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope is not None:
            sin, cos = rope
            q = apply_rope(q, sin, cos, self.num_prefix_tokens).type_as(v)
            k = apply_rope(k, sin, cos, self.num_prefix_tokens).type_as(v)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class SwiGLU(nn.Module):
    """timm's ``SwiGLU`` (separate gate/value fc + LayerNorm on the hidden state)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1_g = nn.Linear(dim, hidden)
        self.fc1_x = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1_g(x)) * self.fc1_x(x)
        return self.fc2(self.norm(h))


class EvaBlock(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        d = spec.embed_dim
        self.norm1 = nn.LayerNorm(d)
        self.attn = EvaAttention(d, spec.num_heads, spec.num_prefix_tokens)
        self.rope_embed = MixedRope3d(spec.head_dim, spec.num_heads, spec.rope_periods, spec.ndim)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = SwiGLU(d, spec.mlp_hidden)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
        rope = self.rope_embed.embed(coords) if coords is not None else None
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class DinoVolViT(nn.Module):
    """Forward-only dinovol_2 backbone.  ``forward_features`` mirrors upstream's
    ``Eva.forward_features`` but returns a plain dict with the token grid shape."""

    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.spec = spec
        d = spec.embed_dim
        if spec.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if spec.num_reg_tokens:
            self.reg_token = nn.Parameter(torch.zeros(1, spec.num_reg_tokens, d))
        if spec.mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.down_projection = PatchEmbed3d(spec.in_channels, d, spec.patch)
        self.blocks = nn.ModuleList([EvaBlock(spec) for _ in range(spec.depth)])
        self.norm = nn.LayerNorm(d)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_features(self, x: torch.Tensor) -> dict[str, Any]:
        """``x`` (B, C, D, H, W), already normalised; D/H/W divisible by the patch size."""
        s = self.spec
        if x.ndim != 5:
            raise ValueError(f"expected (B, C, D, H, W), got {tuple(x.shape)}")
        if any(int(n) % s.patch for n in x.shape[2:]):
            raise ValueError(f"input {tuple(x.shape[2:])} not divisible by patch {s.patch}")
        grid = tuple(int(n) // s.patch for n in x.shape[2:])
        t = self.down_projection(x).flatten(2).transpose(1, 2)  # (B, N, D), (d h w) order
        B = t.shape[0]
        prefix = []
        if s.class_token:
            prefix.append(self.cls_token.expand(B, -1, -1))
        if s.num_reg_tokens:
            prefix.append(self.reg_token.expand(B, -1, -1))
        t = torch.cat(prefix + [t], dim=1) if prefix else t
        coords = rope_coords(grid, device=t.device, dtype=torch.float32)
        for blk in self.blocks:
            t = blk(t, coords)
        t = self.norm(t)
        n = s.num_prefix_tokens
        return {
            "cls": t[:, 0] if s.class_token else None,
            "reg": t[:, int(s.class_token):n],
            "patch_tokens": t[:, n:],
            "grid": grid,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["patch_tokens"]


def build_dinovol(sd: Mapping[str, Any]) -> DinoVolViT:
    """Uninitialised net whose state-dict keys/shapes match ``sd`` (inventory or checkpoint)."""
    return DinoVolViT(infer_vit_spec(sd))


def _extract_state(ckpt: Any, key: str | None) -> Mapping[str, Any]:
    if isinstance(ckpt, Mapping) and key and key in ckpt:
        ckpt = ckpt[key]
    if not isinstance(ckpt, Mapping):
        raise ValueError("checkpoint is not a state dict")
    return ckpt


def load_dinovol(
    path: str | os.PathLike | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    state_key: str | None = "teacher",
) -> DinoVolViT:
    """Load ``dinovol.pt`` strictly into :class:`DinoVolViT` (eval mode, no grad).

    The checkpoint is memory-mapped and released before returning; the
    ``teacher`` sub-dict and the ``backbone.`` prefix are handled here.
    """
    if path is None:
        path = os.path.join(DEFAULT_MODELS_DIR, DEFAULT_DINO_FILE)
    path = os.path.expanduser(os.fspath(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        sd = strip_prefix(_extract_state(ckpt, state_key), ("module.", "_orig_mod.", "backbone."))
        model = build_dinovol(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            for k in list(missing)[:10]:
                print(f"  missing:    {k}")
            for k in list(unexpected)[:10]:
                print(f"  unexpected: {k}")
            raise RuntimeError(f"state dict mismatch for dinovol ({len(missing)} missing, {len(unexpected)} unexpected)")
        model = model.to(device=device, dtype=dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    finally:
        del ckpt
    return model


# --------------------------------------------------------------------------- #
# input normalisation (dinovol RobustNormalization)
# --------------------------------------------------------------------------- #
def normalize_robust(
    x: np.ndarray | torch.Tensor,
    percentile_lower: float = 1.0,
    percentile_upper: float = 99.0,
    clip: bool = True,
) -> np.ndarray:
    """dinovol's default ``robust`` scheme: clip to [p1, p99], then
    ``(x - median) / (1.4826 * MAD)`` with upstream's degenerate-MAD fallbacks
    (pre-clip std, then half the percentile span, then 1.0)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32).copy()
    flat = arr.reshape(-1)
    if flat.size == 0:
        return arr
    std_preclip = float(np.std(flat))
    span = float(flat.max() - flat.min())
    lo = hi = None
    if clip:
        lo = float(np.percentile(flat, percentile_lower))
        hi = float(np.percentile(flat, percentile_upper))
        np.clip(arr, lo, hi, out=arr)
        flat = arr.reshape(-1)
        span = hi - lo
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    scaled = 1.4826 * mad
    if not np.isfinite(scaled) or scaled < 1e-6:
        cands = [abs(v) for v in (std_preclip, span / 2.0) if np.isfinite(v)]
        scaled = next((c for c in cands if c >= 1e-6), 1.0)
    arr -= median
    arr /= scaled
    return arr


# --------------------------------------------------------------------------- #
# windowed token grid (port of dinovol's compute_patch_embedding_grid)
# --------------------------------------------------------------------------- #
def window_starts(axis: int, window: int, overlap: int) -> list[int]:
    """Upstream ``_compute_window_starts`` (in tokens or voxels, same rule)."""
    if window >= axis:
        return [0]
    step = window - overlap
    if step <= 0:
        raise ValueError(f"overlap {overlap} >= window {window}")
    starts = list(range(0, axis - window + 1, step))
    if starts[-1] != axis - window:
        starts.append(axis - window)
    return starts


def _axis_weights(length: int, overlap: int) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return w
    r = np.arange(length, dtype=np.float32)
    edge = np.minimum(r + 1.0, length - r)
    return np.minimum(edge / float(overlap + 1), 1.0).astype(np.float32)


def window_weights(shape: Sequence[int], overlap: Sequence[int]) -> np.ndarray:
    """Separable linear ramp weight grid (upstream ``_window_weight_grid``)."""
    a, b, c = (_axis_weights(int(s), int(o)) for s, o in zip(shape, overlap))
    return (a[:, None, None] * b[None, :, None] * c[None, None, :]).astype(np.float32)


@torch.inference_mode()
def patch_token_grid(
    vol_u8: np.ndarray,
    model: DinoVolViT,
    window: int = 128,
    overlap_tokens: int = 2,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: str = "robust",
    pca: Mapping[str, np.ndarray] | None = None,
    out: np.ndarray | None = None,
    weight_out: np.ndarray | None = None,
    starts: Sequence[Sequence[int]] | None = None,
    offset: Sequence[int] = (0, 0, 0),
    progress: bool = False,
) -> np.ndarray:
    """L2-normalised token grid ``(C, D/p, H/p, W/p)`` for a uint8 volume.

    Windows of ``window`` voxels with ``overlap_tokens`` tokens of overlap are
    run one at a time (memory-bounded: one window's activations + one fp32
    accumulator of the output grid), each window's tokens are L2-normalised and
    blended with upstream's linear ramp weights, and the result is L2-normalised
    again.  ``C`` is 864, or the PCA output dim when ``pca`` is given (the PCA is
    affine and the stitch is a convex combination, so reducing per window and
    stitching is identical to stitching then reducing, up to the final
    normalisation).

    ``out`` / ``weight_out`` / ``starts`` / ``offset`` let the caller drive an
    external accumulator covering a larger grid (used by :func:`run_dino` to tile
    a region without ever materialising the full 864-dim grid).
    """
    p = model.spec.patch
    vol = np.asarray(vol_u8)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {vol.shape}")
    if any(int(n) % p for n in vol.shape):
        raise ValueError(f"volume {vol.shape} not divisible by the patch size {p}")
    if window % p:
        raise ValueError(f"window {window} not divisible by the patch size {p}")
    grid = tuple(int(n) // p for n in vol.shape)
    wt = min(window // p, *grid)
    ov = tuple(min(int(overlap_tokens), wt - 1) if wt > 1 else 0 for _ in range(3))
    if starts is None:
        starts = [window_starts(g, wt, o) for g, o in zip(grid, ov)]
    wgrid = window_weights((wt, wt, wt), ov)
    dim = int(pca["components"].shape[0]) if pca is not None else model.spec.embed_dim
    own = out is None
    if own:
        out = np.zeros((dim, *grid), dtype=np.float32)
        weight_out = np.zeros(grid, dtype=np.float32)
    assert weight_out is not None
    oz, oy, ox = (int(v) for v in offset)
    n = 0
    for tz in starts[0]:
        for ty in starts[1]:
            for tx in starts[2]:
                tile = vol[tz * p:(tz + wt) * p, ty * p:(ty + wt) * p, tx * p:(tx + wt) * p]
                x = normalize_robust(tile) if normalize == "robust" else tile.astype(np.float32)
                t = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(device=device, dtype=dtype)
                tok = model.forward_features(t)["patch_tokens"][0].float()
                tok = F.normalize(tok, dim=-1).reshape(wt, wt, wt, -1)
                arr = tok.cpu().numpy()
                if pca is not None:
                    arr = apply_pca(arr.reshape(-1, arr.shape[-1]), pca, normalize=False).reshape(wt, wt, wt, -1)
                sl = (slice(oz + tz, oz + tz + wt), slice(oy + ty, oy + ty + wt), slice(ox + tx, ox + tx + wt))
                out[(slice(None), *sl)] += np.moveaxis(arr * wgrid[..., None], -1, 0)
                weight_out[sl] += wgrid
                n += 1
                if progress:
                    print(f"[tsm]   window {n} at token {(tz, ty, tx)}", flush=True)
    if not own:
        return out
    out /= np.maximum(weight_out, np.finfo(np.float32).eps)[None]
    norm = np.linalg.norm(out, axis=0, keepdims=True)
    return out / np.maximum(norm, 1e-12)


# --------------------------------------------------------------------------- #
# PCA 864 -> 64
# --------------------------------------------------------------------------- #
def fit_pca(tokens: np.ndarray, dim: int = 64) -> dict[str, np.ndarray]:
    """PCA of ``(N, C)`` tokens: ``{mean (C,), components (dim, C), explained (dim,)}``."""
    x = np.asarray(tokens, dtype=np.float32).reshape(-1, np.asarray(tokens).shape[-1])
    if x.shape[0] < dim:
        raise ValueError(f"need >= {dim} tokens to fit a {dim}-dim PCA, got {x.shape[0]}")
    mean = x.mean(0)
    xc = x - mean
    # economy SVD; N is at most a few 100k so this stays well inside the budget
    _, s, vh = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(1, x.shape[0] - 1)
    total = float(var.sum()) or 1.0
    return {
        "mean": mean.astype(np.float32),
        "components": np.ascontiguousarray(vh[:dim], dtype=np.float32),
        "explained": (var[:dim] / total).astype(np.float32),
        "n_samples": np.asarray(x.shape[0], dtype=np.int64),
    }


def apply_pca(x: np.ndarray, pca: Mapping[str, np.ndarray], normalize: bool = True) -> np.ndarray:
    """Project ``(..., C)`` onto the PCA basis; optionally L2-normalise the result."""
    a = np.asarray(x, dtype=np.float32)
    y = (a.reshape(-1, a.shape[-1]) - pca["mean"]) @ np.asarray(pca["components"]).T
    if normalize:
        y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    return y.reshape(*a.shape[:-1], y.shape[-1])


def save_pca(path: str, pca: Mapping[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in pca.items()})


def load_pca(path: str) -> dict[str, np.ndarray]:
    with np.load(os.path.expanduser(path)) as z:
        return {k: z[k] for k in z.files}


def ink_reference_embedding(path: str | None = None) -> np.ndarray:
    """The expert ink reference embedding (L2-normalised, 864 dims)."""
    if path is None:
        hits = sorted(glob.glob(os.path.expanduser(DEFAULT_AUX_GLOB)))
        if not hits:
            raise FileNotFoundError(DEFAULT_AUX_GLOB)
        path = hits[0]
    v = np.load(os.path.expanduser(path)).astype(np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# --------------------------------------------------------------------------- #
# signed-permutation spatial transforms (augmentation-safe subset)
# --------------------------------------------------------------------------- #
def signed_permutation(L: torch.Tensor | np.ndarray, tol: float = 1e-5) -> tuple[list[int], list[bool]] | None:
    """``(perm, flips)`` if ``L`` (3x3, ``x_out = L x_in``) is a signed permutation, else None.

    ``perm[i]`` is the input axis that becomes output axis ``i``; ``flips[i]``
    says whether that axis is reversed.  Applying it to a scalar field is
    ``x.permute(perm)`` followed by ``flip`` on the marked axes.
    """
    m = np.asarray(L.detach().cpu() if isinstance(L, torch.Tensor) else L, dtype=np.float64)
    if m.shape != (3, 3):
        return None
    a = np.abs(m)
    if not np.all((a < tol) | (np.abs(a - 1.0) < tol)):
        return None
    perm, flips = [], []
    for i in range(3):
        nz = np.flatnonzero(a[i] > 0.5)
        if nz.size != 1:
            return None
        j = int(nz[0])
        perm.append(j)
        flips.append(bool(m[i, j] < 0))
    if sorted(perm) != [0, 1, 2]:
        return None
    return perm, flips


def apply_signed_permutation(x: torch.Tensor, perm: Sequence[int], flips: Sequence[bool]) -> torch.Tensor:
    """Apply a signed permutation to the trailing 3 axes of ``x`` (channels first)."""
    lead = x.ndim - 3
    order = tuple(range(lead)) + tuple(lead + int(p) for p in perm)
    y = x.permute(order)
    axes = [lead + i for i, f in enumerate(flips) if f]
    return torch.flip(y, axes) if axes else y.contiguous()


# --------------------------------------------------------------------------- #
# cached token store + the feat_distill "dino" source
# --------------------------------------------------------------------------- #
class DinoTokenCache:
    """Read-only view of ``<out_dir>/dino/tokens.zarr`` (fp16, (C, D/8, H/8, W/8)).

    ``attrs``: ``origin_zyx`` (level-0 voxel origin of the cached region),
    ``scale`` (voxels per token, 8), ``voxel_um``, ``l2_normalised``.
    """

    def __init__(self, path: str) -> None:
        import zarr

        self.path = os.path.expanduser(path)
        self.array = zarr.open_array(store=self.path, mode="r")
        a = dict(self.array.attrs)
        self.origin_zyx = tuple(int(v) for v in a.get("origin_zyx", (0, 0, 0)))
        self.scale = int(a.get("scale", 8))
        self.voxel_um = float(a.get("voxel_um", 2.4))
        self.channels = int(self.array.shape[0])
        self.shape_zyx = tuple(int(s) for s in self.array.shape[1:])

    def read_crop(self, origin_zyx: Sequence[int], patch: int) -> np.ndarray | None:
        """``(C, patch/8, patch/8, patch/8)`` float32 for a level-0 crop origin, or
        None when the crop is not fully inside the cached region / not aligned."""
        s = self.scale
        if patch % s:
            raise ValueError(f"patch {patch} not divisible by the token scale {s}")
        n = patch // s
        lo = []
        for i in range(3):
            v = int(origin_zyx[i]) - self.origin_zyx[i]
            if v % s:
                return None
            v //= s
            if v < 0 or v + n > self.shape_zyx[i]:
                return None
            lo.append(v)
        z, y, x = lo
        a = np.asarray(self.array[:, z:z + n, y:y + n, x:x + n], dtype=np.float32)
        return a


class DinoFeatSource(nn.Module):
    """``feat_distill`` term against the cached DINO token grid.

    No network runs in the train loop: the tokens for the crop are read from the
    zarr cache (crop origin / 8, patch / 8), the student's ``enc<stage>`` map
    (/8, 256 ch) is projected to the cache's channel count with one 1x1x1 conv
    and scored with a cosine loss.

    Augmentation: the cached tokens are a stack of scalar fields (one per PCA
    channel), so they resample with the crop exactly like ``sdf`` -- *but* the
    ViT is not rotation-equivariant, so a resampled token grid is only the right
    target when the crop's spatial transform is a signed permutation (flips /
    rot90, which the network does see through its own augmentation only as a
    relabelling of the axes).  Any sample whose transform includes a small
    rotation, scaling or an elastic field is dropped from this term; the fraction
    of samples that were kept is logged as ``feat_dino_frac``.
    """

    def __init__(self, cache: Any, student_ch: int, stage: int = 3, groups: int = 8, norm: bool = True) -> None:
        super().__init__()
        self.cache = cache
        self.stage = int(stage)
        self.out_ch = int(cache.channels)
        mods: list[nn.Module] = [nn.Conv3d(int(student_ch), self.out_ch, 1, bias=not norm)]
        if norm:
            mods.append(nn.GroupNorm(math.gcd(int(groups), self.out_ch), self.out_ch))
        self.proj = nn.Sequential(*mods)

    def target(self, origin_zyx: Sequence[int], patch: int, spatial: Any = None) -> np.ndarray | None:
        """Cached tokens for one crop, transformed like the crop, or None to skip."""
        perm, flips = [0, 1, 2], [False, False, False]
        if spatial is not None and not getattr(spatial, "identity", False):
            if spatial.elastic is not None or spatial.scale is not None or spatial.rotation is not None:
                return None
            sp = signed_permutation(spatial.matrix())
            if sp is None:
                return None
            perm, flips = sp
        a = self.cache.read_crop(origin_zyx, patch)
        if a is None:
            return None
        t = torch.from_numpy(a)
        if perm != [0, 1, 2] or any(flips):
            t = apply_signed_permutation(t, perm, flips)
        return t.contiguous()

    def forward(self, feats: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                spatial: Sequence[Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        s = feats[f"enc{self.stage}"]
        B = s.shape[0]
        patch = int(batch["input"].shape[-1])
        origins = batch["origin_zyx"].tolist()
        keep, targets = [], []
        for i in range(B):
            t = self.target(origins[i], patch, spatial[i] if spatial is not None else None)
            if t is not None:
                keep.append(i)
                targets.append(t)
        frac = len(keep) / max(1, B)
        if not keep:
            return s.sum() * 0.0, {"feat_dino": 0.0, "feat_dino_frac": 0.0}
        tgt = torch.stack(targets).to(device=s.device, dtype=torch.float32)
        proj = self.proj(s[keep])
        if proj.shape[2:] != tgt.shape[2:]:
            tgt = F.interpolate(tgt, size=proj.shape[2:], mode="trilinear", align_corners=False)
        loss = (1.0 - F.cosine_similarity(proj.float(), tgt, dim=1, eps=1e-6)).mean()
        return loss, {"feat_dino": float(loss.detach()), "feat_dino_frac": frac}


# --------------------------------------------------------------------------- #
# tsm dino: build the cache over the config region
# --------------------------------------------------------------------------- #
DINO_DEFAULTS: dict[str, Any] = {
    "model_path": None,        # default ~/.cache/tsm-models/dinovol.pt
    "window": 128,             # ViT window in level-0 voxels (128^3 = 16^3 tokens; 256^3 OOMs at 24 GB)
    "overlap_tokens": 2,       # token-level overlap between windows
    "pca_dim": 64,
    "pca_path": None,          # default <out_dir>/dino/pca.npz; reused when it exists
    "pca_sample_windows": 24,  # windows sampled (evenly spaced) to fit the PCA
    "pca_tokens_per_window": 4096,
    "block_windows": 6,        # output tile = block_windows^3 windows (bounds the fp32 accumulator)
    "ink_ref": None,           # default ~/.cache/tsm-production/models/auxiliary/*/avg_ref_embedding.npy
    "device": None,            # default cuda when available
    "dtype": "float16",        # ViT compute dtype on cuda ("float32" on cpu)
    "normalize": "robust",
    "limit_blocks": 0,         # >0: stop after N blocks (smoke runs)
}


def dino_opts(cfg: Any) -> dict[str, Any]:
    raw = (cfg.extra or {}).get("dino", {}) if hasattr(cfg, "extra") else {}
    if not isinstance(raw, dict):
        raise ValueError("extra.dino must be an object")
    unknown = sorted(set(raw) - set(DINO_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.dino keys: {unknown}")
    o = dict(DINO_DEFAULTS)
    o.update(raw)
    if int(o["window"]) % 8:
        raise ValueError("extra.dino.window must be divisible by 8")
    return o


def _axis_blocks(s: Sequence[int], per: int, g: int) -> list[tuple[list[int], int, int]]:
    """``(contributing window starts, core_lo, core_hi)`` per block along one axis.

    Blocks own *disjoint* token cores that tile ``[0, g)``; the contributing starts are every
    window overlapping that core, including windows owned by the neighbouring groups.  Blocks
    that normalise only their own group's windows and then write their whole span overwrite the
    overlap, which makes the result depend on ``block_windows`` (a memory knob)."""
    st = [int(v) for v in s]
    wt = int(g) - st[-1]  # window_starts always ends at g - window
    groups = [st[i:i + per] for i in range(0, len(st), per)]
    out: list[tuple[list[int], int, int]] = []
    for n, grp in enumerate(groups):
        lo = 0 if n == 0 else grp[0]
        hi = int(g) if n == len(groups) - 1 else groups[n + 1][0]
        contrib = [x for x in st if x < hi and x + wt > lo]
        out.append((contrib, lo, hi))
    return out


def _blocks(starts: Sequence[Sequence[int]], per: int, grid: Sequence[int]
            ) -> Iterator[tuple[tuple[list[int], list[int], list[int]], tuple[int, int, int],
                                tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    """``(contributing starts, block origin, per-axis output core)`` for every block."""
    ax = [_axis_blocks(s, per, int(g)) for s, g in zip(starts, grid)]
    for bz in ax[0]:
        for by in ax[1]:
            for bx in ax[2]:
                b = (bz, by, bx)
                sel = (list(b[0][0]), list(b[1][0]), list(b[2][0]))
                org = tuple(int(sel[i][0]) for i in range(3))
                core = tuple((int(b[i][1]), int(b[i][2])) for i in range(3))
                yield sel, org, core  # type: ignore[misc]


def run_dino(cfg: Any, dry_run: bool = False, force: bool = False, model: DinoVolViT | None = None) -> dict[str, Any]:
    """Run the ViT over ``cfg.region`` at level 0 and write the token cache.

    Layout under ``<out_dir>/dino/``::

        tokens.zarr        float16 (pca_dim, D/8, H/8, W/8), chunks (C, 16, 64, 64)
        ink_likeness.zarr  uint8   (1, D/8, H/8, W/8), u8 = round((cos + 1) * 127.5)
        pca.npz            mean / components / explained / n_samples

    Both grids are stored at the /8 token pitch (not upsampled): the consumers
    (``feat_distill`` at the student's /8 stage, ``tsm labels`` agreement) work
    at that pitch, and ×8 nearest upsampling would cost 512× the bytes for no
    information.  ``scale=8`` in the attrs says how to map level-0 voxels to
    token indices (``token = (voxel - origin_zyx) // 8``).
    """
    import zarr

    from tsm.limits import format_table, read_rss_bytes
    from tsm.volume import VolumeReader

    o = dino_opts(cfg)
    p = 8
    win = int(o["window"])
    wt = win // p
    ov = int(o["overlap_tokens"])
    out_dir = os.path.join(os.path.expanduser(cfg.out_dir), "dino")
    dim = int(o["pca_dim"])
    size = tuple(int(v) for v in cfg.region.size_zyx)
    start = tuple(int(v) for v in cfg.region.start_zyx)
    if any(s % p for s in size):
        raise ValueError(f"region.size_zyx {size} must be divisible by {p}")
    grid = tuple(s // p for s in size)
    starts = [window_starts(g, min(wt, g), ov) for g in grid]
    n_windows = int(np.prod([len(s) for s in starts]))
    per = int(o["block_windows"])
    span = [max(c[-1] - c[0] + (int(g) - int(s[-1])) for c, _, _ in _axis_blocks(s, per, int(g)))
            for s, g in zip(starts, grid)]
    embed = model.spec.embed_dim if model is not None else 864
    acc_bytes = int(np.prod(span)) * embed * 4
    tok_bytes = dim * int(np.prod(grid)) * 2
    device = o["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, str(o["dtype"])) if device == "cuda" else torch.float32

    print(f"[tsm] dino region {start} size {size} -> token grid {grid} ({np.prod(grid) / 1e6:.2f} M tokens)")
    print(f"[tsm] dino windows {win}^3 ({wt}^3 tokens, overlap {ov}): {n_windows} on {device}/{str(dtype).split('.')[-1]}")
    n_params = model.num_params() if model is not None else 215_859_168
    print(format_table([
        ["name", "bytes", "GiB"],
        ["ViT weights (fp32)", n_params * 4, f"{n_params * 4 / (1 << 30):.2f}"],
        [f"block accumulator {tuple(span)} x {embed} fp32", acc_bytes, f"{acc_bytes / (1 << 30):.2f}"],
        ["tokens.zarr fp16 (on disk)", tok_bytes, f"{tok_bytes / (1 << 30):.2f}"],
        ["ink_likeness.zarr u8 (on disk)", int(np.prod(grid)), f"{np.prod(grid) / (1 << 30):.4f}"],
    ]))
    if dry_run:
        print("[tsm] dry run: stopping before I/O")
        return {"windows": n_windows, "grid": grid, "tokens_bytes": tok_bytes}

    os.makedirs(out_dir, exist_ok=True)
    from .cli import open_ct  # lazy: cli imports this module
    reader = open_ct(cfg, 0)  # local cache first, then primary/alt url
    model = model if model is not None else load_dinovol(o["model_path"], device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype).eval()
    ref = torch.from_numpy(ink_reference_embedding(o["ink_ref"]))

    pca_path = o["pca_path"] or os.path.join(out_dir, "pca.npz")
    if os.path.exists(pca_path) and not force:
        pca = load_pca(pca_path)
        print(f"[tsm] dino PCA reused from {pca_path} ({pca['components'].shape})")
    else:
        pca = _fit_region_pca(reader, model, start, starts, wt, p, o, device, dtype)
        save_pca(pca_path, pca)
        print(f"[tsm] dino PCA {pca['components'].shape} fitted on {int(pca['n_samples'])} tokens "
              f"(explained {float(pca['explained'].sum()):.3f}) -> {pca_path}")

    tok_path = os.path.join(out_dir, "tokens.zarr")
    ink_path = os.path.join(out_dir, "ink_likeness.zarr")
    attrs = {"origin_zyx": list(start), "scale": p, "voxel_um": float(cfg.volume.voxel_um),
             "l2_normalised": True, "window": win, "overlap_tokens": ov, "model": "dinovol_v2_ps8"}
    tokens = _open_out(zarr, tok_path, (dim, *grid), (dim, 16, 64, 64), np.float16, attrs, force)
    ink = _open_out(zarr, ink_path, (1, *grid), (1, 16, 64, 64), np.uint8,
                    {**attrs, "encoding": "u8 = round((cos + 1) * 127.5)", "l2_normalised": False}, force)

    done = 0
    for sel, org, core in _blocks(starts, per, grid):
        wt_ax = [int(grid[i]) - int(starts[i][-1]) for i in range(3)]
        span = tuple(sel[i][-1] - sel[i][0] + wt_ax[i] for i in range(3))
        vol = reader.read(start[0] + org[0] * p, start[0] + (org[0] + span[0]) * p,
                          start[1] + org[1] * p, start[1] + (org[1] + span[1]) * p,
                          start[2] + org[2] * p, start[2] + (org[2] + span[2]) * p)
        acc = np.zeros((model.spec.embed_dim, *span), dtype=np.float32)
        wsum = np.zeros(span, dtype=np.float32)
        rel = [[s - org[i] for s in sel[i]] for i in range(3)]
        patch_token_grid(vol, model, window=win, overlap_tokens=ov, device=device, dtype=dtype,
                         normalize=str(o["normalize"]), out=acc, weight_out=wsum, starts=rel)
        acc /= np.maximum(wsum, np.finfo(np.float32).eps)[None]
        g = torch.from_numpy(acc)
        g = g / g.norm(dim=0, keepdim=True).clamp_min(1e-12)
        cos = torch.einsum("czyx,c->zyx", g, ref).clamp(-1, 1)
        red = apply_pca(np.moveaxis(g.numpy(), 0, -1), pca, normalize=True)
        # write only this block's disjoint core: the halo tokens are incomplete here (their other
        # contributing windows live in the neighbouring blocks, which own them)
        loc = tuple(slice(core[i][0] - org[i], core[i][1] - org[i]) for i in range(3))
        sl = tuple(slice(core[i][0], core[i][1]) for i in range(3))
        tokens[(slice(None), *sl)] = np.moveaxis(red, -1, 0)[(slice(None), *loc)].astype(np.float16)
        ink[(slice(0, 1), *sl)] = np.round((cos.numpy()[loc] + 1.0) * 127.5).astype(np.uint8)[None]
        del acc, wsum, g, red
        done += 1
        print(f"[tsm] dino block {done} core {core} at token {org} span {span} rss={read_rss_bytes() / (1 << 20):.0f} MiB", flush=True)
        if int(o["limit_blocks"]) and done >= int(o["limit_blocks"]):
            print(f"[tsm] dino stopping after {done} blocks (limit_blocks)")
            break
    print(f"[tsm] dino wrote {tok_path} ({tok_bytes / (1 << 30):.2f} GiB) and {ink_path}")
    return {"windows": n_windows, "grid": grid, "blocks": done, "tokens": tok_path, "ink": ink_path, "pca": pca_path}


def _open_out(zarr: Any, path: str, shape: Sequence[int], chunks: Sequence[int], dtype: Any,
              attrs: Mapping[str, Any], force: bool) -> Any:
    if os.path.exists(os.path.join(path, "zarr.json")) and not force:
        return zarr.open_array(store=path, mode="r+")
    return zarr.create_array(store=path, shape=tuple(int(s) for s in shape), chunks=tuple(int(c) for c in chunks),
                             dtype=np.dtype(dtype), fill_value=0, attributes=dict(attrs), overwrite=True)


def _fit_region_pca(reader: Any, model: DinoVolViT, start: Sequence[int], starts: Sequence[Sequence[int]],
                    wt: int, p: int, o: Mapping[str, Any], device: Any, dtype: Any) -> dict[str, np.ndarray]:
    """Fit the 864 -> pca_dim PCA on tokens from evenly spaced windows."""
    all_windows = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    n = min(int(o["pca_sample_windows"]), len(all_windows))
    idx = np.linspace(0, len(all_windows) - 1, n).round().astype(int)
    rng = np.random.default_rng(0)
    per_win = int(o["pca_tokens_per_window"])
    chunks = []
    for c, i in enumerate(idx):
        tz, ty, tx = all_windows[int(i)]
        vol = reader.read(start[0] + tz * p, start[0] + (tz + wt) * p,
                          start[1] + ty * p, start[1] + (ty + wt) * p,
                          start[2] + tx * p, start[2] + (tx + wt) * p)
        x = normalize_robust(vol) if o["normalize"] == "robust" else vol.astype(np.float32)
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(device=device, dtype=dtype)
            tok = F.normalize(model.forward_features(t)["patch_tokens"][0].float(), dim=-1).cpu().numpy()
        k = min(per_win, tok.shape[0])
        chunks.append(tok[rng.choice(tok.shape[0], size=k, replace=False)])
        print(f"[tsm] dino PCA sample {c + 1}/{n} at token {(tz, ty, tx)}", flush=True)
    return fit_pca(np.concatenate(chunks, 0), dim=int(o["pca_dim"]))
