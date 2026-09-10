"""Pure-PyTorch reimplementation of the ``winding_model_9um`` teacher (villa "H2").

The published checkpoint (``scrollprize/winding_model_9um``, ``ckpt_final.pth``,
state under ``model_ema``) has *no public source code*.  Everything here is
reconstructed from three sources, and every design decision is tagged in the
docstrings as one of:

* **confirmed from weights** -- follows from the tensor inventory / weight
  statistics of the checkpoint itself (hard evidence);
* **confirmed from config** -- follows from ``architecture_config`` stored in
  the checkpoint;
* **derived from reference** -- follows the recovered previous reverse
  engineering (Codex session log, ``WindingTeacher``), which itself was an
  inference; kept because it is self-consistent with the shapes;
* **guess** -- a convention that cannot be verified from shapes at all.

Architecture (``WindingNet``), in the checkpoint's own key names::

    stages.{0,1,2}.layers.{0,1,3,4}      conv3(s=(2,2,1)) LN GELU conv3 LN GELU  [32,64,96]
    trunk_in.{0,1}                        1x1 conv 96->192, LN, GELU
    stem_blocks.{0,1}.{conv1,norm1,conv2,norm2}   residual 3x3x3 blocks @192
    attention_blocks.{0..5}.{ray_norm, ray_attention.{qkv,proj,relative_bias[6,257]},
                             transverse_norm, transverse_attention.{qkv,proj,relative_bias[6,31,31]},
                             mlp_norm, mlp.{0,2}}                 pre-norm axial transformer
    decoder.{0,1,3,4}                    conv3(256->96) LN GELU conv3 LN GELU  (trunk x2 + stage-1 skip)
    full_resolution_head.{0,1,3}         1x1 (96+32 -> 32) LN GELU 1x1 (32 -> 4)  @ transverse/2
                                         -> 2x2 transverse pixel-shuffle -> 1 channel @ full res

Tensor layout: the input tile is ``[B, 2, T, T, R]`` with the *ray axis last*
(``R = ray_length = 384`` samples along a straight ray, ``T = transverse_size =
128`` in the two transverse directions).  Evidence:

* transverse attention bias ``[6, 31, 31]`` = ``2*16-1`` -> 16x16 transverse
  tokens -> transverse 128 downsampled by 8 by the three stride-2 stages
  (confirmed from weights + config ``transverse_size=128``);
* ray attention bias ``[6, 257]`` = ``2*128+1`` (config ``max_relative_distance
  =128``) and *every* offset bin, including |offset| > 100, carries trained
  values of the same magnitude as the centre bins.  With 384/8 = 48 ray tokens
  the bins beyond +-47 would never receive a gradient, so the ray axis is *not*
  downsampled: the encoder strides are ``(2, 2, 1)`` (confirmed from weights);
* the 3x3x3 kernels of ``stages.0.layers.0``, ``stem_blocks.0.conv1`` and
  ``decoder.0`` are statistically identical along their first two spatial axes
  and distinct along the last one -> the anisotropic (ray) axis is the last
  spatial axis (confirmed from weights; the reference also used ray-last);
* ``full_resolution_head.3`` has 4 output channels whose biases
  (-2.839, -2.840, -2.839, -2.840) and weight norms are identical to 3
  decimals -> the 4 channels are four copies of *one* quantity, i.e. a 2x2
  sub-pixel shuffle of the transverse/2 head back to full transverse
  resolution (config ``full_resolution_head=true``, ``use_crossing_head=false``,
  ``use_variance_head=false``) (confirmed from weights, derived from reference).

Output semantics: the single output channel is a per-sample *winding-rate
logit*; ``density = softplus(logit)`` is the winding rate in windings per ray
sample (= per level-2 voxel at ``spacing=1.0``; villa calls the unit "wv",
winding voxels) and ``phase = cumsum(density)`` along the ray is the winding
count (1.0 per sheet) relative to the start of the ray.  Evidence: config
``phase_initial_increment=0.04`` matches a softplus bias init
(``softplus(-3.2)=0.04``); the trained bias ``softplus(-2.84)=0.057``
windings/voxel corresponds to a mean sheet spacing of 17.6 voxels = 170 um at
9.6 um, which is the papyrus sheet pitch; villa's consumer
(``neural_winding_losses.dense_metric_density_loss``) integrates a
``density_windings_per_wv`` field along polylines and compares with integer
winding counts (derived from reference + villa consumer).  ``phase`` is only
relative: villa registers rays against a spiral model; ``WindingTeacher``
returns it relative to the first sample of each ray.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .teachers import DEFAULT_MODELS_DIR, shapes_of, strip_prefix

__all__ = [
    "WINDING_ENABLED",
    "WindingArch",
    "WindingNet",
    "infer_winding_arch",
    "build_winding_net",
    "load_winding_net",
    "prepare_input",
    "decode_outputs",
    "Axis",
    "ArrayReader",
    "RayFrame",
    "ray_frame",
    "sample_ray_tile",
    "RayProfile",
    "WindingTeacher",
    "validate",
]

#: Kill switch (plan: "winding.enabled").  ``validate()`` on the real Paris 4
#: ROI did NOT pass (the reimplemented forward reproduces the checkpoint's keys
#: and runs, but its winding-rate output tracks sheet spacing only weakly, see
#: docs/winding_status.md), so the student's winding head is supervised from
#: lasagna cos/grad_mag + axis geometry instead.  Flip to True only after
#: ``validate()`` passes.
WINDING_ENABLED = False

WINDING_FILE = "winding_model_9um.pth"
WINDING_STATE_KEY = "model_ema"  # confirmed from config (model.json "state_key")
WINDING_VOXEL_UM = 9.6  # training volumes: 2.4 um zarr at volume_scale=2 (confirmed from config)


# --------------------------------------------------------------------------- #
# Architecture description (shape driven)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindingArch:
    """Hyper-parameters of ``WindingNet``; all inferable from the state dict."""

    in_channels: int = 2
    encoder_channels: tuple[int, ...] = (32, 64, 96)
    trunk_dim: int = 192
    trunk_stem_blocks: int = 2
    axial_attention_blocks: int = 6
    attention_heads: int = 6
    decoder_dim: int = 96
    max_relative_distance: int = 128  # ray relative-bias half width
    transverse_tokens: int = 16  # transverse extent at the trunk (128 / 2**3)
    out_channels: int = 4  # 2x2 sub-pixel shuffle of one output quantity
    mlp_ratio: int = 4
    norm_kind: str = "layer"  # 'layer' (per-position channel LayerNorm) | 'group1' (GroupNorm(1)); guess
    ray_first: bool = False  # internal layout [B,C,R,T,T] instead of [B,C,T,T,R] (experiment flag; API is ray-last)
    ray_strides: tuple[int, ...] = (1, 1, 1)  # per-stage stride along the ray (transverse stride is always 2)
    bias_sign: int = 1  # +1: bias index = key - query (guess); -1: query - key
    residual_preact: bool = False  # experiment flag: pre-activation residual blocks
    transverse_first: bool = False  # experiment flag: transverse attention before ray attention
    upsample_mode: str = "trilinear"  # experiment flag: 'trilinear' | 'nearest'
    act: str = "gelu"  # experiment flag: 'gelu' | 'silu' | 'relu' | 'leaky'

    @property
    def n_stages(self) -> int:
        return len(self.encoder_channels)

    @property
    def transverse_size(self) -> int:
        return self.transverse_tokens * 2 ** self.n_stages


def _count(shapes: Mapping[str, Any], fmt: str) -> int:
    i = 0
    while fmt.format(i) in shapes:
        i += 1
    return i


def infer_winding_arch(sd: Mapping[str, Any], norm_kind: str = "layer") -> WindingArch:
    """Infer ``WindingArch`` from state-dict shapes (tensors, ``{'shape':..}`` entries or tuples)."""
    shapes = shapes_of(strip_prefix(sd))
    n_stages = _count(shapes, "stages.{}.layers.0.weight")
    if n_stages == 0:
        raise ValueError("no 'stages.{i}.layers.0.weight' keys: not a winding state dict")
    channels = tuple(int(shapes[f"stages.{s}.layers.0.weight"][0]) for s in range(n_stages))
    in_channels = int(shapes["stages.0.layers.0.weight"][1])
    trunk = int(shapes["trunk_in.0.weight"][0])
    stem_blocks = _count(shapes, "stem_blocks.{}.conv1.weight")
    n_blocks = _count(shapes, "attention_blocks.{}.mlp.0.weight")
    heads, n_ray = (int(v) for v in shapes["attention_blocks.0.ray_attention.relative_bias"])
    tb = shapes["attention_blocks.0.transverse_attention.relative_bias"]
    if int(tb[0]) != heads or int(tb[1]) != int(tb[2]) or int(tb[1]) % 2 == 0 or n_ray % 2 == 0:
        raise ValueError(f"unexpected relative bias shapes {tb} / {n_ray}")
    transverse_tokens = (int(tb[1]) + 1) // 2
    mlp_ratio = int(shapes["attention_blocks.0.mlp.0.weight"][0]) // trunk
    decoder_dim = int(shapes["decoder.0.weight"][0])
    dec_in = int(shapes["decoder.0.weight"][1])
    head_in = int(shapes["full_resolution_head.0.weight"][1])
    out_channels = int(shapes["full_resolution_head.3.weight"][0])
    if dec_in != trunk + channels[-2]:
        raise ValueError(f"decoder.0 expects {dec_in} channels, trunk {trunk} + skip {channels[-2]} != that")
    if head_in != decoder_dim + channels[0]:
        raise ValueError(f"head.0 expects {head_in} channels, decoder {decoder_dim} + skip {channels[0]} != that")
    return WindingArch(
        in_channels=in_channels, encoder_channels=channels, trunk_dim=trunk, trunk_stem_blocks=stem_blocks,
        axial_attention_blocks=n_blocks, attention_heads=heads, decoder_dim=decoder_dim,
        max_relative_distance=(n_ray - 1) // 2, transverse_tokens=transverse_tokens, out_channels=out_channels,
        mlp_ratio=mlp_ratio, norm_kind=norm_kind,
    )


# --------------------------------------------------------------------------- #
# Building blocks (attribute names reproduce the checkpoint keys exactly)
# --------------------------------------------------------------------------- #
class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel dim of an ``[B, C, *spatial]`` tensor.

    guess: the norm layers carry ``weight``/``bias`` of shape ``[C]`` which is
    equally consistent with GroupNorm/InstanceNorm; a per-position channel
    LayerNorm is the natural partner of the GELU/pre-norm transformer trunk and
    is what the reference used.  ``norm_kind='group1'`` swaps in GroupNorm(1).
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(value.movedim(1, -1)).movedim(-1, 1)


def _act(kind: str = "gelu") -> nn.Module:
    return {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.01)}[kind]()


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "layer":
        return ChannelLayerNorm(channels)
    if kind.startswith("group"):
        return nn.GroupNorm(int(kind[5:]), channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"unknown norm_kind {kind!r}")


class WindingEncoderStage(nn.Module):
    """``layers = Sequential(conv3 s=(2,2,1), norm, GELU, conv3, norm, GELU)``.

    Key indices 0,1,3,4 carry parameters (confirmed from weights: the
    parameter-free slots 2 and 5 are the activations).  The stride is
    ``(2, 2, 1)``: transverse halved, ray kept (confirmed from weights, see
    module docstring).
    """

    def __init__(self, cin: int, width: int, norm_kind: str, stride: tuple[int, int, int] = (2, 2, 1),
                 act: str = "gelu"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(cin, width, 3, stride=stride, padding=1),
            _make_norm(norm_kind, width),
            _act(act),
            nn.Conv3d(width, width, 3, padding=1),
            _make_norm(norm_kind, width),
            _act(act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WindingResidualBlock(nn.Module):
    """``x + norm2(conv2(gelu(norm1(conv1(x)))))`` (derived from reference; the
    conv->norm order mirrors the encoder stage's Sequential)."""

    def __init__(self, width: int, norm_kind: str, preact: bool = False, act: str = "gelu"):
        super().__init__()
        self.preact = preact
        self.act = _act(act)
        self.conv1 = nn.Conv3d(width, width, 3, padding=1)
        self.norm1 = _make_norm(norm_kind, width)
        self.conv2 = nn.Conv3d(width, width, 3, padding=1)
        self.norm2 = _make_norm(norm_kind, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act
        if self.preact:
            return x + self.conv2(a(self.norm2(self.conv1(a(self.norm1(x))))))
        h = a(self.norm1(self.conv1(x)))
        return x + self.norm2(self.conv2(h))


def _attend(qkv: torch.Tensor, heads: int, bias: torch.Tensor) -> torch.Tensor:
    """``qkv``: [N, L, 3*D] -> attention output [N, L, D] with additive bias [H, L, L].

    guess: qkv is split as ``(3, heads, head_dim)`` (timm convention) and the
    bias is added to the scaled logits (Swin-style relative position bias).
    """
    n, length, three_d = qkv.shape
    d = three_d // 3
    q, k, v = qkv.reshape(n, length, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None].to(q.dtype))
    return out.transpose(1, 2).reshape(n, length, d)


class RayAttention(nn.Module):
    """Self-attention along the ray with a 1-D relative bias ``[H, 2*max+1]``.

    Offsets ``key - query`` are clamped to ``+-max_relative_distance`` (config)
    and indexed at ``offset + max`` (guess: sign convention; the trained bias
    is mildly asymmetric so this matters only slightly).
    """

    def __init__(self, dim: int, heads: int, max_distance: int, sign: int = 1):
        super().__init__()
        self.num_heads = heads
        self.max_distance = max_distance
        self.sign = sign
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_distance + 1))

    def bias(self, length: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(length, device=device)
        offsets = (self.sign * (pos[None, :] - pos[:, None])).clamp(-self.max_distance, self.max_distance)
        return self.relative_bias[:, offsets + self.max_distance]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [N, L, D]
        return self.proj(_attend(self.qkv(tokens), self.num_heads, self.bias(tokens.shape[1], tokens.device)))


class TransverseAttention(nn.Module):
    """Self-attention over the ``h x w`` transverse token plane with a 2-D
    relative bias ``[H, 2h-1, 2w-1]`` (confirmed from weights: 31 = 2*16-1)."""

    def __init__(self, dim: int, heads: int, tokens: int, sign: int = 1):
        super().__init__()
        self.num_heads = heads
        self.max_distance = tokens - 1
        self.sign = sign
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * tokens - 1, 2 * tokens - 1))

    def bias(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        m = self.max_distance
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        dr = (self.sign * (rows[None, :] - rows[:, None])).clamp(-m, m) + m  # [h, h]
        dc = (self.sign * (cols[None, :] - cols[:, None])).clamp(-m, m) + m  # [w, w]
        b = self.relative_bias[:, dr[:, None, :, None], dc[None, :, None, :]]  # [H, h, w, h, w]
        return b.reshape(self.num_heads, h * w, h * w)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:  # [N, h*w, D]
        return self.proj(_attend(self.qkv(tokens), self.num_heads, self.bias(h, w, tokens.device)))


class AxialAttentionBlock(nn.Module):
    """Pre-norm block: ray attention -> transverse attention -> MLP (derived
    from reference; the order of the two attentions is a guess)."""

    def __init__(self, dim: int, heads: int, ray_distance: int, transverse_tokens: int, mlp_ratio: int = 4,
                 bias_sign: int = 1, transverse_first: bool = False, act: str = "gelu"):
        super().__init__()
        self.transverse_first = transverse_first
        self.ray_norm = nn.LayerNorm(dim)
        self.ray_attention = RayAttention(dim, heads, ray_distance, bias_sign)
        self.transverse_norm = nn.LayerNorm(dim)
        self.transverse_attention = TransverseAttention(dim, heads, transverse_tokens, bias_sign)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), _act(act), nn.Linear(mlp_ratio * dim, dim))

    def _ray(self, tokens: torch.Tensor) -> torch.Tensor:
        b, h, w, r, d = tokens.shape
        cols = self.ray_norm(tokens).reshape(b * h * w, r, d)
        return tokens + self.ray_attention(cols).reshape(b, h, w, r, d)

    def _transverse(self, tokens: torch.Tensor) -> torch.Tensor:
        b, h, w, r, d = tokens.shape
        planes = self.transverse_norm(tokens).permute(0, 3, 1, 2, 4).reshape(b * r, h * w, d)
        planes = self.transverse_attention(planes, h, w).reshape(b, r, h, w, d).permute(0, 2, 3, 1, 4)
        return tokens + planes

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B, h, w, R, D]
        if self.transverse_first:
            tokens = self._ray(self._transverse(tokens))
        else:
            tokens = self._transverse(self._ray(tokens))
        return tokens + self.mlp(self.mlp_norm(tokens))


class WindingNet(nn.Module):
    """The ``winding_model_9um`` network.  ``forward(x)`` with ``x`` =
    ``[B, 2, T, T, R]`` (channels: normalised CT, validity; see
    ``prepare_input``) returns the winding-rate *logit* ``[B, T, T, R]`` at
    full resolution.  Use ``decode_outputs`` for density/phase.
    """

    def __init__(self, arch: WindingArch):
        super().__init__()
        self.arch = arch
        ch = arch.encoder_channels
        nk = arch.norm_kind
        act = arch.act
        stages = []
        cin = arch.in_channels
        for width, rs in zip(ch, arch.ray_strides):
            stride = (rs, 2, 2) if arch.ray_first else (2, 2, rs)
            stages.append(WindingEncoderStage(cin, width, nk, stride, act))
            cin = width
        self.stages = nn.ModuleList(stages)
        self.trunk_in = nn.Sequential(nn.Conv3d(ch[-1], arch.trunk_dim, 1), _make_norm(nk, arch.trunk_dim), _act(act))
        self.stem_blocks = nn.ModuleList(WindingResidualBlock(arch.trunk_dim, nk, arch.residual_preact, act)
                                         for _ in range(arch.trunk_stem_blocks))
        self.attention_blocks = nn.ModuleList(
            AxialAttentionBlock(arch.trunk_dim, arch.attention_heads, arch.max_relative_distance,
                                arch.transverse_tokens, arch.mlp_ratio, arch.bias_sign, arch.transverse_first, act)
            for _ in range(arch.axial_attention_blocks)
        )
        dd = arch.decoder_dim
        self.decoder = nn.Sequential(
            nn.Conv3d(arch.trunk_dim + ch[-2], dd, 3, padding=1), _make_norm(nk, dd), _act(act),
            nn.Conv3d(dd, dd, 3, padding=1), _make_norm(nk, dd), _act(act),
        )
        self.full_resolution_head = nn.Sequential(
            nn.Conv3d(dd + ch[0], ch[0], 1), _make_norm(nk, ch[0]), _act(act),
            nn.Conv3d(ch[0], arch.out_channels, 1),
        )
        self.shuffle = int(round(math.sqrt(arch.out_channels)))
        if self.shuffle ** 2 != arch.out_channels:
            raise ValueError(f"out_channels {arch.out_channels} is not a square (2-D sub-pixel shuffle)")

    def _up(self, h: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        if self.arch.upsample_mode == "nearest":
            return F.interpolate(h, size=tuple(size), mode="nearest")
        return F.interpolate(h, size=tuple(size), mode="trilinear", align_corners=False)

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """[B, s*s, h, w, R] -> [B, s*h, s*w, R]; channel c = s*di + dj (guess)."""
        b, _, h, w, r = x.shape
        s = self.shuffle
        return x.reshape(b, s, s, h, w, r).permute(0, 3, 1, 4, 2, 5).reshape(b, h * s, w * s, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.arch.in_channels:
            raise ValueError(f"expected [B, {self.arch.in_channels}, T, T, R], got {tuple(x.shape)}")
        skips = []
        rf = self.arch.ray_first
        h = x.permute(0, 1, 4, 2, 3) if rf else x  # experiment flag: internal ray-first layout
        for stage in self.stages:
            h = stage(h)
            skips.append(h)
        h = self.trunk_in(h)
        for blk in self.stem_blocks:
            h = blk(h)
        tokens = h.permute(0, 3, 4, 2, 1) if rf else h.permute(0, 2, 3, 4, 1)  # [B, h, w, R, D]
        for blk in self.attention_blocks:
            tokens = blk(tokens)
        h = tokens.permute(0, 4, 3, 1, 2) if rf else tokens.permute(0, 4, 1, 2, 3)
        # derived from reference: trilinear x2 in the transverse plane only.  Concat order
        # [trunk | skip] and [decoder | skip] is confirmed from weights: the input-channel
        # weight norms of decoder.0 / full_resolution_head.0 jump at index 192 / 96.
        h = self._up(h, skips[-2].shape[2:])
        h = self.decoder(torch.cat((h, skips[-2]), dim=1))
        h = self._up(h, skips[0].shape[2:])
        logits = self.full_resolution_head(torch.cat((h, skips[0]), dim=1))
        if rf:
            logits = logits.permute(0, 1, 3, 4, 2)
        out = self._pixel_shuffle(logits)
        if out.shape[-1] != x.shape[-1]:  # ray downsampled by stage 0: linear upsample along the ray
            out = F.interpolate(out, size=(out.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
        return out


def build_winding_net(sd: Mapping[str, Any], norm_kind: str = "layer") -> WindingNet:
    return WindingNet(infer_winding_arch(sd, norm_kind=norm_kind))


def load_winding_net(
    models_dir: str | os.PathLike = DEFAULT_MODELS_DIR,
    device: str | torch.device = "cpu",
    norm_kind: str = "layer",
    file: str = WINDING_FILE,
) -> tuple[WindingNet, dict[str, Any]]:
    """Strictly load the EMA weights (mmap; checkpoint released before return).

    Returns ``(net.eval(), architecture_config)``."""
    path = os.path.join(os.fspath(models_dir), file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        sd = strip_prefix(ckpt[WINDING_STATE_KEY] if isinstance(ckpt, Mapping) and WINDING_STATE_KEY in ckpt else ckpt)
        cfg = dict(ckpt.get("config", {})) if isinstance(ckpt, Mapping) else {}
        net = build_winding_net(sd, norm_kind=norm_kind)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[winding] strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")
            for k in list(missing)[:10]:
                print(f"  missing:    {k}")
            for k in list(unexpected)[:10]:
                print(f"  unexpected: {k}")
            raise RuntimeError("state dict mismatch for winding_model_9um")
        net = net.to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
    finally:
        del ckpt
    return net, cfg


# --------------------------------------------------------------------------- #
# Input normalisation and output decoding
# --------------------------------------------------------------------------- #
def prepare_input(ct: torch.Tensor, valid: torch.Tensor, mode: str = "zscore_masked") -> torch.Tensor:
    """Build the 2-channel input ``[B, 2, T, T, R]`` from CT ``[B, T, T, R]``
    (uint8 scale, 0..255) and a validity mask ``[B, T, T, R]``.

    Channel 0 = normalised CT, channel 1 = validity (float).  The first conv
    uses both input channels with comparable weight norms (0.62 / 0.37,
    confirmed from weights) so the second channel is a real signal, and
    "validity of the sample (inside the volume)" is the reference's reading.
    Normalisation modes (guess; ``zscore_masked`` is the reference's choice and
    the one that validated on real data):

      zscore_masked  (ct - mean_valid) / std_valid, zeroed where invalid
      div255         ct / 255
      zscore         plain per-tile z-score
    """
    ct = ct.float()
    v = valid.to(ct.dtype)
    if mode == "zscore_masked":
        mass = v.sum((1, 2, 3)).clamp_min(1.0)
        mean = (ct * v).sum((1, 2, 3)) / mass
        centered = (ct - mean[:, None, None, None]) * v
        std = (centered.square().sum((1, 2, 3)) / mass).sqrt()
        x0 = centered / (std[:, None, None, None] + 1e-6)
    elif mode == "div255":
        x0 = ct / 255.0 * v
    elif mode == "zscore":
        x0 = (ct - ct.mean((1, 2, 3), keepdim=True)) / (ct.std((1, 2, 3), keepdim=True) + 1e-6)
    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")
    return torch.stack((x0, v), dim=1)


def decode_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """``logits`` [B, T, T, R] -> ``density`` (windings / sample, softplus) and
    ``phase`` (cumulative windings along the ray, relative to sample 0)."""
    density = F.softplus(logits.float())
    return {"logits": logits, "density": density, "phase": density.cumsum(dim=-1)}


# --------------------------------------------------------------------------- #
# Umbilicus axis
# --------------------------------------------------------------------------- #
class Axis:
    """Scroll axis (umbilicus) polyline in *full-resolution* XYZ voxels.

    Accepts the render3d-style JSON (``control_points`` or ``points`` entries
    with ``x, y, z``) and interpolates ``(x, y)`` linearly in ``z``.
    ``xy_at(z, level)`` takes/returns coordinates at zarr ``level``
    (full-res / 2**level)."""

    def __init__(self, points_xyz: Sequence[Sequence[float]], scale_xyz: Sequence[float] = (1.0, 1.0, 1.0)):
        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3) * np.asarray(scale_xyz, dtype=np.float64)
        order = np.argsort(pts[:, 2], kind="stable")
        pts = pts[order]
        if len(pts) < 2:
            raise ValueError("axis needs at least two points")
        self.points = pts
        self.z = pts[:, 2].tolist()

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "Axis":
        d = json.load(open(path))
        entries = d.get("control_points") or d.get("points")
        if entries is None:
            raise ValueError(f"{path}: no 'control_points' or 'points'")
        if isinstance(entries[0], Mapping):
            pts = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in entries]
        else:
            pts = [tuple(float(v) for v in p[:3]) for p in entries]
        return cls(pts)

    def xy_at(self, z: float, level: int = 0) -> tuple[float, float]:
        f = 2.0 ** level
        zf = float(z) * f
        zs = self.z
        i = bisect.bisect_right(zs, zf)
        i = min(max(i, 1), len(zs) - 1)
        z0, z1 = zs[i - 1], zs[i]
        t = 0.0 if z1 == z0 else (zf - z0) / (z1 - z0)
        t = min(max(t, 0.0), 1.0)  # clamp: hold the end points outside the range
        p = self.points[i - 1] * (1 - t) + self.points[i] * t
        return float(p[0] / f), float(p[1] / f)

    def xyz_at(self, z: float, level: int = 0) -> np.ndarray:
        x, y = self.xy_at(z, level)
        return np.array([x, y, float(z)], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Ray-tile sampling
# --------------------------------------------------------------------------- #
class ArrayReader:
    """In-memory stand-in for ``VolumeReader`` (``shape`` + ``read``), optionally
    representing a sub-block of a larger volume at ``origin_zyx``."""

    def __init__(self, data: np.ndarray, origin_zyx: Sequence[int] = (0, 0, 0), full_shape: Sequence[int] | None = None):
        self.data = np.asarray(data)
        if self.data.ndim != 3:
            raise ValueError("ArrayReader needs a 3-D array")
        self.origin = tuple(int(v) for v in origin_zyx)
        self.shape = tuple(int(v) for v in (full_shape if full_shape is not None else self.data.shape))

    def read(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        req = (z1 - z0, y1 - y0, x1 - x0)
        out = np.zeros(req, dtype=np.uint8)
        lo = [max(0, a - o) for a, o in zip((z0, y0, x0), self.origin)]
        hi = [min(s, a - o) for s, a, o in zip(self.data.shape, (z1, y1, x1), self.origin)]
        if any(h <= l for l, h in zip(lo, hi)):
            return out
        blk = self.data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = [l + o - a for l, o, a in zip(lo, self.origin, (z0, y0, x0))]
        out[d[0]:d[0] + blk.shape[0], d[1]:d[1] + blk.shape[1], d[2]:d[2] + blk.shape[2]] = blk
        return out


@dataclass(frozen=True)
class RayFrame:
    """Geometry of one ray tile.

    Sample ``(i, j, k)`` of the ``[T, T, R]`` tile sits at
    ``origin + (i - T//2)*spacing*axis_a + (j - T//2)*spacing*axis_b +
    (r0 + k*spacing)*direction`` (all XYZ, in voxels of the sampled level), so
    tile pixel ``(T//2, T//2)`` lies exactly on the ray.
    ``origin`` is the axis point, ``direction`` the outward radial unit vector,
    ``axis_a`` the scroll-axis (z) direction and ``axis_b = direction x axis_a``
    the in-plane tangent.  Ray samples run *outward* so the winding count
    increases with radius.
    """

    origin_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, float, float]
    axis_a_xyz: tuple[float, float, float]
    axis_b_xyz: tuple[float, float, float]
    r0: float
    spacing: float = 1.0
    transverse: int = 128
    ray_length: int = 384

    def radii(self) -> np.ndarray:
        return self.r0 + self.spacing * np.arange(self.ray_length, dtype=np.float64)

    def points(self) -> np.ndarray:
        """XYZ sample positions, ``[T, T, R, 3]`` float32."""
        t, r = self.transverse, self.ray_length
        i = (np.arange(t, dtype=np.float64) - t // 2)[:, None, None, None]
        j = (np.arange(t, dtype=np.float64) - t // 2)[None, :, None, None]
        k = self.radii()[None, None, :, None]
        o = np.asarray(self.origin_xyz)[None, None, None, :]
        a = np.asarray(self.axis_a_xyz)[None, None, None, :]
        b = np.asarray(self.axis_b_xyz)[None, None, None, :]
        d = np.asarray(self.direction_xyz)[None, None, None, :]
        return (o + self.spacing * (i * a + j * b) + k * d).astype(np.float32)

    def center_points(self) -> np.ndarray:
        """XYZ of the central ray (tile pixel ``(T//2, T//2)``), ``[R, 3]``."""
        return self.points()[self.transverse // 2, self.transverse // 2]


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError("zero direction")
    return a / n


def ray_frame(
    center_xyz: Sequence[float],
    angle: float,
    r0: float,
    ray_length: int = 384,
    transverse: int = 128,
    spacing: float = 1.0,
    up_xyz: Sequence[float] = (0.0, 0.0, 1.0),
) -> RayFrame:
    """Outward radial ray from the axis point ``center_xyz`` at polar ``angle``
    (radians, in the plane perpendicular to ``up_xyz``; angle 0 = +x, pi/2 = +y)."""
    up = _unit(up_xyz)
    ex = _unit(np.cross(up, [0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    ex = _unit(ex - up * float(ex @ up))
    ey = np.cross(up, ex)
    d = _unit(math.cos(angle) * ex + math.sin(angle) * ey)
    b = np.cross(d, up)
    return RayFrame(tuple(float(v) for v in center_xyz), tuple(d.tolist()), tuple(up.tolist()), tuple(b.tolist()),
                    float(r0), float(spacing), int(transverse), int(ray_length))


def sample_ray_tile(reader: Any, frame: RayFrame, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Trilinearly sample a ``[T, T, R]`` tile of CT (float32, 0..255) and its
    in-bounds validity mask from ``reader`` (``VolumeReader``/``ArrayReader``:
    ``shape`` (Z, Y, X) and ``read(z0, z1, y0, y1, x0, x1)`` -> uint8, zero
    padded outside the volume).  Uses ``grid_sample`` with ``align_corners=True``
    so integer XYZ coordinates hit voxel centres exactly.  Sampling is
    trilinear as in the training config (``sampling: trilinear``)."""
    pts = frame.points()  # [T, T, R, 3] xyz
    shape_xyz = np.array(reader.shape[::-1], dtype=np.float64)
    lo = np.floor(pts.reshape(-1, 3).min(0)).astype(int) - 1
    hi = np.ceil(pts.reshape(-1, 3).max(0)).astype(int) + 2
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape_xyz.astype(int))
    valid_np = np.all((pts >= 0) & (pts <= shape_xyz - 1), axis=-1)
    if np.any(hi - lo < 2):
        z = torch.zeros(pts.shape[:3], dtype=torch.float32, device=device)
        return z, torch.zeros_like(z, dtype=torch.bool)
    block = reader.read(int(lo[2]), int(hi[2]), int(lo[1]), int(hi[1]), int(lo[0]), int(hi[0]))
    vol = torch.from_numpy(np.ascontiguousarray(block)).to(device).float()[None, None]  # [1,1,Z,Y,X]
    local = torch.from_numpy(pts).to(device) - torch.tensor(lo, dtype=torch.float32, device=device)
    denom = torch.tensor(np.maximum(hi - lo - 1, 1), dtype=torch.float32, device=device)
    grid = (local / denom * 2.0 - 1.0)[None]  # [1, T, T, R, 3] with (x, y, z) order as grid_sample wants
    ct = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    return ct, torch.from_numpy(valid_np).to(device)


# --------------------------------------------------------------------------- #
# Teacher wrapper: tiles -> rays
# --------------------------------------------------------------------------- #
@dataclass
class RayProfile:
    """Winding-model output along one outward ray (central pixel of the tiles)."""

    radii: np.ndarray  # [N] distance from the axis (level voxels)
    points_xyz: np.ndarray  # [N, 3]
    ct: np.ndarray  # [N] CT along the ray (0..255)
    density: np.ndarray  # [N] windings per voxel
    phase: np.ndarray  # [N] cumulative windings from radii[0]
    valid: np.ndarray  # [N] bool
    tiles: list[dict[str, Any]] = field(default_factory=list)  # per-tile diagnostics (ct/density centre slices)


class WindingTeacher:
    """Convenience wrapper: ``predict_tile`` on ``(ct, valid)`` tiles and
    ``predict_ray`` stitching tiles along a long outward ray."""

    def __init__(self, net: WindingNet, device: str | torch.device = "cpu", normalize: str = "zscore_masked",
                 autocast: bool | None = None):
        self.net = net.eval()
        self.device = torch.device(device)
        self.normalize = normalize
        self.autocast = (self.device.type == "cuda") if autocast is None else autocast
        self.ray_length = 384
        self.transverse = net.arch.transverse_size

    @classmethod
    def load(cls, models_dir: str | os.PathLike = DEFAULT_MODELS_DIR, device: str | torch.device = "cpu",
             normalize: str = "zscore_masked", norm_kind: str = "layer") -> "WindingTeacher":
        net, cfg = load_winding_net(models_dir, device=device, norm_kind=norm_kind)
        t = cls(net, device=device, normalize=normalize)
        t.ray_length = int(cfg.get("ray_length", 384))
        return t

    @torch.inference_mode()
    def predict_tile(self, ct: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """``ct``/``valid``: ``[T, T, R]`` or ``[B, T, T, R]`` -> dict of float32
        ``logits``/``density``/``phase`` (``[B, T, T, R]``) on ``self.device``."""
        if ct.ndim == 3:
            ct, valid = ct[None], valid[None]
        x = prepare_input(ct.to(self.device), valid.to(self.device), self.normalize)
        if self.autocast and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.net(x)
        else:
            logits = self.net(x)
        return decode_outputs(logits.float())

    def predict_ray(
        self,
        reader: Any,
        center_xyz: Sequence[float],
        angle: float,
        r_start: float,
        r_stop: float,
        stride: int | None = None,
        edge: int = 32,
        keep_tiles: bool = False,
        up_xyz: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> RayProfile:
        """Run overlapping tiles of length ``ray_length`` from ``r_start``
        outward until ``r_stop`` and blend their central-pixel densities with a
        trapezoid weight (``edge`` samples of linear ramp at both tile ends,
        which lack context).  ``phase`` is the cumulative density from ``r_start``."""
        n_ray, t = self.ray_length, self.transverse
        stride = stride or (n_ray - 2 * edge - 64)
        n = int(math.ceil(r_stop - r_start)) + 1
        n = max(n, n_ray)
        acc = np.zeros(n, dtype=np.float64)
        wsum = np.zeros(n, dtype=np.float64)
        ct_out = np.zeros(n, dtype=np.float32)
        valid_out = np.zeros(n, dtype=bool)
        w_tile = np.ones(n_ray, dtype=np.float64)
        ramp = (np.arange(edge) + 1) / (edge + 1)
        w_tile[:edge] = ramp
        w_tile[-edge:] = ramp[::-1]
        starts = list(range(0, max(1, n - n_ray + 1), stride))
        if starts[-1] != n - n_ray:
            starts.append(n - n_ray)
        tiles = []
        for s in starts:
            frame = ray_frame(center_xyz, angle, r_start + s, n_ray, t, 1.0, up_xyz)
            ct, valid = sample_ray_tile(reader, frame, self.device)
            out = self.predict_tile(ct, valid)
            c = t // 2
            dens = out["density"][0, c, c].cpu().numpy().astype(np.float64)
            w = w_tile * valid[c, c].cpu().numpy()
            acc[s:s + n_ray] += dens * w
            wsum[s:s + n_ray] += w
            ct_out[s:s + n_ray] = np.where(w > 0, ct[c, c].cpu().numpy(), ct_out[s:s + n_ray])
            valid_out[s:s + n_ray] |= valid[c, c].cpu().numpy()
            if keep_tiles:
                tiles.append({"r0": r_start + s, "ct": ct[:, c].cpu().numpy(), "density": out["density"][0, :, c].cpu().numpy()})
        density = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0.0).astype(np.float32)
        frame = ray_frame(center_xyz, angle, r_start, n, t, 1.0, up_xyz)
        return RayProfile(frame.radii(), frame.center_points(), ct_out, density, np.cumsum(density).astype(np.float32),
                          valid_out, tiles)


# --------------------------------------------------------------------------- #
# Validation gate on a real ROI
# --------------------------------------------------------------------------- #
def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float64)
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x.astype(np.float64), r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _local_maxima(x: np.ndarray, min_height: float, min_distance: int) -> np.ndarray:
    idx = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= min_height)) + 1
    keep: list[int] = []
    for i in sorted(idx.tolist(), key=lambda i: -x[i]):
        if all(abs(i - j) >= min_distance for j in keep):
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


def windings_per_ct_period(ct: np.ndarray, density: np.ndarray, valid: np.ndarray, window: int = 96,
                           step: int = 48, min_period: float = 6.0, max_period: float = 40.0,
                           min_std: float = 1.0, peak_ratio: float = 3.0) -> np.ndarray:
    """Per window along the ray: (mean density) x (dominant CT period), i.e. the
    windings the model accumulates per sheet period; ~1.0 if the density is the
    local winding rate.  The period comes from the FFT peak of the mean-removed,
    smoothed CT profile within [min_period, max_period] voxels.

    A window is only used when the CT actually carries a periodic signal (R24):

    * ``std(ct) >= min_std`` (CT bytes) -- flat or near-flat CT has no sheets, and
    * the in-band FFT peak must exceed ``peak_ratio`` x the in-band median **and** a nonzero
      absolute floor derived from the window's own amplitude.

    Without the floor, a constant profile passes the ratio test (peak = median = 0) and its first
    in-band frequency bin is accepted as "the" period, which turns a constant density predictor into
    seven perfect-looking windings per period.  Windows that fail are skipped, so an all-flat ray
    returns an empty array."""
    out = []
    ct = np.asarray(ct, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    ct = _gaussian_smooth(ct, 1.0)
    n = len(ct)
    freqs = np.fft.rfftfreq(window)
    band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    if not band.any():
        return np.zeros(0, dtype=np.float64)
    for s in range(0, n - window + 1, step):
        v = valid[s:s + window]
        if v.mean() < 0.95:
            continue
        seg = ct[s:s + window]
        std = float(seg.std())
        if not np.isfinite(std) or std < float(min_std):  # flat CT: no sheets to find a period in
            continue
        seg = seg - seg.mean()
        spec = np.abs(np.fft.rfft(seg * np.hanning(window)))
        peak = float(spec[band].max())
        med = float(np.median(spec[band]))
        floor = 0.05 * std * window  # a real sheet train concentrates energy well above this
        if peak < max(float(peak_ratio) * med, floor) or peak <= 0.0:  # no clear periodicity
            continue
        period = 1.0 / freqs[band][int(np.argmax(spec[band]))]
        out.append(float(density[s:s + window].mean() * period))
    return np.asarray(out, dtype=np.float64)


def ray_metrics(profile: RayProfile, ct_sigma: float = 1.5, min_peak_distance: int = 4) -> dict[str, Any]:
    """Semantic checks on one ray: CT sheet peaks vs density peaks and the
    winding count accumulated between consecutive CT sheets (should be ~1)."""
    ok = profile.valid & np.isfinite(profile.density)
    ct = _gaussian_smooth(profile.ct, ct_sigma)
    if ok.sum() < 32:
        return {"n": int(ok.sum()), "ct_peaks": 0}
    thr = ct[ok].mean() + 0.5 * ct[ok].std()
    ct_peaks = _local_maxima(np.where(ok, ct, -np.inf), thr, min_peak_distance)
    dens = _gaussian_smooth(profile.density, 1.0)
    d_peaks = _local_maxima(np.where(ok, dens, -np.inf), dens[ok].mean(), min_peak_distance)
    steps = np.diff(profile.phase[ct_peaks]) if len(ct_peaks) > 1 else np.zeros(0)
    if len(d_peaks) and len(ct_peaks):
        dist = np.abs(d_peaks[:, None] - ct_peaks[None, :]).min(1)
    else:
        dist = np.zeros(0)
    wpp = windings_per_ct_period(profile.ct, profile.density, ok)
    return {
        "n": int(ok.sum()),
        "windings_per_period_median": float(np.median(wpp)) if len(wpp) else float("nan"),
        "windings_per_period_within_0.3": float(np.mean(np.abs(wpp - 1.0) <= 0.3)) if len(wpp) else float("nan"),
        "n_periodic_windows": int(len(wpp)),
        "ct_peaks": int(len(ct_peaks)),
        "density_peaks": int(len(d_peaks)),
        "ct_peak_spacing_median": float(np.median(np.diff(ct_peaks))) if len(ct_peaks) > 1 else float("nan"),
        "phase_step_median": float(np.median(steps)) if len(steps) else float("nan"),
        "phase_step_within_0.3": float(np.mean(np.abs(steps - 1.0) <= 0.3)) if len(steps) else float("nan"),
        "density_peak_to_ct_peak_median": float(np.median(dist)) if len(dist) else float("nan"),
        "density_peak_within_2vox": float(np.mean(dist <= 2)) if len(dist) else float("nan"),
        "density_mean": float(profile.density[ok].mean()),
        "monotone": bool(np.all(np.diff(profile.phase[ok]) >= -1e-6)),
        "finite": bool(np.isfinite(profile.density).all() and np.isfinite(profile.phase).all()),
    }


def validate(
    reader: Any,
    axis: Axis,
    z: float,
    n_rays: int = 8,
    teacher: WindingTeacher | None = None,
    level: int = 2,
    r_start: float = 150.0,
    r_stop: float = 2000.0,
    out_dir: str | os.PathLike | None = "/home/forrest/tsm-output/v1/winding",
    device: str | torch.device | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validation gate (plan, milestone 6) on real data.

    For ``n_rays`` outward rays at ``z`` (coordinates of ``reader``'s level)
    the gate requires: all outputs finite; phase non-decreasing outward on
    >= 80 % of rays (trivially true for a softplus rate, but it also catches
    NaNs); and, since ``density`` is a local winding *rate* (1 / sheet
    spacing) rather than a sheet-peaked signal, the windings accumulated per
    dominant CT sheet period (``windings_per_ct_period``) must be within
    +-0.3 of 1.0 in >= 60 % of periodic windows with a median within +-0.2.
    The CT-peak based ``phase_step`` and density-peak co-location numbers are
    reported but not gated (CT peak picking over-counts sheet faces/fibres).
    Writes ``winding_ray{k}.png`` / ``winding_tile{k}.png`` diagnostics and
    ``winding_validation.json`` to ``out_dir``.  Returns ``(passed, metrics)``.
    """
    if teacher is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        teacher = WindingTeacher.load(device=device)
    center = axis.xyz_at(z, level)
    per_ray = []
    profiles = []
    for k in range(n_rays):
        angle = 2 * math.pi * k / n_rays
        sub = _ray_roi_reader(reader, center, angle, r_start, r_stop, teacher.transverse)
        prof = teacher.predict_ray(sub, center, angle, r_start, r_stop, keep_tiles=out_dir is not None)
        m = ray_metrics(prof)
        m["angle_deg"] = 360.0 * k / n_rays
        per_ray.append(m)
        profiles.append(prof)
        print(f"[winding] ray {k} angle {m['angle_deg']:.0f}: " + ", ".join(
            f"{kk}={vv:.3g}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in m.items() if kk != "angle_deg"))
    finite = all(m.get("finite", False) for m in per_ray)
    monotone_frac = float(np.mean([m.get("monotone", False) for m in per_ray]))
    steps = [m["phase_step_within_0.3"] for m in per_ray if m.get("ct_peaks", 0) > 1 and np.isfinite(m["phase_step_within_0.3"])]
    coloc = [m["density_peak_within_2vox"] for m in per_ray if np.isfinite(m.get("density_peak_within_2vox", float("nan")))]
    wpp_w = [m["windings_per_period_within_0.3"] for m in per_ray if m.get("n_periodic_windows", 0) > 0]
    wpp_m = [m["windings_per_period_median"] for m in per_ray if m.get("n_periodic_windows", 0) > 0]
    metrics = {
        "finite": finite,
        "monotone_fraction": monotone_frac,
        "windings_per_period_within_0.3": float(np.mean(wpp_w)) if wpp_w else float("nan"),
        "windings_per_period_median": float(np.median(wpp_m)) if wpp_m else float("nan"),
        "phase_step_within_0.3": float(np.mean(steps)) if steps else float("nan"),
        "phase_step_median": float(np.nanmedian([m.get("phase_step_median", float("nan")) for m in per_ray])),
        "density_peak_within_2vox": float(np.mean(coloc)) if coloc else float("nan"),
        "ct_peak_spacing_median": float(np.nanmedian([m.get("ct_peak_spacing_median", float("nan")) for m in per_ray])),
        "density_mean": float(np.mean([m.get("density_mean", float("nan")) for m in per_ray])),
        "n_rays": n_rays,
        "z": float(z),
        "center_xyz": [float(v) for v in center],
        "per_ray": per_ray,
    }
    passed = bool(finite and monotone_frac >= 0.8 and wpp_w and metrics["windings_per_period_within_0.3"] >= 0.6
                  and abs(metrics["windings_per_period_median"] - 1.0) <= 0.2)
    metrics["passed"] = passed
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for k, prof in enumerate(profiles):
            _write_png(os.path.join(out_dir, f"winding_ray{k}.png"), _render_profile(prof))
            if prof.tiles:
                _write_png(os.path.join(out_dir, f"winding_tile{k}.png"), _render_tiles(prof))
        with open(os.path.join(out_dir, "winding_validation.json"), "w") as fh:
            json.dump(metrics, fh, indent=1)
    return passed, metrics


def _ray_roi_reader(reader: Any, center: np.ndarray, angle: float, r_start: float, r_stop: float, transverse: int) -> Any:
    """Read the bounding box of one whole ray once (``ArrayReader``), so the
    per-tile ``sample_ray_tile`` calls do not hit the network repeatedly."""
    n = int(math.ceil(r_stop - r_start)) + 1
    fr = ray_frame(center, angle, r_start, max(n, 384), transverse, 1.0)
    pts = fr.points()[:, :, :n]
    lo = np.maximum(np.floor(pts.reshape(-1, 3).min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(pts.reshape(-1, 3).max(0)).astype(int) + 3, np.array(reader.shape[::-1]))
    if np.any(hi <= lo):
        return ArrayReader(np.zeros((1, 1, 1), np.uint8), (0, 0, 0), reader.shape)
    block = reader.read(int(lo[2]), int(hi[2]), int(lo[1]), int(hi[1]), int(lo[0]), int(hi[0]))
    return ArrayReader(block, (int(lo[2]), int(lo[1]), int(lo[0])), reader.shape)


# --------------------------------------------------------------------------- #
# Tiny PNG plotting (numpy + zlib; no matplotlib/PIL dependency)
# --------------------------------------------------------------------------- #
def _write_png(path: str | os.PathLike, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def _draw_curve(img: np.ndarray, y0: int, h: int, x0: int, values: np.ndarray, color: tuple[int, int, int],
                vmin: float | None = None, vmax: float | None = None) -> None:
    v = np.asarray(values, dtype=np.float64)
    fin = np.isfinite(v)
    if not fin.any():
        return
    vmin = float(v[fin].min()) if vmin is None else vmin
    vmax = float(v[fin].max()) if vmax is None else vmax
    span = max(vmax - vmin, 1e-9)
    ys = y0 + h - 1 - np.clip(((v - vmin) / span * (h - 1)), 0, h - 1)
    ys = np.where(fin, ys, np.nan)
    for i in range(1, len(v)):
        if not (np.isfinite(ys[i - 1]) and np.isfinite(ys[i])):
            continue
        a, b = int(ys[i - 1]), int(ys[i])
        lo_, hi_ = min(a, b), max(a, b)
        x = x0 + i
        if 0 <= x < img.shape[1]:
            img[lo_:hi_ + 1, x] = color


def _render_profile(prof: RayProfile) -> np.ndarray:
    n = len(prof.radii)
    margin = 40
    rows = [("ct", prof.ct, (40, 40, 40)), ("density", prof.density, (200, 30, 30)),
            ("phase mod 1", np.mod(prof.phase, 1.0), (30, 90, 200))]
    h = 90
    img = np.full((h * len(rows) + 10, n + 2 * margin, 3), 255, np.uint8)
    for r, (_, vals, col) in enumerate(rows):
        y0 = 5 + r * h
        img[y0:y0 + h - 8, margin:margin + n] = 245
        if r == 0:  # sheet peaks as light bars behind everything
            ct = _gaussian_smooth(prof.ct, 1.5)
            ok = prof.valid
            if ok.sum() > 32:
                pk = _local_maxima(np.where(ok, ct, -np.inf), ct[ok].mean() + 0.5 * ct[ok].std(), 4)
                for p in pk:
                    img[5:5 + h * len(rows) - 8, margin + p] = (210, 235, 210)
        vmin, vmax = (0.0, 1.0) if r == 2 else (None, None)
        _draw_curve(img, y0, h - 8, margin, vals, col, vmin, vmax)
    inval = ~prof.valid
    img[-4:, margin:margin + n][:, inval] = (255, 0, 0)
    return img


def _render_tiles(prof: RayProfile) -> np.ndarray:
    panels = []
    for tile in prof.tiles:
        ct = tile["ct"]  # [T, R]
        dens = tile["density"]  # [T, R]
        g = np.clip(ct, 0, 255).astype(np.uint8)
        rgb = np.stack([g, g, g], -1).astype(np.float32)
        d = np.clip(dens / max(float(np.percentile(dens, 99.5)), 1e-6), 0, 1)[..., None]
        rgb = rgb * (1 - 0.6 * d) + np.array([255, 40, 40], np.float32) * 0.6 * d
        panels.append(np.clip(rgb, 0, 255).astype(np.uint8))
        panels.append(np.full((4, ct.shape[1], 3), 255, np.uint8))
    return np.concatenate(panels, 0)


# --------------------------------------------------------------------------- #
# CLI: uv run python -m tsm.winding --z 8544 --n-rays 8
# --------------------------------------------------------------------------- #
def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .volume import VolumeReader

    p = argparse.ArgumentParser(description="validate the winding teacher on a real ROI")
    p.add_argument("--url", default="s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr")
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--axis", default=os.path.expanduser("~/.cache/tsm-production/axes/PHercParis4/umbilicus-full-resolution.json"))
    p.add_argument("--z", type=float, default=8544.0, help="z at --level")
    p.add_argument("--n-rays", type=int, default=8)
    p.add_argument("--r-start", type=float, default=150.0)
    p.add_argument("--r-stop", type=float, default=2000.0)
    p.add_argument("--normalize", default="zscore_masked")
    p.add_argument("--norm-kind", default="layer")
    p.add_argument("--out", default="/home/forrest/tsm-output/v1/winding")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    reader = VolumeReader(a.url, level=a.level, voxel_um=2.4 * 2 ** a.level)
    axis = Axis.from_json(a.axis)
    teacher = WindingTeacher.load(device=a.device, normalize=a.normalize, norm_kind=a.norm_kind)
    ok, metrics = validate(reader, axis, a.z, a.n_rays, teacher=teacher, level=a.level, r_start=a.r_start,
                           r_stop=a.r_stop, out_dir=a.out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_ray"}, indent=1))
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
