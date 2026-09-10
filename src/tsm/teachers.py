"""Pure-PyTorch reimplementations of the published Vesuvius teacher networks.

Two network families are covered, both descendants of nnU-Net's
``dynamic_network_architectures`` residual-encoder U-Net:

* ``VesuviusUNet``  -- villa's ``NetworkFromConfig`` (``shared_encoder`` +
  per-target ``task_decoders.<t>`` with ``seg_layers``, and/or a features-only
  ``shared_decoder`` + ``task_heads.<t>`` 1x1 conv).  Used by
  ``surface_recto_3dunet``, ``ink_3d_dino_guided`` and ``lasagna``.
* ``NNUNetResEncUNet`` -- stock nnU-Net v2 ``ResidualEncoderUNet``
  (``encoder.*`` / ``decoder.*``; the decoder registers the encoder as a
  submodule so ``decoder.encoder.*`` keys are duplicated).  Used by
  ``surface_m7_nnunet``.

The architecture is *inferred from the state dict's own tensor shapes*
(``infer_vesuvius_arch`` / ``infer_nnunet_arch``), so a shape-only inventory is
enough to build a network with exact state-dict key parity.  Module attribute
names deliberately mirror the reference code:

* ``ConvNormAct`` registers ``conv``, ``norm`` and ``all_modules =
  Sequential(conv, norm, nonlin)`` -> weights appear under both
  ``.conv.weight`` and ``.all_modules.0.weight``.
* ``BasicBlockD.skip = Sequential([AvgPool3d(stride)] + [1x1 ConvNormAct])`` ->
  the projection conv sits at ``skip.1`` when the block is strided.
* scSE (``squeeze_excitation.cSE`` / ``.sSE``) is applied after ``conv2`` and
  before the residual add.

Nothing here imports vesuvius / nnunetv2; see ``dev/parity_villa.py`` for the
numerical parity check against the reference implementation.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ArchSpec",
    "DecoderSpec",
    "ConvNormAct",
    "BasicBlockD",
    "StackedResidualBlocks",
    "StackedConvBlocks",
    "ResidualEncoder",
    "UNetDecoder",
    "VesuviusUNet",
    "NNUNetResEncUNet",
    "Normalizer",
    "encoder_of",
    "encoder_features",
    "encoder_channels",
    "encoder_strides",
    "encoder_bytes_estimate",
    "TeacherSpec",
    "TEACHERS",
    "infer_vesuvius_arch",
    "infer_nnunet_arch",
    "build_teacher",
    "load_teacher",
    "apply_activation",
    "shapes_of",
    "strip_prefix",
]

DEFAULT_MODELS_DIR = os.path.expanduser("~/.cache/tsm-models")
_STRIP_PREFIXES = ("module.", "_orig_mod.")


# --------------------------------------------------------------------------- #
# Architecture description
# --------------------------------------------------------------------------- #
@dataclass
class DecoderSpec:
    """One decoder path (villa ``Decoder`` / nnU-Net ``UNetDecoder``)."""

    out_channels: int | None  # None -> features-only (no seg_layers)
    n_conv_per_stage: list[int]  # per decoder stage, lowest resolution first
    residual: bool = False  # BasicBlockD stages instead of plain conv blocks
    upsample_mode: str = "transpconv"  # transpconv | trilinear | pixelshuffle


@dataclass
class ArchSpec:
    in_channels: int
    features_per_stage: list[int]
    n_blocks_per_stage: list[int]
    strides: list[int]
    kernel_size: int = 3
    conv_bias: bool = True
    norm: str = "instance"  # instance | group | none
    num_groups: int = 32
    squeeze_excitation: str | None = None  # None | 'scse' | 'channel' | 'spatial'
    se_reduction_ratio: float = 1.0 / 16
    # villa: separate decoders (with seg_layers) keyed by target name
    task_decoders: dict[str, DecoderSpec] = field(default_factory=dict)
    # villa: features-only shared decoder + 1x1 heads keyed by target name
    shared_decoder: DecoderSpec | None = None
    task_heads: dict[str, int] = field(default_factory=dict)
    # nnU-Net: single decoder
    decoder: DecoderSpec | None = None

    @property
    def n_stages(self) -> int:
        return len(self.features_per_stage)

    @property
    def targets(self) -> list[str]:
        return sorted(set(self.task_decoders) | set(self.task_heads))


# --------------------------------------------------------------------------- #
# Building blocks (names mirror dynamic_network_architectures / villa)
# --------------------------------------------------------------------------- #
def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None, round_limit: float = 0.9) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def _make_norm(norm: str, channels: int, num_groups: int) -> nn.Module | None:
    if norm == "instance":
        return nn.InstanceNorm3d(channels, eps=1e-5, affine=True)
    if norm == "group":
        return nn.GroupNorm(num_groups, channels, eps=1e-5, affine=True)
    if norm == "none":
        return None
    raise ValueError(f"unknown norm {norm!r}")


class ConvNormAct(nn.Module):
    """villa/nnU-Net ``ConvDropoutNormReLU`` (no dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        conv_bias: bool,
        norm: str,
        nonlin: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, dilation=1, bias=conv_bias
        )
        ops: list[nn.Module] = [self.conv]
        norm_mod = _make_norm(norm, out_channels, num_groups)
        if norm_mod is not None:
            self.norm = norm_mod
            ops.append(self.norm)
        if nonlin:
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            ops.append(self.nonlin)
        self.all_modules = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_modules(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 1.0 / 16, rd_divisor: int = 8):
        super().__init__()
        rd_channels = _make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.fc2(self.act(self.bn(self.fc1(x_se))))
        return x * self.gate(x_se)


class SpatialSE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.conv(x))


class ChannelSpatialSE(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 1.0 / 16):
        super().__init__()
        self.cSE = SqueezeExcite(channels, rd_ratio)
        self.sSE = SpatialSE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cSE(x) + self.sSE(x)


def _make_se(kind: str, channels: int, rd_ratio: float) -> nn.Module:
    if kind == "scse":
        return ChannelSpatialSE(channels, rd_ratio)
    if kind == "channel":
        return SqueezeExcite(channels, rd_ratio)
    if kind == "spatial":
        return SpatialSE(channels)
    raise ValueError(f"unknown squeeze_excitation type {kind!r}")


class BasicBlockD(nn.Module):
    """ResNet-D basic block: conv1(stride) -> conv2 -> [SE] -> + skip -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        conv_bias: bool,
        norm: str,
        squeeze_excitation: str | None = None,
        se_reduction_ratio: float = 1.0 / 16,
        num_groups: int = 32,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.stride = stride
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, stride, conv_bias, norm, True, num_groups)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, 1, conv_bias, norm, False, num_groups)
        self.nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.apply_se = squeeze_excitation is not None
        if self.apply_se:
            self.squeeze_excitation = _make_se(squeeze_excitation, out_channels, se_reduction_ratio)
        has_stride = stride != 1
        requires_projection = in_channels != out_channels
        if has_stride or requires_projection:
            ops: list[nn.Module] = []
            if has_stride:
                ops.append(nn.AvgPool3d(stride, stride))
            if requires_projection:
                ops.append(ConvNormAct(in_channels, out_channels, 1, 1, False, norm, False, num_groups))
            self.skip: nn.Module | None = nn.Sequential(*ops)
        else:
            self.skip = None  # identity (villa uses a lambda; no parameters either way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out = out + residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        initial_stride: int,
        conv_bias: bool,
        norm: str,
        squeeze_excitation: str | None = None,
        se_reduction_ratio: float = 1.0 / 16,
        num_groups: int = 32,
    ):
        super().__init__()
        assert n_blocks > 0
        blocks = [
            BasicBlockD(in_channels, out_channels, kernel_size, initial_stride, conv_bias, norm,
                        squeeze_excitation, se_reduction_ratio, num_groups)
        ]
        blocks += [
            BasicBlockD(out_channels, out_channels, kernel_size, 1, conv_bias, norm,
                        squeeze_excitation, se_reduction_ratio, num_groups)
            for _ in range(1, n_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.initial_stride = initial_stride
        self.output_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        initial_stride: int,
        conv_bias: bool,
        norm: str,
        num_groups: int = 32,
    ):
        super().__init__()
        convs = [ConvNormAct(in_channels, out_channels, kernel_size, initial_stride, conv_bias, norm, True, num_groups)]
        convs += [
            ConvNormAct(out_channels, out_channels, kernel_size, 1, conv_bias, norm, True, num_groups)
            for _ in range(1, num_convs)
        ]
        self.convs = nn.Sequential(*convs)
        self.output_channels = out_channels
        self.initial_stride = initial_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class ResidualEncoder(nn.Module):
    """``stem`` (1 conv block) followed by ``stages`` of ``StackedResidualBlocks``; returns all skips."""

    def __init__(self, spec: ArchSpec):
        super().__init__()
        f = spec.features_per_stage
        self.stem = StackedConvBlocks(1, spec.in_channels, f[0], spec.kernel_size, 1, spec.conv_bias, spec.norm,
                                      spec.num_groups)
        stages = []
        cin = f[0]
        for s in range(spec.n_stages):
            stages.append(
                StackedResidualBlocks(spec.n_blocks_per_stage[s], cin, f[s], spec.kernel_size, spec.strides[s],
                                      spec.conv_bias, spec.norm, spec.squeeze_excitation, spec.se_reduction_ratio,
                                      spec.num_groups)
            )
            cin = f[s]
        self.stages = nn.Sequential(*stages)
        self.output_channels = list(f)
        self.strides = list(spec.strides)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class TrilinearUpsample3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, bias: bool = True):
        super().__init__()
        self.scale_factor = [int(stride)] * 3
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return self.conv(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, bias: bool = True):
        super().__init__()
        r = int(stride)
        self.r = r
        self.conv = nn.Conv3d(in_channels, out_channels * r * r * r, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C_total, D, H, W = x.shape
        r = self.r
        C = C_total // (r * r * r)
        x = x.view(B, C, r, r, r, D, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(B, C, D * r, H * r, W * r)


class UNetDecoder(nn.Module):
    """villa ``Decoder`` / nnU-Net ``UNetDecoder``.

    Stages are ordered from the bottleneck upwards. ``forward`` returns only the
    finest ``seg_layers`` output (or the finest feature map when features-only).
    ``register_encoder`` registers the encoder as ``self.encoder`` the way
    nnU-Net does (duplicating its keys under ``decoder.encoder.*``).
    """

    def __init__(self, spec: ArchSpec, dec: DecoderSpec, register_encoder: ResidualEncoder | None = None):
        super().__init__()
        if register_encoder is not None:
            self.encoder = register_encoder
        f = spec.features_per_stage
        n = spec.n_stages
        assert len(dec.n_conv_per_stage) == n - 1, (len(dec.n_conv_per_stage), n)
        stages: list[nn.Module] = []
        transpconvs: list[nn.Module] = []
        seg_layers: list[nn.Module] = []
        for s in range(1, n):
            below = f[-s]
            skip = f[-(s + 1)]
            stride = spec.strides[-s]
            if dec.upsample_mode == "trilinear":
                transpconvs.append(TrilinearUpsample3d(below, skip, stride, bias=spec.conv_bias))
            elif dec.upsample_mode == "pixelshuffle":
                transpconvs.append(PixelShuffle3d(below, skip, stride, bias=spec.conv_bias))
            elif dec.upsample_mode == "transpconv":
                transpconvs.append(nn.ConvTranspose3d(below, skip, stride, stride, bias=spec.conv_bias))
            else:
                raise ValueError(f"unknown upsample_mode {dec.upsample_mode!r}")
            if dec.residual:
                stages.append(StackedResidualBlocks(dec.n_conv_per_stage[s - 1], 2 * skip, skip, spec.kernel_size, 1,
                                                    spec.conv_bias, spec.norm, None, spec.se_reduction_ratio,
                                                    spec.num_groups))
            else:
                stages.append(StackedConvBlocks(dec.n_conv_per_stage[s - 1], 2 * skip, skip, spec.kernel_size, 1,
                                                spec.conv_bias, spec.norm, spec.num_groups))
            if dec.out_channels is not None:
                seg_layers.append(nn.Conv3d(skip, dec.out_channels, 1, 1, 0, bias=True))
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.return_features_only = dec.out_channels is None

    def forward(self, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
        if self.return_features_only:
            return x
        return self.seg_layers[-1](x)


class VesuviusUNet(nn.Module):
    """villa ``NetworkFromConfig`` (unet branch). ``forward`` -> ``{target: logits}``."""

    def __init__(self, spec: ArchSpec):
        super().__init__()
        self.spec = spec
        self.shared_encoder = ResidualEncoder(spec)
        self.task_decoders = nn.ModuleDict(
            {name: UNetDecoder(spec, dec) for name, dec in sorted(spec.task_decoders.items())}
        )
        self.task_heads = nn.ModuleDict(
            {name: nn.Conv3d(spec.features_per_stage[0], oc, kernel_size=1, stride=1, padding=0, bias=True)
             for name, oc in sorted(spec.task_heads.items())}
        )
        if spec.shared_decoder is not None:
            self.shared_decoder = UNetDecoder(spec, spec.shared_decoder)
        elif spec.task_heads:
            raise ValueError("task_heads given without a shared_decoder")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = self.shared_encoder(x)
        out: dict[str, torch.Tensor] = {}
        for name, dec in self.task_decoders.items():
            out[name] = dec(skips)
        if len(self.task_heads) > 0:
            feats = self.shared_decoder(skips)
            for name, head in self.task_heads.items():
                out[name] = head(feats)
        return out


class NNUNetResEncUNet(nn.Module):
    """nnU-Net v2 ``ResidualEncoderUNet``. ``forward`` -> finest logits tensor."""

    def __init__(self, spec: ArchSpec):
        super().__init__()
        assert spec.decoder is not None
        self.spec = spec
        self.encoder = ResidualEncoder(spec)
        self.decoder = UNetDecoder(spec, spec.decoder, register_encoder=self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Shape-driven architecture inference
# --------------------------------------------------------------------------- #
ShapeDict = Mapping[str, Any]


def shapes_of(sd: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Normalize a state dict / inventory into ``{key: shape tuple}``."""
    out: dict[str, tuple[int, ...]] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = tuple(v.shape)
        elif isinstance(v, Mapping) and "shape" in v:
            out[k] = tuple(int(i) for i in v["shape"])
        elif hasattr(v, "shape"):
            out[k] = tuple(int(i) for i in v.shape)
        else:
            out[k] = tuple(int(i) for i in v)
    return out


def strip_prefix(sd: Mapping[str, Any], prefixes: Sequence[str] = _STRIP_PREFIXES) -> dict[str, Any]:
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _count(shapes: ShapeDict, fmt: str) -> int:
    i = 0
    while fmt.format(i) in shapes:
        i += 1
    return i


def _infer_encoder(shapes: ShapeDict, p: str, norm_type: str | None) -> ArchSpec:
    stem_key = f"{p}stem.convs.0.conv.weight"
    if stem_key not in shapes:
        raise ValueError(f"no encoder stem under prefix {p!r} (looked for {stem_key})")
    stem_w = shapes[stem_key]
    in_channels, kernel = int(stem_w[1]), int(stem_w[2])
    conv_bias = f"{p}stem.convs.0.conv.bias" in shapes
    has_norm = f"{p}stem.convs.0.norm.weight" in shapes
    if norm_type is None:
        norm = "instance" if has_norm else "none"
    else:
        norm = {"instance": "instance", "group": "group", "none": "none", None: "none"}[norm_type]
        if (norm != "none") != has_norm:
            raise ValueError(f"norm_type={norm_type!r} contradicts the state dict (norm params present={has_norm})")
    n_stages = _count(shapes, p + "stages.{}.blocks.0.conv1.conv.weight")
    if n_stages == 0:
        raise ValueError(f"no residual stages under prefix {p!r}")
    features = [int(shapes[f"{p}stages.{s}.blocks.0.conv1.conv.weight"][0]) for s in range(n_stages)]
    n_blocks = [_count(shapes, p + f"stages.{s}.blocks.{{}}.conv1.conv.weight") for s in range(n_stages)]
    se_prefix = f"{p}stages.0.blocks.0.squeeze_excitation."
    se: str | None = None
    if se_prefix + "cSE.fc1.weight" in shapes:
        se = "scse"
    elif se_prefix + "fc1.weight" in shapes:
        se = "channel"
    elif se_prefix + "conv.weight" in shapes:
        se = "spatial"
    # Strides: stage 0 is never strided; a strided block has its projection conv at
    # skip.1 (AvgPool at skip.0). The actual factor comes from the decoder's
    # transposed convs when available (see _infer_decoder); default to 2 otherwise.
    strides = [1] + [2] * (n_stages - 1)
    return ArchSpec(in_channels=in_channels, features_per_stage=features, n_blocks_per_stage=n_blocks,
                    strides=strides, kernel_size=kernel, conv_bias=conv_bias, norm=norm,
                    squeeze_excitation=se)


def _infer_decoder(shapes: ShapeDict, q: str, spec: ArchSpec) -> DecoderSpec | None:
    """Infer one decoder under prefix ``q``; also fixes ``spec.strides`` from transpconv kernels."""
    n_up = _count(shapes, q + "transpconvs.{}.weight")
    mode = "transpconv"
    if n_up == 0:
        n_up = _count(shapes, q + "transpconvs.{}.conv.weight")
        if n_up == 0:
            return None
        w0 = shapes[f"{q}transpconvs.0.conv.weight"]
        skip_ch = spec.features_per_stage[-2]
        if int(w0[0]) == skip_ch:
            mode = "trilinear"
        else:
            r = round((int(w0[0]) / skip_ch) ** (1.0 / 3))
            if r ** 3 * skip_ch != int(w0[0]):
                raise ValueError(f"cannot interpret {q}transpconvs.0.conv.weight shape {w0}")
            mode = "pixelshuffle"
            for i in range(n_up):
                wi = shapes[f"{q}transpconvs.{i}.conv.weight"]
                spec.strides[-(i + 1)] = round((int(wi[0]) / spec.features_per_stage[-(i + 2)]) ** (1.0 / 3))
    else:
        for i in range(n_up):
            wi = shapes[f"{q}transpconvs.{i}.weight"]
            spec.strides[-(i + 1)] = int(wi[2])
    if n_up != spec.n_stages - 1:
        raise ValueError(f"{q}: {n_up} upsampling steps but encoder has {spec.n_stages} stages")
    residual = f"{q}stages.0.blocks.0.conv1.conv.weight" in shapes
    if residual:
        n_conv = [_count(shapes, q + f"stages.{i}.blocks.{{}}.conv1.conv.weight") for i in range(n_up)]
    else:
        n_conv = [_count(shapes, q + f"stages.{i}.convs.{{}}.conv.weight") for i in range(n_up)]
    if min(n_conv) == 0:
        raise ValueError(f"{q}: could not count conv blocks per decoder stage: {n_conv}")
    n_seg = _count(shapes, q + "seg_layers.{}.weight")
    out_channels = int(shapes[f"{q}seg_layers.{n_seg - 1}.weight"][0]) if n_seg else None
    if n_seg not in (0, n_up):
        raise ValueError(f"{q}: expected {n_up} seg_layers (deep supervision heads), found {n_seg}")
    return DecoderSpec(out_channels=out_channels, n_conv_per_stage=n_conv, residual=residual, upsample_mode=mode)


def infer_vesuvius_arch(sd: Mapping[str, Any], norm_type: str | None = None) -> ArchSpec:
    """Infer a villa ``NetworkFromConfig`` architecture from state-dict shapes.

    ``sd`` may hold tensors, ``{'shape': [...]}`` inventory entries or shape
    tuples. ``norm_type`` ('instance' | 'group' | 'none') disambiguates
    InstanceNorm from GroupNorm (identical parameter shapes); default: instance
    when norm parameters are present.
    """
    shapes = shapes_of(strip_prefix(sd))
    spec = _infer_encoder(shapes, "shared_encoder.", norm_type)
    names = sorted({m.group(1) for k in shapes for m in [re.match(r"task_decoders\.([^.]+)\.", k)] if m})
    for name in names:
        dec = _infer_decoder(shapes, f"task_decoders.{name}.", spec)
        if dec is None or dec.out_channels is None:
            raise ValueError(f"task_decoders.{name}: missing transpconvs or seg_layers")
        spec.task_decoders[name] = dec
    spec.shared_decoder = _infer_decoder(shapes, "shared_decoder.", spec)
    if spec.shared_decoder is not None and spec.shared_decoder.out_channels is not None:
        raise ValueError("shared_decoder is expected to be features-only (no seg_layers)")
    heads = sorted({m.group(1) for k in shapes for m in [re.match(r"task_heads\.([^.]+)\.weight$", k)] if m})
    for name in heads:
        spec.task_heads[name] = int(shapes[f"task_heads.{name}.weight"][0])
    if heads and spec.shared_decoder is None:
        raise ValueError("task_heads present but no shared_decoder")
    if not spec.task_decoders and not spec.task_heads:
        raise ValueError("no task_decoders or task_heads found")
    return spec


def infer_nnunet_arch(sd: Mapping[str, Any], norm_type: str | None = None) -> ArchSpec:
    """Infer an nnU-Net v2 ``ResidualEncoderUNet`` architecture from state-dict shapes."""
    shapes = shapes_of(strip_prefix(sd))
    spec = _infer_encoder(shapes, "encoder.", norm_type)
    dec = _infer_decoder(shapes, "decoder.", spec)
    if dec is None or dec.out_channels is None:
        raise ValueError("decoder: missing transpconvs or seg_layers")
    spec.decoder = dec
    return spec


def build_from_spec(kind: str, spec: ArchSpec) -> nn.Module:
    if kind == "vesuvius":
        return VesuviusUNet(spec)
    if kind == "nnunet":
        return NNUNetResEncUNet(spec)
    raise ValueError(f"unknown teacher kind {kind!r}")


def build_teacher(kind: str, sd: Mapping[str, Any], norm_type: str | None = None) -> nn.Module:
    """Build the (uninitialised) network whose state-dict keys/shapes match ``sd``."""
    infer = infer_vesuvius_arch if kind == "vesuvius" else infer_nnunet_arch
    return build_from_spec(kind, infer(sd, norm_type=norm_type))


# --------------------------------------------------------------------------- #
# Input normalisation (matches villa Volume / nnU-Net CTNormalization)
# --------------------------------------------------------------------------- #
def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    """np.percentile(x, q) ('linear' method) for a flattened float tensor."""
    flat = x.reshape(-1)
    n = flat.numel()
    if n == 1:
        return flat[0]
    pos = (n - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    v_lo = torch.kthvalue(flat, lo + 1).values
    if frac == 0.0:
        return v_lo
    v_hi = torch.kthvalue(flat, hi + 1).values
    return v_lo + (v_hi - v_lo) * frac


@dataclass(frozen=True)
class Normalizer:
    """Per-patch intensity normalisation. ``__call__`` accepts a uint8/float numpy
    array or torch tensor of any shape and returns a float32 tensor.

    modes:
      zscore_instance   (x - mean) / max(std, 1e-8)           (villa 'instance_zscore')
      ct_clip           clip(x, lo, hi); (x - mean) / max(std, 1e-8)   (nnU-Net CTNormalization)
      percentile_minmax clip to [p_lo, p_hi] and scale to [0, 1]     (villa 'percentile_minmax')
      div255            x / 255                                    (lasagna)
      none
    """

    mode: str
    mean: float = 0.0
    std: float = 1.0
    lo: float = 0.0
    hi: float = 255.0
    lo_pct: float = 1.0
    hi_pct: float = 99.0

    def __post_init__(self):
        if self.mode not in ("zscore_instance", "ct_clip", "percentile_minmax", "div255", "none"):
            raise ValueError(f"unknown normalizer mode {self.mode!r}")

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
        x = x.to(torch.float32)
        if self.mode == "none":
            return x
        if self.mode == "div255":
            return x / 255.0
        if self.mode == "zscore_instance":
            mean = x.mean()
            std = x.std(unbiased=False)
            return (x - mean) / torch.clamp(std, min=1e-8)
        if self.mode == "ct_clip":
            return (x.clamp(self.lo, self.hi) - self.mean) / max(self.std, 1e-8)
        if self.mode == "percentile_minmax":
            lower = _percentile(x, self.lo_pct)
            upper = _percentile(x, self.hi_pct)
            denom = float(upper - lower)
            if denom <= 1e-8:
                return torch.zeros_like(x)
            return (torch.clamp(x, lower, upper) - lower) / denom
        raise AssertionError(self.mode)


def apply_activation(logits: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "softmax":
        return torch.softmax(logits, dim=1)
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "clamp01":
        return logits.clamp(0, 1)
    if activation == "none":
        return logits
    raise ValueError(f"unknown activation {activation!r}")


# --------------------------------------------------------------------------- #
# Teacher registry and loader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeacherSpec:
    name: str
    file: str
    state_key: str
    kind: str  # 'vesuvius' | 'nnunet'
    patch: tuple[int, int, int]
    normalizer: Normalizer
    activation: str  # 'softmax' | 'sigmoid' | 'clamp01' | 'none'
    fg_channel: int | None
    voxel_um: float
    target: str | None = None  # key of the VesuviusUNet output dict (None for nnunet)
    norm_type: str | None = None  # None -> infer from shapes / ckpt['norm_type']

    def select(self, out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Pick this teacher's logits from a model forward output."""
        if isinstance(out, dict):
            return out[self.target]
        return out


TEACHERS: dict[str, TeacherSpec] = {
    "recto": TeacherSpec(
        name="recto", file="surface_recto_3dunet.pth", state_key="model", kind="vesuvius",
        patch=(256, 256, 256), normalizer=Normalizer("zscore_instance"), activation="softmax",
        fg_channel=1, voxel_um=2.4, target="surface",
    ),
    "m7": TeacherSpec(
        name="m7", file="surface_m7_nnunet.pth", state_key="network_weights", kind="nnunet",
        patch=(192, 192, 192),
        normalizer=Normalizer("ct_clip", mean=87.54424285888672, std=47.74376678466797, lo=0.0, hi=212.0),
        activation="softmax", fg_channel=1, voxel_um=2.4, target=None,
    ),
    "ink": TeacherSpec(
        name="ink", file="ink_3d_dino_guided.pth", state_key="ema_model", kind="vesuvius",
        patch=(256, 256, 256), normalizer=Normalizer("percentile_minmax", lo_pct=1.0, hi_pct=99.0),
        activation="sigmoid", fg_channel=0, voxel_um=2.4, target="ink",
    ),
    # scrollprize/fiber_ink_4class_selfdistill (MIT), step 29000 of p4_4class_ddp8_20260526.
    # 4-way softmax over {background, vertical fiber, horizontal/angular fiber, ink}; the
    # `ema` weights are the ones the card recommends.  Normalisation is villa's
    # `percentile_minmax_normalize` (1st-99th percentile clip, rescaled to [0, 1]) read from
    # scripts/fiber_5class/dataset.py -- the same rule as `ink`.  fg_channel is None: all four
    # softmax channels are written out (the student consumes vt/hz).
    "fiber": TeacherSpec(
        name="fiber", file="fiber_4class_selfdistill.pth", state_key="ema", kind="vesuvius",
        patch=(256, 256, 256), normalizer=Normalizer("percentile_minmax", lo_pct=1.0, hi_pct=99.0),
        activation="softmax", fg_channel=None, voxel_um=2.4, target="labels",
    ),
    # lasagna's checkpoint carries output_sigmoid=False: villa clamps the raw
    # 8-channel output to [0, 1]. load_teacher() overrides `activation` from the
    # checkpoint ('sigmoid' when output_sigmoid is true).
    "lasagna": TeacherSpec(
        name="lasagna", file="lasagna_model_current.pt", state_key="state_dict", kind="vesuvius",
        patch=(192, 192, 192), normalizer=Normalizer("div255"), activation="clamp01",
        fg_channel=None, voxel_um=9.6, target="output",
    ),
}


def _extract_state(ckpt: Any, state_key: str) -> dict[str, torch.Tensor]:
    sd = ckpt
    if isinstance(ckpt, Mapping) and state_key in ckpt:
        sd = ckpt[state_key]
    for inner in ("model", "model_state"):  # ema wrappers: {"model": ...} / {"model_state": ...}
        if isinstance(sd, Mapping) and inner in sd and not torch.is_tensor(sd[inner]):
            sd = sd[inner]
            break
    if not isinstance(sd, Mapping) or not all(torch.is_tensor(v) for v in sd.values()):
        raise ValueError(f"state under {state_key!r} is not a flat tensor dict "
                         f"(top-level keys: {list(ckpt.keys())[:10] if isinstance(ckpt, Mapping) else type(ckpt)})")
    return strip_prefix(sd)


def load_teacher(
    name: str,
    models_dir: str | os.PathLike = DEFAULT_MODELS_DIR,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, TeacherSpec]:
    """Load a teacher checkpoint strictly into our reimplementation (eval mode).

    The checkpoint is memory-mapped, the weights copied into the freshly built
    network, and the checkpoint released before returning.
    """
    spec = TEACHERS[name]
    path = os.path.join(os.fspath(models_dir), spec.file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        sd = _extract_state(ckpt, spec.state_key)
        norm_type = spec.norm_type
        if isinstance(ckpt, Mapping) and ckpt.get("norm_type") is not None and norm_type is None:
            norm_type = str(ckpt["norm_type"])
        if isinstance(ckpt, Mapping) and "output_sigmoid" in ckpt:
            spec = replace(spec, activation="sigmoid" if ckpt["output_sigmoid"] else "clamp01")
        if isinstance(ckpt, Mapping) and isinstance(ckpt.get("model_patch_size"), int):
            p = int(ckpt["model_patch_size"])
            spec = replace(spec, patch=(p, p, p))
        model = build_teacher(spec.kind, sd, norm_type=norm_type)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[teachers] {name}: strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")
            for k in list(missing)[:10]:
                print(f"  missing:    {k}")
            for k in list(unexpected)[:10]:
                print(f"  unexpected: {k}")
            raise RuntimeError(f"state dict mismatch for teacher {name!r}")
        model = model.to(device=device, dtype=dtype).eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
    finally:
        del ckpt
    return model, spec


# --------------------------------------------------------------------------- #
# Encoder-only access (feature distillation into the student, tsm.train)
# --------------------------------------------------------------------------- #
def encoder_of(model: nn.Module) -> ResidualEncoder:
    """The residual encoder of a teacher (``shared_encoder`` for VesuviusUNet, ``encoder`` for nnU-Net);
    a bare :class:`ResidualEncoder` is returned as is."""
    if isinstance(model, ResidualEncoder):
        return model
    enc = getattr(model, "shared_encoder", None) or getattr(model, "encoder", None)
    if enc is None:
        raise TypeError(f"{type(model).__name__} has no shared_encoder / encoder")
    return enc


def encoder_channels(model: nn.Module) -> list[int]:
    """Output channels per encoder stage (stage k = the k-th skip)."""
    return list(encoder_of(model).output_channels)


def encoder_strides(model: nn.Module) -> list[int]:
    """Cumulative downsampling of stage k relative to the input (1, 2, 4, ...)."""
    out, cum = [], 1
    for s in encoder_of(model).strides:
        cum *= int(s if isinstance(s, int) else max(s))
        out.append(cum)
    return out


@torch.no_grad()
def encoder_features(model: nn.Module, x_norm: torch.Tensor) -> list[torch.Tensor]:
    """Run only the teacher encoder on an already normalized ``(B, 1, Z, Y, X)`` input.

    Returns the list of stage outputs (``len == n_stages``; stage ``k`` has
    spatial size ``input / 2**k`` for the published teachers, whose strides are
    ``[1, 2, 2, ...]``).  No decoder is executed and no graph is built.
    """
    enc = encoder_of(model)
    p = next(enc.parameters())
    return enc(x_norm.to(device=p.device, dtype=p.dtype))


def encoder_bytes_estimate(model: nn.Module, patch: int, batch: int = 1, dtype_bytes: int = 2) -> dict[str, int]:
    """Rough VRAM footprint of one no-grad encoder forward: weights + live activations.

    Activations: every stage output is kept (the skip list) plus ~4 tensors of
    the widest live stage (conv1/conv2/skip/residual add inside a block).
    """
    enc = encoder_of(model)
    n_vox = batch * patch ** 3
    skips = 0
    widest = 0
    for c, cum in zip(encoder_channels(model), encoder_strides(model)):
        t = c * n_vox // cum ** 3 * dtype_bytes
        skips += t
        widest = max(widest, t)
    params = sum(p.numel() for p in enc.parameters()) * dtype_bytes
    return {"params": params, "activations": skips + 4 * widest, "total": params + skips + 4 * widest}
