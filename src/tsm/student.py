"""TSMNet: one small multi-head 3D residual U-Net (the student).

Input (B, in_ch + aux_ch, Z, Y, X), ``in_ch`` = 2 or 5:
  ch 0  CT, z-scored per crop: ``(u8/255 - mean) / max(std, 1e-3)`` over the whole
        crop (see :func:`normalize_ct`).  Air/no-data voxels are included in the
        statistics on purpose: the student sees the same crops at inference.
  ch 1  constant scale channel ``log2(voxel_um / 2.4)`` (0 at 2.4 um, 2 at 9.6 um).
  ch 2..4  (``extra.train.input_radial``, default on) the outward radial direction
        ``r_z, r_y, r_x`` at every voxel: the unit vector from the scroll axis
        (umbilicus) to that voxel, in (z, y, x) order, **perpendicular to the local
        axis tangent** (``labels.geometry_frame``).  Without it a 128^3 crop cannot know
        which way is "out", and the sdf sign / winding phase sign / normal orientation --
        all defined as outward -- are unlearnable.  The channels are augmented as tangent
        vectors (pushed forward with ``L``, resampled and renormalised).
  ch 5..7  (``extra.train.input_axis``, default on) the scroll-axis unit direction in
        volume coordinates ``a_z, a_y, a_x`` -- the **local umbilicus tangent**
        (``labels.axis_tangent_field``; a constant (1, 0, 0) with
        ``extra.train.axis_tangent: false``), rotated with the volume by every spatial
        transform, exactly like the radial channels.  "Vertical" fibre means "along this
        axis", so an all-axes rotation makes the fibre classes unlearnable without it.
        Channels 2..7 are an orthonormal pair at every voxel (re-orthonormalised after
        augmentation); the binormal is deliberately not fed (a pseudovector).
  ch 8+ optional aux channels (refiner mode: first-pass heads, 2 + 1 + 8 = 11).

Widths are multiples of 32 (FP8 / TensorRT friendly); the forward has no
in-place ops on views and no Python branching on tensor values.  Use
``channels_last_3d`` memory format on CUDA (``to_channels_last``).

Heads (1x1x1 convs on the decoder features):
  surface (2) = [sdf (voxels, clipped), valid logit]
              (3) = [sdf_in, sdf_out, valid logit] in the two-face mode (``heads_for("faces")``)
              or [d_face, body logit, valid logit] in the "sides" mode (``heads_for("sides")``)
  ink     (1) = ink logit
  fiber   (2) = [vertical fiber logit, horizontal/angular fiber logit] (optional,
              ``heads_for(..., fiber=True)``)
              (4) = [dz, dy, dx, strength logit] with ``fiber_mode="direction"``: one
              axial fibre direction per voxel instead of two axis-relative classes (tsm.fiber)
  winding (8) = [sin 2piw, cos 2piw, density, nx, ny, nz, conf logit, spare]

``forward`` returns ``{head: [full_res, half_res]}`` in train mode (deep
supervision at the two finest decoder levels) and ``{head: full_res}`` in eval.
With ``return_features=True`` it returns ``(out, feats)`` where ``feats`` maps
``"enc<k>"`` to the encoder output at downsampling ``2**k`` (``enc0`` = stem,
``enc4`` = bottleneck for 5 widths); used by the training-only feature
distillation (:class:`FeatureProjector`, never exported).
"""

from __future__ import annotations

import contextlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

BASE_UM = 2.4
IN_CHANNELS = ["ct", "scale"]
RADIAL_CHANNELS = ["r_z", "r_y", "r_x"]  # outward unit radial (perpendicular to the axis), (z, y, x)
# optional scroll-axis direction channels (extra.train.input_axis): the unit scroll axis in
# volume coordinates -- the local umbilicus tangent (a constant (1, 0, 0) with
# extra.train.axis_tangent: false) -- rotated with the volume by every spatial transform.  The fibre classes ("vertical" = along the axis) and the winding field are
# both defined relative to the axis, which an all-axes rotation otherwise hides from the net.
AXIS_CHANNELS = ["a_z", "a_y", "a_x"]  # unit umbilicus tangent (z, y, x); r . a == 0
HEADS: dict[str, int] = {"surface": 2, "ink": 1, "winding": 8}
# two-face surface mode (extra.train.surface_mode = "faces"): [sdf_in, sdf_out, valid logit].
# The same 3-wide head carries surface_mode "sides": [d_face, body logit, valid logit].
FACE_HEADS: dict[str, int] = {"surface": 3, "ink": 1, "winding": 8}
# optional fibre-orientation head (extra.train.heads.fiber): [vertical logit, horizontal/angular
# logit], distilled from the 4-class fiber teacher's softmax channels 1 and 2.
FIBER_HEAD: dict[str, int] = {"fiber": 2}
FIBER_CH = ["vt", "hz"]
# fiber_mode "direction" (tsm.fiber): one axial direction + a strength, instead of two classes
FIBER_DIR_HEAD: dict[str, int] = {"fiber": 4}
FIBER_DIR_CH = ["dz", "dy", "dx", "strength"]
# optional Local Shape Descriptor head (extra.train.heads.lsd, "sides" mode): the five
# descriptors of tsm.data.lsd_targets -- [off_z, off_y, off_x] (voxels, raw), thickness
# (voxels, raw) and the normal-incoherence LOGIT
LSD_HEAD: dict[str, int] = {"lsd": 5}
LSD_CH = ["off_z", "off_y", "off_x", "thick", "ncov"]


def fiber_head(fiber_mode: str = "class") -> dict[str, int]:
    """The fibre head width for a mode: 2 ([vt, hz] logits) or 4 ([dz, dy, dx, strength])."""
    if fiber_mode == "class":
        return dict(FIBER_HEAD)
    if fiber_mode == "direction":
        return dict(FIBER_DIR_HEAD)
    raise ValueError(f"fiber_mode must be 'class' or 'direction', got {fiber_mode!r}")


def heads_for(surface_mode: str = "medial", fiber: bool = False, fiber_mode: str = "class",
              gap_class: bool = False, lsd: bool = False) -> dict[str, int]:
    """Head widths for a surface mode: "medial" (default), "faces" (surface head = 3) or
    "body" (the orientation-free ``min(sdf_in, -sdf_out)``: the *medial* widths, surface head
    = 2 = [sdf_body, valid logit], so the export / TRT layout is unchanged) or "sides"
    (orientation-free with the magnitude and the sign split: surface head = 3 =
    [d_face (raw voxels), body logit, valid logit] -- the same width as the two-face head, so
    again nothing in the export / TRT layout changes);
    ``fiber`` appends the fibre head in either mode (2 channels in the ``"class"`` mode,
    4 in ``"direction"``).  ``gap_class`` (sides mode only, ``extra.train.surface_aux.gap_class``
    > 0) widens the surface head to 4 = [d_face, body logit, valid logit, gap logit].
    ``lsd`` (sides mode only, ``extra.train.heads.lsd``) adds the separate 5-channel Local
    Shape Descriptor head (:func:`tsm.data.lsd_targets`)."""
    if gap_class and surface_mode != "sides":
        raise ValueError(f"gap_class needs surface_mode='sides', got {surface_mode!r}")
    if lsd and surface_mode != "sides":
        raise ValueError(f"heads.lsd needs surface_mode='sides', got {surface_mode!r}")
    if surface_mode in ("faces", "sides"):
        h = dict(FACE_HEADS)
        if gap_class:
            # extra.train.surface_aux.gap_class > 0: one MORE surface channel, the air-gap
            # logit, appended after the valid logit -> [d_face, body, valid, gap].  Only
            # "sides" ever asks for it; every other mode keeps the historical head layout.
            h["surface"] = h["surface"] + 1
    elif surface_mode in ("medial", "body"):
        h = dict(HEADS)
    else:
        raise ValueError(f"surface_mode must be 'medial', 'faces', 'body' or 'sides', "
                         f"got {surface_mode!r}")
    if fiber:
        h.update(fiber_head(fiber_mode))
    if lsd:
        h.update(LSD_HEAD)
    return h
WINDING_CH = ["sin", "cos", "density", "nx", "ny", "nz", "conf", "spare"]


def input_channels(input_radial: bool = False, input_axis: bool = False) -> list[str]:
    """Names of the student input channels in order (2, +3 with the radial field, +3 with the
    scroll-axis direction: 2 / 5 / 8)."""
    return (list(IN_CHANNELS) + (list(RADIAL_CHANNELS) if input_radial else [])
            + (list(AXIS_CHANNELS) if input_axis else []))


def in_channels(input_radial: bool = False, input_axis: bool = False) -> int:
    """Number of student input channels (excluding ``aux_ch``)."""
    return len(input_channels(input_radial, input_axis))


def input_vec_slices(input_radial: bool = False, input_axis: bool = False) -> list[tuple[int, int]]:
    """Channel slices of the input that hold (z, y, x) vector fields and must be transformed
    as vectors by every spatial augmentation / TTA transform (radial, then scroll axis)."""
    out: list[tuple[int, int]] = []
    c = len(IN_CHANNELS)
    for on in (input_radial, input_axis):
        if on:
            out.append((c, c + 3))
            c += 3
    return out


def scale_channel_value(voxel_um: float) -> float:
    return math.log2(float(voxel_um) / BASE_UM)


def normalize_ct(ct: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Per-crop z-score of a CT crop given in [0, 1] (float) or uint8.

    ``ct`` is (..., Z, Y, X); statistics are taken over the last three dims so a
    batch of crops is normalized per crop.
    """
    x = ct.float()
    if ct.dtype == torch.uint8:
        x = x / 255.0
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    std = x.std(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def make_input(ct: torch.Tensor, voxel_um: float | torch.Tensor,
               radial: torch.Tensor | None = None, axis_dir: torch.Tensor | None = None) -> torch.Tensor:
    """(B, Z, Y, X) CT (uint8 or [0,1] float) -> (B, 2, Z, Y, X) network input.

    With ``radial`` (B, 3, Z, Y, X) -- the outward unit radial field in (z, y, x)
    order -- the result is the 5-channel input ``[ct, scale, r_z, r_y, r_x]``; with
    ``axis_dir`` as well (the local scroll-axis tangent, same layout) the 8-channel
    ``[ct, scale, r_z, r_y, r_x, a_z, a_y, a_x]`` (``labels.geometry_frame`` gives both)."""
    if ct.ndim != 4:
        raise ValueError(f"ct must be (B, Z, Y, X), got {tuple(ct.shape)}")
    x = normalize_ct(ct).unsqueeze(1)
    if isinstance(voxel_um, torch.Tensor):
        s = torch.log2(voxel_um.float() / BASE_UM).view(-1, 1, 1, 1, 1).to(x.device)
        s = s.expand(x.shape[0], 1, *x.shape[2:])
    else:
        s = torch.full_like(x, scale_channel_value(voxel_um))
    if axis_dir is not None and radial is None:
        raise ValueError("axis_dir without radial: the input channel order is [ct, scale, radial, axis]")
    if radial is None:
        return torch.cat([x, s], dim=1)
    parts = [x, s]
    for name, v in (("radial", radial), ("axis_dir", axis_dir)):
        if v is None:
            continue
        if v.ndim != 5 or v.shape[1] != 3 or v.shape[0] != x.shape[0] or tuple(v.shape[2:]) != tuple(x.shape[2:]):
            raise ValueError(f"{name} must be (B, 3, Z, Y, X) matching the CT, got {tuple(v.shape)}")
        parts.append(v.to(x.dtype).to(x.device))
    return torch.cat(parts, dim=1)


NORM_KINDS = ("group", "batch")


def _gn(ch: int, groups: int) -> nn.GroupNorm:
    if ch % groups:
        raise ValueError(f"width {ch} must be a multiple of the GroupNorm group count {groups}")
    return nn.GroupNorm(groups, ch)


def _norm(ch: int, groups: int, kind: str = "group") -> nn.Module:
    """Normalisation layer: GroupNorm (default, TTA-safe) or BatchNorm3d (foldable, ablation only).

    GroupNorm is the default because it is deterministic per crop and does not
    interact with test-time augmentation; BatchNorm folds into the preceding conv
    at export time (plan A2) but geometric TTA measurably hurt BN models
    (docs/student_v2_plan.md G), so ``norm="batch"`` exists to ablate, not as the default.
    """
    if kind == "group":
        return _gn(ch, groups)
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    raise ValueError(f"norm must be one of {NORM_KINDS}, got {kind!r}")


@contextlib.contextmanager
def _frozen_norm_stats(mod: nn.Module):
    """Undo any running-stat update made by the wrapped forward on every ``_NormBase`` under ``mod``.

    The buffers are snapshotted and restored rather than switching ``track_running_stats`` off,
    because that would change which tensors ``F.batch_norm`` saves and trip
    ``torch.utils.checkpoint``'s forward/recompute consistency check."""
    bns = [m for m in mod.modules() if isinstance(m, nn.modules.batchnorm._NormBase)]
    saved = [(m, [(b, b.detach().clone()) for b in (m.running_mean, m.running_var, m.num_batches_tracked)
                  if b is not None]) for m in bns]
    try:
        yield
    finally:
        with torch.no_grad():
            for _, bufs in saved:
                for buf, old_val in bufs:
                    buf.copy_(old_val)


class ResBlock(nn.Module):
    """conv-N-SiLU-conv-N + skip, SiLU after the add. Stride on the first conv."""

    def __init__(self, cin: int, cout: int, stride: int = 1, groups: int = 8, norm: str = "group") -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm(cout, groups, norm)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(cout, groups, norm)
        self.act = nn.SiLU()  # not in-place: export/quantization friendly
        if stride != 1 or cin != cout:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False), _norm(cout, groups, norm)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act(y + self.skip(x))


def _stage(cin: int, cout: int, blocks: int, stride: int, groups: int, norm: str = "group") -> nn.Sequential:
    mods = [ResBlock(cin, cout, stride, groups, norm)]
    mods += [ResBlock(cout, cout, 1, groups, norm) for _ in range(blocks - 1)]
    return nn.Sequential(*mods)


class TSMNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        aux_ch: int = 0,
        widths: Sequence[int] = (32, 64, 128, 256, 256),
        blocks: int = 2,
        dec_blocks: int = 1,
        bottleneck_blocks: int = 1,
        heads: Mapping[str, int] | None = None,
        groups: int = 8,
        ds_levels: int = 2,
        act_ckpt: int = 0,
        body_stride: int = 1,
        fullres_width: int = 32,
        norm: str = "group",
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch) + int(aux_ch)
        self.aux_ch = int(aux_ch)
        self.widths = tuple(int(w) for w in widths)
        self.heads = dict(heads if heads is not None else HEADS)
        self.ds_levels = int(ds_levels)
        self.act_ckpt = int(act_ckpt)  # checkpoint stages at resolution levels < act_ckpt (training only)
        self.body_stride = int(body_stride)
        if self.body_stride not in (1, 2):
            raise ValueError(f"body_stride must be 1 or 2, got {body_stride}")
        self.fullres_width = int(fullres_width)
        self.norm_kind = str(norm)
        self.blocks, self.dec_blocks, self.bottleneck_blocks = int(blocks), int(dec_blocks), int(bottleneck_blocks)
        w = self.widths
        if len(w) < 2:
            raise ValueError("need at least 2 widths")
        self.n_down = len(w) - 1
        # with a stride-2 body the decoder's finest level is at 1/2 resolution and the
        # full-resolution block adds one more (level 0) output
        self.n_levels = self.n_down + (1 if self.body_stride == 2 else 0)
        if self.ds_levels < 1 or self.ds_levels > self.n_levels:
            raise ValueError(f"ds_levels must be in [1, {self.n_levels}]")

        # stride-2 stem (conv k3 s2 inside the first residual block; space-to-depth + 1x1 is
        # the cheaper alternative and is interchangeable here) -> the whole encoder-decoder
        # runs at half resolution (2.4 -> 4.8 um), ~8x fewer voxels in the body.
        self.stem = _stage(self.in_ch, w[0], blocks, self.body_stride, groups, norm)
        self.enc = nn.ModuleList(
            _stage(w[i], w[i + 1], bottleneck_blocks if i == self.n_down - 1 else blocks, 2, groups, norm)
            for i in range(self.n_down)
        )
        # decoder level i upsamples w[i+1] -> w[i], concatenates the skip and refines
        self.up = nn.ModuleList(nn.ConvTranspose3d(w[i + 1], w[i], 2, stride=2) for i in range(self.n_down))
        self.dec = nn.ModuleList(_stage(2 * w[i], w[i], dec_blocks, 1, groups, norm) for i in range(self.n_down))
        # full-resolution block: one residual block on [network input, trilinearly upsampled
        # decoder output] so the 1x1x1 SDF head keeps 1-voxel precision
        self.fullres = (
            ResBlock(self.in_ch + w[0], self.fullres_width, 1, math.gcd(groups, self.fullres_width), norm)
            if self.body_stride == 2 else None
        )
        # heads for output levels 0 (full res) .. ds_levels-1; level l is at 1/2**l resolution
        self.head = nn.ModuleDict(
            {
                name: nn.ModuleList(nn.Conv3d(self.level_width(lvl), c, 1) for lvl in range(self.ds_levels))
                for name, c in self.heads.items()
            }
        )
        self._init()

    def level_width(self, lvl: int) -> int:
        """Channels of the feature map feeding the head at output level ``lvl`` (1/2**lvl res)."""
        if self.body_stride == 2:
            return self.fullres_width if lvl == 0 else self.widths[lvl - 1]
        return self.widths[lvl]

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, convs in self.head.items():
            for c in convs:
                nn.init.normal_(c.weight, std=1e-2)
                nn.init.zeros_(c.bias)

    def _run(self, stage: nn.Module, x: torch.Tensor, level: int) -> torch.Tensor:
        """Stage forward, recomputed in backward when activation checkpointing covers this level.

        The recomputed pass must not touch BatchNorm's running buffers a second time (it would
        double-count every batch and make ``act_ckpt`` -- a memory knob -- change eval behaviour),
        so ``track_running_stats`` is switched off for the recompute.  In training mode BN uses the
        batch statistics either way, so the recomputed activations are bit-identical."""
        if self.training and level < self.act_ckpt and torch.is_grad_enabled():
            calls = [0]

            def fn(inp: torch.Tensor) -> torch.Tensor:
                calls[0] += 1
                if calls[0] == 1:
                    return stage(inp)
                with _frozen_norm_stats(stage):  # backward recompute
                    return stage(inp)

            return checkpoint(fn, x, use_reentrant=False)
        return stage(x)

    @property
    def divisor(self) -> int:
        return 2 ** self.n_levels

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encoder_channels(self) -> dict[str, int]:
        """Channels of each ``enc<k>`` feature (k = encoder stage index).

        With ``body_stride=1`` stage k is at downsampling ``2**k``; with
        ``body_stride=2`` the stem already halves, so stage k sits at ``2**(k+1)``."""
        return {f"enc{k}": c for k, c in enumerate(self.widths)}

    # ------------------------------------------------------------------ #
    # receptive field
    # ------------------------------------------------------------------ #
    def receptive_field_radius(self, method: str = "analytic", size: int | None = None,
                               eps: float = 0.0, n_centers: int | None = None) -> int:
        """Radius (in input voxels) of the full-resolution output's receptive field.

        ``method="analytic"`` walks the kernel/stride schedule: a conv of kernel k at
        jump j adds ``j*(k-1)/2`` to the radius and multiplies the jump by its stride; a
        2x transposed conv (k=2, s=2) maps every output voxel onto exactly one input
        voxel, so it halves the jump without growing the radius; a 2x trilinear upsample
        touches two neighbours, so it adds one input-jump; skip connections take the max
        over the two branches.

        ``method="empirical"`` back-propagates from one central output voxel of a zeros
        input of ``size`` (default 96) and returns the radius of the voxels whose
        gradient magnitude exceeds ``eps`` times the maximum (``eps=0`` = the support).
        Normalisation layers are bypassed during the measurement: GroupNorm/BatchNorm
        take statistics over the whole crop, so with them the "receptive field" of every
        output voxel is the whole window and the convolutional radius -- the quantity the
        halo has to cover -- would be unmeasurable.  Returns ``size // 2`` when the
        support reaches the border (i.e. the measurement saturated: use a larger ``size``).
        """
        if method == "analytic":
            return self._analytic_rf()
        if method != "empirical":
            raise ValueError(f"method must be 'analytic' or 'empirical', got {method!r}")
        return self._empirical_rf(int(size or 96), float(eps), n_centers)

    def _analytic_rf(self) -> int:
        """Worst case over output positions, tracked as an exact support interval.

        A feature map at jump ``s`` has, for index ``i``, the input support
        ``[s*i + lo, s*i + hi]``; the radius of the full-resolution output is
        ``max(-lo, hi)`` at ``s = 1``.  Conv (kernel k, stride t, padding (k-1)/2):
        ``lo -= s*(k-1)/2``, ``hi += s*(k-1)/2``, ``s *= t``.  ConvTranspose(2, 2):
        output ``o`` reads input ``floor(o/2)`` only, so ``s /= 2`` and ``lo -= s`` (the
        odd-``o`` alignment), ``hi`` unchanged.  Trilinear 2x (align_corners=False):
        output ``o`` interpolates the two neighbours of ``o/2 - 0.25``, so the support
        grows by ``1.25`` / ``0.75`` source voxels on the two sides.  Skips take the
        union (min / max) of the two branches.
        """
        def conv(iv: tuple[float, float, float], k: int, stride: int = 1) -> tuple[float, float, float]:
            lo, hi, s = iv
            h = s * ((k - 1) / 2.0)
            return lo - h, hi + h, s * stride

        def block(iv, stride: int = 1):
            return conv(conv(iv, 3, stride), 3, 1)

        def stage(iv, n: int, stride: int):
            iv = block(iv, stride)
            for _ in range(n - 1):
                iv = block(iv, 1)
            return iv

        def merge(a, b):
            return min(a[0], b[0]), max(a[1], b[1]), a[2]

        iv = stage((0.0, 0.0, 1.0), self.blocks, self.body_stride)  # stem
        skips = [iv]
        for i in range(self.n_down):
            n = self.bottleneck_blocks if i == self.n_down - 1 else self.blocks
            iv = stage(iv, n, 2)
            skips.append(iv)
        for i in range(self.n_down - 1, -1, -1):
            s = iv[2] / 2.0  # ConvTranspose3d(k=2, s=2)
            iv = (iv[0] - s, iv[1], s)
            iv = stage(merge(iv, skips[i]), self.dec_blocks, 1)
        if self.body_stride == 2:
            s = iv[2] / 2.0  # 2x trilinear upsample, then the full-resolution block
            iv = (iv[0] - 2.5 * s, iv[1] + 1.5 * s, s)
            iv = block(merge(iv, (0.0, 0.0, s)), 1)
        return int(math.ceil(max(-iv[0], iv[1])))

    def _norm_bypass(self):
        """Context manager replacing every norm layer by an identity (see receptive_field_radius)."""

        @contextlib.contextmanager
        def cm():
            saved: list[tuple[nn.Module, str, nn.Module]] = []
            for mod in self.modules():
                for name, child in list(mod.named_children()):
                    if isinstance(child, (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d)):
                        saved.append((mod, name, child))
                        setattr(mod, name, nn.Identity())
            try:
                yield
            finally:
                for mod, name, child in saved:
                    setattr(mod, name, child)

        return cm()

    def _empirical_rf(self, size: int, eps: float, n_centers: int | None = None) -> int:
        """Max over ``n_centers`` neighbouring output voxels (default: one full stride
        period, capped at 8) -- the support is not symmetric around every voxel, the
        down/up-sampling alignment shifts it by up to one coarse voxel."""
        d = self.divisor
        if size % d:
            size = int(math.ceil(size / d)) * d
        was_training = self.training
        self.eval()
        n = int(n_centers if n_centers is not None else min(d, 8))
        c0 = size // 2
        best = 0
        for k in range(n):
            c = c0 + k
            x = torch.zeros(1, self.in_ch, size, size, size, requires_grad=True)
            with self._norm_bypass():
                out = self(x)
            y = out[next(iter(sorted(out)))]
            y = y[0] if isinstance(y, (list, tuple)) else y
            y[0, 0, c, c, c].backward()
            g = x.grad.detach().abs().amax(dim=(0, 1))
            m = float(g.max())
            idx = torch.nonzero(g > (eps * m if m > 0 else 0.0))
            if idx.numel():
                lo, hi = int(idx.amin(dim=0).min()), int(idx.amax(dim=0).max())
                if lo == 0 or hi == size - 1:
                    if was_training:
                        self.train()
                    raise ValueError(
                        f"the receptive field reaches the border of the {size}^3 measurement volume "
                        f"(analytic radius {self._analytic_rf()}): call with size >= "
                        f"{2 * self._analytic_rf() + 2 * n + self.divisor}")
                best = max(best, c - lo, hi - c)
        if was_training:
            self.train()
        return best

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None, return_features: bool = False):
        if aux is not None:
            x = torch.cat([x, aux], dim=1)
        if x.shape[1] != self.in_ch:
            raise ValueError(f"expected {self.in_ch} input channels, got {x.shape[1]}")
        d = self.divisor
        if any(s % d for s in x.shape[2:]):
            raise ValueError(f"spatial shape {tuple(x.shape[2:])} must be divisible by {d}")
        skips = [self._run(self.stem, x, 0)]
        for i, st in enumerate(self.enc):
            skips.append(self._run(st, skips[-1], i))
        y = skips[-1]
        off = 1 if self.body_stride == 2 else 0  # decoder level i is output level i + off
        feats: dict[int, torch.Tensor] = {}
        for i in range(self.n_down - 1, -1, -1):
            y = self.up[i](y)
            y = self._run(self.dec[i], torch.cat([y, skips[i]], dim=1), i)
            if i + off < self.ds_levels:
                feats[i + off] = y
        if off:
            up = nn.functional.interpolate(y, scale_factor=2.0, mode="trilinear", align_corners=False)
            feats[0] = self._run(self.fullres, torch.cat([x, up], dim=1), 0)
            del up
        out: dict[str, list[torch.Tensor]] = {}
        levels = range(self.ds_levels) if self.training else range(1)
        for name, convs in self.head.items():
            out[name] = [convs[lvl](feats[lvl]) for lvl in levels]
        result = out if self.training else {k: v[0] for k, v in out.items()}
        if return_features:
            return result, {f"enc{k}": t for k, t in enumerate(skips)}
        return result


class FeatureProjector(nn.Module):
    """Training-only 1x1x1 conv (+GroupNorm) per (student stage -> teacher stage) pair.

    ``pairs`` are ``(student_k, teacher_k)`` downsampling powers; ``student_ch``
    / ``teacher_ch`` are the channel widths indexed by those powers.  Lives in
    the trainer state, not in the exported student.  With ``identity=True`` and
    equal widths the conv starts as an identity map (no norm), for tests.
    """

    def __init__(
        self,
        pairs: Sequence[Sequence[int]],
        student_ch: Mapping[int, int] | Sequence[int],
        teacher_ch: Mapping[int, int] | Sequence[int],
        groups: int = 8,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.pairs = [(int(a), int(b)) for a, b in pairs]
        self.proj = nn.ModuleDict()
        for sk, tk in self.pairs:
            cin, cout = int(student_ch[sk]), int(teacher_ch[tk])
            mods: list[nn.Module] = [nn.Conv3d(cin, cout, 1, bias=not norm)]
            if norm:
                mods.append(_gn(cout, math.gcd(groups, cout)))
            self.proj[self.key(sk, tk)] = nn.Sequential(*mods)

    @staticmethod
    def key(sk: int, tk: int) -> str:
        return f"s{sk}_t{tk}"

    def forward(self, feats: Mapping[str, torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        return {(sk, tk): self.proj[self.key(sk, tk)](feats[f"enc{sk}"]) for sk, tk in self.pairs}


def feature_distill_loss(
    proj: Mapping[tuple[int, int], torch.Tensor],
    teacher: Sequence[torch.Tensor],
    kind: str = "cosine",
) -> torch.Tensor:
    """Mean over pairs of ``1 - cos_sim`` per voxel (``cosine``) or MSE between the
    projected student feature and the (detached) teacher stage ``tk``; the
    teacher map is trilinearly resampled to the student's spatial size if needed."""
    if kind not in ("cosine", "mse"):
        raise ValueError(f"feat_distill.loss must be 'cosine' or 'mse', got {kind!r}")
    terms = []
    for (sk, tk), s in proj.items():
        t = teacher[tk].detach().float()
        s = s.float()
        if t.shape[2:] != s.shape[2:]:
            t = nn.functional.interpolate(t, size=s.shape[2:], mode="trilinear", align_corners=False)
        if t.shape[1] != s.shape[1]:
            raise ValueError(f"pair ({sk},{tk}): projected {s.shape[1]} ch vs teacher {t.shape[1]} ch")
        if kind == "cosine":
            terms.append((1.0 - nn.functional.cosine_similarity(s, t, dim=1, eps=1e-6)).mean())
        else:
            terms.append(nn.functional.mse_loss(s, t))
    return torch.stack(terms).mean() if terms else torch.zeros(())


def conv_macs(net: nn.Module, x: torch.Tensor, **kw) -> int:
    """Multiply-accumulates of every Conv3d / ConvTranspose3d in one ``net(x)`` forward.

    Hook based, so it counts what actually runs (deep-supervision heads included in
    train mode).  Conv3d: ``out_voxels * cout * cin/groups * prod(k)``; ConvTranspose3d
    scatters each *input* voxel over the kernel, so it is
    ``in_voxels * cin * cout/groups * prod(k)``."""
    macs = [0]

    def hook(m, inp, out):
        k = math.prod(m.kernel_size)
        if isinstance(m, nn.ConvTranspose3d):
            n = int(inp[0].numel() // inp[0].shape[1])  # batch * input voxels
            macs[0] += n * m.in_channels * (m.out_channels // m.groups) * k
        else:
            n = int(out.numel() // out.shape[1])
            macs[0] += n * m.out_channels * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d))]
    try:
        with torch.no_grad():
            net(x, **kw)
    finally:
        for h in hs:
            h.remove()
    return macs[0]


def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last_3d) if x.ndim == 5 else x


def load_student_state(model: nn.Module, sd: Mapping[str, torch.Tensor], what: str = "checkpoint") -> None:
    """``load_state_dict`` that tolerates a *head* mismatch only, with a printed notice.

    Optional heads come and go (``heads.fiber``): a checkpoint written before the fibre head
    existed has no ``head.fiber.*`` tensors, and a fibre checkpoint loaded into a model built
    without the head has tensors nothing consumes.  Both are reported and accepted; any other
    missing / unexpected key is a real mismatch and still raises."""
    missing, unexpected = model.load_state_dict(sd, strict=False)  # type: ignore[arg-type]
    bad_m = [k for k in missing if not k.startswith("head.")]
    bad_u = [k for k in unexpected if not k.startswith("head.")]
    if bad_m or bad_u:
        raise RuntimeError(f"{what}: state dict mismatch outside the heads: "
                           f"{len(bad_m)} missing {bad_m[:5]}, {len(bad_u)} unexpected {bad_u[:5]}")
    def _heads(keys):  # noqa: E306
        return sorted({k.split(".")[1] for k in keys})
    if missing:
        print(f"[tsm] {what}: head(s) {_heads(missing)} are NOT in the checkpoint and keep their "
              f"random initialisation ({len(missing)} tensors)", flush=True)
    if unexpected:
        print(f"[tsm] {what}: head(s) {_heads(unexpected)} are in the checkpoint but not in this "
              f"model and were dropped ({len(unexpected)} tensors)", flush=True)


def build_model(
    widths: Sequence[int] = (32, 64, 128, 256, 256),
    aux_ch: int = 0,
    in_ch: int = 2,
    compile: bool = False,
    act_ckpt: int = 0,
    channels_last: bool = False,
    surface_mode: str = "medial",
    body_stride: int = 1,
    fullres_width: int = 32,
    norm: str = "group",
    fiber: bool = False,
    fiber_mode: str = "class",
    gap_class: bool = False,
    lsd: bool = False,
    **kw,
) -> nn.Module:
    kw.setdefault("heads", heads_for(surface_mode, fiber, fiber_mode, gap_class, lsd))
    net = TSMNet(widths=widths, in_ch=in_ch, aux_ch=aux_ch, act_ckpt=act_ckpt,
                 body_stride=body_stride, fullres_width=fullres_width, norm=norm, **kw)
    if channels_last:
        net = net.to(memory_format=torch.channels_last_3d)
    if compile:
        net = torch.compile(net)  # type: ignore[assignment]
    return net


def activation_bytes_estimate(patch: int, batch: int, widths: Sequence[int], blocks: int = 2, dec_blocks: int = 1, act_ckpt: int = 0) -> int:
    """Rough forward-activation footprint kept for backward under bf16 autocast.

    Convs emit bf16 (2 B) but GroupNorm/SiLU run in fp32 (4 B), so a residual
    block keeps ~2 bf16 + ~4 fp32 tensors of its width.  Levels covered by
    activation checkpointing keep only their stage inputs (recomputed in backward).
    """
    total = 0
    n = patch ** 3
    w = list(widths)
    for lvl, c in enumerate(w):
        vox = n // (8 ** lvl)
        nb = blocks + (dec_blocks if lvl < len(w) - 1 else 0)
        if lvl < act_ckpt:
            per_block = 0
            total += 3 * (2 * c) * vox * 2  # stage inputs (stem in, enc in, dec concat in) + head features
        else:
            per_block = 2 * c * vox * 2 + 4 * c * vox * 4
        total += nb * per_block + (2 * c * vox * 2 if lvl < len(w) - 1 else 0)
    return int(total * batch)


if __name__ == "__main__":  # params / MACs / receptive field per body_stride (CPU)
    p = 128
    for bs in (1, 2):
        net = TSMNet(in_ch=5, body_stride=bs).eval()
        macs = conv_macs(net, torch.zeros(1, 5, p, p, p))
        with torch.no_grad():
            o = net(torch.zeros(1, 5, 64, 64, 64))
        shapes = {k: tuple(v.shape) for k, v in o.items()}
        print(f"TSMNet body_stride={bs}: {net.num_params() / 1e6:.2f} M params, {macs / 1e9:.1f} GMAC "
              f"at {p}^3, receptive-field radius {net.receptive_field_radius()} vox, out {shapes}")
