"""Student inference (``tsm infer``) and export to villa's consumers (``tsm export``).

``run_infer``: TSMNet with its EMA weights, sliding window (``tsm.sliding``) over the
config region at the config pitch (2.4 um) -> ``<out_dir>/student/pred.zarr`` +
``pred.summary.json`` + mid-slice preview PNGs.  ``export_lasagna`` / ``export_spiral``
turn ``pred.zarr`` into the formats villa's lasagna optimizer and the spiral scripts
consume.

pred.zarr: (C, Z, Y, X) uint8, chunks (1, 128, 128, 128), attrs like the label store
(``channels``, ``voxel_um``, ``origin_zyx``, ``scale``); encodings = docs/label_store.md:

  0 sdf       round(128 + clip(sdf)*127/clip); 0 = no data (empty / uncovered tile)
  1 valid     sigmoid(valid logit) * 255
  2 ink       sigmoid(ink logit) * 255
  3 sin 4 cos (sin, cos) of 2*pi*w renormalised to the unit circle: 127.5 + 127.5 v
  5 density   relu(density) * 1000, wraps per voxel of the inference pitch (clipped 255)
  6 nx 7 ny 8 nz  unit normal (renormalised): 127.5 + 127.5 v
  9 conf      sigmoid(conf logit) * 255
 10 spare     sigmoid(spare) * 255 (unsupervised, kept so the head tensor is complete)
 11 surface1  255 on the 1-voxel zero-crossing surface (``extract_surface``), else 0

The sliding engine blends windows with gaussian weights and quantises ``round(p*255)``;
``StudentNet`` therefore maps every activated head into [0, 1] (``to_unit``) so that
the quantised byte *is* the label-store encoding.  Blending happens on those unit
values: linear in sdf / sin / cos / normal / density, on probabilities for the logits;
(sin, cos) and the normal may lose unit length in the blend -- consumers renormalise.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

from tsm.config import RunCfg
from tsm.data import CLIP, decode_density, decode_prob, decode_sdf, decode_signed, encode_density, encode_prob, encode_signed
from tsm.labels import (DEFAULT_AXIS, axis_yx_at, hist_percentiles, iter_cores, load_axis, read_box_padded,
                        surface_stats)
from tsm.limits import cuda_guard, estimate_and_assert, free_cuda, peak_rss_mb
from tsm.student import (BASE_UM, build_model, in_channels, input_channels, load_student_state, normalize_ct,
                         scale_channel_value)
from tsm.train import EMA, TRAIN_DEFAULTS
from tsm.volume import BrickWriter

__all__ = [
    "PRED_CHANNELS",
    "FACE_PRED_CHANNELS",
    "FIBER_PRED_CHANNELS",
    "pred_channels",
    "n_head_ch",
    "N_HEAD_CH",
    "BODY_PRED_CHANNELS",
    "SIDES_PRED_CHANNELS",
    "N_BODY_HEAD_CH",
    "INFER_DEFAULTS",
    "EXPORT_DEFAULTS",
    "activate_heads",
    "to_unit",
    "decode_pred",
    "StudentNet",
    "TTA_MODES",
    "tta_transforms",
    "tta_forward",
    "tta_inverse",
    "load_student",
    "student_build_kw",
    "checkpoint_axis_tangent",
    "student_rf_radius",
    "rf_tiling",
    "extract_surface",
    "write_surface_channel",
    "write_fiber_class_channels",
    "run_infer",
    "pool_mean",
    "encode_pred_dt",
    "hemisphere_nxny",
    "export_lasagna",
    "export_spiral",
    "run_export",
]

PRED_CHANNELS = ["sdf", "valid", "ink", "sin", "cos", "density", "nx", "ny", "nz", "conf", "spare", "surface1"]
N_HEAD_CH = 11  # surface 2 + ink 1 + winding 8
# two-face mode (extra.train.surface_mode = "faces"): surface 3 + ink 1 + winding 8 head channels,
# plus the two thin surfaces and the derived thickness
FACE_PRED_CHANNELS = ["sdf_in", "sdf_out", "valid", "ink", "sin", "cos", "density", "nx", "ny", "nz", "conf", "spare",
                      "surface_in1", "surface_out1", "thickness"]
N_FACE_HEAD_CH = 12
# orientation-free body mode (extra.train.surface_mode = "body"): the medial head layout
# (surface 2 + ink 1 + winding 8) with the single SDF being tsm.data.body_sdf, plus the one
# derived thin surface of its zero set.  No thickness channel: the body SDF does not name sides.
BODY_PRED_CHANNELS = ["sdf_body", "valid", "ink", "sin", "cos", "density", "nx", "ny", "nz", "conf", "spare",
                      "surface_body1"]
N_BODY_HEAD_CH = 11
# orientation-free "sides" mode (extra.train.surface_mode = "sides"): the TWO-FACE head layout
# (surface 3 + ink 1 + winding 8) with the three surface channels being the UNSIGNED distance to
# the nearest face, the sheet-body probability and the validity -- so the export / TRT layout is
# identical to the two-face one.  `d_face` is written with the SAME byte encoding as any SDF
# (128 + clip(d)*127/clip), so an existing sdf reader decodes it as a non-negative distance; the
# derived `surface_side1` is the 2-voxel shell `d_face <= 1` (both sides of every face at once).
# No thickness channel: the mode names no sides.
SIDES_PRED_CHANNELS = ["d_face", "body", "valid", "ink", "sin", "cos", "density", "nx", "ny", "nz", "conf",
                       "spare", "surface_side1"]
N_SIDES_HEAD_CH = 12
# optional fibre head (2 logits): appended after the winding channels, before the derived
# surface1 / thickness channels, so an old store is a prefix of a new one only up to `spare`.
FIBER_PRED_CHANNELS = ["fiber_vt", "fiber_hz"]
N_FIBER_HEAD_CH = 2
# fiber_mode "direction" (tsm.fiber): the head writes the axial direction + strength, and the
# two class probabilities are *derived* afterwards from the direction and the predicted sheet
# normal (grad sdf_in) -- so downstream consumers and dev/eval_region.py keep working.
FIBER_DIR_PRED_CHANNELS = ["fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"]
N_FIBER_DIR_HEAD_CH = 4


def pred_channels(surface_mode: str = "medial", fiber: bool = False, fiber_mode: str = "class") -> list[str]:
    base = {"faces": FACE_PRED_CHANNELS, "body": BODY_PRED_CHANNELS,
            "sides": SIDES_PRED_CHANNELS}.get(surface_mode, PRED_CHANNELS)
    base = list(base)
    if not fiber:
        return base
    cut = base.index("spare") + 1  # the derived (non-head) channels follow `spare`
    if fiber_mode == "direction":
        # head channels stay contiguous; the derived vt/hz go last, with surface1 / thickness
        return base[:cut] + list(FIBER_DIR_PRED_CHANNELS) + base[cut:] + list(FIBER_PRED_CHANNELS)
    return base[:cut] + list(FIBER_PRED_CHANNELS) + base[cut:]


def n_head_ch(surface_mode: str = "medial", fiber: bool = False, fiber_mode: str = "class") -> int:
    n = {"faces": N_FACE_HEAD_CH, "body": N_BODY_HEAD_CH,
         "sides": N_SIDES_HEAD_CH}.get(surface_mode, N_HEAD_CH)
    if not fiber:
        return n
    return n + (N_FIBER_DIR_HEAD_CH if fiber_mode == "direction" else N_FIBER_HEAD_CH)
SPIRAL_CHANNELS = ["sin", "cos", "density", "nx", "ny", "nz", "conf", "valid"]
LASAGNA_CHANNELS = ["cos", "grad_mag", "nx", "ny", "pred_dt"]
GRAD_MAG_ENCODE_SCALE = 1000.0
PRED_DT_MAX = 48  # villa: distances clipped to 48 voxels, outside=[80,127] inside=[128,175]

INFER_DEFAULTS: dict[str, Any] = {
    "checkpoint": None,  # default <out_dir>/train/latest.pt
    "widths": None,  # default: from the checkpoint config
    "device": None,  # "cuda" | "cpu"; default cuda when available
    "patch": 128,
    "step": None,  # default patch // 2
    "out_tile": 192,
    # int, None (= patch - step) or "rf": receptive-field halo (see rf_window_spec) --
    # halo = net.receptive_field_radius() + 2, step = patch - 2*halo, uniform core blending
    "halo": None,
    "weight": None,  # None = "uniform" with halo "rf", else "gaussian"; "gaussian" | "uniform"
    "backend": "torch",  # "torch" | "trt" (TensorRT engine for the student, CUDA only)
    "trt_precision": "fp16",  # fp16 (bf16 has no TensorRT 11 tactic for 3D ConvTranspose)
    "trt_workspace_gb": 6.0,
    "batch": 1,
    "surface_mode": None,  # None = from the checkpoint config ("medial" | "faces" | "body")
    "tta": "none",  # "none" | "flip8" | "flip8_rot4" (student-side, per-channel inverse rules; bool: False/True = none/flip8)
    "clip": CLIP,
    "empty_max": 0,
    "empty_window_max": 0,
    "channels_last": True,
    "cudnn_benchmark": True,
    "compile": None,
    "prefetch": True,
    "chunk": 128,
    "previews": True,
    "min_component": 100,
    "stats_box": [256, 512, 512],  # surface_stats on the central sub-box of this size (RAM)
    "probe_brick": 64,
    "axis_path": None,  # umbilicus JSON for the radial input channels; default: the checkpoint's
}

EXPORT_DEFAULTS: dict[str, Any] = {
    "pred": None,  # default <out_dir>/student/pred.zarr
    "out_dir": None,  # default <out_dir>/export
    "axis": DEFAULT_AXIS,  # umbilicus control points (level-0 voxels); null = no rays / no umbilicus_json
    "clip": None,  # default: pred.summary.json clip, else data.CLIP
    "lasagna": {
        "name": "tsm",
        "level_shift": 2,  # the lasagna "input" level: 2 = 9.6 um for a 2.4 um prediction
        "cos_scaledown": 2,  # villa defaults: cos (+ pred_dt) at input/2, grad_mag/nx/ny at input/4
        "scaledown": 4,
        "base_shape_zyx": None,  # default: pred.summary.json volume_shape_zyx, else region end
        "pred_dt_channel": None,  # default: "sdf_in" (two-face) / "sdf_body" (body), else "sdf"
        "sheet_half_vox": 2.0,  # |sdf| <= this (fine voxels) counts as "inside" the sheet for pred_dt
        "min_cover": 0.5,
        "ome_chunk": 32,
        "n_levels": 5,
    },
    "spiral": {
        "level_shift": 2,
        "ray_spacing": 8,  # rays every N level-shift voxels in z
        "ray_angle_deg": 5.0,
        "ray_length": None,  # level-shift voxels; default: to the far corner of the store
        "ray_step": 1.0,
        "min_cover": 0.5,
    },
    "brick": [64, 512, 512],  # fine-voxel brick for the pooling passes
}


def _checkpoint_surface_mode(ckpt: str, override: Any = None) -> str:
    """Surface mode of a student checkpoint ("medial" | "faces" | "body"); ``override`` wins when set."""
    if override:
        return str(override)
    if not os.path.exists(ckpt):
        return "medial"
    try:
        cfgb = torch.load(ckpt, map_location="cpu", weights_only=False).get("config") or {}
    except Exception:  # a corrupt / foreign checkpoint fails later with a better message
        return "medial"
    return str(cfgb.get("surface_mode") or (cfgb.get("train") or {}).get("surface_mode") or "medial")


def _checkpoint_fiber(ckpt: str) -> bool:
    """True when the student checkpoint was trained with the optional fibre head."""
    if not os.path.exists(ckpt):
        return False
    try:
        cfgb = torch.load(ckpt, map_location="cpu", weights_only=False).get("config") or {}
    except Exception:
        return False
    heads = cfgb.get("heads") or (cfgb.get("train") or {}).get("heads") or {}
    return bool(heads.get("fiber", False))


def _checkpoint_fiber_mode(ckpt: str) -> str:
    """``extra.train.fiber_mode`` of a student checkpoint ("class" for anything older)."""
    if not os.path.exists(ckpt):
        return "class"
    try:
        cfgb = torch.load(ckpt, map_location="cpu", weights_only=False).get("config") or {}
    except Exception:
        return "class"
    return str(cfgb.get("fiber_mode") or (cfgb.get("train") or {}).get("fiber_mode") or "class")


def _opts(cfg: RunCfg, key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.extra.get(key, {})
    if not isinstance(raw, dict):
        raise ValueError(f"extra.{key} must be an object")
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ValueError(f"unknown extra.{key} keys: {unknown}")
    opts = dict(defaults)
    for k, v in raw.items():
        if isinstance(defaults.get(k), dict) and isinstance(v, dict):
            unknown = sorted(set(v) - set(defaults[k]))
            if unknown:
                raise ValueError(f"unknown extra.{key}.{k} keys: {unknown}")
            opts[k] = {**defaults[k], **v}
        else:
            opts[k] = v
    return opts


# --------------------------------------------------------------------------- #
# head activations and the [0, 1] mapping used by the sliding engine
# --------------------------------------------------------------------------- #
def activate_heads(out: dict[str, torch.Tensor], surface_mode: str = "medial",
                   fiber_mode: str = "class") -> torch.Tensor:
    """TSMNet eval output dict -> (B, 11 | 12 [+2 with the fibre head], Z, Y, X) float32 physical values.

    [sdf, valid, ink, sin, cos, density, nx, ny, nz, conf, spare] (or, in the two-face mode,
    [sdf_in, sdf_out, valid, ...]): sdf raw (voxels), probabilities via sigmoid, (sin, cos) and
    the normal renormalised, density relu (training used the raw head output; relu keeps it >= 0)."""
    s, i, w = out["surface"].float(), out["ink"].float(), out["winding"].float()
    ns = 2 if surface_mode == "faces" else 1  # "body" is one channel, like "medial"
    sc = F.normalize(w[:, 0:2], dim=1, eps=1e-6)
    n = F.normalize(w[:, 3:6], dim=1, eps=1e-6)
    if surface_mode == "sides":
        # [d_face, body logit, valid logit]: the distance is relu'd (training used the raw head
        # output, relu only enforces the >= 0 the target already guarantees), the other two are
        # probabilities
        head = [torch.relu(s[:, 0:1]), torch.sigmoid(s[:, 1:3])]
    else:
        head = [s[:, 0:ns], torch.sigmoid(s[:, ns:ns + 1])]
    parts = head + [torch.sigmoid(i[:, 0:1]), sc, torch.relu(w[:, 2:3]), n,
                    torch.sigmoid(w[:, 6:7]), torch.sigmoid(w[:, 7:8])]
    if "fiber" in out:
        f = out["fiber"].float()
        if fiber_mode == "direction":  # [dz, dy, dx, strength], appended after `spare`
            parts += [F.normalize(f[:, 0:3], dim=1, eps=1e-6), torch.sigmoid(f[:, 3:4])]
        else:  # [vt, hz] probabilities, appended after `spare`
            parts.append(torch.sigmoid(f[:, 0:2]))
    return torch.cat(parts, dim=1)


def to_unit(phys: torch.Tensor, clip: float, surface_mode: str = "medial",
            fiber_mode: str = "class") -> torch.Tensor:
    """(B, 11 | 12, ...) physical values -> [0, 1] so that round(u*255) is the label-store byte.

    sdf -> (128 + clip(sdf)*127/clip)/255 (bytes 1..255, 0 stays "no data"); in the two-face mode
    both sdf channels use that encoding, and in the "sides" mode ``d_face`` uses it too (clamped
    to [0, clip], so its bytes are always >= 128 and any sdf reader sees a non-negative distance)
    while ``body`` is a probability.  sin/cos/normal -> (v+1)/2 (= 127.5 + 127.5 v);
    density -> v*1000/255; probabilities as is."""
    clip = float(clip)
    ns = 2 if surface_mode == "faces" else 1
    signed = lambda t: (t + 1.0) * 0.5  # noqa: E731
    if surface_mode == "sides":
        # d_face uses the SAME sdf byte encoding, clamped to [0, clip] so every byte is >= 128:
        # any reader that decodes an sdf channel reads it back as a non-negative distance.
        # `body` is a plain probability.
        o = 1
        head = [(128.0 + phys[:, 0:1].clamp(0.0, clip) * (127.0 / clip)) / 255.0, phys[:, 1:4]]
    else:
        o = ns - 1
        head = [(128.0 + phys[:, 0:ns].clamp(-clip, clip) * (127.0 / clip)) / 255.0,
                phys[:, 1 + o:3 + o]]
    tail = phys[:, 9 + o:]  # conf, spare, then the fibre head
    if fiber_mode == "direction" and tail.shape[1] >= 6:
        # [conf, spare, dz, dy, dx, strength]: the direction is signed like a normal
        tail = torch.cat([tail[:, 0:2], signed(tail[:, 2:5]), tail[:, 5:]], dim=1)
    u = torch.cat(
        head + [signed(phys[:, 3 + o:5 + o]), phys[:, 5 + o:6 + o] * (GRAD_MAG_ENCODE_SCALE / 255.0),
                signed(phys[:, 6 + o:9 + o]), tail], dim=1,
    )
    return u.clamp(0.0, 1.0)


def pred_dt_field(dec: Mapping[str, np.ndarray], dt_ch: str) -> np.ndarray:
    """The **signed** distance field the lasagna ``pred_dt`` encoding wants, from ``decode_pred``.

    Every mode but ``"sides"`` already has one (``sdf_in`` / ``sdf_body`` / ``sdf``).  In sides
    mode the store carries the unsigned ``d_face`` plus the body probability, so the sign is put
    back here: ``where(body > 0.5, d_face, -d_face)`` -- positive inside the papyrus, negative
    outside, 0 on the faces, exactly like the body SDF."""
    if dt_ch == "d_face" and "body" in dec:
        return np.where(dec["body"] > 0.5, dec["d_face"], -dec["d_face"]).astype(np.float32)
    return dec[dt_ch]


def decode_pred(u8: np.ndarray, clip: float, channels: Sequence[str] = PRED_CHANNELS) -> dict[str, np.ndarray]:
    """(C, ...) pred.zarr bytes -> float32 fields + ``data`` (sdf byte != 0) and ``surface1`` (bool)."""
    ch = {name: u8[k] for k, name in enumerate(channels)}
    sdf_names = [k for k in ("sdf", "sdf_in", "sdf_out", "sdf_body", "d_face") if k in ch]
    out: dict[str, np.ndarray] = {"data": ch[sdf_names[0]] != 0}
    for k in sdf_names:
        out[k] = decode_sdf(ch[k], clip) * out["data"]
    if "sdf" not in out:
        # the default "the surface" field of a store that has no plain `sdf` channel: the in
        # face (two-face), the body SDF (body mode) or the unsigned face distance (sides mode)
        alias = next((k for k in ("sdf_in", "sdf_body", "d_face") if k in out), None)
        if alias is not None:
            out["sdf"] = out[alias]
    for k in ("valid", "body", "ink", "conf", "spare", "fiber_vt", "fiber_hz", "fiber_strength"):
        if k in ch:
            out[k] = decode_prob(ch[k])
    for k in ("fiber_dz", "fiber_dy", "fiber_dx"):
        if k in ch:
            out[k] = decode_signed(ch[k])
    for k in ("sin", "cos", "nx", "ny", "nz"):
        out[k] = decode_signed(ch[k])
    out["density"] = decode_density(ch["density"])
    for k in ("surface1", "surface_in1", "surface_out1", "surface_body1", "surface_side1"):
        if k in ch:
            out[k] = ch[k] > 127
    if "thickness" in ch:
        out["thickness"] = ch["thickness"].astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# student test-time augmentation (flips + in-plane rot90), per-channel inverse rules
# --------------------------------------------------------------------------- #
TTA_MODES = ("none", "flip8", "flip8_rot4")


def tta_transforms(mode: str) -> list[tuple[tuple[int, ...], bool]]:
    """``(flip_axes, transpose_yx)`` list: flip8 = every subset of {z, y, x}; flip8_rot4 adds the
    y<->x transpose (rot90 = transpose o flip, so flips x {id, transpose} are the 16 distinct
    elements generated by flips and in-plane rot90s)."""
    if isinstance(mode, bool):
        mode = "flip8" if mode else "none"
    mode = str(mode or "none")
    if mode not in TTA_MODES:
        raise ValueError(f"tta must be one of {TTA_MODES}, got {mode!r}")
    if mode == "none":
        return [((), False)]
    flips = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    out = [(f, False) for f in flips]
    if mode == "flip8_rot4":
        out += [(f, True) for f in flips]
    return out


def tta_forward(x: torch.Tensor, axes: Sequence[int], transpose: bool,
                vec: Any = None) -> torch.Tensor:
    """Apply the transform to a (B, C, Z, Y, X) tensor: flips (spatial axes 0=z, 1=y, 2=x) then y<->x.

    ``vec`` = ``(start, stop)`` (or a sequence of such slices) marks the channel slices holding
    vector fields in (z, y, x) component order (the radial and scroll-axis input channels):
    their components move with the grid -- the transpose swaps the y and x components, a flip of
    axis ``a`` negates component ``a``."""
    if axes:
        x = torch.flip(x, [a + 2 for a in axes])
    if transpose:
        x = x.transpose(3, 4)
    if vec is None:
        return x
    slices = [vec] if (len(vec) == 2 and np.isscalar(vec[0])) else list(vec)  # one slice or several
    parts: list[torch.Tensor] = []
    cut = 0
    for lo, hi in slices:
        lo, hi = int(lo), int(hi)
        if lo > cut:
            parts.append(x[:, cut:lo])
        v = x[:, lo:hi]
        # the flips are applied to the grid before the transpose, so the component signs (indexed by
        # the *original* axes) must be applied before the component swap -- the two do not commute.
        sign = torch.ones(hi - lo, device=x.device, dtype=x.dtype)
        for a in axes:
            sign[a] = -1.0
        v = v * sign.view(1, -1, 1, 1, 1)
        if transpose:
            v = v[:, [0, 2, 1]]  # (z, y, x) components: swap y <-> x
        parts.append(v)
        cut = hi
    return torch.cat(parts + [x[:, cut:]], dim=1)


def tta_inverse(out: dict[str, torch.Tensor], axes: Sequence[int], transpose: bool) -> dict[str, torch.Tensor]:
    """Undo ``tta_forward`` on raw head outputs: every channel moves back with the grid (sdf, valid,
    ink, sin, cos, density, conf, spare and the fibre logits are scalars); the normal
    (winding 3:6 = nx, ny, nz) is a vector: the transpose swaps nx <-> ny, a flip of axis a negates the matching component."""
    res = {}
    for k, v in out.items():
        if transpose:
            v = v.transpose(3, 4)
        if axes:
            v = torch.flip(v, [a + 2 for a in axes])
        if k == "winding":
            n = v[:, 3:6]
            if transpose:
                n = n[:, [1, 0, 2]]
            sign = torch.ones(3, device=v.device, dtype=v.dtype)
            for a in axes:
                sign[2 - a] = -1.0  # spatial axis a (z, y, x) <-> component (nz, ny, nx)
            n = n * sign.view(1, 3, 1, 1, 1)
            v = torch.cat([v[:, :3], n, v[:, 6:]], dim=1)
        elif k == "fiber" and v.shape[1] >= 4:
            # the direction head [dz, dy, dx, strength] is a (z, y, x) vector; the strength and
            # the two class logits of the other mode are scalars and move with the grid alone
            d = v[:, 0:3]
            if transpose:
                d = d[:, [0, 2, 1]]  # (z, y, x) components: swap y <-> x
            sign = torch.ones(3, device=v.device, dtype=v.dtype)
            for a in axes:
                sign[a] = -1.0
            v = torch.cat([d * sign.view(1, 3, 1, 1, 1), v[:, 3:]], dim=1)
        res[k] = v
    return res


class StudentNet(nn.Module):
    """``sliding.run_sliding``-compatible wrapper: (B, 1, p, p, p) z-scored CT -> (B, 11, p, p, p) in [0, 1].

    With ``tta`` != "none" every pass transforms the input (``tta_forward``), runs the net and
    maps the raw heads back (``tta_inverse``); the passes are averaged in the pre-activation
    space and activated once (``activate_heads``)."""

    def __init__(self, net: nn.Module, voxel_um: float, clip: float, tta: str | bool = "none",
                 surface_mode: str = "medial", input_radial: bool = False, axis: Any = None,
                 input_axis: bool = False, fiber_mode: str = "class",
                 axis_tangent: bool = True) -> None:
        super().__init__()
        self.net = net
        self.scale = float(scale_channel_value(voxel_um))
        self.clip = float(clip)
        self.tta = tta_transforms(tta)
        self.surface_mode = str(surface_mode)
        self.input_radial = bool(input_radial)
        self.input_axis = bool(input_axis)
        # True: the geometry channels follow the real umbilicus (local tangent + perpendicular
        # radial, labels.geometry_frame).  False: the legacy constant (1, 0, 0) axis and the
        # in-plane radial -- what every checkpoint written before 2026-09-12 was trained with.
        self.axis_tangent = bool(axis_tangent)
        self.fiber_mode = str(fiber_mode)
        # sliding.predict_box hands the absolute origin of every window to a net that asks for it
        self.needs_box_origin = self.input_radial or self.input_axis
        self.axis = None
        if self.needs_box_origin:
            from tsm.data import load_axis_spec

            self.axis = load_axis_spec(axis)

    def frame(self, x: torch.Tensor, box_origin_zyx: Sequence[Sequence[int]] | Sequence[int]) -> dict[str, torch.Tensor]:
        """``{"radial", "axis_dir"}`` (B, 3, p, p, p) for the windows starting at ``box_origin_zyx``
        -- the same fields ``CropDataset`` feeds at training time (``labels.geometry_frame``)."""
        from tsm.labels import geometry_frame, radial_field

        B = int(x.shape[0])
        shape = tuple(int(v) for v in x.shape[2:])
        origins = list(box_origin_zyx)
        if origins and np.isscalar(origins[0]):
            origins = [origins] * B  # one origin for the whole batch
        if len(origins) != B:
            raise ValueError(f"{len(origins)} window origins for a batch of {B}")
        if self.axis_tangent:
            fr = [geometry_frame(self.axis, o, shape, scale=1) for o in origins]
        else:
            a = np.zeros((3,) + shape, np.float32)
            a[0] = 1.0
            fr = [{"radial": radial_field(self.axis, o, shape, scale=1), "axis_dir": a} for o in origins]
        return {k: torch.from_numpy(np.stack([f[k] for f in fr])).to(x.device, x.dtype)
                for k in ("radial", "axis_dir")}

    def radial(self, x: torch.Tensor, box_origin_zyx: Sequence[Sequence[int]] | Sequence[int]) -> torch.Tensor:
        """(B, 3, p, p, p) outward radial field for the windows starting at ``box_origin_zyx``."""
        return self.frame(x, box_origin_zyx)["radial"]

    def raw(self, x: torch.Tensor, box_origin_zyx: Any = None) -> dict[str, torch.Tensor]:
        """TTA-averaged raw head dict (pre-activation)."""
        s = torch.full_like(x, self.scale)
        parts = [x, s]
        vecs: list[tuple[int, int]] = []
        if self.input_radial or self.input_axis:
            if box_origin_zyx is None:
                raise ValueError("a student with geometry inputs needs box_origin_zyx "
                                 "(the window's absolute (z, y, x) origin)")
            fr = self.frame(x, box_origin_zyx)
            # both are (z, y, x) vector fields and are moved by the TTA transform as such
            for on, k in ((self.input_radial, "radial"), (self.input_axis, "axis_dir")):
                if not on:
                    continue
                start = 2 + 3 * len(vecs)
                parts.append(fr[k])
                vecs.append((start, start + 3))
        inp = torch.cat(parts, dim=1)
        acc: dict[str, torch.Tensor] | None = None
        for axes, tr in self.tta:
            out = self.net(tta_forward(inp, axes, tr, vec=vecs or None))
            out = {k: (v[0] if isinstance(v, (list, tuple)) else v).float() for k, v in out.items()}
            out = tta_inverse(out, axes, tr)
            acc = out if acc is None else {k: acc[k] + out[k] for k in acc}
        assert acc is not None
        return {k: v / len(self.tta) for k, v in acc.items()}

    def forward(self, x: torch.Tensor, box_origin_zyx: Any = None) -> torch.Tensor:
        phys = activate_heads(self.raw(x, box_origin_zyx), self.surface_mode, self.fiber_mode)
        return to_unit(phys, self.clip, self.surface_mode, self.fiber_mode)


def student_build_kw(cfgb: dict[str, Any], widths: Sequence[int] | None = None) -> dict[str, Any]:
    """``build_model`` keyword arguments recorded in a student checkpoint's ``config`` blob.

    Missing keys fall back to the defaults of the checkpoint's era: 2 input channels (no
    radial field), medial surface mode, full-resolution body, GroupNorm.  ``axis_tangent`` is
    read by :func:`checkpoint_axis_tangent` instead -- it is a property of the *input fields*,
    not of the architecture, so it is not a ``build_model`` argument."""
    tb = cfgb.get("train") or {}
    g = lambda k, d: cfgb.get(k, tb.get(k, d))  # noqa: E731
    w = widths or g("widths", None) or TRAIN_DEFAULTS["widths"]
    return {
        "widths": tuple(int(v) for v in w),
        "surface_mode": str(g("surface_mode", None) or "medial"),
        "in_ch": in_channels(bool(g("input_radial", False)), bool(g("input_axis", False))),
        "body_stride": int(g("body_stride", 1)),
        "fullres_width": int(g("fullres_width", 32)),
        "norm": str(g("norm", "group")),
        # the fibre head is optional and absent from every checkpoint written before it existed
        "fiber": bool((g("heads", None) or {}).get("fiber", False)),
        # ... and its width depends on the fibre target encoding (2 = class, 4 = direction);
        # a checkpoint written before fiber_mode existed is a 2-channel class head
        "fiber_mode": str(g("fiber_mode", None) or "class"),
    }


def checkpoint_axis_tangent(cfgb: dict[str, Any]) -> bool:
    """``extra.train.axis_tangent`` of a checkpoint config blob; **False** when absent (a
    checkpoint from before the real-tangent frame existed was trained on the constant axis)."""
    tb = cfgb.get("train") or {}
    return bool(cfgb.get("axis_tangent", tb.get("axis_tangent", False)))


def student_rf_radius(ckpt: str, widths: Sequence[int] | None = None) -> int:
    """Analytic receptive-field radius (input voxels) of the student in ``ckpt``.

    The net is built on the ``meta`` device -- the radius follows from the kernel/stride
    schedule alone, so no weights are read and nothing is allocated."""
    cfgb: dict[str, Any] = {}
    if os.path.exists(ckpt):
        try:
            cfgb = torch.load(ckpt, map_location="meta", weights_only=False).get("config") or {}
        except Exception:
            cfgb = {}
    with torch.device("meta"):
        net = build_model(**student_build_kw(cfgb, widths))
    return int(net.receptive_field_radius())


def rf_tiling(patch: int, rf_radius: int, pad: int = 2) -> tuple[int, int]:
    """(halo, step) for receptive-field tiling: halo = rf_radius + pad, step = patch - 2*halo.

    Instead of 50 % overlap (8 windows per voxel) the window cores tile the volume and
    only the halo is recomputed, so a voxel is covered ``(patch/step)**3`` times.  The
    halo has to fit twice into the patch, which for the canonical student (rf radius 122
    at ``body_stride=1``, 248 at ``body_stride=2``) means patch >= 256 (resp. >= 512)."""
    halo = int(rf_radius) + int(pad)
    step = int(patch) - 2 * halo
    if step <= 0:
        raise ValueError(
            f"receptive-field halo {halo} (rf radius {rf_radius} + {pad}) does not fit twice into patch {patch}: "
            f"use patch > {2 * halo} (e.g. {1 << (2 * halo + 1).bit_length()}) or a fixed halo")
    return halo, step


def load_student(path: str, device: str | torch.device = "cpu", widths: Sequence[int] | None = None) -> tuple[nn.Module, dict[str, Any]]:
    """Build TSMNet from the checkpoint's config, load the EMA weights (else the raw weights), eval."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfgb = ck.get("config") or {}
    tb = cfgb.get("train") or {}
    kw = student_build_kw(cfgb, widths)
    w = kw["widths"]
    smode = kw["surface_mode"]
    # checkpoints written before extra.train.input_radial existed are 2-channel
    radial = bool(cfgb.get("input_radial", tb.get("input_radial", False)))
    ax_in = bool(cfgb.get("input_axis", tb.get("input_axis", False)))
    # ... and checkpoints written before extra.train.axis_tangent existed were trained with the
    # constant (1, 0, 0) axis and the in-plane radial, so the default here is False, not the
    # training default: feeding them the real tangent would be a different input distribution.
    ax_tan = checkpoint_axis_tangent(cfgb)
    axis_path = cfgb.get("axis_path") or tb.get("axis_path") or None
    model = build_model(**kw)
    load_student_state(model, ck["model"], what=f"checkpoint {os.path.basename(path)}")
    has_ema = ck.get("ema") is not None
    if has_ema:
        ema = EMA(model)
        try:
            ema.load_state_dict(ck["ema"])
            ema.copy_to(model)
        except ValueError as exc:  # the head set changed since the checkpoint: use the raw weights
            print(f"[tsm] WARNING checkpoint EMA not loaded ({exc}); using the raw weights", flush=True)
            has_ema = False
    model.eval().to(device)
    return model, {"checkpoint": os.path.abspath(path), "step": int(ck.get("step", -1)), "widths": [int(v) for v in w],
                   "ema": has_ema, "surface_mode": smode, "input_radial": radial, "input_axis": ax_in,
                   "axis_tangent": ax_tan, "axis_path": axis_path,
                   "in_ch": in_channels(radial, ax_in), "input_channels": input_channels(radial, ax_in),
                   "body_stride": kw["body_stride"], "fullres_width": kw["fullres_width"], "norm": kw["norm"],
                   "fiber": bool(kw["fiber"]), "fiber_mode": str(kw["fiber_mode"]),
                   "rf_radius": int(model.receptive_field_radius())}


# --------------------------------------------------------------------------- #
# surface extraction
# --------------------------------------------------------------------------- #
def _any6(mask: np.ndarray) -> np.ndarray:
    """True where any of the 6 face neighbours is True (outside the box = False)."""
    p = np.pad(mask, 1)
    out = p[:-2, 1:-1, 1:-1] | p[2:, 1:-1, 1:-1]
    out |= p[1:-1, :-2, 1:-1] | p[1:-1, 2:, 1:-1]
    out |= p[1:-1, 1:-1, :-2] | p[1:-1, 1:-1, 2:]
    return out


def extract_surface(sdf_u8: np.ndarray, valid_u8: np.ndarray, clip: float) -> np.ndarray:
    """1-voxel surface: data voxels with sdf <= 0 that have a 6-neighbour with sdf > 0
    (the zero crossing, negative side; byte 128 = exactly 0 counts as the surface),
    restricted to valid > 0.5.  For a digital plane this is the inner boundary of the
    non-positive half-space: 1 voxel per column along the dominant axis, 6-connected."""
    data = np.asarray(sdf_u8) != 0
    sdf = decode_sdf(np.asarray(sdf_u8), clip)
    neg = data & (sdf <= 0.0)
    pos = data & (sdf > 0.0)
    return neg & _any6(pos) & (np.asarray(valid_u8) >= 128)


def write_fiber_class_channels(pred_path: str, clip: float, brick: int = 128,
                               log: Callable[[str], None] | None = None, axis: Any = None,
                               origin_zyx: Sequence[int] | None = None) -> dict[str, Any]:
    """Fill the derived ``fiber_vt`` / ``fiber_hz`` channels of a ``fiber_mode="direction"``
    prediction store, brick-wise with a 1-voxel halo.

    The head predicts an axial direction ``d`` and a strength ``s``; the two class
    probabilities everything downstream expects (``dev/eval_region.py``, the ablation report)
    are ``s * |d^ . t_v|`` and ``s * |d^ . t_h|``, with the in-plane basis built from the
    **predicted** sheet normal ``grad sdf_in`` (``sdf`` in the medial mode) and the scroll axis
    (:mod:`tsm.fiber`).  Voxels where the normal is parallel to the axis (no basis) get 0.

    Only the *direction* of ``grad sdf`` matters here (it is normalised), so -- unlike the
    label-side derivation, which insists on ``|grad|`` in [0.5, 1.5] before it trusts the field
    -- any non-zero gradient is used; gating on the magnitude would silently zero the derived
    classes wherever the predicted SDF is flatter or steeper than a true distance field.

    ``axis`` (an umbilicus JSON path or (N, 3) control points) makes "vertical" the **local**
    axis tangent, built per brick at the store's absolute origin (``origin_zyx``, by default the
    store's own ``origin_zyx`` attribute); without it the historical constant (1, 0, 0) is used."""
    from tsm.fiber import class_from_direction, fiber_basis, sheet_normal
    from tsm.labels import axis_tangent_field

    log = log or (lambda m: print(m, flush=True))
    arr = zarr.open_array(store=pred_path, mode="r+")
    ch = list(arr.attrs["channels"])
    for need in ("fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength", "fiber_vt", "fiber_hz"):
        if need not in ch:
            raise ValueError(f"{pred_path} has no {need!r} channel (channels={ch})")
    sdf_name = next(k for k in ("sdf_in", "sdf_body", "sdf", "d_face") if k in ch)
    idx = {n: ch.index(n) for n in ("fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength",
                                    "fiber_vt", "fiber_hz", sdf_name)}
    shape = tuple(int(s) for s in arr.shape[1:])
    ax_pts = None
    if axis is not None:
        from tsm.data import load_axis_spec

        ax_pts = load_axis_spec(axis)
    org = [int(v) for v in (origin_zyx if origin_zyx is not None else arr.attrs.get("origin_zyx", (0, 0, 0)))]
    t0 = time.perf_counter()
    n_ok = n_tot = 0
    for lo, hi in iter_cores(shape, (brick,) * 3):
        lo_h, hi_h = [v - 1 for v in lo], [v + 1 for v in hi]
        sdf = decode_sdf(read_box_padded(arr, lo_h, hi_h, channels=idx[sdf_name])[0], clip)
        d = np.stack([decode_signed(read_box_padded(arr, lo_h, hi_h, channels=idx[k])[0])
                      for k in ("fiber_dz", "fiber_dy", "fiber_dx")])
        st = decode_prob(read_box_padded(arr, lo_h, hi_h, channels=idx["fiber_strength"])[0])
        n, _ = sheet_normal(torch.from_numpy(sdf[None].astype(np.float32)), lo=1e-6, hi=float("inf"))
        ax_fld = None
        if ax_pts is not None:
            ax_fld = torch.from_numpy(axis_tangent_field(
                ax_pts, [o + l for o, l in zip(org, lo_h)], [h - l for l, h in zip(lo_h, hi_h)], scale=1))
        tv, th, ok = fiber_basis(n, ax_fld)
        pv, ph = class_from_direction(torch.from_numpy(d.astype(np.float32)),
                                      torch.from_numpy(st[None].astype(np.float32)), tv, th)
        core = (slice(1, -1),) * 3
        for name, v in (("fiber_vt", pv), ("fiber_hz", ph)):
            u = np.clip(np.rint(v[0].numpy() * 255.0), 0, 255).astype(np.uint8)[core]
            arr[idx[name], lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = u
        n_ok += int(ok[0].numpy()[core].sum())
        n_tot += int(np.prod([h - l for l, h in zip(lo, hi)]))
        del sdf, d, st, n, tv, th, ok, pv, ph
    out = {"seconds": time.perf_counter() - t0, "basis_defined_frac": (n_ok / n_tot) if n_tot else 0.0,
           "sdf_channel": sdf_name, "axis_tangent": ax_pts is not None}
    log(f"[infer] derived fiber_vt/fiber_hz from the direction head "
        f"(in-plane basis defined on {out['basis_defined_frac']:.4f} of the region) in {out['seconds']:.1f}s")
    return out


def write_surface_channel(pred_path: str, clip: float, brick: int = 128, stats_box: Sequence[int] = (256, 512, 512),
                          min_component: int = 100, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Fill the thin-surface channel(s) (and, in the two-face mode, ``thickness``) in pred.zarr
    brick-wise (1-voxel halo, edge replicated); stats on the central box.

    Medial mode: ``surface1`` from ``sdf``.  Two-face mode: ``surface_in1`` from ``sdf_in``,
    ``surface_out1`` from ``sdf_out`` and ``thickness`` = round(clip(sdf_in - sdf_out, 0, 255))
    (the gap between the two zero sets, in voxels, 0 outside the sheet / no data).  Body mode:
    ``surface_body1`` from ``sdf_body`` and no thickness pass (the body SDF names no sides).
    Sides mode: ``surface_side1`` = ``(d_face <= 1) & (valid >= 0.5)``, a 2-voxel-thick shell
    around every face (pointwise -- an unsigned distance has no zero crossing to extract), and
    again no thickness pass."""
    log = log or (lambda m: print(m, flush=True))
    arr = zarr.open_array(store=pred_path, mode="r+")
    ch = list(arr.attrs["channels"])
    faces = "sdf_in" in ch
    sides = "d_face" in ch
    if faces:
        pairs = [("surface_in1", "sdf_in"), ("surface_out1", "sdf_out")]
    elif sides:
        pairs = [("surface_side1", "d_face")]
    elif "sdf_body" in ch:
        pairs = [("surface_body1", "sdf_body")]
    else:
        pairs = [("surface1", "sdf")]
    ival = ch.index("valid")
    shape = tuple(int(s) for s in arr.shape[1:])
    t0 = time.perf_counter()
    out: dict[str, Any] = {"seconds": 0.0}
    for surf_name, sdf_name in pairs:
        ci, isdf = ch.index(surf_name), ch.index(sdf_name)
        n_surf = n_data = 0
        for lo, hi in iter_cores(shape, (brick,) * 3):
            lo_h, hi_h = [v - 1 for v in lo], [v + 1 for v in hi]
            sdf_box = read_box_padded(arr, lo_h, hi_h, channels=isdf)[0]
            val_box = read_box_padded(arr, lo_h, hi_h, channels=ival)[0]
            if sides:
                # no zero CROSSING to find: d_face is unsigned, so the sheet sides are the
                # 2-voxel-thick shell d_face <= 1 (pointwise, both sides of every face at once,
                # matching the 2-voxel tolerance of the upstream Dice)
                surf = ((sdf_box != 0) & (decode_sdf(sdf_box, clip) <= 1.0)
                        & (val_box >= 128))[1:-1, 1:-1, 1:-1]
            else:
                surf = extract_surface(sdf_box, val_box, clip)[1:-1, 1:-1, 1:-1]
            n_surf += int(surf.sum())
            n_data += int((sdf_box[1:-1, 1:-1, 1:-1] != 0).sum())
            arr[ci, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = surf.astype(np.uint8) * 255
            del sdf_box, val_box, surf
        # connectivity / thickness on a central sub-box (a full-region bool mask may not fit in RAM)
        sb = [min(int(s), int(b)) for s, b in zip(shape, stats_box)]
        slo = [(s - b) // 2 for s, b in zip(shape, sb)]
        sub = np.asarray(arr[ci, slo[0]:slo[0] + sb[0], slo[1]:slo[1] + sb[1], slo[2]:slo[2] + sb[2]]) > 127
        stats = surface_stats(sub, min_component)
        del sub
        res = {"n_voxels": n_surf, "n_data_voxels": n_data, "frac_of_data": (n_surf / n_data) if n_data else 0.0,
               "stats_box_lo": slo, "stats_box_shape": sb, "stats": stats}
        out[surf_name] = res
        if len(pairs) == 1:
            out.update(res)
        log(f"[infer] {surf_name}: {n_surf} voxels ({res['frac_of_data']:.4f} of data), central box "
            f"{sb}: deg6_mean={stats.get('deg6_mean', 0):.2f} thickness_median={stats.get('thickness', {}).get('median', 0)} "
            f"components_6={stats.get('components_6', {}).get('n_components', 0)}")
    if faces:
        ti, ii, io = ch.index("thickness"), ch.index("sdf_in"), ch.index("sdf_out")
        hist = np.zeros(256, np.int64)
        for lo, hi in iter_cores(shape, (brick,) * 3):
            a = np.asarray(arr[ii, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            b = np.asarray(arr[io, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            data = (a != 0) & (b != 0)
            th = np.clip(np.rint(decode_sdf(a, clip) - decode_sdf(b, clip)), 0, 255).astype(np.uint8)
            th[~data] = 0
            arr[ti, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = th
            hist += np.bincount(th.ravel(), minlength=256)[:256]
            del a, b, th, data
        # exact quantiles from the 256-bin histogram: never materialise one int64 per voxel
        med, p10, p90 = hist_percentiles(hist, (50, 10, 90))
        out["thickness"] = {"median": med, "p10": p10, "p90": p90,
                            "n_positive": int(hist[1:].sum())}
        log(f"[infer] thickness (voxels): {out['thickness']}")
    out["seconds"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- #
# previews
# --------------------------------------------------------------------------- #
def _preview_slices(pred_path: str, clip: float, max_dim: int = 2048) -> dict[str, np.ndarray]:
    arr = zarr.open_array(store=pred_path, mode="r")
    zm = int(arr.shape[1]) // 2
    sl = np.asarray(arr[:, zm])  # (C, Y, X)
    st = max(1, int(math.ceil(max(sl.shape[1:]) / max_dim)))
    sl = sl[:, ::st, ::st]
    ch = list(arr.attrs["channels"])
    g = lambda k: sl[ch.index(k)]  # noqa: E731
    n_rgb = np.stack([g("nx"), g("ny"), g("nz")], axis=-1)
    common = {"valid": g("valid"), "ink": g("ink"), "cos": g("cos"), "normal": n_rgb}
    if "sdf_in" in ch:
        rgb = np.repeat(g("sdf_in")[..., None], 3, axis=-1).copy()
        rgb[g("surface_in1") > 127] = (255, 0, 0)
        rgb[g("surface_out1") > 127] = (0, 0, 255)
        return {"sdf_in": g("sdf_in"), "sdf_out": g("sdf_out"), "faces": rgb, "thickness": g("thickness"), **common}
    if "sdf_body" in ch:
        rgb = np.repeat(g("sdf_body")[..., None], 3, axis=-1).copy()
        rgb[g("surface_body1") > 127] = (255, 0, 0)
        return {"sdf_body": rgb, **common}
    if "d_face" in ch:
        # grey face distance, red sheet sides, a blue tint where the body head fires
        rgb = np.repeat(g("d_face")[..., None], 3, axis=-1).astype(np.uint16)
        bod = g("body") > 127
        rgb[bod, 2] = np.minimum(255, rgb[bod, 2] + 80)
        rgb = rgb.astype(np.uint8)
        rgb[g("surface_side1") > 127] = (255, 0, 0)
        return {"d_face": rgb, "body": g("body"), **common}
    sdf_rgb = np.repeat(g("sdf")[..., None], 3, axis=-1).copy()
    sdf_rgb[g("surface1") > 127] = (255, 0, 0)
    return {"sdf": sdf_rgb, **common}


def write_previews(pred_path: str, out_dir: str, clip: float) -> list[str]:
    try:
        from PIL import Image
    except ImportError:
        print("[infer] preview skipped: PIL not installed", flush=True)
        return []
    paths = []
    panels = _preview_slices(pred_path, clip)
    for name, img in panels.items():
        p = os.path.join(out_dir, f"preview_{name}.png")
        Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(p)
        paths.append(p)
    rgb = [im if im.ndim == 3 else np.repeat(im[..., None], 3, axis=-1) for im in panels.values()]
    h = max(a.shape[0] for a in rgb)
    w = sum(a.shape[1] for a in rgb) + 4 * (len(rgb) - 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    xx = 0
    for a in rgb:
        canvas[: a.shape[0], xx: xx + a.shape[1]] = a
        xx += a.shape[1] + 4
    p = os.path.join(out_dir, "preview_side_by_side.png")
    Image.fromarray(canvas).save(p)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# tsm infer
# --------------------------------------------------------------------------- #
class _HeadView:
    """The sliding engine sees the 11 head channels only; ``surface1`` is filled afterwards."""

    def __init__(self, writer: BrickWriter, n: int) -> None:
        self.w = writer
        self.channels = writer.channels[:n]

    def write(self, c: int, z0: int, y0: int, x0: int, a: np.ndarray) -> None:
        self.w.write(c, z0, y0, x0, a)

    def has_brick(self, z0: int, y0: int, x0: int) -> bool:
        done = set(self.w._done.get(self.w._key(z0, y0, x0), []))
        return set(range(len(self.channels))) <= done


def run_infer(cfg: RunCfg, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    from tsm.cli import open_ct
    from tsm.sliding import WindowSpec, plan_bytes, run_sliding

    opts = _opts(cfg, "infer", INFER_DEFAULTS)
    device = str(opts["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_guard(cfg.budget)
    out_dir = os.path.join(cfg.out_dir, "student")
    pred_path = os.path.join(out_dir, "pred.zarr")
    ckpt = opts["checkpoint"] or os.path.join(cfg.out_dir, "train", "latest.pt")
    clip = float(opts["clip"])
    region = cfg.region
    voxel_um = float(cfg.volume.voxel_um)
    tta = tta_transforms(opts["tta"])  # validated here; applied inside StudentNet (never sliding's flip-average)
    tta_name = {1: "none", 8: "flip8", 16: "flip8_rot4"}[len(tta)]
    smode = _checkpoint_surface_mode(ckpt, opts.get("surface_mode"))
    fiber = _checkpoint_fiber(ckpt)
    fmode = _checkpoint_fiber_mode(ckpt)
    channels, nhead = pred_channels(smode, fiber, fmode), n_head_ch(smode, fiber, fmode)
    if fiber:
        print(f"[infer] fibre head present (fiber_mode={fmode}): writing "
              f"{FIBER_DIR_PRED_CHANNELS + FIBER_PRED_CHANNELS if fmode == 'direction' else FIBER_PRED_CHANNELS}",
              flush=True)
    backend = str(opts["backend"])
    if backend not in ("torch", "trt"):
        raise ValueError(f"extra.infer.backend must be 'torch' or 'trt', got {backend!r}")
    rf_halo = isinstance(opts["halo"], str) and str(opts["halo"]).lower() == "rf"
    rf_radius = student_rf_radius(ckpt, opts["widths"]) if rf_halo else None
    halo, step = opts["halo"], opts["step"]
    if rf_halo:
        halo, step = rf_tiling(int(opts["patch"]), int(rf_radius))
    weight = str(opts["weight"] or ("uniform" if rf_halo else "gaussian"))
    spec = WindowSpec(
        patch=int(opts["patch"]), step=step, out_tile=int(opts["out_tile"]), halo=halo,
        batch=int(opts["batch"]), tta=False,
        dtype=torch.float32 if backend == "trt" else torch.bfloat16, empty_max=int(opts["empty_max"]),
        norm_scope="window", empty_window_max=int(opts["empty_window_max"]), channels_last=bool(opts["channels_last"]),
        cudnn_benchmark=bool(opts["cudnn_benchmark"]), compile=opts["compile"], prefetch=bool(opts["prefetch"]),
        backend=backend, weight=weight,
    )
    if rf_halo:
        print(f"[infer] receptive-field halo: rf radius {rf_radius} -> halo {spec.halo}, step {spec.step} "
              f"(patch {spec.patch}), uniform core blending, "
              f"{(spec.patch / spec.step) ** 3:.2f} windows per core volume", flush=True)
    rows, tiles = plan_bytes(spec, region, nhead, "none")
    estimate_and_assert(rows, cfg.budget)
    n_win = sum(tp.n_windows(spec.patch, spec.step) for tp in tiles)
    print(f"[infer] checkpoint={ckpt} exists={os.path.exists(ckpt)} device={device} region start={region.start_zyx} "
          f"size={region.size_zyx} pitch={voxel_um} um clip={clip} tta={tta_name} ({len(tta)} passes) "
          f"surface_mode={smode} -> {pred_path}", flush=True)
    print(f"[infer] {len(tiles)} tiles of {spec.out_tile} (halo {spec.halo}), {n_win} windows of {spec.patch}^3, "
          f"channels={channels}", flush=True)
    if dry_run:
        return {"tiles": len(tiles), "windows": n_win, "checkpoint": ckpt, "pred": pred_path}
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"student checkpoint {ckpt} not found (train first or set extra.infer.checkpoint)")
    if force and os.path.exists(pred_path):
        shutil.rmtree(pred_path)
    os.makedirs(out_dir, exist_ok=True)

    t = time.perf_counter()
    model, info = load_student(ckpt, device, opts["widths"])
    body: Any = model
    if backend == "trt":
        if device != "cuda":
            raise RuntimeError("extra.infer.backend 'trt' needs a CUDA device")
        from tsm.trt import TRTStudent

        body = TRTStudent.build_or_load(
            model, patch=spec.patch, batch=spec.batch, precision=str(opts["trt_precision"]),
            device=device, workspace_gb=float(opts["trt_workspace_gb"]),
        )
        info["trt"] = {"engine": body.engine_path, "precision": body.precision,
                       "build_seconds": round(body.build_seconds, 1), "device_memory_mb": body.device_memory_mb}
        print(f"[infer] TensorRT engine {os.path.basename(body.engine_path)} "
              f"({body.precision}, in {body.in_shape} -> out {body.out_shape})", flush=True)
    net = StudentNet(body, voxel_um, clip, tta=tta_name, surface_mode=info.get("surface_mode", smode),
                     input_radial=bool(info.get("input_radial", False)),
                     axis=opts.get("axis_path") or info.get("axis_path"),
                     input_axis=bool(info.get("input_axis", False)),
                     axis_tangent=bool(info.get("axis_tangent", False)),
                     fiber_mode=str(info.get("fiber_mode", fmode))).to(device)
    print(f"[infer] loaded {info} in {time.perf_counter() - t:.1f}s, params={model.num_params() / 1e6:.2f}M", flush=True)
    reader = open_ct(cfg, 0, int(opts["probe_brick"]))
    writer = BrickWriter(pred_path, channels, region.size_zyx, chunk=int(opts["chunk"]),
                         origin_zyx=region.start_zyx, voxel_um=voxel_um, scale=1.0)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        summary = run_sliding(reader, net, normalize_ct, spec, region, _HeadView(writer, nhead), nhead, "none",
                              cfg.budget, device=device, log=lambda m: print(m.replace("[sliding]", "[infer]"), flush=True))
    finally:
        del net, model, body
        free_cuda()
    surf = write_surface_channel(pred_path, clip, brick=int(opts["chunk"]), stats_box=opts["stats_box"],
                                 min_component=int(opts["min_component"]))
    fiber_derived = None
    if fiber and fmode == "direction":
        fiber_derived = write_fiber_class_channels(
            pred_path, clip, brick=int(opts["chunk"]),
            # the real axis tangent iff the student was trained against it
            axis=((opts.get("axis_path") or info.get("axis_path") or DEFAULT_AXIS)
                  if info.get("axis_tangent") else None),
            origin_zyx=region.start_zyx)
    summary.update({
        "pred": pred_path, "channels": channels, "clip": clip, "voxel_um": voxel_um, "tta": tta_name,
        "surface_mode": smode,
        "region_start_zyx": list(region.start_zyx), "region_size_zyx": list(region.size_zyx),
        "volume_shape_zyx": list(reader.shape), "source": reader.url, "student": info, "surface1": surf,
        "fiber_mode": fmode if fiber else None,
        **({"fiber_derived": fiber_derived} if fiber_derived else {}),
        "encodings": {
            "sdf": f"round(128 + clip(sdf, +-{clip:g})*127/{clip:g}); 0 = no data",
            "valid|ink|conf|spare|fiber_vt|fiber_hz|fiber_strength": "sigmoid * 255",
            "sin|cos|nx|ny|nz|fiber_dz|fiber_dy|fiber_dx": "127.5 + 127.5 v",
            "density": "relu(v) * 1000, wraps per voxel at voxel_um",
            "surface1|surface_in1|surface_out1|surface_body1": "255 on the 1-voxel surface",
            "d_face": (f"round(128 + clip(d_face, 0, {clip:g})*127/{clip:g}); the UNSIGNED distance "
                       f"to the nearest sheet face, bytes >= 128, 0 = no data (sides mode)"),
            "body": "sigmoid * 255: P(this voxel is papyrus body) (sides mode)",
            "surface_side1": "255 where d_face <= 1 and valid >= 0.5 (sides mode)",
            "thickness": "round(clip(sdf_in - sdf_out, 0, 255)) voxels (two-face mode)",
        },
        "peak_rss_mb": peak_rss_mb(),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1 << 20)) if device == "cuda" else 0.0,
    })
    if bool(opts["previews"]):
        try:
            summary["previews"] = write_previews(pred_path, out_dir, clip)
        except Exception as exc:  # best effort
            print(f"[infer] preview failed: {type(exc).__name__}: {exc}", flush=True)
    with open(os.path.join(out_dir, "pred.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[infer] done: {summary['done']} tiles, {summary['windows']} windows in {summary['seconds']:.1f}s, "
          f"peak RSS {summary['peak_rss_mb']:.0f} MiB, peak VRAM {summary['peak_vram_mb']:.0f} MiB", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# export: shared pooling
# --------------------------------------------------------------------------- #
def _open_pred(path: str) -> tuple[zarr.Array, dict[str, Any]]:
    arr = zarr.open_array(store=path, mode="r")
    attrs = dict(arr.attrs)
    attrs.setdefault("channels", PRED_CHANNELS)
    attrs["origin_zyx"] = [int(v) for v in attrs.get("origin_zyx", (0, 0, 0))]
    attrs["voxel_um"] = float(attrs.get("voxel_um", BASE_UM))
    attrs["shape_zyx"] = [int(s) for s in arr.shape[1:]]
    return arr, attrs


def _pred_clip(path: str, clip: float | None) -> float:
    if clip is not None:
        return float(clip)
    sp = os.path.join(path, "..", "pred.summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            return float(json.load(fh).get("clip", CLIP))
    return float(CLIP)


def _pred_summary(path: str) -> dict[str, Any]:
    sp = os.path.join(path, "..", "pred.summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            return json.load(fh)
    return {}


def _read_zero(arr: zarr.Array, lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
    """(C, ...) box in local store coordinates, zero (= no data) outside the store."""
    shape = [int(s) for s in arr.shape[1:]]
    out = np.zeros((int(arr.shape[0]),) + tuple(h - l for l, h in zip(lo, hi)), dtype=np.uint8)
    a = [max(0, int(l)) for l in lo]
    b = [min(s, int(h)) for s, h in zip(shape, hi)]
    if all(bb > aa for aa, bb in zip(a, b)):
        blk = np.asarray(arr[:, a[0]:b[0], a[1]:b[1], a[2]:b[2]])
        d = [aa - int(l) for aa, l in zip(a, lo)]
        out[:, d[0]:d[0] + blk.shape[1], d[1]:d[1] + blk.shape[2], d[2]:d[2] + blk.shape[3]] = blk
    return out


def pool_mean(vals: np.ndarray, mask: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask-aware block mean by ``f`` on the last three axes: (mean (C, Z/f, Y/f, X/f), cover (Z/f, ...)).

    Only ``mask`` voxels contribute; cells with no contributing voxel get 0 and cover 0."""
    f = int(f)
    C = vals.shape[0]
    Z, Y, X = vals.shape[1:]
    if Z % f or Y % f or X % f:
        raise ValueError(f"block {vals.shape[1:]} not a multiple of {f}")
    m = mask.astype(np.float32).reshape(Z // f, f, Y // f, f, X // f, f)
    cnt = m.sum(axis=(1, 3, 5))
    v = (vals.astype(np.float32) * mask.astype(np.float32)).reshape(C, Z // f, f, Y // f, f, X // f, f)
    s = v.sum(axis=(2, 4, 6))
    mean = s / np.maximum(cnt, 1.0)[None]
    return mean, cnt / float(f ** 3)


def _renorm(v: np.ndarray, axis: int = 0, eps: float = 1e-6) -> np.ndarray:
    n = np.sqrt((v * v).sum(axis=axis, keepdims=True))
    return np.where(n > eps, v / np.maximum(n, eps), 0.0).astype(np.float32)


def _iter_fine_bricks(arr: zarr.Array, attrs: dict[str, Any], f: int, brick: Sequence[int]) -> Iterator[tuple[tuple[int, int, int], np.ndarray]]:
    """Yield (global fine origin, (C, bz, by, bx) bytes) bricks aligned to ``f`` in global fine coordinates,
    covering the store; voxels outside the store are 0 (no data)."""
    origin = attrs["origin_zyx"]
    shape = attrs["shape_zyx"]
    g0 = [(o // f) * f for o in origin]
    g1 = [int(math.ceil((o + s) / f)) * f for o, s in zip(origin, shape)]
    b = [max(f, (int(v) // f) * f) for v in brick]
    for z in range(g0[0], g1[0], b[0]):
        for y in range(g0[1], g1[1], b[1]):
            for x in range(g0[2], g1[2], b[2]):
                lo = (z, y, x)
                hi = (min(z + b[0], g1[0]), min(y + b[1], g1[1]), min(x + b[2], g1[2]))
                blk = _read_zero(arr, [l - o for l, o in zip(lo, origin)], [h - o for h, o in zip(hi, origin)])
                yield lo, blk


def _valid_mask(dec: dict[str, np.ndarray]) -> np.ndarray:
    return dec["data"] & (dec["valid"] > 0.5)


def _pool_winding(dec: dict[str, np.ndarray], mask: np.ndarray, f: int) -> dict[str, np.ndarray]:
    """Pooled winding field at factor f: sin/cos and normal renormalised, density in wraps per pooled voxel."""
    vals = np.stack([dec["sin"], dec["cos"], dec["density"], dec["nx"], dec["ny"], dec["nz"], dec["conf"], dec["sdf"]])
    m, cover = pool_mean(vals, mask, f)
    sc = _renorm(m[0:2])
    n = _renorm(m[3:6])
    return {"sin": sc[0], "cos": sc[1], "density": m[2] * f, "nx": n[0], "ny": n[1], "nz": n[2], "conf": m[6],
            "sdf": m[7], "cover": cover}


# --------------------------------------------------------------------------- #
# export: lasagna (.lasagna.json + per-channel OME-Zarr, villa preprocess_cos_omezarr layout)
# --------------------------------------------------------------------------- #
def encode_pred_dt(sdf: np.ndarray, data: np.ndarray, half: float) -> np.ndarray:
    """villa pred_dt bytes from the signed distance to the sheet's medial surface (fine voxels).

    "inside" = |sdf| <= half (the sheet band): 127 + clip(round(half - |sdf| + 1), 1, 48) in [128, 175];
    outside: 128 - clip(round(|sdf| - half), 1, 48) in [80, 127]; no data = 0.
    The jump across the band edge is exactly 1 (127 -> 128), like villa's EDT encoding."""
    a = np.abs(np.asarray(sdf, dtype=np.float32))
    inside = a <= float(half)
    inner = 127 + np.clip(np.rint(float(half) - a + 1.0), 1, PRED_DT_MAX)
    outer = 128 - np.clip(np.rint(a - float(half)), 1, PRED_DT_MAX)
    out = np.where(inside, inner, outer).astype(np.uint8)
    out[~np.asarray(data, dtype=bool)] = 0
    return out


def hemisphere_nxny(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """villa's nx/ny bytes: flip the normal so nz >= 0, byte = round(v*127 + 128); nz = sqrt(1 - nx^2 - ny^2)."""
    flip = np.where(nz < 0, -1.0, 1.0).astype(np.float32)
    enc = lambda v: np.clip(np.rint(v * flip * 127.0 + 128.0), 0, 255).astype(np.uint8)  # noqa: E731
    return enc(nx), enc(ny)


def omezarr_level_shape(base_shape: Sequence[int], level: int) -> tuple[int, int, int]:
    """villa: shape at a pyramid level = repeated ceil-halving of the base shape."""
    z, y, x = (int(v) for v in base_shape)
    for _ in range(max(0, int(level))):
        z, y, x = max(1, (z + 1) // 2), max(1, (y + 1) // 2), max(1, (x + 1) // 2)
    return z, y, x


def _create_omezarr(path: str, base_shape: Sequence[int], first_level: int, n_levels: int, chunk: int, name: str) -> zarr.Group:
    """OME-Zarr (v2, '/' chunk keys) group with 3D uint8 levels first_level..n_levels-1, like villa's ``_create_omezarr``."""
    if os.path.exists(path):
        shutil.rmtree(path)
    g = zarr.open_group(path, mode="w", zarr_format=2)
    datasets = []
    for lv in range(first_level, n_levels):
        sh = omezarr_level_shape(base_shape, lv)
        g.create_array(str(lv), shape=sh, chunks=tuple(min(s, chunk) for s in sh), dtype="uint8", fill_value=0,
                       chunk_key_encoding={"name": "v2", "separator": "/"})
        datasets.append({"path": str(lv), "coordinateTransformations": [{"type": "scale", "scale": [float(2 ** lv)] * 3}]})
    g.attrs["multiscales"] = [{
        "version": "0.4", "name": name,
        "axes": [{"name": a, "type": "space", "unit": "pixel"} for a in "zyx"],
        "datasets": datasets,
    }]
    g.attrs["lasagna_pyramid_downsample"] = "mean_pool2x"
    return g


def _pool2(a: np.ndarray) -> np.ndarray:
    """2x mean pool (ceil sizes, zero padded) of a 3D float array."""
    z, y, x = a.shape
    p = np.zeros(((z + 1) // 2 * 2, (y + 1) // 2 * 2, (x + 1) // 2 * 2), dtype=np.float32)
    c = np.zeros_like(p)
    p[:z, :y, :x] = a
    c[:z, :y, :x] = 1.0
    s = p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2, p.shape[2] // 2, 2).sum(axis=(1, 3, 5))
    n = c.reshape(s.shape[0], 2, s.shape[1], 2, s.shape[2], 2).sum(axis=(1, 3, 5))
    return s / np.maximum(n, 1.0)


def _build_pyramid(groups: dict[str, zarr.Group], name: str, first_level: int, n_levels: int, lo: Sequence[int], hi: Sequence[int]) -> None:
    """Fill levels > first_level over the written box [lo, hi) (level ``first_level`` indices).
    Scalars: 2x mean of the bytes; normals (nx, ny together): decode, average, hemisphere, re-encode."""
    for lv in range(first_level, n_levels - 1):
        src = [groups[k][str(lv)] for k in ((name,) if name != "normal" else ("nx", "ny"))]
        dst = [groups[k][str(lv + 1)] for k in ((name,) if name != "normal" else ("nx", "ny"))]
        # pool on the global even grid: read [2*(lo//2), 2*ceil(hi/2)) clipped to the level shape
        lo = [(v // 2) * 2 for v in lo]
        hi = [min(int(s), ((v + 1) // 2) * 2) for v, s in zip(hi, src[0].shape)]
        blk = [np.asarray(s[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32) for s in src]
        if name == "normal":
            nx = (blk[0] - 128.0) / 127.0
            ny = (blk[1] - 128.0) / 127.0
            nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))
            n = _renorm(np.stack([_pool2(nx), _pool2(ny), _pool2(nz)]))
            out = list(hemisphere_nxny(n[0], n[1], n[2]))
        else:
            out = [np.clip(np.rint(_pool2(blk[0])), 0, 255).astype(np.uint8)]
        lo = [v // 2 for v in lo]
        hi = [(v + 1) // 2 for v in hi]
        for d, o in zip(dst, out):
            d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = o[: hi[0] - lo[0], : hi[1] - lo[1], : hi[2] - lo[2]]


def export_lasagna(
    pred_store: str,
    out_path: str,
    name: str = "tsm",
    level_shift: int = 2,
    cos_scaledown: int = 2,
    scaledown: int = 4,
    base_shape_zyx: Sequence[int] | None = None,
    axis: str | None = None,
    sheet_half_vox: float = 2.0,
    pred_dt_channel: str | None = None,
    min_cover: float = 0.5,
    ome_chunk: int = 32,
    n_levels: int = 5,
    clip: float | None = None,
    brick: Sequence[int] = (64, 512, 512),
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """pred.zarr -> ``<out_path>/<name>.lasagna.json`` + ``<name>_{cos,grad_mag,nx,ny,pred_dt}.ome.zarr``.

    Layout = villa ``preprocess_cos_omezarr.run_preprocess_3d`` run on a level-``level_shift``
    volume (input_sd = 2**level_shift): ``cos`` and ``pred_dt`` at OME level
    ``level_shift + log2(cos_scaledown)``, ``grad_mag``/``nx``/``ny`` at ``level_shift + log2(scaledown)``;
    each channel group is a zarr-v2 OME group whose level arrays span the whole base volume
    (``base_shape_zyx``, ceil-halved per level) and are filled only over the prediction region.

    Bytes: cos = round(255 * (0.5 + 0.5 cos)); grad_mag = round(1000 * density) with density in
    wraps per level-``level_shift`` voxel (``grad_mag_factor = 1 / 2**level_shift`` converts to base
    voxels, as villa's ``fit_data.load_3d`` does: ``grad_mag / (1000 / grad_mag_factor)``);
    nx, ny = ``hemisphere_nxny`` (nz >= 0, (u8 - 128) / 127); pred_dt = ``encode_pred_dt`` of
    ``pred_dt_channel`` (default ``sdf_in`` for a two-face store, ``sdf_body`` for a body store,
    else ``sdf``)
    (fine voxels) mean-pooled to the cos level like villa's INTER_AREA.  No data (cover <
    ``min_cover``): cos/grad_mag/pred_dt 0, nx/ny 128.  Pooling is mask-aware over
    valid > 0.5 voxels; sin/cos and normals are renormalised after averaging.
    """
    log = log or (lambda m: print(m, flush=True))
    if not out_path.endswith(".lasagna.json"):
        manifest = os.path.join(out_path, f"{name}.lasagna.json")
    else:
        manifest = out_path
        name = os.path.basename(out_path)[: -len(".lasagna.json")]
    out_dir = os.path.dirname(os.path.abspath(manifest))
    if os.path.exists(manifest) and not force:
        raise FileExistsError(f"{manifest} exists; pass --force to overwrite")
    arr, attrs = _open_pred(pred_store)
    clip = _pred_clip(pred_store, clip)
    ch = list(attrs["channels"])
    dt_ch = str(pred_dt_channel or next(k for k in ("sdf_in", "sdf_body", "sdf", "d_face") if k in ch))
    if dt_ch not in ch:
        raise ValueError(f"pred_dt_channel {dt_ch!r} not in the prediction store ({ch})")
    origin, shape = attrs["origin_zyx"], attrs["shape_zyx"]
    f_in = 2 ** int(level_shift)
    f_cos, f_oth = f_in * int(cos_scaledown), f_in * int(scaledown)
    for f, what in ((f_cos, "cos"), (f_oth, "grad_mag")):
        if f & (f - 1):
            raise ValueError(f"{what} factor {f} is not a power of two")
    if f_oth % f_cos:
        raise ValueError("scaledown must be a multiple of cos_scaledown")
    cos_level, oth_level = int(round(math.log2(f_cos))), int(round(math.log2(f_oth)))
    n_levels = max(int(n_levels), oth_level + 2)
    if base_shape_zyx is None:
        base_shape_zyx = _pred_summary(pred_store).get("volume_shape_zyx") or [o + s for o, s in zip(origin, shape)]
    base_shape = tuple(int(v) for v in base_shape_zyx)
    if any(o + s > b for o, s, b in zip(origin, shape, base_shape)):
        raise ValueError(f"prediction region {origin}+{shape} exceeds base_shape_zyx {base_shape}")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{name}_"
    levels = {"cos": cos_level, "grad_mag": oth_level, "nx": oth_level, "ny": oth_level, "pred_dt": cos_level}
    groups = {k: _create_omezarr(os.path.join(out_dir, f"{prefix}{k}.ome.zarr"), base_shape, lv, n_levels, int(ome_chunk), k)
              for k, lv in levels.items()}
    lv_arr = {k: groups[k][str(lv)] for k, lv in levels.items()}
    log(f"[export] lasagna {manifest}: base_shape={base_shape} cos level {cos_level} (x{f_cos}) "
        f"other level {oth_level} (x{f_oth}) region {origin}+{shape}")

    brick = [max(int(v), f_oth) for v in brick]
    t0 = time.perf_counter()
    n_bricks = 0
    for g0, blk in _iter_fine_bricks(arr, attrs, f_oth, brick):
        dec = decode_pred(blk, clip, ch)
        mask = _valid_mask(dec)
        # cos level: cos (renormalised with sin) and pred_dt bytes
        vals = np.stack([dec["sin"], dec["cos"],
                         encode_pred_dt(pred_dt_field(dec, dt_ch), dec["data"], sheet_half_vox).astype(np.float32)])
        m, cover = pool_mean(vals, mask, f_cos)
        ok = cover >= float(min_cover)
        sc = _renorm(m[0:2])
        cos_u8 = np.where(ok, np.clip(np.rint(255.0 * (0.5 + 0.5 * sc[1])), 0, 255), 0).astype(np.uint8)
        dt_u8 = np.where(ok, np.clip(np.rint(m[2]), 0, 255), 0).astype(np.uint8)
        c0 = [g // f_cos for g in g0]
        for k, a in (("cos", cos_u8), ("pred_dt", dt_u8)):
            lv_arr[k][c0[0]:c0[0] + a.shape[0], c0[1]:c0[1] + a.shape[1], c0[2]:c0[2] + a.shape[2]] = a
        # other level: grad_mag, nx, ny
        pw = _pool_winding(dec, mask, f_oth)
        ok = pw["cover"] >= float(min_cover)
        gm = np.where(ok, encode_density(pw["density"] / f_oth * f_in), 0).astype(np.uint8)  # wraps per input voxel
        nx_u8, ny_u8 = hemisphere_nxny(pw["nx"], pw["ny"], pw["nz"])
        nx_u8 = np.where(ok, nx_u8, 128).astype(np.uint8)
        ny_u8 = np.where(ok, ny_u8, 128).astype(np.uint8)
        o0 = [g // f_oth for g in g0]
        for k, a in (("grad_mag", gm), ("nx", nx_u8), ("ny", ny_u8)):
            lv_arr[k][o0[0]:o0[0] + a.shape[0], o0[1]:o0[1] + a.shape[1], o0[2]:o0[2] + a.shape[2]] = a
        n_bricks += 1
        del blk, dec, mask, vals, m, pw
    log(f"[export] lasagna: {n_bricks} bricks pooled in {time.perf_counter() - t0:.1f}s; building pyramids")
    for k, f in (("cos", f_cos), ("pred_dt", f_cos), ("grad_mag", f_oth), ("normal", f_oth)):
        lo = [o // f for o in origin]
        hi = [int(math.ceil((o + s) / f)) for o, s in zip(origin, shape)]
        _build_pyramid(groups, k, levels["cos" if f == f_cos else "grad_mag"], n_levels, lo, hi)

    umb = ""
    if axis:
        umb = f"{prefix}umbilicus.json"
        shutil.copyfile(os.path.expanduser(axis), os.path.join(out_dir, umb))
    crop = [int(origin[2]), int(origin[1]), int(origin[0]), int(shape[2]), int(shape[1]), int(shape[0])]  # x,y,z,w,h,d
    doc: dict[str, Any] = {
        "version": 2,
        "source_to_base": 1.0,
        "grad_mag_encode_scale": GRAD_MAG_ENCODE_SCALE,
        "grad_mag_factor": 1.0 / f_in,
        "groups": {k: {"zarr": f"{prefix}{k}.ome.zarr/{lv}", "scaledown": lv, "channels": [k]} for k, lv in levels.items()},
        "crops": [crop],
        "base_shape_zyx": list(base_shape),
        # tsm extras (ignored by villa's LasagnaVolume.load)
        "preprocess_params": {
            "source": "tsm", "pred_store": os.path.abspath(pred_store), "input_sd": f_in, "level_shift": int(level_shift),
            "scaledown": int(scaledown), "cos_scaledown": int(cos_scaledown), "grad_mag_encode_scale": GRAD_MAG_ENCODE_SCALE,
            "channels": LASAGNA_CHANNELS, "crop_xyzwhd": crop, "base_voxel_um": attrs["voxel_um"],
            "sdf_clip_vox": clip, "sheet_half_vox": float(sheet_half_vox), "min_cover": float(min_cover),
            "pred_dt_channel": dt_ch,
            "pred_dt": "outside=[80,127] inside=[128,175] no-data=0, distances in base voxels, band |sdf|<=sheet_half_vox",
            "nx_ny": "(u8-128)/127, hemisphere nz>=0, nz=sqrt(1-nx^2-ny^2); axes x,y,z of the volume",
            "grad_mag": "round(1000 * wraps per input voxel); consumer: grad_mag/(1000/grad_mag_factor) = wraps per base voxel",
            "cos": "round(255*(0.5+0.5*cos(2 pi w)))",
        },
    }
    if umb:
        doc["umbilicus_json"] = umb
    tmp = manifest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, manifest)
    log(f"[export] wrote {manifest} in {time.perf_counter() - t0:.1f}s")
    return {"manifest": manifest, "groups": doc["groups"], "base_shape_zyx": list(base_shape), "crop_xyzwhd": crop,
            "pred_dt_channel": dt_ch, "bricks": n_bricks, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# export: spiral (winding field zarr + JSON + dense ray observations)
# --------------------------------------------------------------------------- #
def _sample_field(field: torch.Tensor, pts_zyx: torch.Tensor) -> torch.Tensor:
    """Trilinear sample of (C, Z, Y, X) at float voxel coordinates (N, 3) (zyx) -> (N, C); 0 outside."""
    Z, Y, X = field.shape[1:]
    denom = torch.tensor([max(X - 1, 1), max(Y - 1, 1), max(Z - 1, 1)], dtype=torch.float32)
    g = 2.0 * pts_zyx[:, [2, 1, 0]].float() / denom - 1.0
    out = F.grid_sample(field[None].float(), g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, :, 0, 0].T


def _unwrap_phase(angle: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-ray unwrap over the valid samples (invalid ones are bridged, then set to NaN); wraps, first valid = 0."""
    out = np.full(angle.shape, np.nan, dtype=np.float32)
    for r in range(angle.shape[0]):
        idx = np.flatnonzero(valid[r])
        if idx.size == 0:
            continue
        u = np.unwrap(angle[r, idx])
        out[r, idx] = (u - u[0]) / (2.0 * np.pi)
    return out


def _rays(store: zarr.Array, attrs: dict[str, Any], axis: np.ndarray, f: int, spacing: int, angle_deg: float,
          length: float | None, step: float, min_cover: float) -> dict[str, np.ndarray]:
    """Outward radial rays from the axis at every ``spacing``-th z of the level-``f`` store.

    Coordinates are level-f voxel indices (global, i.e. store origin included); level-f voxel i
    sits at level-0 coordinate f*i + (f-1)/2.  Samples are trilinear (``grid_sample``) of the
    float field; a sample is valid when it is inside the store and the pooled ``valid`` >= 0.5.
    """
    ch = list(attrs["channels"])
    origin = attrs["origin_zyx"]
    shape = attrs["shape_zyx"]
    angles = np.deg2rad(np.arange(0.0, 360.0, float(angle_deg)))
    if length is None:
        length = float(math.hypot(shape[1], shape[2]))
    n = int(math.floor(float(length) / float(step))) + 1
    ks = np.arange(n, dtype=np.float32) * float(step)
    starts, dirs, phase, dens, conf, valid = [], [], [], [], [], []
    for zl in range(0, shape[0], int(spacing)):
        z0 = (origin[0] + zl) * f + (f - 1) / 2.0  # level-0 z of this slice
        ay, ax = axis_yx_at(axis, np.array([z0]))
        cy, cx = (float(ay[0]) - (f - 1) / 2.0) / f, (float(ax[0]) - (f - 1) / 2.0) / f  # global level-f
        zs = slice(max(0, zl - 1), min(shape[0], zl + 2))
        blk = np.asarray(store[:, zs])
        fld = torch.from_numpy(np.stack([
            decode_signed(blk[ch.index("sin")]), decode_signed(blk[ch.index("cos")]), decode_density(blk[ch.index("density")]),
            decode_prob(blk[ch.index("conf")]), (blk[ch.index("valid")] > 0).astype(np.float32),
        ]))
        for th in angles:
            d = np.array([0.0, math.sin(th), math.cos(th)], dtype=np.float32)
            p = np.array([origin[0] + zl, cy, cx], dtype=np.float32)[None] + ks[:, None] * d[None]
            local = p - np.array([origin[0] + zs.start, origin[1], origin[2]], dtype=np.float32)
            inside = (local[:, 1] >= 0) & (local[:, 1] <= shape[1] - 1) & (local[:, 2] >= 0) & (local[:, 2] <= shape[2] - 1)
            s = _sample_field(fld, torch.from_numpy(local)).numpy()
            ok = inside & (s[:, 4] >= float(min_cover))
            starts.append(p[0])
            dirs.append(d)
            phase.append(np.arctan2(s[:, 0], s[:, 1]).astype(np.float32))
            dens.append(s[:, 2].astype(np.float32))
            conf.append(s[:, 3].astype(np.float32))
            valid.append(ok)
    starts_a = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
    dirs_a = np.asarray(dirs, dtype=np.float32).reshape(-1, 3)
    valid_a = np.asarray(valid, dtype=bool).reshape(-1, n)
    angle_a = np.asarray(phase, dtype=np.float32).reshape(-1, n)
    points = starts_a[:, None, :] + ks[None, :, None] * dirs_a[:, None, :]
    return {
        "starts_zyx": starts_a, "centers_zyx": starts_a + (0.5 * (n - 1) * float(step)) * dirs_a, "directions_zyx": dirs_a,
        "points_zyx": points.astype(np.float32), "spacing": np.float32(step),
        "phase": _unwrap_phase(angle_a, valid_a), "phase_angle": angle_a,
        "density": np.asarray(dens, dtype=np.float32).reshape(-1, n), "conf": np.asarray(conf, dtype=np.float32).reshape(-1, n),
        "valid": valid_a,
    }


def export_spiral(
    pred_store: str,
    out_path: str,
    level_shift: int = 2,
    axis: str | None = None,
    ray_spacing: int = 8,
    ray_angle_deg: float = 5.0,
    ray_length: float | None = None,
    ray_step: float = 1.0,
    min_cover: float = 0.5,
    clip: float | None = None,
    brick: Sequence[int] = (64, 512, 512),
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """pred.zarr -> ``<out_path>/winding.zarr`` (level-``level_shift`` winding field, label-store
    encodings), ``winding.json`` (encodings, origin, pitch, axis) and, with an axis, ``rays.npz``.

    winding.zarr: (8, Z, Y, X) uint8 channels sin, cos (127.5 + 127.5 v), density (round(1000 *
    wraps per level-``level_shift`` voxel)), nx, ny, nz (127.5 + 127.5 v, outward-oriented),
    conf (p*255), valid (1 where >= ``min_cover`` of the fine voxels were valid, else 0); attrs
    ``channels``, ``voxel_um``, ``origin_zyx`` (level-``level_shift`` voxels), ``scale``.

    rays.npz mirrors what ``neural_winding_losses`` caches per straight ray: ``centers_zyx``
    (ray midpoint), ``directions_zyx`` (unit, outward radial, in-plane), ``spacing`` (sample
    step), ``points_zyx`` [rays, samples, 3], ``phase`` [rays, samples] (unwrapped wraps, first
    valid sample = 0, NaN where invalid), ``phase_angle`` (raw atan2), ``density``, ``conf``,
    ``valid``, plus ``starts_zyx``; all coordinates in level-``level_shift`` voxels (see
    ``winding.json`` for the level-0 mapping).
    """
    log = log or (lambda m: print(m, flush=True))
    os.makedirs(out_path, exist_ok=True)
    store_path = os.path.join(out_path, "winding.zarr")
    if os.path.exists(store_path):
        if not force:
            raise FileExistsError(f"{store_path} exists; pass --force to overwrite")
        shutil.rmtree(store_path)
    arr, attrs = _open_pred(pred_store)
    clip = _pred_clip(pred_store, clip)
    ch = list(attrs["channels"])
    origin, shape = attrs["origin_zyx"], attrs["shape_zyx"]
    f = 2 ** int(level_shift)
    c_lo = [o // f for o in origin]
    c_hi = [int(math.ceil((o + s) / f)) for o, s in zip(origin, shape)]
    c_shape = [h - l for l, h in zip(c_lo, c_hi)]
    voxel_um = attrs["voxel_um"] * f
    writer = BrickWriter(store_path, SPIRAL_CHANNELS, c_shape, chunk=max(1, 128 // f), origin_zyx=c_lo, voxel_um=voxel_um, scale=float(f))
    log(f"[export] spiral {store_path}: level shift {level_shift} (x{f}) origin {c_lo} shape {c_shape} pitch {voxel_um:g} um")
    t0 = time.perf_counter()
    n_bricks = 0
    for g0, blk in _iter_fine_bricks(arr, attrs, f, [max(int(v), f) for v in brick]):
        dec = decode_pred(blk, clip, ch)
        pw = _pool_winding(dec, _valid_mask(dec), f)
        ok = pw["cover"] >= float(min_cover)
        enc = {
            "sin": encode_signed(pw["sin"]), "cos": encode_signed(pw["cos"]), "density": encode_density(pw["density"]),
            "nx": encode_signed(pw["nx"]), "ny": encode_signed(pw["ny"]), "nz": encode_signed(pw["nz"]),
            "conf": encode_prob(pw["conf"]), "valid": ok.astype(np.uint8),
        }
        c0 = [g // f for g in g0]
        for ci, k in enumerate(SPIRAL_CHANNELS):
            a = enc[k] if k == "valid" else np.where(ok, enc[k], 0).astype(np.uint8)
            writer.write(ci, c0[0], c0[1], c0[2], np.ascontiguousarray(a))
        n_bricks += 1
        del blk, dec, pw
    log(f"[export] spiral: {n_bricks} bricks pooled in {time.perf_counter() - t0:.1f}s")

    rays_path = None
    n_rays = 0
    if axis:
        ax = load_axis(axis)
        rays = _rays(zarr.open_array(store=store_path, mode="r"), {"channels": SPIRAL_CHANNELS, "origin_zyx": c_lo, "shape_zyx": c_shape},
                     ax, f, int(ray_spacing), float(ray_angle_deg), ray_length, float(ray_step), float(min_cover))
        rays_path = os.path.join(out_path, "rays.npz")
        np.savez_compressed(rays_path, **rays)
        n_rays = int(rays["centers_zyx"].shape[0])
        log(f"[export] spiral: {n_rays} rays x {rays['phase'].shape[1]} samples -> {rays_path} "
            f"({float(rays['valid'].mean()):.2f} valid)")
    doc = {
        "source": "tsm", "pred_store": os.path.abspath(pred_store), "winding_zarr": "winding.zarr", "level_shift": int(level_shift),
        "factor": f, "voxel_um": voxel_um, "origin_zyx": c_lo, "shape_zyx": c_shape,
        "origin_level0_zyx": [int(v) * f for v in c_lo], "coordinate_note": f"level-{level_shift} voxel i is at level-0 coordinate {f}*i + {(f - 1) / 2}",
        "channels": SPIRAL_CHANNELS,
        "encodings": {
            "sin|cos": "127.5 + 127.5 v; phase w = atan2(sin, cos) / 2pi, increasing outward",
            "density": f"round(1000 * wraps per level-{level_shift} voxel)",
            "nx|ny|nz": "127.5 + 127.5 v; unit sheet normal oriented outward from the axis, axes x,y,z of the volume",
            "conf": "p * 255", "valid": "1 where >= min_cover of the fine voxels were valid, else 0 (all other channels 0 there)",
        },
        "axis_json": os.path.abspath(os.path.expanduser(axis)) if axis else None,
        "rays": None if rays_path is None else {
            "file": "rays.npz", "n_rays": n_rays, "ray_spacing": int(ray_spacing), "ray_angle_deg": float(ray_angle_deg),
            "ray_length": ray_length, "ray_step": float(ray_step),
            "fields": "starts_zyx/centers_zyx/directions_zyx [rays,3], points_zyx [rays,samples,3], phase (wraps, unwrapped, "
                      "NaN invalid), phase_angle (rad), density, conf, valid [rays,samples]; level-shift voxel coordinates",
        },
        "min_cover": float(min_cover), "sdf_clip_vox": clip,
    }
    with open(os.path.join(out_path, "winding.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
    return {"winding_zarr": store_path, "json": os.path.join(out_path, "winding.json"), "rays": rays_path, "n_rays": n_rays,
            "shape_zyx": c_shape, "origin_zyx": c_lo, "bricks": n_bricks, "seconds": time.perf_counter() - t0}


def run_export(cfg: RunCfg, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    opts = _opts(cfg, "export", EXPORT_DEFAULTS)
    pred = opts["pred"] or os.path.join(cfg.out_dir, "student", "pred.zarr")
    out_dir = opts["out_dir"] or os.path.join(cfg.out_dir, "export")
    axis = opts["axis"]
    if axis and not os.path.exists(os.path.expanduser(axis)):
        print(f"[export] axis {axis} not found: no umbilicus_json / rays", flush=True)
        axis = None
    print(f"[export] pred={pred} exists={os.path.exists(os.path.join(pred, 'zarr.json'))} out_dir={out_dir} axis={axis}", flush=True)
    if dry_run:
        return {"pred": pred, "out_dir": out_dir}
    out: dict[str, Any] = {}
    if opts["lasagna"]:
        L = opts["lasagna"]
        out["lasagna"] = export_lasagna(
            pred, os.path.join(out_dir, "lasagna"), name=str(L["name"]), level_shift=int(L["level_shift"]),
            cos_scaledown=int(L["cos_scaledown"]), scaledown=int(L["scaledown"]), base_shape_zyx=L["base_shape_zyx"], axis=axis,
            sheet_half_vox=float(L["sheet_half_vox"]), pred_dt_channel=L.get("pred_dt_channel"),
            min_cover=float(L["min_cover"]), ome_chunk=int(L["ome_chunk"]),
            n_levels=int(L["n_levels"]), clip=opts["clip"], brick=opts["brick"], force=force,
        )
    if opts["spiral"]:
        S = opts["spiral"]
        out["spiral"] = export_spiral(
            pred, os.path.join(out_dir, "spiral"), level_shift=int(S["level_shift"]), axis=axis, ray_spacing=int(S["ray_spacing"]),
            ray_angle_deg=float(S["ray_angle_deg"]), ray_length=S["ray_length"], ray_step=float(S["ray_step"]),
            min_cover=float(S["min_cover"]), clip=opts["clip"], brick=opts["brick"], force=force,
        )
    with open(os.path.join(out_dir, "export.summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out
