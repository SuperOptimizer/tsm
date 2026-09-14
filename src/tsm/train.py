"""Student training: losses, loop, EMA, checkpoints. ``tsm train config.json``.

Training is at the fine (2.4 um) pitch only; winding targets are upsampled from
the coarse store by the dataset.  Losses (all masked by the target valid masks):
  surface: gaussian-weighted L1 on sdf (w = 1 + 8 exp(-sdf^2 / (2 * 4^2)), valid==1)
           + BCE(valid logit, valid==1) on valid!=2
           + surface-band soft Dice between exp(-|sdf|/2) maps (valid==1)
           ``extra.train.surface_aux`` (all weights 0 by default = unchanged loss) adds
           the zero-set terms ``shell`` / ``crest`` / ``far`` per face (``surface_aux_terms``),
           logged as ``surface/shell_in`` etc., and the two *body* topology terms ``gap``
           (anti-merge: no predicted body inside a label air gap) and ``cldice`` (soft clDice on
           the sheet body) which need both faces at once (``surface_body_terms``), logged as
           ``surface/gap`` / ``surface/cldice``.
           ``extra.train.surface_mode = "faces"`` instead trains a 3-channel head
           [sdf_in, sdf_out, valid] against the two-face labels (``faces_loss``):
           the same L1 + band Dice per face plus the one valid BCE.
           ``extra.train.surface_mode = "body"`` trains the *orientation-free* 2-channel head
           [sdf_body, valid] against ``data.body_sdf`` = min(sdf_in, -sdf_out) (``body_loss``):
           the medial terms on the single channel, plus the same optional aux terms with the
           body as its own "face" (``shell_body`` / ``crest_body`` / ``far_body``) and the
           single-target form of ``gap`` / ``cldice``.
           ``extra.train.surface_mode = "sides"`` trains the *orientation-free* 3-channel head
           [d_face, body logit, valid] (``sides_loss``): the magnitude and the sign of the body
           SDF split apart, ``data.face_dist`` = min(|sdf_in|, |sdf_out|) regressed (L1 + band
           Dice) and ``data.body_mask_np`` = (sdf_in>0)&(sdf_out<0) classified (BCE + soft
           Dice), so neither part hedges near zero the way the signed body SDF does.  Default
           "medial" (the single SDF to the recto medial surface) is unchanged.
  ink:     BCE (``pos_weight`` from ``extra.train.ink_pos_weight``: "auto" = this batch's
           sum(1-p)/sum(p) over valid==1, clamped to [1, 50]; a number; or null/0 = off)
           + soft Dice on the soft teacher probability (valid==1).  ``loss_weights.ink``
           defaults to 2.0 -- ink is a small positive class.
  winding: 1 - cos(dphi) on the normalized (sin, cos), Huber on 10*density,
           1 - n_pred.n_tgt (oriented normals), BCE(conf logit, conf > 0.5);
           all on valid==1, phase/density/normal weighted by the target conf.
Deep supervision at [full, half] resolution with weights [1.0, 0.5]; continuous
targets are mask-aware average pooled, masks nearest-subsampled.

The heads share one total loss under one global gradient clip, so the head with the
largest loss (or gradient) scale owns the step.  ``extra.train.balance`` (default
null = off, byte-identical) turns on :class:`LossBalancer`: ``"scale"`` divides each
head by an EMA of its own magnitude (frozen after ``balance_warmup`` steps unless
``balance_freeze`` is false) and rescales so the total keeps its magnitude;
``"gradnorm_lite"`` measures the per-head gradient norm at the shared trunk every
``balance_every`` steps and equalises them up to ``loss_weights``.  Multipliers and
normalised contributions are logged as ``balance/<head>``; per-head gradient norms on
the head parameters are logged as ``gradnorm/<head>`` every ``grad_log_every`` steps
in every mode.

Input channels (``extra.train.input_axis``, default **true**, adds ``(a_z, a_y, a_x)`` -- the
scroll-axis direction in volume coordinates, moved as a vector by every spatial transform).
Input channels (``extra.train.input_radial``, default true): ``[ct z-score, scale const,
r_z, r_y, r_x]`` -- the last three are the outward unit radial direction from the scroll
axis (axis from ``extra.train.axis_path``).  The sdf sign, the
winding phase sign and the normal orientation are all defined as "away from the axis",
which a 128^3 crop cannot infer, so the direction is given as input; it is augmented
exactly like the winding normal target.  With ``extra.train.axis_tangent`` (default true)
the two are the orthonormal (radial, local umbilicus tangent) pair of
``labels.geometry_frame`` instead of a constant (1, 0, 0) axis and an in-plane radial.
``input_radial`` / ``input_axis`` / ``axis_tangent`` / ``in_ch`` / ``axis_path`` go
into the checkpoint config so ``tsm infer`` rebuilds the same input.

Augmentation: ``extra.train.augment`` = True ("strong"), a preset name, a dict (``data.AugmentConfig``),
"v1" (the CPU flips/rot90 + jitter path) or False.  v2 runs on the device per batch after the
DataLoader (``data.Augment``); the number of samples each transform touched per optimizer step
is logged as ``aug/<name>`` in log.jsonl.  The one exception is the ``volcomp`` family (preset
"strong_volcomp"), which round-trips the raw uint8 crop through the lossy CT-mirror codec on the
CPU in the dataset worker -- its ``aug/volcomp`` count rides along with the batch.
``holdout_origins`` holds crops out of training
(``data.split_holdout``); the EMA model is scored on them at the end (``evaluate`` ->
``holdout_metrics.json``).

Feature distillation (``extra.train.feat_distill``, training only): the frozen
teacher *encoders* (recto / ink, bf16, no decoder) are run on the same CT crop
normalized the teacher's own way (``teachers.Normalizer`` on the u8 crop, i.e.
``input[:, 0] * 255`` -- the intensity-augmented crop before the student
z-score), and ``1 - cos`` (or MSE) between a 1x1x1-projected student encoder
stage and the teacher stage is added with ``weight``.  Pairs are downsampling
powers ``[student_k, teacher_k]``: with widths (32,64,128,256,256) and the
vesuvius encoders [32,64,128,256,320,320,320], ``[3,3]`` matches 256 -> 256 ch
at /8 and ``[4,4]`` 256 -> 320 ch at /16.  The projectors live in the
checkpoint (``extra.feat_proj``), never in the exported student.  Logged as
``feat_<teacher>`` (unweighted) and ``feat`` (weighted sum) in log.jsonl.
``feat_distill.source`` (or ``sources`` per teacher) switches a teacher from
``"live"`` to ``"cached"``: the target is then read from ``<out_dir>/feats/``
(``tsm feats``, :mod:`tsm.feats`) instead of running the encoder, which removes
the teacher forward from the step but only matches crops whose spatial transform
is a signed permutation (kept fraction logged as ``feat_<teacher>_frac``).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from tsm.config import RegionCfg, RunCfg, VolumeCfg, _region, _volume
from tsm.fiber import FIBER_MODES, class_from_direction, fiber_basis, sheet_normal
from tsm.labels import DEFAULT_AXIS
from tsm.limits import GB, MB, BudgetError, cuda_guard, estimate_and_assert, format_table, free_cuda, peak_rss_mb, read_rss_bytes
from tsm.student import (HEADS, FeatureProjector, TSMNet, activation_bytes_estimate, build_model,
                         feature_distill_loss, heads_for, in_channels, input_channels, input_vec_slices,
                         load_student_state, normalize_ct, to_channels_last)

FEAT_DISTILL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "teachers": ["recto", "ink"],
    "pairs": [[3, 3], [4, 4]],  # [student enc stage, teacher enc stage] (downsampling powers)
    "weight": 0.1,
    "loss": "cosine",  # cosine | mse
    "every": 1,  # apply on every N-th optimizer step
    "models_dir": None,  # default teachers.DEFAULT_MODELS_DIR
    # "dino" in teachers reads the cached DINO token grid (tsm dino) instead of running a
    # network: student enc<dino_stage> (/8) -> 1x1x1 projector -> cosine vs the cached tokens.
    "dino_cache": None,  # default <out_dir>/dino/tokens.zarr
    "dino_stage": 3,     # student encoder stage matching the token pitch (/8)
    # "live" runs the frozen teacher encoders in the loop; "cached" reads the grids written
    # by `tsm feats` (<out_dir>/feats/<teacher>_s<stage>.zarr), which removes the encoder
    # forward entirely at the cost of clean-volume targets (flips/rot90 only, no intensity
    # function matching).  ``sources`` overrides it per teacher, e.g. {"recto": "cached"}.
    "source": "live",   # live | cached
    "sources": {},      # {teacher: "live" | "cached"}
    "feats_dir": None,  # default <out_dir>/feats
}

TRAIN_DEFAULTS: dict[str, Any] = {
    "steps": 20000,
    "patch": 128,
    "batch": 2,
    "accum": 4,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup": 500,
    "ema": 0.999,
    "widths": [32, 64, 128, 256, 256],
    "ckpt_every": 1000,
    "log_every": 10,
    "num_workers": 4,
    "compile": False,
    "overfit_one": False,
    "loss_weights": {"surface": 1.0, "ink": 2.0, "winding": 1.0, "fiber": 1.0},
    # optional per-head loss balancing on top of loss_weights (see LossBalancer):
    # null = off (byte-identical to no balancing), "scale" | "gradnorm_lite"
    "balance": None,
    "balance_warmup": 200,   # steps of unbalanced training that seed the running statistics
    "balance_decay": 0.99,   # EMA decay of the per-head magnitudes (per optimizer step)
    "balance_freeze": True,  # freeze the EMA (and hence the multipliers) after the warmup
    "balance_every": 50,     # gradnorm_lite: measure the per-head gradient norms every N steps
    "grad_log_every": 100,   # log gradnorm/<head> every N steps (0 = never), any balance mode
    # optional heads: {"fiber": true} adds the 2-channel fibre-orientation head (needs a fine
    # store built with the fiber_vt / fiber_hz / fiber_valid channels); {"lsd": true} adds the
    # 5-channel Local Shape Descriptor head ("sides" mode only, tsm.data.lsd_targets)
    "heads": {"fiber": False, "lsd": False},
    # gaussian window (voxels) of the Local Shape Descriptors
    "lsd_sigma": 6.0,  # = tsm.data.LSD_SIGMA
    # fibre target encoding (tsm.fiber): "direction" (default: an axial fibre direction +
    # strength derived on the fly against the local axis tangent; a 4-channel head) or "class"
    # (legacy ablation: the two axis-relative probabilities, swapped by an augmentation that
    # moves the volume z axis -- exact only for the cube rotations)
    "fiber_mode": "direction",
    # class mode only: false reproduces the pre-2026-09-07 bug (classes augmented as invariant
    # scalars).  Kept for the ablation baseline; there is no reason to set it in a real run.
    "fiber_swap_fix": True,
    # direction mode: per-voxel loss weight of the human-traced hzvt_class band voxels
    "fiber_band_weight": 5.0,
    "ink_pos_weight": "auto",  # "auto" = positives/negatives of the batch (capped), a float, or null/0 = off
    "ds_weights": [1.0, 0.5],
    "grad_clip": 1.0,
    "seed": 0,
    "stride": 64,
    "min_valid_frac": 0.05,
    "fine_store": None,     # default <out_dir>/labels/fine.zarr
    "coarse_store": None,   # default <out_dir>/labels/coarse.zarr
    # several label stores (regions / scrolls) at once; replaces fine_store/coarse_store (see
    # ``store_opts`` and docs/label_store.md "Several stores").  Each entry is self-contained:
    #   {"name", "fine_store", "coarse_store", "axis", "voxel_um", "weight", "volume", "region"}
    "stores": None,
    "holdout_stores": None,  # [name, ...]: those stores train on nothing and are evaluated whole
    # "medial" (single SDF to the recto medial surface) | "faces" (two-face) |
    # "body" (orientation-free min(sdf_in, -sdf_out), one channel, tsm.data.body_sdf)
    "surface_mode": "medial",
    # optional auxiliary surface terms (faces / body modes); null / all-zero weights = unchanged loss
    "surface_aux": None,   # see SURFACE_AUX_DEFAULTS
    # train-time umbilicus core mask: voxels within this in-plane radius (fine voxels) of the
    # scroll axis get surface_valid = 2 (ignore) and winding_valid = 0 (see data.CropDataset).
    # 0 = off (unchanged).  The label-build-time equivalent is extra.labels.core_radius_vox.
    "core_radius_vox": 0.0,
    # CT-gated body label (surface_mode "body" / "sides"), null = off.  A threshold in RAW
    # uint8 CT units (e.g. 60): label-body voxels whose gaussian-smoothed (sigma 2) CT is below
    # it get surface_valid = 2 (ignore), so the air the label calls papyrus stops supervising
    # the surface targets.  The label STORE is never touched (data.CropDataset.load).
    "body_ct_gate": None,
    # where the winding targets come from: "coarse" (default, unchanged: the 9.6 um store
    # upsampled on the fly), "fine" (the native 2.4 um wf_* block only, tsm.winding_fine) or
    # "merge" (fine where wf_valid == 1, coarse elsewhere).  A store without wf_* is "coarse".
    "winding_source": "coarse",
    "input_radial": True,  # feed the outward radial direction (r_z, r_y, r_x) as input channels 2..4
    # feed the scroll-axis direction (a_z, a_y, a_x) as three more input channels: the local
    # umbilicus tangent, rotated with the volume by every spatial transform.  The fibre classes
    # ("vertical" = along the axis) are unlearnable under all-axes rotations without it -- the
    # analogue of input_radial for "outward".
    "input_axis": True,
    # True (default): the geometry inputs are the orthonormal (radial, local axis tangent) pair
    # of labels.geometry_frame, and the direction fibre targets are built against that tangent.
    # False: the pre-2026-09-12 constant (1, 0, 0) axis and purely in-plane radial (ablation).
    "axis_tangent": True,
    "axis_path": None,  # umbilicus JSON for input_radial; default extra.labels.axis_path / labels.DEFAULT_AXIS
    "augment": True,  # True -> "strong" (v2, GPU) | False -> off | "v1" (CPU flips/rot90 + jitter) | preset name | dict
    # None | [[z,y,x],...] | {"<z|y|x>_frac": f} | {"<z|y|x>_range": [lo, hi]} | {"yx_frac": f} (data.split_holdout)
    "holdout_origins": None,
    "eval_holdout": True,  # evaluate the EMA model on the holdout crops at the end of training
    "eval_max_crops": 64,
    # architecture (docs/student_v2_plan.md A1): body_stride 2 = stride-2 stem, the whole
    # encoder-decoder at half resolution + one full-resolution residual block of
    # ``fullres_width`` before the 1x1x1 heads (~3.3x fewer MACs); norm "batch" folds into
    # the convs at export but conflicts with geometric TTA -- GroupNorm stays the default.
    "body_stride": 1,
    "fullres_width": 32,
    "norm": "group",
    "act_ckpt": 2,  # activation checkpointing on the 2 finest resolution levels (VRAM)
    "channels_last": None,  # channels_last_3d for model + inputs; None = on when CUDA
    "feat_distill": None,  # see FEAT_DISTILL_DEFAULTS
}
#: ``extra.train.surface_aux``: optional auxiliary surface terms, faces mode only.  All
#: weights default to 0, which makes the surface loss byte-identical to the historical one
#: (the terms are not even computed).  See :func:`surface_aux_terms`.
SURFACE_AUX_DEFAULTS: dict[str, Any] = {
    "shell": 0.0,          # weight of the "de-blob" separation term
    "shell_radius": 2,     # dilation radius (voxels) of the label zero set
    "shell_margin": 3.0,   # label distance above which a shell voxel is supervised, and the
                           # |sdf_pred| the term pushes it to
    "crest": 0.0,          # weight of the existence-recall term on the label zero set
    "crest_tol": 1.0,      # |sdf_pred| allowed on a label face voxel before it costs anything
    "far": 0.0,            # weight of the precision term (competes with crest -- keep it small)
    "far_margin": 6.0,     # label distance above which a voxel counts as "far from any face"
    # body (two-face) topology terms -- see :func:`surface_body_terms`
    "gap": 0.0,            # weight of the gap-preservation / anti-merge penalty
    "gap_radius": 3,       # closing radius (voxels): gaps narrower than ~2*radius are protected
    "gap_tau": 2.0,        # softness (voxels) of the predicted body probability
    "cldice": 0.0,         # weight of the soft-clDice (topology) term on the sheet body
    "cldice_iters": 5,     # soft skeletonisation iterations
    # explicit air-gap supervision (sides mode; see :func:`surface_body_terms`)
    "gap_dilate": 0,       # dilate the LABEL gap mask by this many voxels before it is used by
                           # `gap` and `gap_class` (bias toward under-connection); 0 = unchanged
    "gap_class": 0.0,      # weight of the BCE + soft Dice of a dedicated gap HEAD channel
                           # against that label gap mask (widens the sides head to 4 channels)
    "gap_border_weight": 0.0,  # Ronneberger-style w0: extra per-voxel weight on the gap voxels
                           # that touch a body, added to the d_face L1 and body BCE masks
    # eikonal regulariser on the predicted unsigned distance (sides mode)
    "eikonal": 0.0,        # weight of mean (|grad d_pred| - 1)^2 away from faces / ridge
    "eikonal_min_d": 3.0,  # label d_face above which a voxel is supervised by it
}
BALANCE_MODES = (None, "scale", "gradnorm_lite")
SDF_SIGMA = 4.0
SDF_WMAX = 8.0
BAND_TAU = 2.0
EPS = 1e-6


def train_opts(cfg: RunCfg) -> dict[str, Any]:
    raw = cfg.extra.get("train", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.train must be an object")
    unknown = sorted(set(raw) - set(TRAIN_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.train keys: {unknown}")
    opts = dict(TRAIN_DEFAULTS)
    opts.update(raw)
    lw = dict(TRAIN_DEFAULTS["loss_weights"])
    lw.update(opts.get("loss_weights") or {})
    opts["loss_weights"] = lw
    opts["input_radial"] = bool(opts.get("input_radial", True))
    opts["input_axis"] = bool(opts.get("input_axis", True))
    opts["axis_tangent"] = bool(opts.get("axis_tangent", True))
    if opts["fiber_mode"] not in FIBER_MODES:
        raise ValueError(f"extra.train.fiber_mode must be one of {list(FIBER_MODES)}, got {opts['fiber_mode']!r}")
    opts["fiber_swap_fix"] = bool(opts["fiber_swap_fix"])
    if float(opts["fiber_band_weight"]) < 0:
        raise ValueError("extra.train.fiber_band_weight must be >= 0")
    hd = dict(TRAIN_DEFAULTS["heads"])
    raw_heads = opts.get("heads") or {}
    if not isinstance(raw_heads, dict):
        raise ValueError("extra.train.heads must be an object")
    unknown_h = sorted(set(raw_heads) - set(hd))
    if unknown_h:
        raise ValueError(f"unknown extra.train.heads keys: {unknown_h} (known: {sorted(hd)})")
    hd.update({k: bool(v) for k, v in raw_heads.items()})
    opts["heads"] = hd
    if float(opts["lsd_sigma"]) <= 0:
        raise ValueError(f"extra.train.lsd_sigma must be > 0, got {opts['lsd_sigma']!r}")
    if hd["lsd"]:
        if str(opts["surface_mode"]) != "sides":
            raise ValueError("extra.train.heads.lsd needs extra.train.surface_mode='sides', got "
                             f"{opts['surface_mode']!r}")
        # the head's default weight, only when it is actually built (so a run without it keeps
        # exactly the historical loss_weights, and the balancer keeps the historical head set)
        opts["loss_weights"].setdefault("lsd", 1.0)
    pw = opts.get("ink_pos_weight", "auto")
    if not (pw in (None, "auto") or isinstance(pw, (int, float))):
        raise ValueError(f"extra.train.ink_pos_weight must be 'auto', a number or null, got {pw!r}")
    if not opts.get("axis_path"):
        opts["axis_path"] = (cfg.extra.get("labels", {}) or {}).get("axis_path") or DEFAULT_AXIS
    opts["stores"] = store_opts(cfg, opts)
    if int(opts["num_workers"]) > 4:
        raise ValueError("num_workers must be <= 4 (RAM budget)")
    if int(opts["patch"]) % 16:
        raise ValueError("patch must be divisible by 16")
    if opts["balance"] not in BALANCE_MODES:
        raise ValueError(f"extra.train.balance must be one of {list(BALANCE_MODES)}, got {opts['balance']!r}")
    if int(opts["balance_warmup"]) < 0 or int(opts["balance_every"]) < 1:
        raise ValueError("extra.train.balance_warmup must be >= 0 and balance_every >= 1")
    if not 0.0 < float(opts["balance_decay"]) < 1.0:
        raise ValueError("extra.train.balance_decay must be in (0, 1)")
    opts["balance_freeze"] = bool(opts["balance_freeze"])
    if int(opts["grad_log_every"]) < 0:
        raise ValueError("extra.train.grad_log_every must be >= 0")
    from tsm.data import SURFACE_MODES

    if opts["surface_mode"] not in SURFACE_MODES:
        raise ValueError(f"extra.train.surface_mode must be one of {SURFACE_MODES}, "
                         f"got {opts['surface_mode']!r}")
    opts["surface_aux"] = surface_aux_opts(opts.get("surface_aux"), opts["surface_mode"])
    if float(opts["core_radius_vox"]) < 0:
        raise ValueError("extra.train.core_radius_vox must be >= 0")
    opts["core_radius_vox"] = float(opts["core_radius_vox"])
    g = opts.get("body_ct_gate")
    if g is not None:
        if isinstance(g, bool) or not isinstance(g, (int, float)) or not 0 <= float(g) <= 255:
            raise ValueError(f"extra.train.body_ct_gate must be null or a CT threshold in [0, 255], got {g!r}")
        if opts["surface_mode"] not in ("body", "sides"):
            raise ValueError("extra.train.body_ct_gate needs extra.train.surface_mode 'body' or 'sides'")
        opts["body_ct_gate"] = float(g)
    from tsm.data import WINDING_SOURCES

    if opts["winding_source"] not in WINDING_SOURCES:
        raise ValueError(f"extra.train.winding_source must be one of {list(WINDING_SOURCES)}, "
                         f"got {opts['winding_source']!r}")
    if int(opts["body_stride"]) not in (1, 2):
        raise ValueError(f"extra.train.body_stride must be 1 or 2, got {opts['body_stride']!r}")
    if opts["norm"] not in ("group", "batch"):
        raise ValueError(f"extra.train.norm must be 'group' or 'batch', got {opts['norm']!r}")
    opts["feat_distill"] = feat_distill_opts(opts.get("feat_distill"))
    check_augment_feat_distill(opts)
    return opts


STORE_KEYS = {"name", "fine_store", "coarse_store", "axis", "voxel_um", "weight", "volume", "region",
              "holdout_box_zyx", "holdout_boxes_zyx"}


def _holdout_box(v: Any, what: str) -> list[int]:
    """One ``[z0, y0, x0, dz, dy, dx]`` box in scroll (level-0) voxels -> list of 6 ints.

    Same convention as ``region`` (``start_zyx`` + ``size_zyx``), flattened."""
    if not isinstance(v, (list, tuple)) or len(v) != 6:
        raise ValueError(f"{what} must be a list of 6 ints [z0, y0, x0, dz, dy, dx], got {v!r}")
    out = []
    for t in v:
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError(f"{what} must be a list of 6 ints [z0, y0, x0, dz, dy, dx], got {v!r}")
        out.append(int(t))
    if min(out[3:]) <= 0:
        raise ValueError(f"{what} sizes (dz, dy, dx) must be positive, got {out[3:]}")
    return out


def _holdout_boxes(e: dict[str, Any], what: str) -> list[list[int]]:
    """Normalise ``holdout_box_zyx`` / ``holdout_boxes_zyx`` of one store entry to a list of boxes."""
    boxes: list[list[int]] = []
    one = e.get("holdout_box_zyx")
    if one is not None:
        boxes.append(_holdout_box(one, f"{what}.holdout_box_zyx"))
    many = e.get("holdout_boxes_zyx")
    if many is not None:
        if not isinstance(many, (list, tuple)) or not many:
            raise ValueError(f"{what}.holdout_boxes_zyx must be a non-empty list of boxes")
        for j, b in enumerate(many):
            boxes.append(_holdout_box(b, f"{what}.holdout_boxes_zyx[{j}]"))
    return boxes


def store_opts(cfg: RunCfg, opts: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Validate ``extra.train.stores`` / ``holdout_stores`` (None when the single-store path is used).

    ``stores`` replaces ``fine_store`` / ``coarse_store``: giving both is an error.  Every entry is
    self-contained -- the fine store, its coarse store, the umbilicus ``axis`` used for the radial
    input channels, the ``voxel_um`` pitch of the fine store, an optional sampling ``weight``
    (default: the store's number of training origins, i.e. uniform over crops) and an optional
    per-store ``volume`` / ``region`` (default: the top-level ones for ``volume``, the fine store's
    own extent for ``region``, which is what the local CT cache must cover).

    ``holdout_box_zyx`` (or ``holdout_boxes_zyx``, a list of them) is ``[z0, y0, x0, dz, dy, dx]`` in
    scroll (level-0) voxels -- the same convention as ``region``, flattened.  Every crop that
    intersects the box is moved out of the store's training origins and into its holdout origins,
    so an evaluation region inside a training slab is never trained on.
    """
    raw = opts.get("stores")
    hold = opts.get("holdout_stores") or []
    if raw is None:
        if hold:
            raise ValueError("extra.train.holdout_stores needs extra.train.stores")
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("extra.train.stores must be a non-empty list of objects")
    if opts.get("fine_store") or opts.get("coarse_store"):
        raise ValueError("extra.train.stores replaces fine_store / coarse_store; give one or the other")
    out: list[dict[str, Any]] = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(f"extra.train.stores[{i}] must be an object")
        unknown = sorted(set(e) - STORE_KEYS)
        if unknown:
            raise ValueError(f"unknown extra.train.stores[{i}] keys: {unknown} (known: {sorted(STORE_KEYS)})")
        name = e.get("name") or f"store{i}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"extra.train.stores[{i}].name must be a non-empty string")
        fine = e.get("fine_store")
        if not isinstance(fine, str) or not fine:
            raise ValueError(f"extra.train.stores[{i}].fine_store must be a non-empty string")
        coarse = e.get("coarse_store")
        if coarse is not None and (not isinstance(coarse, str) or not coarse):
            raise ValueError(f"extra.train.stores[{i}].coarse_store must be a string or null")
        w = e.get("weight")
        if w is not None and (isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0):
            raise ValueError(f"extra.train.stores[{i}].weight must be a non-negative number or null")
        vu = e.get("voxel_um")
        if vu is not None and (isinstance(vu, bool) or not isinstance(vu, (int, float)) or vu <= 0):
            raise ValueError(f"extra.train.stores[{i}].voxel_um must be a positive number or null")
        vol = e.get("volume")
        if vol is not None:
            _volume(vol)  # validates now, parsed again in build_dataset
        reg = e.get("region")
        if reg is not None:
            _region(reg)
        boxes = _holdout_boxes(e, f"extra.train.stores[{i}]")
        out.append({"name": name, "fine_store": os.path.expanduser(fine),
                    "coarse_store": os.path.expanduser(coarse) if coarse else None,
                    "axis": e.get("axis"), "voxel_um": float(vu) if vu is not None else None,
                    "weight": float(w) if w is not None else None,
                    "volume": vol, "region": reg, "holdout_boxes_zyx": boxes})
    names = [e["name"] for e in out]
    if len(set(names)) != len(names):
        raise ValueError(f"extra.train.stores names must be unique, got {names}")
    if not isinstance(hold, (list, tuple)):
        raise ValueError("extra.train.holdout_stores must be a list of store names")
    stray = sorted(set(map(str, hold)) - set(names))
    if stray:
        raise ValueError(f"extra.train.holdout_stores names unknown stores: {stray} (known: {names})")
    if set(map(str, hold)) == set(names):
        raise ValueError("extra.train.holdout_stores holds out every store: nothing to train on")
    opts["holdout_stores"] = [str(h) for h in hold]
    return out


def check_augment_feat_distill(opts: dict[str, Any]) -> None:
    """Reject legacy ("v1") augmentation together with feature supervision read from a cache.

    The v1 CPU dataset flips/rotates the crop but returns no transform, so cached teacher / DINO
    features -- which are sampled by global coordinate in the *original* frame -- cannot be aligned
    with what the student sees.  Live teachers see the same transformed CT and are fine; v2
    augmentation carries the transform.  Raising here (config validation, before any training
    starts) is deliberate: silently misaligned supervision is worse than a refused run."""
    mode, _ = augment_mode(opts)
    fd = opts["feat_distill"]
    if mode != "v1" or not fd["enabled"]:
        return
    bad = sorted(t for t in fd["teachers"]
                 if t == "dino" or str(fd["sources"].get(t, fd["source"])) == "cached")
    if bad:
        raise ValueError(
            f"unsupported combination: extra.train.augment='v1' with cached feat_distill features {bad} -- "
            "cached features are sampled in the original frame and v1 augmentation does not report its "
            "spatial transform, so the supervision would be misaligned; use augment=true / a v2 preset, "
            "or feat_distill.source='live' (dino is always cached)")


def feat_distill_opts(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return dict(FEAT_DISTILL_DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError("extra.train.feat_distill must be an object")
    unknown = sorted(set(raw) - set(FEAT_DISTILL_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.train.feat_distill keys: {unknown}")
    fd = dict(FEAT_DISTILL_DEFAULTS)
    fd.update(raw)
    fd["enabled"] = bool(fd["enabled"])
    if fd["loss"] not in ("cosine", "mse"):
        raise ValueError("feat_distill.loss must be 'cosine' or 'mse'")
    if int(fd["every"]) < 1:
        raise ValueError("feat_distill.every must be >= 1")
    fd["pairs"] = [[int(a), int(b)] for a, b in fd["pairs"]]
    fd["dino_stage"] = int(fd["dino_stage"])
    if fd["source"] not in ("live", "cached"):
        raise ValueError("feat_distill.source must be 'live' or 'cached'")
    if not isinstance(fd["sources"], dict):
        raise ValueError("feat_distill.sources must be an object {teacher: 'live'|'cached'}")
    fd["sources"] = {str(k): str(v) for k, v in fd["sources"].items()}
    bad = sorted(k for k, v in fd["sources"].items() if v not in ("live", "cached"))
    if bad:
        raise ValueError(f"feat_distill.sources values must be 'live' or 'cached' (bad: {bad})")
    stray = sorted(set(fd["sources"]) - set(fd["teachers"]))
    if stray:
        raise ValueError(f"feat_distill.sources names teachers that are not listed: {stray}")
    if fd["enabled"] and not fd["teachers"]:
        raise ValueError("feat_distill needs at least one teacher")
    if fd["enabled"] and not fd["pairs"] and set(fd["teachers"]) != {"dino"}:
        raise ValueError("feat_distill needs at least one pair for the network teachers")
    return fd


def surface_aux_opts(raw: Any, surface_mode: str = "faces") -> dict[str, Any]:
    """``extra.train.surface_aux`` -> a full :data:`SURFACE_AUX_DEFAULTS` dict (None = all off).

    Every weight defaults to 0; with all three at 0 the surface loss is byte-identical to the
    historical one, so this is safe to leave in a config.  The terms only exist in
    ``surface_mode="faces"``, ``"body"`` and ``"sides"`` -- asking for them in ``"medial"`` mode
    is an error
    rather than a silent no-op."""
    if raw is None or raw is False:
        return dict(SURFACE_AUX_DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError("extra.train.surface_aux must be an object or null")
    unknown = sorted(set(raw) - set(SURFACE_AUX_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.train.surface_aux keys: {unknown} "
                         f"(known: {sorted(SURFACE_AUX_DEFAULTS)})")
    out = {**SURFACE_AUX_DEFAULTS, **raw}
    for k in ("shell", "crest", "far", "shell_margin", "crest_tol", "far_margin", "gap", "gap_tau", "cldice",
              "gap_class", "gap_border_weight", "eikonal", "eikonal_min_d"):
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) < 0:
            raise ValueError(f"extra.train.surface_aux.{k} must be a non-negative number, got {v!r}")
        out[k] = float(v)
    for k in ("shell_radius", "gap_radius", "cldice_iters"):
        r = out[k]
        if isinstance(r, bool) or not isinstance(r, (int, float)) or int(r) < 1:
            raise ValueError(f"extra.train.surface_aux.{k} must be an integer >= 1, got {r!r}")
        out[k] = int(r)
    d = out["gap_dilate"]
    if isinstance(d, bool) or not isinstance(d, (int, float)) or int(d) < 0:
        raise ValueError(f"extra.train.surface_aux.gap_dilate must be an integer >= 0, got {d!r}")
    out["gap_dilate"] = int(d)
    if out["gap_class"] > 0 and surface_mode != "sides":
        raise ValueError("extra.train.surface_aux.gap_class needs extra.train.surface_mode='sides' "
                         "(it is a dedicated head channel of that mode)")
    if (out["gap_border_weight"] > 0 or out["eikonal"] > 0) and surface_mode != "sides":
        raise ValueError("extra.train.surface_aux.gap_border_weight / eikonal need "
                         "extra.train.surface_mode='sides'")
    if out["gap"] > 0 and out["gap_tau"] <= 0:
        raise ValueError("extra.train.surface_aux.gap_tau must be > 0 when gap > 0")
    if surface_aux_active(out) and surface_mode not in ("faces", "body", "sides"):
        raise ValueError("extra.train.surface_aux needs extra.train.surface_mode='faces' "
                         "or 'body' or 'sides'")
    if out["shell"] > 0 and out["shell_radius"] < out["shell_margin"]:
        print(f"[tsm] WARNING surface_aux.shell_radius {out['shell_radius']} < shell_margin "
              f"{out['shell_margin']}: the shell is the *cubic* dilation of the label zero set, so "
              f"on a smooth face only its corners (up to {out['shell_radius'] * 3 ** 0.5:.2f} vox "
              f"away) reach the |sdf| >= {out['shell_margin']} gate -- a thin, anisotropic sliver "
              f"plus whatever label discontinuities (face edges, masked regions) contribute.  Set "
              f"shell_radius >= shell_margin for the intended de-blob behaviour.", flush=True)
    return out


def surface_aux_active(aux: Mapping[str, Any] | None) -> bool:
    """True when at least one auxiliary surface weight is non-zero (per-face *or* body terms)."""
    return bool(aux) and any(float(aux.get(k, 0.0)) > 0.0
                             for k in ("shell", "crest", "far", "gap", "cldice", "gap_class",
                                       "gap_border_weight", "eikonal"))


def volcomp_spec(opts: dict[str, Any]):
    """``extra.train.augment``'s ``volcomp`` family (``data.VolcompCfg``) or None when it is off.

    The codec round trip is the one intensity transform that does NOT run in ``Augment``: it
    works on the raw uint8 crop, so it is applied on the CPU in the dataset worker and has to
    be handed to the dataset instead of to the GPU pipeline."""
    mode, spec = augment_mode(opts)
    if mode != "v2":
        return None
    from tsm.data import as_augment_config

    vc = as_augment_config(spec).volcomp
    return vc if float(vc.p) > 0 else None


def augment_mode(opts: dict[str, Any]) -> tuple[str, Any]:
    """``("off" | "v1" | "v2", spec)`` from ``extra.train.augment`` (spec = preset name / dict for v2)."""
    a = opts.get("augment", True)
    if a is False or a is None or a == "none" or a == "off":
        return "off", None
    if a is True:
        return "v2", "strong"
    if a == "v1":
        return "v1", None
    if isinstance(a, (str, dict)):
        from tsm.data import as_augment_config

        cfg = as_augment_config(a)  # validates now (unknown keys / presets)
        return ("off" if cfg.is_identity() else "v2"), cfg
    raise ValueError(f"extra.train.augment must be a bool, 'v1', a preset name or an object, got {a!r}")


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp_min(1.0)


def soft_dice(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """1 - 2<a,b>/(<a,a>+<b,b>) over the masked voxels (zero for a == b, right for soft targets)."""
    dims = tuple(range(1, a.ndim))
    inter = (a * b * m).sum(dims)
    denom = (a * a * m).sum(dims) + (b * b * m).sum(dims)
    has = (m.sum(dims) > 0).float()
    d = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return (d * has).sum() / has.sum().clamp_min(1.0)


def surface_band_dice(sdf_pred: torch.Tensor, sdf_tgt: torch.Tensor, m: torch.Tensor, tau: float = BAND_TAU) -> torch.Tensor:
    """Soft Dice between exp(-|sdf|/tau) 'surface band' maps (sign-agnostic)."""
    return soft_dice(torch.exp(-sdf_pred.abs() / tau), torch.exp(-sdf_tgt.abs() / tau), m)


def _weighted(m: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """Fold an optional per-voxel weight into a 0/1 loss mask.

    Every surface term is a mask-weighted mean (``masked_mean``) or a mask-weighted soft Dice,
    and both use the mask purely multiplicatively -- so multiplying the mask by the weight *is*
    the per-voxel multiplier, and a weight of exactly 1.0 leaves the arithmetic bit-identical
    (``m * 1.0 == m`` in IEEE754).  ``None`` (no ``faces_weight`` channel) skips it entirely.
    """
    return m if weight is None else m * weight


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of a 0/1 (B, 1, Z, Y, X) mask by a cube of ``radius`` voxels."""
    r = int(radius)
    return (F.max_pool3d(mask, kernel_size=2 * r + 1, stride=1, padding=r) > 0).float()


def per_sample_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over ``m`` **per sample**, then averaged over the samples that have any.

    Unlike :func:`masked_mean` (one mean over the whole batch's voxels) a sample with few masked
    voxels counts as much as a dense one -- what a sparse, existence-style target needs."""
    dims = tuple(range(1, x.ndim))
    den = m.sum(dims)
    has = (den > 0).float()
    return (((x * m).sum(dims) / den.clamp_min(1.0)) * has).sum() / has.sum().clamp_min(1.0)


def surface_aux_terms(p_sdf: torch.Tensor, t_sdf: torch.Tensor, mv: torch.Tensor, m1: torch.Tensor,
                      aux: Mapping[str, Any], face: str) -> dict[str, torch.Tensor]:
    """Optional auxiliary terms for one face (``extra.train.surface_aux``, all off by default).

    The main per-face loss (gaussian-weighted L1 + band Dice) is a regression on the SDF *value*.
    Measured on the student (2026-09), that leaves two failure modes it barely penalises: the
    predicted in-face has excellent recall of the human bands (band -> pred median 2 vox) but poor
    precision (45 % of predicted in-face voxels are > 5 vox from any band) and fragments in
    crumpled regions.  These three terms attack the zero *set* directly.

    ``shell`` -- "de-blob" separation.  The label zero set ``|sdf_tgt| <= 0.5`` (supervised voxels
    only) is dilated by ``shell_radius``; the dilation minus the zero set is the *shell*, and the
    term supervises the shell voxels whose label distance is already ``>= shell_margin``, pushing
    ``|sdf_pred|`` there up to ``shell_margin``::

        mean over shell of relu(shell_margin - |sdf_pred|)

    A perfect prediction scores exactly 0 (its ``|sdf_pred| = |sdf_tgt| >= shell_margin`` there),
    which is why the margin gate is on the *label*.  Note the geometry: the dilation is cubic
    (one ``max_pool3d``), so a smooth face contributes shell voxels out to ``shell_radius *
    sqrt(3)``; with ``shell_radius < shell_margin`` only those corners pass the gate and the term
    is a thin, anisotropic sliver.  Set ``shell_radius >= shell_margin`` to actually push
    predicted faces off the sides of true ones; :func:`surface_aux_opts` warns otherwise.

    ``crest`` -- existence recall on the label zero set: ``relu(|sdf_pred| - crest_tol)`` averaged
    **per sample** over the label face voxels (:func:`per_sample_mean`), so a crop with a thin,
    sparse face is not drowned by the dense crops in the batch.

    ``far`` -- the direct precision term: over voxels the label puts ``>= far_margin`` from the
    face, ``relu(far_margin / 2 - |sdf_pred|)`` penalises a predicted zero crossing that has no
    business being there.  It pulls against ``crest`` (one wants zero crossings, the other does
    not), so keep its weight small -- 0.1 against 0.5 in the shipped config.

    Every term is masked by ``faces_valid == 1`` (times the optional ``faces_weight``); the
    returned values are already multiplied by their weight, so the caller just sums them.
    """
    out: dict[str, torch.Tensor] = {}
    tabs = t_sdf.abs()
    pabs = p_sdf.abs()
    zero = (tabs <= 0.5).float() * mv          # the label face, supervised voxels only
    w_shell, w_crest, w_far = (float(aux.get(k, 0.0)) for k in ("shell", "crest", "far"))
    if w_shell > 0:
        margin = float(aux["shell_margin"])
        shell = _dilate(zero, int(aux["shell_radius"])) * (1.0 - zero) * (tabs >= margin).float() * m1
        out[f"shell_{face}"] = w_shell * masked_mean(F.relu(margin - pabs), shell)
    if w_crest > 0:
        out[f"crest_{face}"] = w_crest * per_sample_mean(F.relu(pabs - float(aux["crest_tol"])), zero * m1)
    if w_far > 0:
        fm = float(aux["far_margin"])
        out[f"far_{face}"] = w_far * masked_mean(F.relu(0.5 * fm - pabs), (tabs >= fm).float() * m1)
    return out


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary erosion of a 0/1 mask by a cube of ``radius`` voxels (dilation of the complement)."""
    return 1.0 - _dilate(1.0 - mask, radius)


def _close(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary morphological *closing* (dilate then erode) with a cube of ``radius`` voxels.

    Closing fills every hole / gap narrower than about ``2 * radius`` voxels and leaves everything
    else alone, so ``close(x) - x`` is exactly the set of thin gaps inside and between the
    components of ``x``."""
    return _erode(_dilate(mask, radius), radius)


def body_mask(t_in: torch.Tensor, t_out: torch.Tensor | None = None,
              m: torch.Tensor | float = 1.0, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The label sheet *body* (papyrus material) as a 0/1 map.

    Two-face target (``t_out`` given): ``(sdf_in > 0) & (sdf_out < 0)`` -- the same definition as
    the ``inside`` mask of :func:`evaluate`'s thickness metric: the body is the slab between the
    two faces, and its thickness is ``sdf_in - sdf_out``.  Body target (``t_out is None``,
    ``surface_mode="body"``): the single channel already *is* ``min(sdf_in, -sdf_out)``
    (:func:`tsm.data.body_sdf`), so the body is simply ``sdf_body > 0``.  ``mask`` given
    (``surface_mode="sides"``): the body is already a 0/1 target channel
    (:func:`tsm.data.body_mask_np`) and is used as is, ``t_in`` / ``t_out`` ignored.  ``m``
    restricts it to the supervised voxels."""
    if mask is not None:
        return mask.float() * m
    inside = (t_in > 0) if t_out is None else ((t_in > 0) & (t_out < 0))
    return inside.float() * m


def narrow_gap_mask(body_t: torch.Tensor, m1: torch.Tensor, radius: int, dilate: int = 0) -> torch.Tensor:
    """``close_radius(body) * (1 - body) * m1`` -- the thin air gaps a weld would fill.

    ``dilate > 0`` (``extra.train.surface_aux.gap_dilate``) grows that mask by a cube of
    ``dilate`` voxels and then removes the body again, so the protected air widens by a voxel
    or two on each side without ever covering labelled papyrus: the bias is deliberately toward
    *under*-connection (a hair too much "keep this apart" is cheaper than a weld).  ``dilate=0``
    returns exactly the historical mask."""
    gap = _close(body_t, radius) * (1.0 - body_t) * m1
    if int(dilate) > 0:
        gap = _dilate(gap, int(dilate)) * (1.0 - body_t) * m1
    return gap


def gap_border_mask(body_t: torch.Tensor, gap_t: torch.Tensor, dilate: int = 0) -> torch.Tensor:
    """The gap voxels that *touch* a body -- a cheap stand-in for Ronneberger's border weight map.

    U-Net (Ronneberger et al. 2015) weights a voxel by ``w0 * exp(-(d1 + d2)^2 / (2 sigma^2))``
    with ``d1``, ``d2`` the distances to the two nearest *different* objects, which needs two
    full distance transforms per crop.  The voxels that map keeps are exactly the thin air
    between two bodies, close to both -- and here the thin air is already known
    (:func:`narrow_gap_mask`), so the approximation is a single morphological test::

        border = gap * dilate_{dilate + 1}(body)

    i.e. the label-gap voxels whose (chebyshev) distance to the nearest body is <=
    ``gap_dilate + 1``.  It is 0/1 rather than a smooth bump; the caller scales it by
    ``surface_aux.gap_border_weight`` (the ``w0``) and adds it to the loss mask."""
    return gap_t * _dilate(body_t, int(dilate) + 1)


def _central_diff(x: torch.Tensor) -> torch.Tensor:
    """Central finite differences of a (B, 1, Z, Y, X) field -> (B, 3, Z, Y, X), voxel units.

    Replicate padding, so the 1-voxel border carries a half-width difference; every caller
    excludes that border from its mask."""
    p = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    dz = (p[:, :, 2:, 1:-1, 1:-1] - p[:, :, :-2, 1:-1, 1:-1]) * 0.5
    dy = (p[:, :, 1:-1, 2:, 1:-1] - p[:, :, 1:-1, :-2, 1:-1]) * 0.5
    dx = (p[:, :, 1:-1, 1:-1, 2:] - p[:, :, 1:-1, 1:-1, :-2]) * 0.5
    return torch.cat([dz, dy, dx], dim=1)


def _interior(x: torch.Tensor) -> torch.Tensor:
    """0/1 mask that is 0 on the 1-voxel border of the crop (where central differences are half)."""
    m = torch.zeros_like(x)
    m[:, :, 1:-1, 1:-1, 1:-1] = 1.0
    return m


def _ridge_mask(g: torch.Tensor) -> torch.Tensor:
    """Voxels a central difference of the LABEL distance changes sign *across*, along any axis.

    An unsigned distance field is |grad| = 1 everywhere except on the faces (where it is 0, a
    minimum) and on the medial ridge of the sheet and of the air gap (a maximum): at both the
    derivative along some axis flips sign from one neighbour to the other.  ``g`` is
    :func:`_central_diff` of the label field; the test is ``g[i-1] * g[i+1] < 0`` per axis."""
    p = F.pad(g, (1, 1, 1, 1, 1, 1), mode="replicate")
    cz = p[:, 0:1, 2:, 1:-1, 1:-1] * p[:, 0:1, :-2, 1:-1, 1:-1]
    cy = p[:, 1:2, 1:-1, 2:, 1:-1] * p[:, 1:2, 1:-1, :-2, 1:-1]
    cx = p[:, 2:3, 1:-1, 1:-1, 2:] * p[:, 2:3, 1:-1, 1:-1, :-2]
    return ((cz < 0) | (cy < 0) | (cx < 0)).float()


def eikonal_term(p_d: torch.Tensor, t_d: torch.Tensor, m1: torch.Tensor,
                 weight: float, min_d: float = 3.0) -> torch.Tensor:
    """``mean over the eikonal mask of (|grad d_pred| - 1)^2`` (sides mode only), times ``weight``.

    ``d_face`` is an UNSIGNED distance to the nearest sheet face, so the true field satisfies
    the eikonal equation ``|grad d| = 1`` almost everywhere -- everywhere except three sets,
    all of which the mask removes:

    * **near the faces** -- label ``d_face <= eikonal_min_d`` (the gaussian-weighted L1 already
      owns that band, and the field has a kink at the face itself),
    * **saturated labels** -- label ``d_face >= clip - 1``, where the store clipped the distance
      flat (gradient 0).  The clip is not passed in; it is read off the batch as the maximum
      label distance over supervised voxels, which *is* ``extra.labels.clip`` whenever anything
      in the batch saturates,
    * **the medial ridge** -- the locus where the nearest face switches and the distance has a
      local maximum along some axis.  Approximated by :func:`_ridge_mask` (a central difference
      of the *label* field changing sign across the voxel) dilated by a cube of radius 2, which
      covers the "within 1.5 voxels of a sign change" band the plan asks for.

    plus the 1-voxel crop border, where the replicate-padded central difference is halved.  The
    gradient is central finite differences in voxel units (:func:`_central_diff`); the mean is
    taken over the mask (``masked_mean``), so an empty mask contributes exactly 0."""
    g_t = _central_diff(t_d)
    clip = float(t_d.mul(m1 > 0).max()) if float((m1 > 0).sum()) > 0 else 0.0
    m = (m1 > 0).float() * _interior(t_d)
    m = m * (t_d > float(min_d)).float() * (t_d < clip - 1.0).float()
    m = m * (1.0 - _dilate(_ridge_mask(g_t), 2))
    gm = _central_diff(p_d).pow(2).sum(dim=1, keepdim=True).clamp_min(EPS).sqrt()
    return float(weight) * masked_mean((gm - 1.0) ** 2, m)


def soft_body(p_in: torch.Tensor, p_out: torch.Tensor | None = None, tau: float = 2.0,
              prob: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable body probability, the soft counterpart of :func:`body_mask`.

    Two faces: ``sigmoid(sdf_in / tau) * sigmoid(-sdf_out / tau)`` -- the product of the two
    half-space indicators the hard body mask ANDs.  One body channel (``p_out is None``):
    ``sigmoid(sdf_body / tau)``.  ``prob`` given (``surface_mode="sides"``): the head already
    emits a body *probability*, so it is returned as is and ``tau`` plays no part.  ``tau``
    (voxels, the same scale as :data:`BAND_TAU`) is how sharply the faces cut."""
    if prob is not None:
        return prob
    p = torch.sigmoid(p_in / tau)
    return p if p_out is None else p * torch.sigmoid(-p_out / tau)


def soft_skel(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Soft skeletonisation of a [0, 1] map (Shit et al. 2021, "clDice"), 3D, differentiable.

    ``soft_erode = -maxpool3d(-x)`` and ``soft_dilate = maxpool3d(x)`` on a 3x3x3 window are the
    grey-scale morphology min/max; ``soft_open = dilate(erode(x))``.  Each iteration collects the
    part of the current map that opening removes -- the thin structure -- and then erodes::

        skel = relu(x - open(x)); repeat i times: x = erode(x); skel += relu(x - open(x)) * (1 - skel)

    so after ``iters`` erosions ``skel`` is a thin, one-voxel-ish medial map of ``x``."""
    def _ero(t: torch.Tensor) -> torch.Tensor:
        return -F.max_pool3d(-t, kernel_size=3, stride=1, padding=1)

    def _open(t: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(_ero(t), kernel_size=3, stride=1, padding=1)

    skel = F.relu(x - _open(x))
    for _ in range(int(iters)):
        x = _ero(x)
        delta = F.relu(x - _open(x))
        skel = skel + delta * (1.0 - skel)
    return skel


def surface_body_terms(p_in: torch.Tensor, p_out: torch.Tensor | None, t_in: torch.Tensor,
                       t_out: torch.Tensor | None,
                       mv: torch.Tensor, m1: torch.Tensor, aux: Mapping[str, Any],
                       p_body: torch.Tensor | None = None,
                       t_body: torch.Tensor | None = None,
                       p_gap: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Optional *body* terms of ``extra.train.surface_aux`` (``gap``, ``cldice``), both off by default.

    Unlike ``shell`` / ``crest`` / ``far`` (:func:`surface_aux_terms`, one face at a time) these
    two look at the region *between* the faces -- the sheet body ``(sdf_in > 0) & (sdf_out < 0)``,
    whose thickness is ``sdf_in - sdf_out`` -- so they are computed once per crop, not per face.
    They are adapted from the anti-merge loss of the hercUNet project (separation penalty + soft
    skeleton recall).  The failure they attack is the student's dominant one (measured 2026-09):
    **over-connection** -- adjacent wraps welded together across the thin air gap that separates
    them, plus invented sheets, with essentially zero breaks.  The per-face L1 + band Dice cannot
    see a weld: filling a 2-voxel gap costs a couple of voxels of SDF error.

    ``gap`` -- gap preservation / anti-merge.  On the *label* side (no gradients) the body mask is
    morphologically CLOSED with a cube of radius ``gap_radius`` (default 3) and the body itself
    subtracted::

        narrow_gap = close_r(body_t) * (1 - body_t) * m1

    Closing fills exactly the gaps narrower than ~``2 * gap_radius``, so this is the thin air
    between two nearby sheets (plus small concavities) -- precisely the voxels a weld would fill.
    The prediction is scored there with the soft body probability of :func:`soft_body`::

        gap * per_sample_mean(sigmoid(p_in / gap_tau) * sigmoid(-p_out / gap_tau), narrow_gap)

    A perfect prediction has no body in a label gap and scores exactly 0.  The mask is label-side
    because the gap is a property of the ground truth: deriving it from the prediction would let a
    model that has already welded the sheets erase its own penalty.  ``per_sample_mean`` (not
    ``masked_mean``): the narrow gaps are a tiny, very unevenly distributed fraction of a crop, so
    a crop containing one thin gap must not be drowned by a batch-mate full of open space.

    ``cldice`` -- soft clDice on the body (Shit et al. 2021), the topology term.  With
    ``p_body`` the soft body and ``t_body`` the label body, both restricted to ``m1``::

        tprec = <soft_skel(p_body), t_body> / <soft_skel(p_body), 1>     (precision: no invented mass)
        tsens = <soft_skel(t_body), p_body> / <soft_skel(t_body), 1>     (recall: no breaks)
        cldice_loss = 1 - 2 * tprec * tsens / (tprec + tsens)

    A predicted bridge, weld or extra sheet puts skeleton where the label body is not and drops
    ``tprec``; a missing or broken sheet leaves label skeleton uncovered and drops ``tsens``.  The
    sums are over the whole batch (masked); a batch whose label body is empty everywhere returns a
    constant 0 rather than the degenerate ``1 - 0/0``.

    In ``surface_mode="body"`` the primary target is already the body SDF
    (:func:`tsm.data.body_sdf`): call this with ``p_out = t_out = None`` and both the label body
    (``sdf_body > 0``) and the soft predicted body (``sigmoid(sdf_body / tau)``) come from the
    single channel -- every term below is otherwise identical.

    In ``surface_mode="sides"`` the body is a head of its own: pass the already-sigmoided body
    probability as ``p_body`` and the 0/1 label body as ``t_body`` (``tsm.data.body_mask_np``)
    and neither ``tau`` nor the SDF channels are consulted at all.  That mode can also supervise
    the gap *explicitly*: ``gap_dilate`` widens the label gap mask both terms use
    (:func:`narrow_gap_mask`) and, with ``p_gap`` (the 4th surface head channel) and a non-zero
    ``gap_class`` weight, the mask becomes a classification target of its own::

        gap_class * (BCE_with_logits(p_gap, gap_mask) + soft_dice(sigmoid(p_gap), gap_mask))

    both masked by ``valid == 1`` (times ``faces_weight``).

    Returned values are already multiplied by their weight, so the caller just sums them."""
    out: dict[str, torch.Tensor] = {}
    w_gap, w_cld = float(aux.get("gap", 0.0)), float(aux.get("cldice", 0.0))
    w_gc = float(aux.get("gap_class", 0.0)) if p_gap is not None else 0.0
    if w_gap <= 0 and w_cld <= 0 and w_gc <= 0:
        return out
    dil = int(aux.get("gap_dilate", 0))
    rad = int(aux.get("gap_radius", 3))
    body_t = body_mask(t_in, t_out, mv, mask=t_body)   # label side: hard, no gradients
    if w_gap > 0:
        tau = float(aux.get("gap_tau", 2.0))
        gap_m = narrow_gap_mask(body_t, m1, rad, dil)
        out["gap"] = w_gap * per_sample_mean(soft_body(p_in, p_out, tau, prob=p_body), gap_m)
    if w_gc > 0:
        # the same label gap mask, now as a 0/1 TARGET for a head channel of its own (sides
        # mode): BCE + soft Dice on valid == 1, exactly the pair the ink / body heads use.
        # Making the air an explicit class gives the net a gradient that says "this is a gap"
        # instead of only "do not put body here".
        gap_t = narrow_gap_mask(body_t, mv, rad, dil)
        out["gap_class"] = w_gc * (
            masked_mean(F.binary_cross_entropy_with_logits(p_gap, gap_t, reduction="none"), m1)
            + soft_dice(torch.sigmoid(p_gap), gap_t, m1))
    if w_cld > 0:
        iters = int(aux.get("cldice_iters", 5))
        tau = float(aux.get("gap_tau", 2.0))
        t_b = body_t * m1
        p_b = soft_body(p_in, p_out, tau, prob=p_body) * m1
        if float(t_b.sum()) <= 0.0:              # nothing to be topologically right about
            out["cldice"] = torch.zeros((), device=p_in.device, dtype=p_in.dtype)
        else:
            sk_p, sk_t = soft_skel(p_b, iters), soft_skel(t_b, iters)
            tprec = (sk_p * t_b).sum() / (sk_p.sum() + EPS)
            tsens = (sk_t * p_b).sum() / (sk_t.sum() + EPS)
            out["cldice"] = w_cld * (1.0 - 2.0 * tprec * tsens / (tprec + tsens + EPS))
    return out


def surface_loss(pred: torch.Tensor, sdf: torch.Tensor, valid: torch.Tensor,
                 weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    p_sdf, p_val = pred[:, 0:1].float(), pred[:, 1:2].float()
    m1 = _weighted((valid == 1).float(), weight)
    m_not2 = _weighted((valid != 2).float(), weight)
    w = 1.0 + SDF_WMAX * torch.exp(-(sdf ** 2) / (2.0 * SDF_SIGMA ** 2))
    l1 = masked_mean(w * (p_sdf - sdf).abs(), m1)
    bce = masked_mean(F.binary_cross_entropy_with_logits(p_val, (valid == 1).float(), reduction="none"), m_not2)
    dice = surface_band_dice(p_sdf, sdf, m1)
    return {"sdf_l1": l1, "valid_bce": bce, "band_dice": dice}


def faces_loss(pred: torch.Tensor, sdf: torch.Tensor, valid: torch.Tensor,
               weight: torch.Tensor | None = None,
               aux: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Two-face surface head: pred = [sdf_in, sdf_out, valid logit], target sdf = [sdf_in, sdf_out].

    Same terms as :func:`surface_loss` but per face (gaussian-weighted L1 + band Dice on each
    of the two zero sets) plus one BCE on the shared valid logit.

    ``weight`` is the optional per-voxel multiplier from the store's ``faces_weight`` channel
    (:data:`tsm.data.FACES_WEIGHT_CHANNEL`, written by ``dev/annot_import.py`` on human-corrected
    voxels).  It multiplies every term's mask, so ``weight is None`` -- and a weight that is 1
    everywhere -- reproduce the previous loss bit for bit.

    ``aux`` (``extra.train.surface_aux``) adds the optional zero-set terms of
    :func:`surface_aux_terms` per face (``shell_in``, ``crest_in``, ``far_in``, ...) and, once for
    both faces together, the body topology terms of :func:`surface_body_terms` (``gap``,
    ``cldice``).  With no ``aux`` -- or all its weights 0 -- nothing is computed and the returned
    dict is exactly the historical one."""
    if pred.shape[1] < 3 or sdf.shape[1] != 2:
        raise ValueError(f"faces mode needs a 3-channel surface head and a 2-channel target, got {tuple(pred.shape)} / {tuple(sdf.shape)}")
    mv = (valid == 1).float()
    m1 = _weighted(mv, weight)
    m_not2 = _weighted((valid != 2).float(), weight)
    do_aux = surface_aux_active(aux)
    out: dict[str, torch.Tensor] = {}
    for i, face in enumerate(("in", "out")):
        p_f, t_f = pred[:, i:i + 1].float(), sdf[:, i:i + 1]
        w = 1.0 + SDF_WMAX * torch.exp(-(t_f ** 2) / (2.0 * SDF_SIGMA ** 2))
        out[f"sdf_{face}_l1"] = masked_mean(w * (p_f - t_f).abs(), m1)
        out[f"band_dice_{face}"] = surface_band_dice(p_f, t_f, m1)
        if do_aux:
            out.update(surface_aux_terms(p_f, t_f, mv, m1, aux or {}, face))
    if do_aux:  # body terms: both faces at once, once per crop
        out.update(surface_body_terms(pred[:, 0:1].float(), pred[:, 1:2].float(),
                                      sdf[:, 0:1], sdf[:, 1:2], mv, m1, aux or {}))
    out["valid_bce"] = masked_mean(
        F.binary_cross_entropy_with_logits(pred[:, 2:3].float(), (valid == 1).float(), reduction="none"), m_not2)
    return out


def body_loss(pred: torch.Tensor, sdf: torch.Tensor, valid: torch.Tensor,
              weight: torch.Tensor | None = None,
              aux: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Orientation-free surface head (``surface_mode="body"``): pred = [sdf_body, valid logit],
    target = the single :func:`tsm.data.body_sdf` channel ``min(sdf_in, -sdf_out)``.

    The primary terms are exactly :func:`surface_loss` (gaussian-weighted L1 + band Dice on the
    zero set + the valid BCE on ``valid != 2``), so with no ``aux`` -- or all its weights 0 --
    the returned dict is bit-identical to the medial one.  ``aux``
    (``extra.train.surface_aux``) adds the zero-set terms of :func:`surface_aux_terms` with the
    body as its own "face" (``shell_body``, ``crest_body``, ``far_body``) and the single-target
    form of the body topology terms (``gap``, ``cldice``, :func:`surface_body_terms`)."""
    if pred.shape[1] < 2 or sdf.shape[1] != 1:
        raise ValueError(f"body mode needs a 2-channel surface head and a 1-channel target, got "
                         f"{tuple(pred.shape)} / {tuple(sdf.shape)}")
    out = surface_loss(pred, sdf, valid, weight)
    if surface_aux_active(aux):
        mv = (valid == 1).float()
        m1 = _weighted(mv, weight)
        p_sdf = pred[:, 0:1].float()
        out.update(surface_aux_terms(p_sdf, sdf, mv, m1, aux or {}, "body"))
        out.update(surface_body_terms(p_sdf, None, sdf, None, mv, m1, aux or {}))
    return out


def sides_loss(pred: torch.Tensor, sdf: torch.Tensor, valid: torch.Tensor,
               body: torch.Tensor, weight: torch.Tensor | None = None,
               aux: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Orientation-free surface head with the magnitude and the sign **split**
    (``surface_mode="sides"``): pred = [d_face (raw voxels), body logit, valid logit].

    Targets, both derived on the fly from the same two face labels (no relabelling):
    ``sdf`` = ``tsm.data.face_dist`` = ``min(|sdf_in|, |sdf_out|)``, the UNSIGNED distance to
    the nearest face of either kind, and ``body`` = ``tsm.data.body_mask_np`` =
    ``(sdf_in > 0) & (sdf_out < 0)`` as a 0/1 mask.

    Why the split: the signed body SDF of :func:`body_loss` flips sign twice per sheet and its
    interior magnitude is the (noisy, 20-70 voxel) half-thickness, so an L1 regression hedges
    near zero -- measured on the first 30k run, the predicted body SDF had a compressed range
    (+0.2 median inside sheets against labels of +12..+20) and thresholding at 0 fragmented the
    bodies.  Here neither half can hedge: the distance is non-negative and single-valued, and
    the side question is a classification with a Dice term that cares about the *set*.

    Terms: the gaussian-weighted L1 on ``d_face`` (:data:`SDF_WMAX` / :data:`SDF_SIGMA`; the
    target is >= 0, so the weight peaks exactly on the faces) + :func:`surface_band_dice` on
    ``exp(-d / tau)`` (sign-agnostic, so it works unchanged on an unsigned field) + BCE and
    soft Dice on the body logit (as in the ink head) + the valid BCE on ``valid != 2``.  Every
    term but the valid BCE is masked by ``valid == 1`` (times the optional ``faces_weight``).

    ``aux`` (``extra.train.surface_aux``) adds the zero-set terms of :func:`surface_aux_terms`
    on the face distance (``shell_face`` / ``crest_face`` / ``far_face``) and the body topology
    terms of :func:`surface_body_terms` (``gap``, ``cldice``) evaluated on the *predicted body
    probability* ``sigmoid(body logit)`` and the 0/1 label body.  Three more terms exist only
    here: ``gap_class`` (a 4th head channel classified against the label air gap,
    :func:`surface_body_terms`), ``gap_border_weight`` (a Ronneberger-style extra weight on the
    gap voxels that touch a body, :func:`gap_border_mask`, folded into the ``sdf_l1`` /
    ``body_bce`` masks) and ``eikonal`` (``(|grad d_pred| - 1)^2`` away from the faces, the
    label clip and the medial ridge, :func:`eikonal_term`).  All default to 0."""
    if pred.shape[1] < 3 or sdf.shape[1] != 1 or body.shape[1] != 1:
        raise ValueError(f"sides mode needs a 3-channel surface head and 1-channel d_face / body "
                         f"targets, got {tuple(pred.shape)} / {tuple(sdf.shape)} / {tuple(body.shape)}")
    p_d, p_b, p_v = pred[:, 0:1].float(), pred[:, 1:2].float(), pred[:, 2:3].float()
    aux = aux or {}
    w_gc = float(aux.get("gap_class", 0.0))
    if w_gc > 0 and pred.shape[1] < 4:
        raise ValueError(f"surface_aux.gap_class needs the 4-channel sides head "
                         f"[d_face, body, valid, gap], got {tuple(pred.shape)}")
    p_g = pred[:, 3:4].float() if pred.shape[1] > 3 else None
    mv = (valid == 1).float()
    m1 = _weighted(mv, weight)
    m_not2 = _weighted((valid != 2).float(), weight)
    w = 1.0 + SDF_WMAX * torch.exp(-(sdf ** 2) / (2.0 * SDF_SIGMA ** 2))
    # Ronneberger-style border weight map: the label-gap voxels that touch a body get
    # `gap_border_weight` added to the mask of the two dense terms (d_face L1, body BCE), so a
    # voxel in the thin air between two wraps counts for (1 + w0) of an ordinary voxel.  With
    # the weight at 0 (the default) `m_b` IS `m1`, so the arithmetic is bit-identical.
    w_border = float(aux.get("gap_border_weight", 0.0))
    m_b = m1
    if w_border > 0:
        body_t = body_mask(sdf, None, mv, mask=body)
        dil = int(aux.get("gap_dilate", 0))
        gap_t = narrow_gap_mask(body_t, mv, int(aux.get("gap_radius", 3)), dil)
        m_b = m1 * (1.0 + w_border * gap_border_mask(body_t, gap_t, dil))
    out: dict[str, torch.Tensor] = {
        "sdf_l1": masked_mean(w * (p_d - sdf).abs(), m_b),
        "band_dice": surface_band_dice(p_d, sdf, m1),
        "body_bce": masked_mean(F.binary_cross_entropy_with_logits(p_b, body, reduction="none"), m_b),
        "body_dice": soft_dice(torch.sigmoid(p_b), body, m1),
        "valid_bce": masked_mean(
            F.binary_cross_entropy_with_logits(p_v, (valid == 1).float(), reduction="none"), m_not2),
    }
    if surface_aux_active(aux):
        out.update(surface_aux_terms(p_d, sdf, mv, m1, aux, "face"))
        out.update(surface_body_terms(p_b, None, sdf, None, mv, m1, aux,
                                      p_body=torch.sigmoid(p_b), t_body=body,
                                      p_gap=p_g if w_gc > 0 else None))
        w_eik = float(aux.get("eikonal", 0.0))
        if w_eik > 0:
            out["eikonal"] = eikonal_term(p_d, sdf, m1, w_eik, float(aux.get("eikonal_min_d", 3.0)))
    return out


INK_POS_WEIGHT_MAX = 50.0


def ink_pos_weight(prob: torch.Tensor, valid: torch.Tensor, spec: Any = "auto") -> torch.Tensor | None:
    """BCE ``pos_weight`` for the ink head: ``"auto"`` = negatives/positives of this batch.

    Positives are the soft target mass ``sum(prob)`` over ``valid == 1`` voxels, negatives
    ``sum(1 - prob)``; the ratio is clamped to [1, ``INK_POS_WEIGHT_MAX``] so an all-negative
    crop cannot blow the loss up.  A number is used as given; ``None`` / 0 disables it."""
    if spec is None or spec == 0:
        return None
    m1 = (valid == 1).float()
    if spec == "auto":
        pos = (prob * m1).sum()
        neg = ((1.0 - prob) * m1).sum()
        return (neg / pos.clamp_min(1.0)).clamp(1.0, INK_POS_WEIGHT_MAX).detach()
    return torch.as_tensor(float(spec), device=prob.device, dtype=torch.float32)


def ink_loss(pred: torch.Tensor, prob: torch.Tensor, valid: torch.Tensor, pos_weight: Any = None) -> dict[str, torch.Tensor]:
    """BCE (optionally with ``pos_weight``, see :func:`ink_pos_weight`) + soft Dice, on ``valid == 1``."""
    p = pred[:, 0:1].float()
    m1 = (valid == 1).float()
    pw = pos_weight if isinstance(pos_weight, torch.Tensor) or pos_weight is None else ink_pos_weight(prob, valid, pos_weight)
    bce = masked_mean(F.binary_cross_entropy_with_logits(p, prob, reduction="none", pos_weight=pw), m1)
    dice = soft_dice(torch.sigmoid(p), prob, m1)
    return {"ink_bce": bce, "ink_dice": dice}


def fiber_loss(pred: torch.Tensor, prob: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """BCE + soft Dice of the 2-channel fibre head against the soft teacher probabilities
    ([vt, hz]), masked by ``fiber_valid == 1``.  Each channel is an independent sigmoid: the
    teacher's 4-way softmax also carries background and ink, so vt + hz <= 1 rather than == 1."""
    if pred.shape[1] < 2 or prob.shape[1] != 2:
        raise ValueError(f"fiber needs a 2-channel head and a 2-channel target, got {tuple(pred.shape)} / {tuple(prob.shape)}")
    m1 = (valid == 1).float()
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(("vt", "hz")):
        p_i, t_i = pred[:, i:i + 1].float(), prob[:, i:i + 1]
        out[f"{name}_bce"] = masked_mean(F.binary_cross_entropy_with_logits(p_i, t_i, reduction="none"), m1)
        out[f"{name}_dice"] = soft_dice(torch.sigmoid(p_i), t_i, m1)
    return out


def fiber_dir_loss(pred: torch.Tensor, d: torch.Tensor, strength: torch.Tensor, valid: torch.Tensor,
                   weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Axial-direction fibre loss (``fiber_mode="direction"``, tsm.fiber).

    ``pred`` is the 4-channel head ``[dz, dy, dx, strength logit]``, ``d`` the unit target
    direction ((z, y, x), sign irrelevant) and ``strength`` the target ``max(p_vt, p_hz)``.
    Two terms, both masked by ``fiber_valid == 1`` and by the per-voxel ``weight`` (the human
    bands carry ``fiber_band_weight``):

    * ``dir_cos`` = ``1 - (d^ . d)^2`` -- the **axial** cosine, invariant to a sign flip of
      either vector -- additionally weighted by the target strength, so a voxel the teacher
      calls neither vertical nor horizontal pulls on nothing;
    * ``str_bce`` = BCE of the strength logit against ``strength``.
    """
    if pred.shape[1] < 4 or d.shape[1] != 3:
        raise ValueError(f"fiber direction needs a 4-channel head and a 3-channel target, got "
                         f"{tuple(pred.shape)} / {tuple(d.shape)}")
    m1 = (valid == 1).float()
    w = m1 if weight is None else m1 * weight.float()
    dp = F.normalize(pred[:, 0:3].float(), dim=1, eps=EPS)
    dt = F.normalize(d.float(), dim=1, eps=EPS)
    cos2 = (dp * dt).sum(1, keepdim=True) ** 2
    dcos = masked_mean(1.0 - cos2, w * strength)
    sbce = masked_mean(F.binary_cross_entropy_with_logits(pred[:, 3:4].float(), strength.clamp(0, 1),
                                                          reduction="none"), w)
    return {"dir_cos": dcos, "str_bce": sbce}


def lsd_loss(pred: torch.Tensor, lsd: torch.Tensor, valid: torch.Tensor,
             clip: float = 20.0) -> dict[str, torch.Tensor]:  # clip = tsm.data.CLIP
    """Local Shape Descriptor loss (``extra.train.heads.lsd``, :func:`tsm.data.lsd_targets`).

    ``pred`` is the 5-channel head ``[off_z, off_y, off_x, thick, ncov logit]`` and ``lsd`` the
    matching target; everything is masked by ``lsd_valid == 1`` (the label body).  Masked L1 on
    all five channels, per-channel scale-normalised so the three terms are O(1) and comparable:

    * ``off_l1``   -- the offset to the sheet centre plane, in voxels / ``clip``;
    * ``thick_l1`` -- the windowed sheet thickness, in voxels / ``clip``;
    * ``ncov_l1``  -- the windowed normal incoherence; the head channel is a logit, so the
      comparison is against ``sigmoid(pred)``, which is also what inference writes.
    """
    if pred.shape[1] < 5 or lsd.shape[1] != 5:
        raise ValueError(f"lsd needs a 5-channel head and a 5-channel target, got "
                         f"{tuple(pred.shape)} / {tuple(lsd.shape)}")
    p, t = pred.float(), lsd.float()
    m = (valid == 1).float()
    inv = 1.0 / float(clip)
    off = masked_mean((p[:, 0:3] - t[:, 0:3]).abs().mean(dim=1, keepdim=True) * inv, m)
    thick = masked_mean((p[:, 3:4] - t[:, 3:4]).abs() * inv, m)
    ncov = masked_mean((torch.sigmoid(p[:, 4:5]) - t[:, 4:5]).abs(), m)
    return {"off_l1": off, "thick_l1": thick, "ncov_l1": ncov}


def winding_loss(pred: torch.Tensor, w: torch.Tensor, conf: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    pred = pred.float()
    m1 = (valid == 1).float()
    mc = m1 * conf
    sc = F.normalize(pred[:, 0:2], dim=1, eps=EPS)
    sct = F.normalize(w[:, 0:2], dim=1, eps=EPS)
    phase = masked_mean(1.0 - (sc * sct).sum(1, keepdim=True), mc)
    dens = masked_mean(F.huber_loss(10.0 * pred[:, 2:3], 10.0 * w[:, 2:3], reduction="none", delta=1.0), mc)
    n = F.normalize(pred[:, 3:6], dim=1, eps=EPS)
    nt = F.normalize(w[:, 3:6], dim=1, eps=EPS)
    normal = masked_mean(1.0 - (n * nt).sum(1, keepdim=True), mc)
    cbce = masked_mean(F.binary_cross_entropy_with_logits(pred[:, 6:7], (conf > 0.5).float(), reduction="none"), m1)
    return {"phase": phase, "density": dens, "normal": normal, "conf_bce": cbce}


def _pool_mask(m: torch.Tensor) -> torch.Tensor:
    return m[:, :, ::2, ::2, ::2]


def _pool_cont(x: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
    if m is None:
        return F.avg_pool3d(x, 2)
    num = F.avg_pool3d(x * m, 2)
    den = F.avg_pool3d(m, 2)
    return torch.where(den > 0, num / den.clamp_min(1e-6), F.avg_pool3d(x, 2))


def downsample_targets(t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sv = (t["surface_valid"] == 1).float()
    iv = (t["ink_valid"] == 1).float()
    wv = (t["winding_valid"] == 1).float()
    out = {
        "surface_sdf": _pool_cont(t["surface_sdf"], sv),
        "surface_valid": _pool_mask(t["surface_valid"]),
        "ink_prob": _pool_cont(t["ink_prob"], iv),
        "ink_valid": _pool_mask(t["ink_valid"]),
        "winding": _pool_cont(t["winding"], wv),
        "winding_conf": _pool_cont(t["winding_conf"], wv),
        "winding_valid": _pool_mask(t["winding_valid"]),
    }
    if "surface_weight" in t:
        out["surface_weight"] = _pool_cont(t["surface_weight"], sv)
    if "surface_body" in t:
        # surface_mode="sides": the 0/1 body occupancy pools like ink_prob (mask-aware average
        # of the mask), while surface_sdf (the face distance) pools like any other SDF above
        out["surface_body"] = _pool_cont(t["surface_body"], sv)
    if "lsd" in t:
        # the descriptors live on their own (body) mask: masked average, mask by decimation.
        # Like surface_sdf, the two length channels stay in FINE voxels at every level.
        lv = (t["lsd_valid"] == 1).float()
        out["lsd"] = _pool_cont(t["lsd"], lv)
        out["lsd_valid"] = _pool_mask(t["lsd_valid"])
    if "fiber_valid" in t:
        fv = (t["fiber_valid"] == 1).float()
        out["fiber_valid"] = _pool_mask(t["fiber_valid"])
        if "fiber_prob" in t:
            out["fiber_prob"] = _pool_cont(t["fiber_prob"], fv)
        if "fiber_dir" in t:
            # the direction is axial: pool it weighted by the strength (so a half-empty block
            # does not average a strong direction with nothing) and renormalise
            s_half = _pool_cont(t["fiber_str"], fv)
            d = _pool_cont(t["fiber_dir"] * t["fiber_str"], fv)
            out["fiber_dir"] = F.normalize(d, dim=1, eps=EPS)
            out["fiber_str"] = s_half
            out["fiber_weight"] = _pool_cont(t["fiber_weight"], fv)
    if "fiber_band" in t:
        out["fiber_band"] = _pool_mask(t["fiber_band"])
    return out


def compute_losses(
    out: dict[str, list[torch.Tensor]] | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    ds_weights: Sequence[float] = (1.0, 0.5),
    surface_mode: str = "medial",
    surface_aux: Mapping[str, Any] | None = None,
    ink_pos_weight_spec: Any = None,
    head_scale: Mapping[str, float] | None = None,
    head_losses: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss + per-term floats. ``out`` as returned by TSMNet (train or eval form).

    ``head_scale`` replaces ``weights[head]`` in the sum with the multiplier of a
    :class:`LossBalancer` (``None`` = plain ``loss_weights``, the historical path).
    ``head_losses``, when given, is filled with the *unweighted* per-head loss tensors
    (deep-supervision-summed) so the caller can measure per-head gradients."""
    weights = {**dict(TRAIN_DEFAULTS["loss_weights"]), **(weights or {})}
    outs = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in out.items()}
    n_lvl = len(next(iter(outs.values())))
    t: dict[str, torch.Tensor] = {k: batch[k] for k in ("surface_sdf", "surface_valid", "ink_prob", "ink_valid", "winding", "winding_conf", "winding_valid")}
    if "surface_weight" in batch:
        t["surface_weight"] = batch["surface_weight"]
    if "surface_body" in batch:
        t["surface_body"] = batch["surface_body"]
    if "lsd" in outs and "lsd" in batch:
        t["lsd"], t["lsd_valid"] = batch["lsd"], batch["lsd_valid"]
    if "fiber" in outs and "fiber_valid" in batch:
        t["fiber_valid"] = batch["fiber_valid"]
        for k in ("fiber_prob", "fiber_dir", "fiber_str", "fiber_weight"):
            if k in batch:
                t[k] = batch[k]
    total = torch.zeros((), device=t["surface_sdf"].device)
    pw = ink_pos_weight(t["ink_prob"], t["ink_valid"], ink_pos_weight_spec)
    parts: dict[str, float] = {}
    if pw is not None:
        parts["ink/pos_weight"] = float(pw)
    head_tot: dict[str, torch.Tensor] = {}
    for lvl in range(n_lvl):
        if lvl > 0:
            t = downsample_targets(t)
        lw = float(ds_weights[lvl]) if lvl < len(ds_weights) else 0.0
        terms: dict[str, dict[str, torch.Tensor]] = {}
        if "surface" in outs:
            if surface_mode == "faces":
                terms["surface"] = faces_loss(outs["surface"][lvl], t["surface_sdf"], t["surface_valid"],
                                              t.get("surface_weight"), aux=surface_aux)
            elif surface_mode == "body":
                terms["surface"] = body_loss(outs["surface"][lvl], t["surface_sdf"], t["surface_valid"],
                                             t.get("surface_weight"), aux=surface_aux)
            elif surface_mode == "sides":
                if "surface_body" not in t:
                    raise ValueError("surface_mode='sides' needs the 'surface_body' target "
                                     "(tsm.data.target_keys('sides'))")
                terms["surface"] = sides_loss(outs["surface"][lvl], t["surface_sdf"], t["surface_valid"],
                                              t["surface_body"], t.get("surface_weight"), aux=surface_aux)
            else:
                terms["surface"] = surface_loss(outs["surface"][lvl], t["surface_sdf"], t["surface_valid"],
                                                t.get("surface_weight"))
        if "ink" in outs:
            terms["ink"] = ink_loss(outs["ink"][lvl], t["ink_prob"], t["ink_valid"], pos_weight=pw)
        if "winding" in outs:
            terms["winding"] = winding_loss(outs["winding"][lvl], t["winding"], t["winding_conf"], t["winding_valid"])
        if "lsd" in outs and "lsd" in t:
            terms["lsd"] = lsd_loss(outs["lsd"][lvl], t["lsd"], t["lsd_valid"])
        if "fiber" in outs and "fiber_valid" in t:
            if "fiber_dir" in t:  # fiber_mode "direction": [dz, dy, dx, strength]
                terms["fiber"] = fiber_dir_loss(outs["fiber"][lvl], t["fiber_dir"], t["fiber_str"],
                                                t["fiber_valid"], t.get("fiber_weight"))
            else:
                terms["fiber"] = fiber_loss(outs["fiber"][lvl], t["fiber_prob"], t["fiber_valid"])
        for head, d in terms.items():
            s = sum(d.values())
            head_tot[head] = head_tot.get(head, 0.0) + lw * s
            if lvl == 0:
                for k, v in d.items():
                    parts[f"{head}/{k}"] = float(v.detach())
    for head, s in head_tot.items():
        w = weights.get(head, 1.0) if head_scale is None else float(head_scale.get(head, weights.get(head, 1.0)))
        total = total + w * s
        parts[head] = float(s.detach())
        if head_scale is not None:
            parts[f"balance/{head}"] = w                            # multiplier used this step
            parts[f"balance/{head}_contrib"] = w * parts[head]      # normalised contribution
    if head_losses is not None:
        head_losses.update(head_tot)
    parts["total"] = float(total.detach())
    return total, parts


# --------------------------------------------------------------------------- #
# per-head loss balancing (extra.train.balance)
# --------------------------------------------------------------------------- #
def _grad_norm(grads: Sequence[torch.Tensor | None]) -> float:
    """L2 norm over a list of gradients, ``None`` (unused parameter) counting as zero."""
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.detach().float().pow(2).sum())
    return math.sqrt(sq)


def head_grad_norms(head_losses: Mapping[str, torch.Tensor], head_params: Mapping[str, Sequence[torch.nn.Parameter]],
                    shared: Sequence[torch.nn.Parameter] = ()) -> dict[str, tuple[float, float]]:
    """``{head: (||d loss_h / d head params||, ||... / d (head + shared) params||)}``.

    One ``torch.autograd.grad`` per head with ``retain_graph=True``: the graph and the caller's
    ``.grad`` buffers are untouched, so this can be called before the normal ``backward()``
    without changing the optimizer step at all (only its cost).  The losses are the *unweighted*
    per-head totals, so the norms do not depend on the current multipliers."""
    shared = list(shared)
    out: dict[str, tuple[float, float]] = {}
    for h, loss in head_losses.items():
        own = [p for p in head_params.get(h, ()) if p.requires_grad]
        ps = own + [p for p in shared if p.requires_grad]
        if not ps or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            continue
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        out[h] = (_grad_norm(gs[: len(own)]), _grad_norm(gs))
    return out


class LossBalancer:
    """Optional per-head loss balancing on top of ``loss_weights`` (``extra.train.balance``).

    All heads share one total loss under one global gradient clip, so the head with the largest
    loss (or gradient) scale owns the step.  ``mode``:

    ``None``
        off: the multiplier of head *h* is exactly ``loss_weights[h]`` (byte-identical training).
    ``"scale"``
        running-scale normalisation.  An EMA (``decay``, per optimizer step) of the detached
        per-head loss magnitude ``ema_h`` turns the contribution into
        ``w_h * S * loss_h / (ema_h + eps)`` with ``S = sum_h w_h ema_h / sum_h w_h``.  Without
        ``S`` the total would have magnitude ``sum_h w_h`` instead of ``sum_h w_h ema_h``;
        rescaling by the sum of the loss weights keeps the *expected total* -- and hence the
        meaning of the learning rate and of the 1.0 gradient clip -- what it was.
        The first ``warmup`` steps run unbalanced (multipliers = ``loss_weights``) and only feed
        the EMA.  ``freeze=True`` (default) then freezes ``ema_h`` and ``S``: the multipliers
        become constants, so the run stays a fixed reweighting of the original objective and
        cannot drift.  ``freeze=False`` keeps updating the EMA forever, which tracks heads whose
        scale changes during training but is a feedback loop (a head that improves gets a larger
        multiplier), so the effective weights keep moving.
    ``"gradnorm_lite"``
        gradient-norm equalisation at the shared trunk.  Every ``every`` steps the per-head
        gradient norm ``r_h`` of the *unweighted* head loss is measured on the head parameters
        plus the finest decoder block (:func:`head_grad_norms`) -- one cheap backward per head
        instead of a full one -- and the multipliers are set to ``m_h = c * w_h / (r_h + eps)``
        with ``c = sum_h w_h r_h / sum_h w_h`` so the *post-multiplier* norms ``m_h r_h`` are
        proportional to ``loss_weights`` and their weighted sum is unchanged.  Updates are
        EMA-smoothed with ``decay ** every`` (the per-step decay compounded over the measurement
        interval).  The global clip stays in place.

    Heads with weight 0 (ablations) keep multiplier 0 and are excluded from the normalisation.
    """

    def __init__(self, mode: str | None, weights: Mapping[str, float], decay: float = 0.99, warmup: int = 200,
                 freeze: bool = True, every: int = 50, eps: float = 1e-8) -> None:
        if mode not in BALANCE_MODES:
            raise ValueError(f"balance mode must be one of {list(BALANCE_MODES)}, got {mode!r}")
        self.mode = mode
        self.weights = {k: float(v) for k, v in weights.items()}
        self.decay, self.warmup, self.freeze = float(decay), int(warmup), bool(freeze)
        self.every, self.eps = max(1, int(every)), float(eps)
        self.mult: dict[str, float] = dict(self.weights)
        self.ema: dict[str, float] = {}
        self.norms: dict[str, float] = {}
        self.steps = 0
        self.n_updates = 0

    @property
    def enabled(self) -> bool:
        return self.mode is not None

    def scales(self) -> dict[str, float] | None:
        """Multipliers for the next step (``None`` when off, so ``compute_losses`` is unchanged)."""
        return dict(self.mult) if self.enabled else None

    def _active(self) -> dict[str, float]:
        return {h: w for h, w in self.weights.items() if w != 0.0}

    def update(self, parts: Mapping[str, float]) -> None:
        """Feed the accumulated per-head losses of one optimizer step ("scale" mode)."""
        self.steps += 1
        if self.mode != "scale":
            return
        warm = self.steps <= self.warmup or not self.freeze
        for h in self._active():
            v = parts.get(h)
            if v is None or (h in self.ema and not warm):
                continue  # frozen after the warmup, except to seed a head never seen (resume)
            v = abs(float(v))
            self.ema[h] = v if h not in self.ema else self.decay * self.ema[h] + (1.0 - self.decay) * v
        if self.steps < self.warmup:
            self.mult = dict(self.weights)  # warmup: unbalanced, exactly as with balance=null
            return
        act = {h: w for h, w in self._active().items() if h in self.ema}
        wsum = sum(act.values())
        if not act or wsum <= 0:
            return
        s = sum(w * self.ema[h] for h, w in act.items()) / wsum
        self.mult = {h: (w * s / (self.ema[h] + self.eps) if h in act else w) for h, w in self.weights.items()}

    def needs_measure(self, step: int) -> bool:
        """True on the steps where "gradnorm_lite" measures the per-head gradient norms."""
        return self.mode == "gradnorm_lite" and (step == 1 or step % self.every == 0)

    def measure(self, norms: Mapping[str, float]) -> None:
        """Per-head gradient norms of the unweighted head losses -> new multipliers."""
        if self.mode != "gradnorm_lite":
            return
        self.norms = {h: float(v) for h, v in norms.items()}
        act = {h: self.weights[h] for h, v in self.norms.items()
               if self.weights.get(h, 0.0) != 0.0 and v > 0.0}
        wsum = sum(act.values())
        if not act or wsum <= 0:
            return
        c = sum(w * self.norms[h] for h, w in act.items()) / wsum
        self.n_updates += 1
        a = self.decay ** self.every
        for h, w in act.items():
            tgt = c * w / (self.norms[h] + self.eps)
            self.mult[h] = tgt if self.n_updates == 1 else a * self.mult.get(h, w) + (1.0 - a) * tgt

    def log(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        out = {f"balance/{h}": m for h, m in self.mult.items()}
        out.update({f"balance/{h}_ema": v for h, v in self.ema.items()})
        out.update({f"balance/{h}_gnorm": v for h, v in self.norms.items()})
        return out

    def describe(self) -> str:
        return " ".join(f"{h}={m:.3g}" for h, m in self.mult.items())


# --------------------------------------------------------------------------- #
# EMA / schedule / checkpoints
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(self.params, self.decay)
        torch._foreach_add_(self.params, src, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for p, e in zip(model.parameters(), self.params):
            p.copy_(e)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "params": [p.cpu() for p in self.params]}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = float(sd["decay"])
        # the shadow is a flat list in model.parameters() order, so a checkpoint whose head set
        # differs (an optional head added / removed) would zip out of alignment -- refuse it.
        if len(sd["params"]) != len(self.params):
            raise ValueError(f"EMA shadow has {len(sd['params'])} tensors, this model has {len(self.params)} "
                             f"(the head set changed); load the raw weights instead")
        for p, e in zip(self.params, sd["params"]):
            if tuple(p.shape) != tuple(e.shape):
                raise ValueError(f"EMA shadow shape {tuple(e.shape)} != model parameter {tuple(p.shape)}")
            p.copy_(e.to(p.device))


def lr_lambda(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path: str, model: torch.nn.Module, ema: EMA | None, opt: torch.optim.Optimizer, sched: Any, step: int, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    """``extra`` maps a name to an object with ``state_dict()`` (training-only state, e.g. feature projectors)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {
        "model": _unwrap(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "step": int(step),
        "config": config,
        "extra": {k: v.state_dict() for k, v in (extra or {}).items()},
    }
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: torch.nn.Module, ema: EMA | None = None, opt: torch.optim.Optimizer | None = None, sched: Any = None, device: str = "cpu", extra: dict[str, Any] | None = None) -> int:
    sd = torch.load(path, map_location=device, weights_only=False)
    load_student_state(_unwrap(model), sd["model"], what=f"checkpoint {os.path.basename(path)}")
    for k, v in (extra or {}).items():
        if k in sd.get("extra", {}):
            v.load_state_dict(sd["extra"][k])
        else:
            print(f"[tsm] WARNING checkpoint has no extra state {k!r}; keeping fresh init", flush=True)
    if ema is not None and sd.get("ema") is not None:
        try:
            ema.load_state_dict(sd["ema"])
        except ValueError as exc:  # head set changed: keep the EMA seeded from the loaded weights
            print(f"[tsm] WARNING checkpoint EMA not loaded ({exc}); the EMA restarts from the raw weights",
                  flush=True)
    if opt is not None and sd.get("opt") is not None:
        opt.load_state_dict(sd["opt"])
    if sched is not None and sd.get("sched") is not None:
        sched.load_state_dict(sd["sched"])
    return int(sd["step"])


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #
class RangeSampler(Sampler[int]):
    def __init__(self, start: int, end: int) -> None:
        self.start, self.end = int(start), int(end)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return max(0, self.end - self.start)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def model_input(batch: dict[str, torch.Tensor], channels_last: bool = False) -> torch.Tensor:
    """[ct01, scale, (radial)] -> [zscore(ct), scale, (radial)] (z-scored on the device, after augmentation)."""
    inp = batch["input"]
    x = torch.cat([normalize_ct(inp[:, 0]).unsqueeze(1), inp[:, 1:]], dim=1)
    return to_channels_last(x) if channels_last else x


#: the tilted constant scroll axis of ``synthetic_batch`` (z, y, x), normalised on use
SYNTH_AXIS = (1.0, 0.3, -0.2)


def synthetic_batch(batch: int, patch: int, device: str = "cpu", seed: int = 0, surface_mode: str = "medial",
                    input_radial: bool = False, fiber: bool = False, fiber_mode: str = "class",
                    fiber_band: bool = False, input_axis: bool = False,
                    lsd: bool = False) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sides = surface_mode == "sides"
    surface_ch = 2 if surface_mode == "faces" else 1
    P = patch
    r = lambda c: torch.rand((batch, c, P, P, P), generator=g)  # noqa: E731
    inp = [r(1), torch.zeros(batch, 1, P, P, P)]
    if input_radial:
        rad = torch.cat([torch.zeros(batch, 1, P, P, P), r(2) * 2 - 1], 1)
        inp.append(F.normalize(rad, dim=1, eps=1e-6))
    if input_axis:
        # a *tilted* constant axis (not (1, 0, 0)): a synthetic batch must not accidentally
        # satisfy "the scroll axis is exactly volume z", which is what the real tangent breaks
        ax = torch.zeros(batch, 3, P, P, P)
        for i, v in enumerate(SYNTH_AXIS):
            ax[:, i] = v
        inp.append(F.normalize(ax, dim=1, eps=1e-6))
    b = {
        "input": torch.cat(inp, 1),
        "voxel_um": torch.full((batch,), 2.4),
        "origin_zyx": torch.zeros(batch, 3, dtype=torch.int64),
        # "sides": surface_sdf is the UNSIGNED face distance, so it is >= 0 by construction
        "surface_sdf": (r(surface_ch) * 20.0) if sides else ((r(surface_ch) * 2 - 1) * 20.0),
        "surface_valid": (r(1) * 3).floor(),
        "ink_prob": r(1),
        "ink_valid": (r(1) > 0.3).float(),
        "winding": torch.cat([r(2) * 2 - 1, r(1) * 0.2, F.normalize(r(3) * 2 - 1, dim=1)], 1),
        "winding_conf": r(1),
        "winding_valid": (r(1) * 3).floor(),
    }
    if sides:
        b["surface_body"] = (r(1) > 0.5).float()
    if lsd:
        # [off_z, off_y, off_x] (voxels, |off| <= clip), thickness (voxels) and the [0, 1]
        # normal incoherence, on the label-body mask
        b["lsd"] = torch.cat([(r(3) * 2 - 1) * 10.0, r(1) * 40.0, r(1)], 1)
        b["lsd_valid"] = (r(1) > 0.5).float()
    if fiber:
        b["fiber_valid"] = (r(1) > 0.3).float()
        if fiber_mode == "direction":
            b["fiber_dir"] = F.normalize(r(3) * 2 - 1, dim=1, eps=1e-6)
            b["fiber_str"] = r(1)
            b["fiber_weight"] = torch.ones(batch, 1, P, P, P)
        else:
            b["fiber_prob"] = r(2)
        if fiber_band:
            b["fiber_band"] = (r(1) * 4).floor()
    return to_device(b, device)


# --------------------------------------------------------------------------- #
# feature distillation (training only)
# --------------------------------------------------------------------------- #
class FeatDistiller:
    """Frozen teacher encoders + trainable projectors; ``__call__`` -> (weighted loss, parts).

    ``loader(name, models_dir, device=, dtype=)`` must return ``(model, spec)``
    like :func:`tsm.teachers.load_teacher` (tests monkeypatch it with a tiny
    random encoder).  Only the encoder is kept; decoders are dropped after
    loading.  Teachers run sequentially under ``no_grad`` + bf16 autocast and
    their feature maps are freed before the next one.
    """

    def __init__(self, fd: dict[str, Any], widths: Sequence[int], device: str, loader: Any = None,
                 out_dir: str | None = None, dino_cache: Any = None,
                 feat_caches: Mapping[str, Mapping[int, Any]] | None = None) -> None:
        from tsm.teachers import DEFAULT_MODELS_DIR, encoder_channels, encoder_of, encoder_strides, load_teacher

        self.fd = fd
        self.device = device
        self.kind = str(fd["loss"])
        self.weight = float(fd["weight"])
        self.every = int(fd["every"])
        self.pairs = [(int(a), int(b)) for a, b in fd["pairs"]]
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        loader = loader or load_teacher
        models_dir = fd.get("models_dir") or DEFAULT_MODELS_DIR
        student_ch = {k: int(c) for k, c in enumerate(widths)}
        for sk, _ in self.pairs:
            if sk not in student_ch:
                raise ValueError(f"feat_distill pair student stage {sk} outside widths {list(widths)}")
        self.encoders: dict[str, torch.nn.Module] = {}
        self.normalizers: dict[str, Any] = {}
        self.channels: dict[str, list[int]] = {}
        self.strides: dict[str, list[int]] = {}
        self.proj = torch.nn.ModuleDict()
        self.dino = None
        if "dino" in fd["teachers"]:
            from tsm.dino import DinoFeatSource, DinoTokenCache

            path = fd.get("dino_cache") or os.path.join(out_dir or ".", "dino", "tokens.zarr")
            cache = dino_cache if dino_cache is not None else DinoTokenCache(path)
            sk = int(fd["dino_stage"])
            if sk not in student_ch:
                raise ValueError(f"feat_distill dino_stage {sk} outside widths {list(widths)}")
            self.dino = DinoFeatSource(cache, student_ch[sk], stage=sk).to(device)
            self.proj["dino"] = self.dino
        self.cached: dict[str, Any] = {}
        feats_dir = fd.get("feats_dir") or os.path.join(out_dir or ".", "feats")
        for name in [n for n in fd["teachers"] if n != "dino"]:
            if str(fd["sources"].get(name, fd["source"])) == "cached":
                from tsm.feats import CachedFeatSource, open_caches

                stages = sorted({tk for _, tk in self.pairs})
                caches = dict((feat_caches or {}).get(name) or open_caches(feats_dir, name, stages))
                src = CachedFeatSource(name, caches, self.pairs, student_ch, kind=self.kind).to(device)
                self.cached[name] = src
                self.proj[name] = src
                self.channels[name] = [int(caches[k].channels) if k in caches else 0
                                       for k in range(max(stages) + 1)]
                self.strides[name] = [int(caches[k].scale) if k in caches else 0 for k in range(max(stages) + 1)]
                continue
            model, spec = loader(name, models_dir, device=device, dtype=self.dtype)
            enc = encoder_of(model).eval()
            enc.requires_grad_(False)
            del model
            tch, tst = encoder_channels(enc), encoder_strides(enc)
            for _, tk in self.pairs:
                if tk >= len(tch):
                    raise ValueError(f"feat_distill pair teacher stage {tk} outside {name} encoder ({len(tch)} stages)")
            self.encoders[name] = enc
            self.normalizers[name] = spec.normalizer
            self.channels[name], self.strides[name] = tch, tst
            self.proj[name] = FeatureProjector(self.pairs, student_ch, tch).to(device)
        free_cuda()

    def describe(self, widths: Sequence[int]) -> list[str]:
        lines = []
        if self.dino is not None:
            d = self.dino
            lines.append(f"dino: student enc{d.stage} ({widths[d.stage]} ch, /{2 ** d.stage}) -> cached tokens "
                         f"({d.out_ch} ch, /{d.cache.scale}) from {d.cache.path}")
        for name, src in self.cached.items():
            for sk, tk in self.pairs:
                c = src.caches[tk]
                lines.append(f"{name}: student enc{sk} ({widths[sk]} ch, /{2 ** sk}) -> CACHED stage {tk} "
                             f"({c.channels} ch, /{c.scale}, pca={c.pca}) from {c.path}")
        for name in self.encoders:
            for sk, tk in self.pairs:
                lines.append(f"{name}: student enc{sk} ({widths[sk]} ch, /{2 ** sk}) -> teacher stage {tk} "
                             f"({self.channels[name][tk]} ch, /{self.strides[name][tk]})")
        return lines

    def parameters(self):
        return self.proj.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.proj.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.proj.load_state_dict(sd)

    def active(self, step: int) -> bool:
        return step % self.every == 0

    def bytes_estimate(self, patch: int, batch: int) -> list[tuple[str, int]]:
        from tsm.teachers import encoder_bytes_estimate

        rows = []
        for name, enc in self.encoders.items():
            e = encoder_bytes_estimate(enc, patch, batch, dtype_bytes=2 if self.dtype == torch.bfloat16 else 4)
            rows.append((f"teacher {name} encoder weights ({'bf16' if self.dtype == torch.bfloat16 else 'fp32'})", e["params"]))
            rows.append((f"teacher {name} encoder activations {batch}x{patch}^3 (transient)", e["activations"]))
        if self.dino is not None:
            n = self.dino.out_ch * (patch // self.dino.cache.scale) ** 3 * batch * 4
            rows.append((f"dino cached tokens {batch}x{patch // self.dino.cache.scale}^3 (transient)", n))
        for name, src in self.cached.items():
            n = sum(int(c.channels) * (patch // int(c.scale)) ** 3 for c in src.caches.values()) * batch * 4
            rows.append((f"cached {name} features {batch}x{patch}^3 crop (transient)", n))
        rows.append(("feature projectors + grad + adam", sum(p.numel() for p in self.proj.parameters()) * 4 * 4))
        return rows

    def teacher_input(self, name: str, ct01: torch.Tensor) -> torch.Tensor:
        """Per-crop teacher normalization of the (B, Z, Y, X) CT in [0, 1] (u8 scale restored)."""
        norm = self.normalizers[name]
        u8 = (ct01.float() * 255.0).round_()
        return torch.stack([norm(u8[i]) for i in range(u8.shape[0])]).unsqueeze(1)

    def __call__(self, feats: dict[str, torch.Tensor], ct01: torch.Tensor, batch: Mapping[str, Any] | None = None,
                 spatial: Sequence[Any] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        from tsm.teachers import encoder_features

        parts: dict[str, float] = {}
        total = torch.zeros((), device=ct01.device)
        n_terms = len(self.encoders) + len(self.cached) + (1 if self.dino is not None else 0)
        if self.dino is not None:
            if batch is None:
                raise ValueError("feat_distill 'dino' needs the batch (origin_zyx) -- pass batch=")
            loss, dparts = self.dino(feats, batch, spatial)
            parts.update(dparts)
            total = total + loss
        for name, src in self.cached.items():
            if batch is None:
                raise ValueError(f"feat_distill cached teacher {name!r} needs the batch (origin_zyx) -- pass batch=")
            loss, cparts = src(feats, batch, spatial)
            parts.update(cparts)
            total = total + loss
        for name, enc in self.encoders.items():
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                t = encoder_features(enc, self.teacher_input(name, ct01))
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                proj = self.proj[name](feats)
            loss = feature_distill_loss(proj, t, self.kind)
            del t, proj
            parts[f"feat_{name}"] = float(loss.detach())
            total = total + loss
        total = self.weight * total / max(1, n_terms)
        parts["feat"] = float(total.detach())
        return total, parts


def _conv_macs(module: torch.nn.Module, fn: Any) -> int:
    """Multiply-accumulates of all Conv3d / ConvTranspose3d layers during ``fn()`` (forward hooks)."""
    macs = [0]

    def hook(m, inp, out):
        k = math.prod(m.kernel_size)
        macs[0] += int(out.numel() // out.shape[0]) * out.shape[0] * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d))]
    try:
        fn()
    finally:
        for h in hs:
            h.remove()
    return macs[0]


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def _vram_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / MB, torch.cuda.max_memory_allocated() / MB


def _open_ct(cfg: RunCfg, region_start: Sequence[int], probe: int = 32,
             volume: VolumeCfg | None = None, region: RegionCfg | None = None):
    """Zero-arg level-0 reader factory: local cache, else primary URL, else alt.

    ``volume`` / ``region`` override the top-level ones (a per-store entry of
    ``extra.train.stores`` may name its own scroll volume and the region its cache must cover)."""
    from tsm.volume import VolumeReader, open_cached_reader

    vol = volume if volume is not None else cfg.volume
    reg = region if region is not None else cfg.region
    z0, y0, x0 = (int(v) for v in region_start)

    def make() -> VolumeReader:
        last: Exception | None = None
        cached = open_cached_reader(vol.cache_root, vol.url, vol.level, vol.voxel_um, reg, cfg.budget)
        if cached is not None:
            return cached
        for url in (vol.url, vol.alt_url):
            if not url:
                continue
            try:
                r = VolumeReader(url, vol.level, vol.voxel_um, cfg.budget)
                if r.read(z0, z0 + probe, y0, y0 + probe, x0, x0 + probe).any():
                    return r
            except Exception as exc:
                last = exc
        raise RuntimeError(f"no CT source with data at level {vol.level} ({last})")

    return make


def _box_hits(origins: np.ndarray, patch: int, boxes: Sequence[Sequence[int]],
              store_origin_zyx: Sequence[int]) -> np.ndarray:
    """Bool mask: which ``patch``-cube crops (local origins) intersect any global holdout box.

    ``boxes`` are ``[z0, y0, x0, dz, dy, dx]`` in scroll (level-0) voxels; crop origins are local to
    the fine store, so the box is shifted by ``store_origin_zyx`` before testing."""
    o = np.asarray(origins, np.int64).reshape(-1, 3)
    hit = np.zeros(len(o), bool)
    if not len(o):
        return hit
    off = np.asarray([int(v) for v in store_origin_zyx], np.int64)
    for b in boxes:
        bb = np.asarray([int(v) for v in b], np.int64)
        lo = bb[:3] - off
        hi = lo + bb[3:]
        hit |= ((o < hi[None, :]) & (o + int(patch) > lo[None, :])).all(axis=1)
    return hit


def _store_dataset(cfg: RunCfg, opts: dict[str, Any], entry: dict[str, Any], augment: bool,
                   length: int, hold_all: bool):
    """One store of ``extra.train.stores`` -> (CropDataset, train origins, holdout origins).

    The entry is self-contained: its own fine/coarse stores, umbilicus ``axis`` (radial input
    channels), ``voxel_um`` and -- optionally -- its own CT ``volume`` / ``region`` (default: the
    top-level volume and the fine store's own extent).  ``hold_all`` (``holdout_stores``) makes the
    whole store an evaluation set: it trains on nothing and every origin is held out."""
    from tsm.data import CropDataset, LabelStore, build_origins, split_holdout

    name = entry["name"]
    fine_path = entry["fine_store"]
    if not os.path.exists(os.path.join(fine_path, "zarr.json")):
        raise FileNotFoundError(f"store {name!r}: no fine label store at {fine_path}")
    fine = LabelStore(fine_path)
    coarse_path = entry["coarse_store"]
    coarse = None
    if coarse_path:
        if os.path.exists(os.path.join(coarse_path, "zarr.json")):
            coarse = LabelStore(coarse_path)
        else:
            print(f"[tsm] WARNING store {name!r}: coarse store {coarse_path} missing; "
                  f"winding head unsupervised on it", flush=True)
    vol = _volume(entry["volume"]) if entry["volume"] else cfg.volume
    vu = float(entry["voxel_um"]) if entry["voxel_um"] is not None else fine.voxel_um
    if abs(fine.voxel_um - vu) > 1e-6:
        raise ValueError(f"store {name!r}: voxel_um {vu} != fine store pitch {fine.voxel_um}")
    if abs(vu - vol.voxel_um) > 1e-6:
        raise ValueError(f"store {name!r}: voxel_um {vu} != volume pitch {vol.voxel_um}")
    region = _region(entry["region"]) if entry["region"] else RegionCfg(start_zyx=fine.origin_zyx,
                                                                       size_zyx=fine.shape_zyx)
    P = int(opts["patch"])
    smode = str(opts.get("surface_mode", "medial"))
    valid_ch = "faces_valid" if smode == "faces" else "sdf_valid"
    origins = build_origins(fine, valid_ch, P, int(opts["stride"]), float(opts["min_valid_frac"]))
    boxes = entry.get("holdout_boxes_zyx") or []
    n_box = 0
    if hold_all:
        train_o, hold_o = np.zeros((0, 3), np.int32), origins
    else:
        train_o, hold_o = split_holdout(origins, P, opts.get("holdout_origins"))
        if boxes:
            hit = _box_hits(train_o, P, boxes, fine.origin_zyx)
            n_box = int(hit.sum())
            hold_o = np.concatenate([np.asarray(hold_o, np.int32).reshape(-1, 3),
                                     np.asarray(train_o, np.int32).reshape(-1, 3)[hit]], axis=0)
            train_o = np.asarray(train_o, np.int32).reshape(-1, 3)[~hit]
    extra_msg = f", {n_box} in holdout box" if boxes else ""
    print(f"[tsm] store {name!r}: fine {fine.path} shape={fine.shape_zyx} origin={fine.origin_zyx} "
          f"voxel_um={fine.voxel_um}; {len(train_o)} train / {len(hold_o)} holdout origins "
          f"(of {len(origins)}, patch {P}{extra_msg})", flush=True)
    ds = CropDataset(
        _open_ct(cfg, fine.origin_zyx, volume=vol, region=region), fine, coarse, patch=P,
        seed=int(opts["seed"]), length=length, stride=int(opts["stride"]),
        min_frac=float(opts["min_valid_frac"]), augment=augment,
        # a fully held-out store still needs a non-empty origin list to load its crops
        origins=train_o if len(train_o) else origins, surface_mode=smode,
        input_radial=bool(opts.get("input_radial", True)), axis=entry["axis"] or opts.get("axis_path"),
        fiber=bool((opts.get("heads") or {}).get("fiber", False)) or None,
        winding_source=str(opts.get("winding_source", "coarse")),
        fiber_mode=str(opts.get("fiber_mode", "class")),
        fiber_band_weight=float(opts.get("fiber_band_weight", 5.0)),
        input_axis=bool(opts.get("input_axis", True)),
        core_radius_vox=float(opts.get("core_radius_vox", 0.0) or 0.0),
        axis_tangent=bool(opts.get("axis_tangent", True)),
        body_ct_gate=opts.get("body_ct_gate"),
        lsd=bool((opts.get("heads") or {}).get("lsd", False)),
        lsd_sigma=float(opts.get("lsd_sigma", 6.0)),
        volcomp=volcomp_spec(opts),
    )
    return ds, train_o, hold_o


def build_multi_dataset(cfg: RunCfg, opts: dict[str, Any], augment: bool = True):
    """``MultiStoreDataset`` over ``extra.train.stores`` (several regions / scrolls)."""
    from tsm.data import MultiStoreDataset

    entries = opts["stores"]
    hold_names = set(opts.get("holdout_stores") or [])
    length = int(opts["steps"]) * int(opts["batch"]) * int(opts["accum"]) + 1
    dss, trains, holds, weights = [], [], [], []
    for e in entries:
        hold_all = e["name"] in hold_names
        d, tr, ho = _store_dataset(cfg, opts, e, augment, length, hold_all)
        dss.append(d)
        trains.append(tr)
        holds.append(ho)
        weights.append(0.0 if hold_all else e["weight"])
    if sum(len(t) for t in trains) == 0:
        raise ValueError("stores / holdout_origins leave no training origins")
    if any(w is None for w in weights):
        weights = [float(len(t)) if w is None else float(w) for w, t in zip(weights, trains)]
    ds = MultiStoreDataset(dss, [e["name"] for e in entries], weights=weights, seed=int(opts["seed"]),
                           length=length, holdout=holds, train_origins=trains)
    if bool(opts.get("eval_holdout", True)) and len(ds.holdout_origins) == 0 and (
            opts.get("holdout_origins") or hold_names):
        raise ValueError("holdout_origins / holdout_stores select 0 crop origins, so the held-out "
                         "evaluation cannot run; use another split or set eval_holdout=false")
    print(f"[tsm] {len(ds.datasets)} stores: " + ", ".join(
        f"{n} w={w:g} p={p:.3f} ({len(t)} train, {len(h)} holdout)"
        for n, w, p, t, h in zip(ds.names, ds.weights, ds.p, trains, holds)), flush=True)
    print(f"[tsm] {len(ds.origins)} crop origins (patch {ds.patch}, stride {opts['stride']}), "
          f"{len(ds.holdout_origins)} held out; winding_source={ds.winding_source}", flush=True)
    return ds


def build_dataset(cfg: RunCfg, opts: dict[str, Any], augment: bool = True):
    """CropDataset over the label stores; ``augment`` enables the v1 CPU path only.

    ``opts["holdout_origins"]`` splits the crop origins (``data.split_holdout``): the
    dataset samples the training part and carries the rest as ``ds.holdout_origins``.
    ``opts["stores"]`` (several regions / scrolls) instead builds a ``MultiStoreDataset``."""
    from tsm.data import CropDataset, LabelStore, build_origins, split_holdout

    if opts.get("stores"):
        return build_multi_dataset(cfg, opts, augment)

    fine_path = opts.get("fine_store") or os.path.join(cfg.out_dir, "labels", "fine.zarr")
    coarse_path = opts.get("coarse_store") or os.path.join(cfg.out_dir, "labels", "coarse.zarr")
    if not os.path.exists(os.path.join(fine_path, "zarr.json")):
        raise FileNotFoundError(f"no fine label store at {fine_path}")
    fine = LabelStore(fine_path)
    coarse = LabelStore(coarse_path) if os.path.exists(os.path.join(coarse_path, "zarr.json")) else None
    if coarse is None:
        print(f"[tsm] WARNING coarse store {coarse_path} missing; winding head unsupervised", flush=True)
    for name, st in (("fine", fine), ("coarse", coarse)):
        if st is not None:
            print(f"[tsm] {name} store {st.path} shape={st.shape_zyx} origin={st.origin_zyx} voxel_um={st.voxel_um}", flush=True)
    if abs(fine.voxel_um - cfg.volume.voxel_um) > 1e-6:
        raise ValueError(f"fine store pitch {fine.voxel_um} != volume pitch {cfg.volume.voxel_um}")
    P = int(opts["patch"])
    smode = str(opts.get("surface_mode", "medial"))
    valid_ch = "faces_valid" if smode == "faces" else "sdf_valid"
    origins = build_origins(fine, valid_ch, P, int(opts["stride"]), float(opts["min_valid_frac"]))
    spec = opts.get("holdout_origins")
    train_o, hold_o = split_holdout(origins, P, spec)
    print(f"[tsm] holdout split {spec}: {len(train_o)} train / {len(hold_o)} holdout origins "
          f"(of {len(origins)}, patch {P})", flush=True)
    if len(train_o) == 0:
        raise ValueError("holdout_origins leaves no training origins")
    if spec and len(hold_o) == 0 and bool(opts.get("eval_holdout", True)):
        raise ValueError(
            f"holdout_origins {spec} selects 0 of {len(origins)} crop origins, so the held-out evaluation "
            f"cannot run.  The origin grid spans z={sorted(set(origins[:, 0].tolist()))[:6]}... "
            f"y[{origins[:, 1].min()}..{origins[:, 1].max()}] x[{origins[:, 2].min()}..{origins[:, 2].max()}]; "
            f'use {{"y_frac": f}} or {{"yx_frac": f}} for a thin region, or set eval_holdout=false.')
    ds = CropDataset(
        _open_ct(cfg, fine.origin_zyx), fine, coarse, patch=P, seed=int(opts["seed"]),
        length=int(opts["steps"]) * int(opts["batch"]) * int(opts["accum"]) + 1, stride=int(opts["stride"]),
        min_frac=float(opts["min_valid_frac"]), augment=augment, origins=train_o, surface_mode=smode,
        input_radial=bool(opts.get("input_radial", True)), axis=opts.get("axis_path"),
        fiber=bool((opts.get("heads") or {}).get("fiber", False)) or None,
        winding_source=str(opts.get("winding_source", "coarse")),
        fiber_mode=str(opts.get("fiber_mode", "class")),
        fiber_band_weight=float(opts.get("fiber_band_weight", 5.0)),
        input_axis=bool(opts.get("input_axis", True)),
        core_radius_vox=float(opts.get("core_radius_vox", 0.0) or 0.0),
        axis_tangent=bool(opts.get("axis_tangent", True)),
        body_ct_gate=opts.get("body_ct_gate"),
        lsd=bool((opts.get("heads") or {}).get("lsd", False)),
        lsd_sigma=float(opts.get("lsd_sigma", 6.0)),
        volcomp=volcomp_spec(opts),
    )
    ds.holdout_origins = hold_o
    print(f"[tsm] {len(ds.origins)} crop origins (patch {ds.patch}, stride {opts['stride']}, coarse factor {ds.factor}), "
          f"{len(hold_o)} held out; winding_source={ds.winding_source}", flush=True)
    return ds


def _overfit_batch(ds, batch: int) -> dict[str, torch.Tensor]:
    from torch.utils.data.dataloader import default_collate

    o = ds.origins
    return default_collate([ds.to_tensors(ds.load(o[(len(o) // 2 + i) % len(o)])) for i in range(batch)])


def train_step_batches(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for b in loader:
            yield b


def run_train(cfg: RunCfg, dry_run: bool = False, resume: bool = False, force: bool = False) -> dict[str, Any]:
    opts = train_opts(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_guard(cfg.budget)
    torch.manual_seed(int(opts["seed"]))
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join(cfg.out_dir, "train")
    P, B, A = int(opts["patch"]), int(opts["batch"]), int(opts["accum"])
    widths = tuple(int(w) for w in opts["widths"])

    cl = bool(device == "cuda") if opts["channels_last"] is None else bool(opts["channels_last"])
    smode = str(opts["surface_mode"])
    do_fiber = bool(opts["heads"]["fiber"])
    fmode = str(opts["fiber_mode"])
    # extra.train.surface_aux.gap_class > 0 widens the sides surface head by one channel
    gap_cls = float((opts.get("surface_aux") or {}).get("gap_class", 0.0)) > 0
    do_lsd = bool(opts["heads"]["lsd"])
    heads = heads_for(smode, do_fiber, fmode, gap_cls, do_lsd)
    radial = bool(opts["input_radial"])
    ax_in = bool(opts["input_axis"])
    ax_tan = bool(opts["axis_tangent"])
    n_in = in_channels(radial, ax_in)
    model = build_model(widths=widths, in_ch=n_in, compile=bool(opts["compile"]), act_ckpt=int(opts["act_ckpt"]),
                        channels_last=cl, surface_mode=smode, body_stride=int(opts["body_stride"]),
                        fullres_width=int(opts["fullres_width"]), norm=str(opts["norm"]), fiber=do_fiber,
                        fiber_mode=fmode, gap_class=gap_cls, lsd=do_lsd).to(device)
    n_params = _unwrap(model).num_params()
    print(f"[tsm] TSMNet widths={widths} params={n_params / 1e6:.2f}M surface_mode={smode} heads={heads} "
          f"in_ch={n_in} {input_channels(radial, ax_in)} device={device} channels_last={cl} "
          f"body_stride={opts['body_stride']} fullres_width={opts['fullres_width']} norm={opts['norm']} "
          f"rf_radius={_unwrap(model).receptive_field_radius()}", flush=True)
    if radial:
        print(f"[tsm] input_radial: outward radial field from axis {opts['axis_path']}", flush=True)
    if ax_in:
        print(f"[tsm] input_axis: the scroll-axis direction (a_z, a_y, a_x) as input channels "
              f"({'local umbilicus tangent' if ax_tan else 'constant (1, 0, 0)'})", flush=True)
    if do_lsd:
        from tsm.data import LSD_CHANNELS

        print(f"[tsm] heads.lsd: Local Shape Descriptors {LSD_CHANNELS} "
              f"(gaussian window sigma {float(opts['lsd_sigma']):g} voxels)", flush=True)
    if do_fiber:
        print(f"[tsm] fiber_mode={fmode}" + ("" if fmode == "direction" else
              f" (class swap under axis-moving transforms: {bool(opts['fiber_swap_fix'])})"), flush=True)
    items_bytes = (n_in + sum(_target_ch(smode, do_fiber, fmode, True, do_lsd))) * P ** 3 * 4
    print("[tsm] RAM:", flush=True)
    estimate_and_assert(
        [("loader prefetch (RAM)", (max(1, int(opts["num_workers"])) * 2 * B, items_bytes // 4), np.float32)], cfg.budget
    )
    vram_rows = [
        ("model+grad+adam(fp32)", n_params * 4 * 4),
        ("ema(fp32)", n_params * 4),
        (f"activations(est, act_ckpt={opts['act_ckpt']})", activation_bytes_estimate(P, B, widths, act_ckpt=int(opts["act_ckpt"]))),
        ("batch on gpu", B * items_bytes),
    ]
    fd = opts["feat_distill"]
    distill: FeatDistiller | None = None
    if fd["enabled"]:
        distill = FeatDistiller(fd, widths, device, loader=feat_teacher_loader, out_dir=cfg.out_dir)
        for line in distill.describe(widths):
            print(f"[tsm] feat_distill {line}", flush=True)
        print(f"[tsm] feat_distill loss={fd['loss']} weight={fd['weight']} every={fd['every']}", flush=True)
        vram_rows += distill.bytes_estimate(P, B)
    vram_total = sum(b for _, b in vram_rows)
    print("[tsm] VRAM (estimate):", flush=True)
    print(format_table([["name", "bytes", "GiB"]] + [[n, b, f"{b / GB:.3f}"] for n, b in vram_rows] + [["TOTAL", vram_total, f"{vram_total / GB:.3f}"]]), flush=True)
    if device == "cuda":
        cap = torch.cuda.get_device_properties(0).total_memory * cfg.budget.vram_frac
        if vram_total > cap:
            raise BudgetError(f"VRAM estimate {vram_total / GB:.2f} GiB > vram_frac x device {cap / GB:.2f} GiB; reduce batch/patch/widths")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if dry_run:
        model.train()
        batch = synthetic_batch(B, P, device, surface_mode=smode, input_radial=radial, fiber=do_fiber,
                                fiber_mode=fmode, input_axis=ax_in, lsd=do_lsd)
        aug_mode, aug_spec = augment_mode(opts)
        aug_dt = 0.0
        if aug_mode == "v2":
            from tsm.data import Augment

            aug = Augment(aug_spec, seed=int(opts["seed"]), fiber_swap=bool(opts["fiber_swap_fix"]),
                          input_vec=input_vec_slices(radial, ax_in))
            t = time.perf_counter()
            for _ in range(3):
                batch, fired = aug(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            aug_dt = (time.perf_counter() - t) / 3
            print(f"[tsm] dry run augment v2: {aug_dt * 1000:.1f} ms per batch of {B}x{P}^3 ({aug.describe()}); last fired {fired}", flush=True)
        t = time.perf_counter()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out, feats = model(model_input(batch, cl), return_features=True)
        loss, parts = compute_losses(out, batch, opts["loss_weights"], opts["ds_weights"], surface_mode=smode,
                                        surface_aux=opts.get("surface_aux"),
                                        ink_pos_weight_spec=opts["ink_pos_weight"])
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        _, peak = _vram_mb()
        print(f"[tsm] dry run: fwd+bwd {B}x{P}^3 in {dt:.2f}s, peak VRAM {peak:.0f} MiB, RSS {read_rss_bytes() / MB:.0f} MiB", flush=True)
        print("[tsm] dry run losses: " + json.dumps({k: round(v, 4) for k, v in parts.items()}), flush=True)
        if distill is not None:
            feats = {k: v.detach().requires_grad_(True) for k, v in feats.items()}
            t = time.perf_counter()
            fl, fparts = distill(feats, batch["input"][:, 0], batch=batch)
            fl.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            dft = time.perf_counter() - t
            _, peak = _vram_mb()
            with torch.no_grad():
                s_macs = _conv_macs(_unwrap(model), lambda: model(model_input(batch, cl)))
                t_macs = {n: _conv_macs(e, lambda e=e, n=n: distill.encoders[n](distill.teacher_input(n, batch["input"][:, 0]).to(distill.dtype)))
                          for n, e in distill.encoders.items()}
            print(f"[tsm] dry run feat_distill: {len(distill.encoders)} teacher encoder fwd + proj in {dft:.2f}s "
                  f"(+{100 * dft / max(dt, 1e-9):.0f}% of the student fwd+bwd), peak VRAM {peak:.0f} MiB", flush=True)
            print(f"[tsm] dry run conv MACs: student fwd {s_macs / 1e9:.1f} G; " + ", ".join(f"{n} encoder fwd {m / 1e9:.1f} G" for n, m in t_macs.items()), flush=True)
            print("[tsm] dry run feat losses: " + json.dumps({k: round(v, 4) for k, v in fparts.items()}), flush=True)
            parts.update(fparts)
            parts["feat_time_s"] = dft
        model.zero_grad(set_to_none=True)
        free_cuda()
        return {"params": n_params, "peak_vram_mb": peak, "losses": parts, "augment_time_s": aug_dt, "step_time_s": dt}

    os.makedirs(out_dir, exist_ok=True)
    groups = [{"params": list(model.parameters())}]
    extra_state: dict[str, Any] = {}
    if distill is not None:
        groups.append({"params": list(distill.parameters())})
        extra_state["feat_proj"] = distill
    opt = torch.optim.AdamW(groups, lr=float(opts["lr"]), weight_decay=float(opts["weight_decay"]), betas=(0.9, 0.99))
    clip_params = [p for g in groups for p in g["params"]]
    steps = int(opts["steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, int(opts["warmup"]), steps))
    ema = EMA(model, float(opts["ema"]))
    step = 0
    latest = os.path.join(out_dir, "latest.pt")
    if resume and os.path.exists(latest):
        step = load_checkpoint(latest, model, ema, opt, sched, device, extra=extra_state)
        print(f"[tsm] resumed from {latest} at step {step}", flush=True)
    elif os.path.exists(latest) and not force:
        raise FileExistsError(f"{latest} exists; pass --resume to continue or --force to overwrite")
    config_blob = {"train": opts, "widths": list(widths), "surface_mode": smode,
                   "body_stride": int(opts["body_stride"]), "fullres_width": int(opts["fullres_width"]),
                   "norm": str(opts["norm"]),
                   "input_radial": radial, "input_axis": ax_in, "axis_tangent": ax_tan, "fiber_mode": fmode,
                   "gap_class": gap_cls, "lsd": do_lsd,
                   "in_ch": n_in, "axis_path": opts["axis_path"], "region": [list(cfg.region.start_zyx), list(cfg.region.size_zyx)], "volume": cfg.volume.__dict__}

    aug_mode, aug_spec = augment_mode(opts)
    if bool(opts["overfit_one"]):
        aug_mode = "off"
    aug = None
    if aug_mode == "v2":
        from tsm.data import Augment

        aug = Augment(aug_spec, seed=int(opts["seed"]) + step, fiber_swap=bool(opts["fiber_swap_fix"]),
                      input_vec=input_vec_slices(radial, ax_in))
        print(f"[tsm] augment v2 (GPU, per batch): {aug.describe()}", flush=True)
    else:
        print(f"[tsm] augment: {aug_mode}", flush=True)
    ds = build_dataset(cfg, opts, augment=aug_mode == "v1")
    if len(ds.holdout_origins):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "holdout.npy"), ds.holdout_origins)
    if bool(opts["overfit_one"]):
        fixed = to_device(_overfit_batch(ds, B), device)
        batches: Iterator[dict[str, torch.Tensor]] = iter(lambda: fixed, None)  # type: ignore[arg-type]
    else:
        nw = int(opts["num_workers"])
        loader = DataLoader(
            ds, batch_size=B, sampler=RangeSampler(step * B * A, len(ds)), num_workers=nw,
            prefetch_factor=2 if nw > 0 else None, persistent_workers=nw > 0, pin_memory=device == "cuda", drop_last=True,
        )
        batches = train_step_batches(loader)

    log_path = os.path.join(out_dir, "log.jsonl")
    log_fh = open(log_path, "a")
    # per-head loss balancing + per-head gradient-norm logging (see LossBalancer)
    bal = LossBalancer(opts["balance"], opts["loss_weights"], decay=float(opts["balance_decay"]),
                       warmup=int(opts["balance_warmup"]), freeze=bool(opts["balance_freeze"]),
                       every=int(opts["balance_every"]))
    bal.steps = step  # a resumed run does not repeat the warmup
    gl_every = int(opts["grad_log_every"])
    mod = _unwrap(model)
    head_params = {h: list(mod.head[h].parameters()) for h in mod.head}
    # "shared trunk" for gradnorm_lite: the finest decoder block (+ the full-res block when the
    # body runs at stride 2) -- the last features every head reads from.
    trunk_params = list(mod.dec[0].parameters()) + (list(mod.fullres.parameters()) if mod.fullres is not None else [])
    if bal.enabled:
        print(f"[tsm] loss balance={bal.mode} warmup={bal.warmup} decay={bal.decay} freeze={bal.freeze} "
              f"every={bal.every} (weights {bal.weights})", flush=True)
    model.train()
    t_last = time.perf_counter()
    nonfinite_grads = 0
    last_parts: dict[str, float] = {}
    print(f"[tsm] training steps {step}..{steps} batch {B} x accum {A} x {P}^3", flush=True)
    while step < steps:
        opt.zero_grad(set_to_none=True)
        acc: dict[str, float] = {}
        use_feat = distill is not None and distill.active(step + 1)
        scales = bal.scales()
        measure = bal.needs_measure(step + 1)
        want_gn = measure or (gl_every > 0 and ((step + 1) % gl_every == 0 or step == 0))
        for ai in range(A):
            batch = to_device(next(batches), device)
            if "volcomp_q" in batch:
                # the codec round trip fires in the dataset worker (CPU, on the raw uint8
                # crop), so its count comes with the batch rather than from aug(batch)
                acc["aug/volcomp"] = acc.get("aug/volcomp", 0.0) + float((batch["volcomp_q"] > 0).sum())
            if aug is not None:
                batch, fired = aug(batch)
                for k, v in fired.items():
                    acc[f"aug/{k}"] = acc.get(f"aug/{k}", 0.0) + float(v)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                out, feats = model(model_input(batch, cl), return_features=True)
            hl: dict[str, torch.Tensor] | None = {} if (want_gn and ai == 0) else None
            loss, parts = compute_losses(out, batch, opts["loss_weights"], opts["ds_weights"], surface_mode=smode,
                                        surface_aux=opts.get("surface_aux"),
                                        ink_pos_weight_spec=opts["ink_pos_weight"], head_scale=scales,
                                        head_losses=hl)
            if use_feat:
                fl, fparts = distill(feats, batch["input"][:, 0], batch=batch,
                                     spatial=getattr(aug, "last_params", None))
                loss = loss + fl
                parts.update(fparts)
                parts["total"] = float(loss.detach())
            if hl:
                # extra autograd.grad calls (retain_graph, no .grad written): the step itself is
                # unchanged, so this is safe to run with balance=null too.
                norms = head_grad_norms(hl, head_params, trunk_params if measure else ())
                for h, (n_head, n_all) in norms.items():
                    acc[f"gradnorm/{h}"] = n_head
                if measure:
                    bal.measure({h: n_all for h, (_, n_all) in norms.items()})
            del feats, out, hl
            (loss / A).backward()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v / A
        # aug/<name> = number of samples the transform touched in this optimizer step (B x A samples)
        gn = torch.nn.utils.clip_grad_norm_(clip_params, float(opts["grad_clip"]))
        # A single bad batch (or a dropped NaN on flaky virtualised CUDA) yields a non-finite
        # grad norm; taking the step would push inf/NaN into the weights and permanently
        # collapse the model (seen on the Thunder A6000 at step 5470).  Skip the update instead:
        # zero the grads, count it, and abort only if it becomes chronic rather than a one-off.
        if not math.isfinite(float(gn)):
            nonfinite_grads += 1
            print(f"[tsm] step {step + 1}: non-finite grad norm ({gn}); skipping optimizer step "
                  f"({nonfinite_grads} so far)", flush=True)
            opt.zero_grad(set_to_none=True)
            if nonfinite_grads > int(opts.get("max_nonfinite_grads", 50)):
                raise FloatingPointError(f"aborting: {nonfinite_grads} non-finite grad steps")
            step += 1
            continue
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        last_parts = acc
        if step % int(opts["log_every"]) == 0 or step == 1 or step == steps:
            now = time.perf_counter()
            n = int(opts["log_every"]) if step % int(opts["log_every"]) == 0 else 1
            its = n / max(1e-9, now - t_last)
            t_last = now
            cur, peak = _vram_mb()
            rec = {
                "step": step, "lr": float(opt.param_groups[0]["lr"]), "it_s": round(its, 3), "grad_norm": float(gn),
                "vram_mb": round(cur), "vram_peak_mb": round(peak), "rss_mb": round(read_rss_bytes() / MB), "time": time.time(),
                # balancer state (ema / measured norms); acc below overrides balance/<head> with
                # the multipliers that were actually applied this step
                **{k: round(v, 6) for k, v in bal.log().items()},
                **{k: (int(v) if k.startswith("aug/") else round(v, 5)) for k, v in acc.items()},
            }
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            head_str = " ".join(f"{h}={acc.get(h, 0.0):.4f}" for h in heads)
            if "feat" in acc:
                head_str += f" feat={acc['feat']:.4f}"
            if bal.enabled:
                head_str += " bal[" + " ".join(
                    f"{h}={acc.get(f'balance/{h}', bal.mult.get(h, 0.0)):.3g}"
                    f"*{acc.get(h, 0.0):.3g}={acc.get(f'balance/{h}_contrib', 0.0):.3g}" for h in heads) + "]"
            print(f"[tsm] step {step} loss={acc.get('total', 0.0):.4f} {head_str} lr={rec['lr']:.2e} {its:.2f}it/s vram={peak:.0f}MiB rss={rec['rss_mb']}MiB", flush=True)
        bal.update(acc)  # after the log, so balance/<head> is the multiplier this step used
        if step % int(opts["ckpt_every"]) == 0 or step == steps:
            p = os.path.join(out_dir, f"ckpt_{step}.pt")
            save_checkpoint(p, model, ema, opt, sched, step, config_blob, extra_state)
            save_checkpoint(latest, model, ema, opt, sched, step, config_blob, extra_state)
            print(f"[tsm] saved {p}", flush=True)
    log_fh.close()
    _, peak = _vram_mb()
    summary = {"step": step, "params": n_params, "peak_vram_mb": peak, "peak_rss_mb": peak_rss_mb(), "losses": last_parts,
               "n_train_origins": int(len(ds.origins)), "n_holdout_origins": int(len(ds.holdout_origins))}
    if bool(opts["eval_holdout"]) and len(ds.holdout_origins):
        ema.copy_to(model)  # EMA weights (the checkpoint is already saved)
        t = time.perf_counter()
        metrics = evaluate_holdout(_unwrap(model), ds, device=device, batch=B,
                                   max_crops=int(opts["eval_max_crops"]), surface_mode=smode)
        metrics["seconds"] = time.perf_counter() - t
        with open(os.path.join(out_dir, "holdout_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        summary["holdout"] = metrics
        print("[tsm] holdout (EMA): " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}), flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    free_cuda()
    return summary


# --------------------------------------------------------------------------- #
# evaluation on held-out crops (deterministic, no augmentation)
# --------------------------------------------------------------------------- #
def evaluate_holdout(model: torch.nn.Module, ds: Any, device: str = "cpu", batch: int = 1,
                     max_crops: int | None = None, surface_mode: str = "medial") -> dict[str, Any]:
    """``evaluate`` on ``ds.holdout_origins``, plus per-store metrics for a MultiStoreDataset.

    The pooled keys are exactly the single-store ones; every store additionally contributes
    ``<store>/<metric>`` (so ``holdout/<store>/<metric>`` in the summary / holdout_metrics.json)."""
    metrics = evaluate(model, ds, ds.holdout_origins, device=device, batch=batch,
                       max_crops=max_crops, surface_mode=surface_mode)
    names = list(getattr(ds, "names", []) or [])
    if len(names) > 1:
        o = np.asarray(ds.holdout_origins, np.int32).reshape(-1, 4)
        for k, name in enumerate(names):
            oi = o[o[:, 0] == k]
            if not len(oi):
                continue
            m = evaluate(model, ds, oi, device=device, batch=batch, max_crops=max_crops,
                         surface_mode=surface_mode)
            metrics.update({f"{name}/{kk}": v for kk, v in m.items()})
    return metrics



def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (grouped thresholds); NaN when empty or with a single class.

    Thin wrapper around the one project-wide implementation,
    :func:`tsm.equivariance.average_precision` (equal scores = one threshold, so the value does not
    depend on voxel order), converting its ``None`` for degenerate input into ``nan``."""
    from tsm.equivariance import average_precision as _ap

    v = _ap(scores, labels)
    return float("nan") if v is None else float(v)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged); NaN without both classes."""
    from scipy.stats import rankdata

    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels, bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _any6_t(m: torch.Tensor) -> torch.Tensor:
    """(B, 1, D, H, W) bool -> True where any 6-neighbour is True (outside = False)."""
    p = F.pad(m.float(), (1, 1, 1, 1, 1, 1))
    out = p[:, :, :-2, 1:-1, 1:-1] + p[:, :, 2:, 1:-1, 1:-1] + p[:, :, 1:-1, :-2, 1:-1] + p[:, :, 1:-1, 2:, 1:-1] \
        + p[:, :, 1:-1, 1:-1, :-2] + p[:, :, 1:-1, 1:-1, 2:]
    return out > 0


def zero_crossing(sdf: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """1-voxel surface: sdf <= 0 voxels with a 6-neighbour > 0, restricted to ``m`` (like infer.extract_surface)."""
    neg = (sdf <= 0) & m
    pos = (sdf > 0) & m
    return neg & _any6_t(pos)


class _Sampler:
    """Deterministic simple random sample of (score, label) pairs over the whole eval stream.

    Bottom-k ("Algorithm L"-equivalent) reservoir: every voxel gets one independent uniform key
    drawn in arrival order and the ``cap`` smallest keys are kept, so every eligible voxel has the
    same inclusion probability and the sample is the pooled voxel population -- not one quota per
    crop.  Because the keys are drawn per voxel in stream order, the result depends only on the
    order of the voxels, **not** on how they are split into ``add`` calls (the eval batch size)."""

    def __init__(self, cap: int, seed: int = 0) -> None:
        self.cap = max(1, int(cap))
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros(0, np.float32)
        self.y = np.zeros(0, bool)
        self.k = np.zeros(0, np.float64)  # the kept keys
        self.n = 0                        # eligible voxels seen

    def add(self, score: torch.Tensor, label: torch.Tensor, n_crops: int | None = None) -> None:
        """Add a batch of eligible voxels (``n_crops`` is accepted for call compatibility and unused)."""
        s = score.reshape(-1).float().cpu().numpy()
        y = label.reshape(-1).bool().cpu().numpy()
        if s.size == 0:
            return
        self.n += s.size
        k = self.rng.random(s.size)
        s, y, k = np.concatenate([self.s, s]), np.concatenate([self.y, y]), np.concatenate([self.k, k])
        if k.size > self.cap:
            keep = np.argpartition(k, self.cap - 1)[: self.cap]
            s, y, k = s[keep], y[keep], k[keep]
        self.s, self.y, self.k = s, y, k

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.s, self.y


def _surface_distances(pred_surf: np.ndarray, tgt_surf: np.ndarray, censor: float | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(distances of pred surface voxels to the target surface, and vice versa), voxels.

    A nonempty surface whose counterpart is **empty** is a total miss: every one of its voxels
    contributes the censored distance ``censor`` (default: the crop diagonal, the largest distance
    that can occur inside the crop) instead of contributing nothing.  Without this a crop the model
    missed entirely would silently disappear from the pooled percentiles.  Both empty -> two empty
    arrays (nothing to measure)."""
    from tsm.labels import _edt

    if pred_surf.any() and tgt_surf.any():
        return _edt(~tgt_surf)[pred_surf], _edt(~pred_surf)[tgt_surf]
    c = float(np.sqrt(sum(float(v) ** 2 for v in tgt_surf.shape))) if censor is None else float(censor)
    d_p2t = np.full(int(pred_surf.sum()), c, np.float32) if pred_surf.any() else np.zeros(0, np.float32)
    d_t2p = np.full(int(tgt_surf.sum()), c, np.float32) if tgt_surf.any() else np.zeros(0, np.float32)
    return d_p2t, d_t2p


def _dist_stats(d: np.ndarray, prefix: str) -> dict[str, float]:
    if d.size == 0:
        return {f"{prefix}_median": float("nan"), f"{prefix}_p90": float("nan"), f"{prefix}_frac_gt3": float("nan")}
    return {f"{prefix}_median": float(np.median(d)), f"{prefix}_p90": float(np.percentile(d, 90)), f"{prefix}_frac_gt3": float((d > 3.0).mean())}


#: minimum size (voxels) of a body component counted by ``surface/body_n_components_ratio``
BODY_MIN_COMPONENT = 32


def _n_components(mask: np.ndarray, min_size: int = BODY_MIN_COMPONENT) -> int:
    """Number of 26-connected components of a boolean mask with at least ``min_size`` voxels."""
    from scipy import ndimage as ndi

    if not mask.any():
        return 0
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), np.uint8))
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= int(min_size)).sum())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    ds: Any,
    origins: Sequence[Sequence[int]] | np.ndarray,
    device: str = "cpu",
    batch: int = 1,
    clip: float = 20.0,
    max_crops: int | None = None,
    sample_cap: int = 2_000_000,
    surface_mode: str = "medial",
) -> dict[str, Any]:
    """Per-head metrics of ``model`` on the fixed, un-augmented crops at ``origins`` of ``ds``.

    surface: ``sdf_mae`` (valid==1 and |target| < clip), ``zc_dice`` (zero-crossing Dice vs the
    label zero crossing), medial-surface distances ``surf_p2t_*`` (pred -> label; precision-like),
    ``surf_t2p_*`` (label -> pred; recall-like) and ``surf_sym_*`` (both), each median / p90 /
    frac > 3 vox (a crop whose surface is missed entirely contributes the crop diagonal as a
    censored distance and is counted in ``missed_surfaces`` / ``spurious_surfaces``), ``valid_auprc`` (valid logit vs valid==1 on valid != 2); ink: ``ink_auprc``
    (logit vs soft target > 0.5 on ink_valid==1); winding (valid==1): ``phase_err_deg`` (mean
    circular error) + ``phase_err_deg_median``, ``density_mae`` (wraps/voxel, raw head vs target),
    ``normal_err_deg`` (mean + median), ``conf_auroc`` (conf logit vs conf > 0.5); fiber (optional head, ``fiber_valid == 1``):
    ``fiber/vt_auprc`` / ``fiber/hz_auprc`` (logit -- or, in the direction mode, the derived
    class probability -- vs teacher probability > 0.5); direction mode also reports the axial
    angular error ``fiber/angle_deg`` (mean + median, on valid voxels whose target strength is
    > 0.5) and, where the human ``hzvt_class`` bands cover a voxel, the teacher-independent
    ``fiber/band_angle_deg`` and ``fiber/band_class_acc``; both modes also report the class
    **exclusivity** ``fiber/overlap_frac`` = n(vt > .5 & hz > .5) / n(vt > .5 | hz > .5) and
    ``fiber/firing_frac`` = n(vt > .5 | hz > .5) / n(fiber_valid == 1), on the (derived) class
    probabilities.  With the optional Local Shape Descriptor head (``extra.train.heads.lsd``)
    it also reports, on ``lsd_valid == 1`` and in the descriptors' own units,
    ``lsd/off_mae`` (voxels, the mean over the three offset components), ``lsd/thick_mae``
    (voxels) and ``lsd/ncov_mae`` ([0, 1], against ``sigmoid`` of the head).  ``surface_mode="body"`` goes through the single-face branch (sdf MAE, zc
    Dice, distances, missed / spurious, valid AUPRC) and adds ``surface/body_dice``,
    ``surface/body_iou``, ``surface/body_vol_ratio`` (predicted / label body voxels) and
    ``surface/body_n_components_ratio`` (26-connected components of at least 32 voxels, the
    per-crop median of pred / label counts); there is no ``thickness_mae`` in body mode.
    ``surface_mode="sides"`` reports the same block with ``surface/sdf_mae`` measured on the
    UNSIGNED face distance ``d_face`` (``valid == 1``) and ``surface/zc_dice`` (plus the
    ``surf_p2t`` / ``surf_t2p`` / ``surf_sym`` distances and the missed / spurious counts)
    measured on the "zero set" ``d_face <= 1.0`` -- a 2-voxel-thick shell around every face,
    taken the same way on the prediction and on the label, which is also what the derived
    ``surface_side1`` channel writes; the body numbers come from the dedicated body head
    (``sigmoid(body logit) > 0.5``) against the 0/1 label body mask, and the fibre direction
    basis uses ``sheet_normal(prefer_fallback=True)`` because the gradient of an unsigned
    distance is degenerate on the ridge.
    Deterministic
    (fixed crop order, no augmentation, eval mode); AUPRC/AUROC use a seeded reservoir of at most
    ``sample_cap`` voxels drawn with equal probability from the whole pooled voxel stream (the
    sample, and hence the metric, does not depend on ``batch``)."""
    from torch.utils.data.dataloader import default_collate

    from tsm.data import CLIP as _CLIP  # noqa: F401  (clip default documented in data)

    # (N, 3) local origins of a CropDataset, or (N, 4) [store, z, y, x] of a MultiStoreDataset
    o = np.asarray(origins, np.int32)
    o = o.reshape(-1, int(o.shape[-1]) if o.ndim >= 2 and int(o.shape[-1]) == 4 else 3)
    if max_crops is not None:
        o = o[: int(max_crops)]
    n = len(o)
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    cnt: dict[str, float] = {}
    samp = {k: _Sampler(sample_cap // 3, seed=i) for i, k in enumerate(("valid", "ink", "conf", "fiber_vt", "fiber_hz"))}
    fiber_ang: list[np.ndarray] = []
    band_ang: list[np.ndarray] = []
    band_hit = [0.0, 0.0]  # correct / total human-band voxels
    faces = surface_mode == "faces"
    body = surface_mode == "body"
    sides = surface_mode == "sides"
    face_names = ("in", "out") if faces else ("",)
    # the valid logit is the LAST surface channel: index 2 for the 3-wide heads (faces: after
    # [sdf_in, sdf_out]; sides: after [d_face, body logit]), 1 for the 2-wide ones
    i_valid = 2 if (faces or sides) else 1
    body_sums = [0.0, 0.0, 0.0]   # |pred & label|, |pred|, |label| body voxels (valid == 1)
    body_comp: list[float] = []   # per-crop ratio of 26-connected component counts
    fib_excl = [0.0, 0.0, 0.0]    # n(both), n(either), n(fiber_valid == 1)
    d_p2t: dict[str, list[np.ndarray]] = {f: [] for f in face_names}
    d_t2p: dict[str, list[np.ndarray]] = {f: [] for f in face_names}
    missed = {f: [0, 0] for f in face_names}  # crops with (label surface, no prediction) / (prediction, no label)
    zc = {f: [0.0, 0.0, 0.0] for f in face_names}  # inter, pred, label
    phase_med: list[np.ndarray] = []
    normal_med: list[np.ndarray] = []
    dev = torch.device(device)
    B = max(1, int(batch))

    def acc(key: str, v: torch.Tensor, m: torch.Tensor) -> None:
        sums[key] = sums.get(key, 0.0) + float((v * m).sum())
        cnt[key] = cnt.get(key, 0.0) + float(m.sum())

    for i0 in range(0, n, B):
        items = [ds.to_tensors(ds.load(o[i])) for i in range(i0, min(n, i0 + B))]
        b = to_device(default_collate(items), device)
        if getattr(ds, "input_axis", False):
            # the local axis tangent the crops were built with, back out of the input channels:
            # "vertical" must mean the same thing here as it did in the target derivation
            lo_a, hi_a = input_vec_slices(bool(getattr(ds, "input_radial", False)), True)[-1]
            b["axis_dir"] = b["input"][:, lo_a:hi_a]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(model_input(b))
        out = {k: (v[0] if isinstance(v, (list, tuple)) else v).float() for k, v in out.items()}
        # surface
        sv = b["surface_valid"]
        m1 = (sv == 1)
        for fi, fname in enumerate(face_names):
            pre = f"surface/{fname}_" if fname else "surface/"
            sdf_t, sdf_p = b["surface_sdf"][:, fi:fi + 1], out["surface"][:, fi:fi + 1]
            band = m1 & (sdf_t.abs() < float(clip) - 1e-3)
            acc(f"{pre}sdf_mae", (sdf_p - sdf_t).abs(), band.float())
            if sides:
                # the face distance is UNSIGNED, so it has no zero *crossing*: the "zero set"
                # scored here is the 2-voxel-thick shell d_face <= 1.0 on both sides, the same
                # rule the derived `surface_side1` channel writes at inference time
                zp, zt = (sdf_p <= 1.0) & m1, (sdf_t <= 1.0) & m1
            else:
                zp, zt = zero_crossing(sdf_p, m1), zero_crossing(sdf_t, m1)
            zc[fname][0] += float((zp & zt).sum())
            zc[fname][1] += float(zp.sum())
            zc[fname][2] += float(zt.sum())
            for bi in range(zp.shape[0]):
                pn, tn = zp[bi, 0].cpu().numpy(), zt[bi, 0].cpu().numpy()
                a, c = _surface_distances(pn, tn)
                d_p2t[fname].append(a)
                d_t2p[fname].append(c)
                missed[fname][0] += int(tn.any() and not pn.any())
                missed[fname][1] += int(pn.any() and not tn.any())
        if body or sides:
            # the orientation-free body.  body mode: sdf_body > 0 is "this voxel is sheet
            # material"; sides mode: the dedicated body head, sigmoid(logit) > 0.5, against the
            # 0/1 label mask (tsm.data.body_mask_np)
            if sides:
                p_b = (torch.sigmoid(out["surface"][:, 1:2]) > 0.5) & m1
                t_b = (b["surface_body"][:, 0:1] > 0.5) & m1
            else:
                p_b = (out["surface"][:, 0:1] > 0) & m1
                t_b = (b["surface_sdf"][:, 0:1] > 0) & m1
            body_sums[0] += float((p_b & t_b).sum())
            body_sums[1] += float(p_b.sum())
            body_sums[2] += float(t_b.sum())
            for bi in range(p_b.shape[0]):
                np_, nt_ = (_n_components(x[bi, 0].cpu().numpy()) for x in (p_b, t_b))
                if nt_ > 0:
                    body_comp.append(np_ / nt_)
        if faces:  # predicted vs label sheet thickness (sdf_in - sdf_out between the faces)
            th_p = out["surface"][:, 0:1] - out["surface"][:, 1:2]
            th_t = b["surface_sdf"][:, 0:1] - b["surface_sdf"][:, 1:2]
            inside = m1 & (th_t > 0) & (b["surface_sdf"][:, 0:1] > 0) & (b["surface_sdf"][:, 1:2] < 0)
            acc("surface/thickness_mae", (th_p - th_t).abs(), inside.float())
        if "lsd" in out and "lsd" in b:
            # Local Shape Descriptors, on the label body (lsd_valid == 1): the mean absolute
            # error of each descriptor in its OWN units (voxels / voxels / [0, 1]), not the
            # scale-normalised loss
            ml = (b["lsd_valid"] == 1).float()
            pl, tl = out["lsd"].float(), b["lsd"].float()
            acc("lsd/off_mae", (pl[:, 0:3] - tl[:, 0:3]).abs().mean(dim=1, keepdim=True), ml)
            acc("lsd/thick_mae", (pl[:, 3:4] - tl[:, 3:4]).abs(), ml)
            acc("lsd/ncov_mae", (torch.sigmoid(pl[:, 4:5]) - tl[:, 4:5]).abs(), ml)
        m_not2 = sv != 2
        samp["valid"].add(out["surface"][:, i_valid:i_valid + 1][m_not2], m1[m_not2], n)
        # ink
        mi = b["ink_valid"] == 1
        samp["ink"].add(out["ink"][:, 0:1][mi], (b["ink_prob"] > 0.5)[mi], n)
        # fiber (optional head): AUPRC of each (derived) probability against the teacher's > 0.5,
        # plus, in the direction mode, the axial angular error and the human-band numbers
        if "fiber" in out and "fiber_valid" in b:
            mf = b["fiber_valid"] == 1
            fdir = "fiber_dir" in b
            if fdir:
                fb_out = out["fiber"]
                # the in-plane basis at every voxel, from the LABEL geometry (grad sdf_in, the
                # winding normal where that is not a unit vector) -- the same rule the dataset
                # used to build the target, so the derived classes are comparable to the teacher
                # ``prefer_fallback`` in body mode: grad of a body SDF is degenerate on the
                # medial ridge, so the winding normal leads there (tsm.fiber.sheet_normal)
                nrm, _ = sheet_normal(b["surface_sdf"][:, 0:1], b["winding"][:, 3:6].flip(1),
                                      prefer_fallback=body or sides)
                tv, th, ok = fiber_basis(nrm, b.get("axis_dir"))
                dp = F.normalize(fb_out[:, 0:3].float(), dim=1, eps=EPS)
                sp = torch.sigmoid(fb_out[:, 3:4].float())
                pv, ph = class_from_direction(dp, sp, tv, th)
                logits = [torch.logit(pv.clamp(1e-6, 1 - 1e-6)), torch.logit(ph.clamp(1e-6, 1 - 1e-6))]
                dt = F.normalize(b["fiber_dir"], dim=1, eps=EPS)
                cos = (dp * dt).sum(1, keepdim=True).abs().clamp(0, 1)
                ma = mf & ok & (b["fiber_str"] > 0.5)
                if bool(ma.any()):
                    fiber_ang.append(torch.rad2deg(torch.acos(cos))[ma].cpu().numpy())
                if "fiber_band" in b:  # teacher-independent: the human hz/vt bands
                    band = b["fiber_band"]
                    human = (band == 1) | (band == 2)
                    mb = human & ok & (b["fiber_valid"] > 0)
                    if bool(mb.any()):
                        tgt = torch.where(band == 2, tv, th)  # 1 hz, 2 vt
                        cb = (dp * tgt).sum(1, keepdim=True).abs().clamp(0, 1)
                        band_ang.append(torch.rad2deg(torch.acos(cb))[mb].cpu().numpy())
                        pred_vt = pv > ph
                        band_hit[0] += float((pred_vt == (band == 2))[mb].sum())
                        band_hit[1] += float(mb.sum())
            else:
                logits = [out["fiber"][:, i:i + 1] for i in range(2)]
                if "fiber_band" in b:  # the same teacher-independent class accuracy, class mode
                    band = b["fiber_band"]
                    mb = ((band == 1) | (band == 2)) & (b["fiber_valid"] > 0)
                    if bool(mb.any()):
                        band_hit[0] += float(((logits[0] > logits[1]) == (band == 2))[mb].sum())
                        band_hit[1] += float(mb.sum())
            # class exclusivity, on the probabilities both modes can produce: the student's
            # two channels should almost never fire together (the labels never do)
            q_vt, q_hz = ((pv, ph) if fdir else (torch.sigmoid(logits[0]), torch.sigmoid(logits[1])))
            hot_v, hot_h = (q_vt > 0.5) & mf, (q_hz > 0.5) & mf
            fib_excl[0] += float((hot_v & hot_h).sum())
            fib_excl[1] += float((hot_v | hot_h).sum())
            fib_excl[2] += float(mf.sum())
            if "fiber_prob" in b:
                for i, name in enumerate(("vt", "hz")):
                    samp[f"fiber_{name}"].add(logits[i][mf], (b["fiber_prob"][:, i:i + 1] > 0.5)[mf], n)
        # winding
        mw = (b["winding_valid"] == 1)
        w, wp = b["winding"], out["winding"]
        sc_p, sc_t = F.normalize(wp[:, 0:2], dim=1, eps=EPS), F.normalize(w[:, 0:2], dim=1, eps=EPS)
        dphi = torch.rad2deg(torch.acos((sc_p * sc_t).sum(1, keepdim=True).clamp(-1, 1)))
        acc("winding/phase_err_deg", dphi, mw.float())
        acc("winding/density_mae", (wp[:, 2:3] - w[:, 2:3]).abs(), mw.float())
        n_p, n_t = F.normalize(wp[:, 3:6], dim=1, eps=EPS), F.normalize(w[:, 3:6], dim=1, eps=EPS)
        dn = torch.rad2deg(torch.acos((n_p * n_t).sum(1, keepdim=True).clamp(-1, 1)))
        acc("winding/normal_err_deg", dn, mw.float())
        if bool(mw.any()):
            phase_med.append(dphi[mw].cpu().numpy())
            normal_med.append(dn[mw].cpu().numpy())
        samp["conf"].add(wp[:, 6:7][mw], (b["winding_conf"] > 0.5)[mw], n)
    if was_training:
        model.train()
    res: dict[str, Any] = {"n_crops": int(n)}
    for k in sums:
        res[k] = sums[k] / cnt[k] if cnt[k] > 0 else float("nan")
    for fname in face_names:
        pre = f"surface/{fname}_" if fname else "surface/"
        inter, za, zb = zc[fname]
        res[f"{pre}zc_dice"] = 2.0 * inter / (za + zb) if (za + zb) > 0 else float("nan")
        res[f"{pre}zc_pred_voxels"] = za
        res[f"{pre}zc_label_voxels"] = zb
        a = np.concatenate(d_p2t[fname]) if d_p2t[fname] else np.zeros(0)
        c = np.concatenate(d_t2p[fname]) if d_t2p[fname] else np.zeros(0)
        res.update(_dist_stats(a, f"{pre}surf_p2t"))
        res.update(_dist_stats(c, f"{pre}surf_t2p"))
        res.update(_dist_stats(np.concatenate([a, c]), f"{pre}surf_sym"))
        res[f"{pre}missed_surfaces"] = float(missed[fname][0])    # label surface, empty prediction
        res[f"{pre}spurious_surfaces"] = float(missed[fname][1])  # prediction, empty label surface
    if body or sides:
        inter, npred, nlab = body_sums
        res["surface/body_dice"] = 2.0 * inter / (npred + nlab) if (npred + nlab) > 0 else float("nan")
        union = npred + nlab - inter
        res["surface/body_iou"] = inter / union if union > 0 else float("nan")
        res["surface/body_vol_ratio"] = npred / nlab if nlab > 0 else float("nan")
        res["surface/body_pred_voxels"] = npred
        res["surface/body_label_voxels"] = nlab
        res["surface/body_n_components_ratio"] = float(np.median(body_comp)) if body_comp else float("nan")
    res["surface/valid_auprc"] = average_precision(*samp["valid"].arrays())
    res["ink/ink_auprc"] = average_precision(*samp["ink"].arrays())
    for name in ("vt", "hz"):
        if samp[f"fiber_{name}"].n:
            res[f"fiber/{name}_auprc"] = average_precision(*samp[f"fiber_{name}"].arrays())
    if fib_excl[2] > 0:
        res["fiber/overlap_frac"] = fib_excl[0] / fib_excl[1] if fib_excl[1] > 0 else 0.0
        res["fiber/firing_frac"] = fib_excl[1] / fib_excl[2]
    if fiber_ang:
        a = np.concatenate(fiber_ang)
        res["fiber/angle_deg"] = float(a.mean())
        res["fiber/angle_deg_median"] = float(np.median(a))
        res["fiber/angle_n"] = float(a.size)
    if band_ang:
        a = np.concatenate(band_ang)
        res["fiber/band_angle_deg"] = float(a.mean())
        res["fiber/band_angle_deg_median"] = float(np.median(a))
    if band_hit[1] > 0:
        res["fiber/band_class_acc"] = band_hit[0] / band_hit[1]
        res["fiber/band_n"] = band_hit[1]
    res["winding/phase_err_deg_median"] = float(np.median(np.concatenate(phase_med))) if phase_med else float("nan")
    res["winding/normal_err_deg_median"] = float(np.median(np.concatenate(normal_med))) if normal_med else float("nan")
    res["winding/conf_auroc"] = auroc(*samp["conf"].arrays())
    return res


def feat_teacher_loader(name: str, models_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Indirection so tests can monkeypatch ``tsm.train.feat_teacher_loader`` with a tiny encoder."""
    from tsm.teachers import load_teacher

    return load_teacher(name, models_dir, device=device, dtype=dtype)


def _target_ch(surface_mode: str = "medial", fiber: bool = False, fiber_mode: str = "class",
               band: bool = False, lsd: bool = False) -> list[int]:
    from tsm.data import target_keys

    return list(target_keys(surface_mode, fiber, fiber_mode, band, lsd=lsd).values())
