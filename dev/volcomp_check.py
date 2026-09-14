"""Does volcomp (q=8, lossy) CT change what the teachers and the student predict?

The mirror at ``volume.url`` is a lossy zarr-v3 re-encoding of the lossless S3 volume at
``volume.alt_url``.  On a 256^3 box of Paris 4 the byte error is mean 3.0 grey levels /
p99 11 / max 47 (PSNR 36 dB).  The question this script answers is not "how big is the
byte error" but "does any consumer of the CT notice": every model is run **twice, on the
very same box**, once from the lossless volume and once from the mirror, and the two
outputs are compared in the space each model is actually used in.

    uv run python dev/volcomp_check.py --config configs/paris4_eval_long30k.json \\
        --origin 34432,15360,18688 --models recto,fiber,lasagna \\
        --pair /home/forrest/tsm-output/volcomp_check_box.npy \\
        --student /home/forrest/tsm-output/faces30k_gf/train/latest.pt \\
        --out /tmp/volcomp_check.json

Geometry.  ``--origin`` / ``--size`` name a box in the **level-0** (2.4 um) grid; that box
is what recto / ink / fiber and the student see (one window, no tiling -- the point is the
model's sensitivity to the input bytes, not the blending).  ``lasagna`` is a 9.6 um model,
so it is run on a ``--lasagna-patch`` box of level 2 **centred on the same physical place**,
and its metrics are reported on the central core that corresponds to the level-0 box.

Inputs.  Either ``--pair a.npy`` (a saved ``(2, Z, Y, X)`` uint8 array: ``[0]`` lossless,
``[1]`` mirror -- avoids re-reading S3) or live reads: lossless from ``volume.alt_url``
(preferring ``volume.cache_root``) and mirror from ``volume.url``.  The level-2 lasagna box
is always read live unless ``--pair-l2`` is given.

Metrics (per model, lossless vs mirror):
  * ``mean|d|`` / ``p99|d|`` / ``max|d|``: absolute difference of the *probabilities*
    (post-activation), averaged over every output channel and voxel.
  * ``dice@0.5``: Dice between the two foreground masks at threshold 0.5 (the channel the
    consumer thresholds: recto fg, fiber vt+hz, lasagna cos, student zero set).
  * ``argmax dt``: fraction of voxels whose winning class changes (softmax teachers).
  * lasagna also reports the cos channel on its own and the angle between the decoded
    unit normals (a line field: the angle is taken modulo pi).
  * the student reports sdf / d_face mean |d| in voxels and the zero-set Dice@1.

Everything runs on the CPU by default: **the laptop GPU is never used**; ``--device cuda``
needs ``TSM_ALLOW_LOCAL_GPU=1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tsm.config import load_config  # noqa: E402
from tsm.teachers import DEFAULT_MODELS_DIR, TEACHERS, apply_activation, load_teacher  # noqa: E402
from tsm.volume import VolumeReader, open_cached_reader  # noqa: E402

ALLOW_LOCAL_GPU_ENV = "TSM_ALLOW_LOCAL_GPU"
#: teachers that take a level-0 (2.4 um) box; `lasagna` is handled separately at level 2.
L0_TEACHERS = ("recto", "ink", "fiber", "m7")


def log(msg: str) -> None:
    print(f"[volcomp] {msg}", flush=True)


def pick_device(requested: str | None = None) -> torch.device:
    """CPU unless ``TSM_ALLOW_LOCAL_GPU`` is set; an explicit ``--device cuda`` without it fails."""
    allowed = bool(os.environ.get(ALLOW_LOCAL_GPU_ENV))
    req = str(requested or "").strip().lower()
    if req.startswith("cuda"):
        if not allowed:
            raise SystemExit(
                f"--device {requested} refused: the laptop GPU must never be used.  Run this on "
                f"forlindesk2 / the A6000, or set {ALLOW_LOCAL_GPU_ENV}=1 if this machine has a "
                f"spare GPU; --device cpu always works.")
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {requested} but no CUDA device is available")
        return torch.device(req)
    if req:
        return torch.device(req)
    if allowed and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Metrics (pure numpy; tests/test_volcomp_check.py covers these)
# --------------------------------------------------------------------------- #
def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two boolean masks; two empty masks are identical (1.0)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return 1.0
    return float(2.0 * int(np.logical_and(a, b).sum()) / s)


def delta_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """mean / p90 / p99 / max of ``|a - b|`` (float32), optionally restricted to ``mask``.

    ``mask`` broadcasts against the trailing spatial dims of ``a`` (so one spatial mask
    covers every channel).  An all-false mask gives zeros and ``n = 0``."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    if mask is not None:
        m = np.broadcast_to(np.asarray(mask, bool), d.shape)
        d = d[m]
    d = np.asarray(d, np.float32).ravel()
    if d.size == 0:
        return {"n": 0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "rms": 0.0}
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "rms": float(np.sqrt(np.mean(np.square(d, dtype=np.float64)))),
    }


def argmax_change(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of voxels whose argmax over axis 0 differs between ``a`` and ``b``."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.ndim < 2:
        raise ValueError("argmax_change needs a channel axis")
    ch = a.argmax(0) != b.argmax(0)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.sum() == 0:
            return 0.0
        ch = ch[m]
    return float(ch.mean()) if ch.size else 0.0


def line_angle_deg(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """Angle in degrees between two (3, ...) fields of unit *lines* (sign is arbitrary)."""
    n1 = np.asarray(n1, np.float64)
    n2 = np.asarray(n2, np.float64)
    if n1.shape != n2.shape or n1.shape[0] != 3:
        raise ValueError(f"expected matching (3, ...) fields, got {n1.shape} and {n2.shape}")
    e = 1e-12
    n1 = n1 / np.maximum(np.linalg.norm(n1, axis=0, keepdims=True), e)
    n2 = n2 / np.maximum(np.linalg.norm(n2, axis=0, keepdims=True), e)
    dot = np.abs((n1 * n2).sum(0)).clip(0.0, 1.0)
    return np.degrees(np.arccos(dot))


def angle_stats(ang: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    a = np.asarray(ang, np.float64)
    if mask is not None:
        a = a[np.asarray(mask, bool)]
    a = a.ravel()
    if a.size == 0:
        return {"n": 0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {"n": int(a.size), "mean": float(a.mean()), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def face_dist_from_sdf(sdf: np.ndarray) -> np.ndarray:
    """Unsigned distance to the nearest sheet face from the two-face student head.

    ``sdf`` is ``(2, ...)`` = ``[sdf_in, sdf_out]``; the faces are the zero sets of the two
    signed distances, so the distance to the nearest face is ``min(|sdf_in|, |sdf_out|)``."""
    sdf = np.asarray(sdf, np.float32)
    if sdf.shape[0] != 2:
        raise ValueError(f"expected 2 sdf channels, got {sdf.shape[0]}")
    return np.minimum(np.abs(sdf[0]), np.abs(sdf[1]))


# --------------------------------------------------------------------------- #
# Reading the two versions of one box
# --------------------------------------------------------------------------- #
def read_pair(cfg, origin: tuple[int, int, int], size: tuple[int, int, int],
              level: int, budget=None, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(lossless, mirror) uint8 boxes at ``origin`` of the given level, read live."""
    from tsm.config import RegionCfg

    vol = cfg.volume
    if not vol.alt_url:
        raise SystemExit("the config has no volume.alt_url: nothing lossless to compare against")
    region = RegionCfg(start_zyx=tuple(origin), size_zyx=tuple(size))
    lossless = None
    if use_cache and level == vol.level:
        lossless = open_cached_reader(vol.cache_root, vol.alt_url, level, vol.voxel_um, region, budget)
        if lossless is None:
            lossless = open_cached_reader(vol.cache_root, vol.url, level, vol.voxel_um, region, budget)
    src = "cache" if lossless is not None else "s3"
    if lossless is None:
        lossless = VolumeReader(vol.alt_url, level, vol.voxel_um, budget)
    mirror = VolumeReader(vol.url, level, vol.voxel_um, budget)
    z0, y0, x0 = origin
    dz, dy, dx = size
    t0 = time.time()
    a = lossless.read(z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
    b = mirror.read(z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
    info = {"level": level, "origin_zyx": list(origin), "size_zyx": list(size),
            "lossless_source": src, "read_s": round(time.time() - t0, 1)}
    return a, b, info


def ct_stats(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Byte-level difference between the two CT boxes (the input side of the question)."""
    st = delta_stats(a.astype(np.float32), b.astype(np.float32))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    st["psnr_db"] = float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))
    st["lossless_mean"] = float(a.mean())
    st["lossless_std"] = float(a.std())
    st["nonzero_frac"] = float((a > 0).mean())
    return st


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def teacher_forward(net, spec, box: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    """One window, normalised the way ``sliding.predict_box`` normalises it (per window)."""
    x = spec.normalizer(box)[None, None].to(device)
    t0 = time.time()
    out = apply_activation(spec.select(net(x)).float(), spec.activation)
    dt = time.time() - t0
    return out[0].cpu().numpy(), dt


def run_teacher(name: str, box_a: np.ndarray, box_b: np.ndarray, models_dir: str,
                device: torch.device, mask: np.ndarray | None = None) -> dict[str, Any]:
    log(f"{name}: loading")
    net, spec = load_teacher(name, models_dir, device=device)
    log(f"{name}: forward on {box_a.shape} (lossless)")
    pa, ta = teacher_forward(net, spec, box_a, device)
    log(f"{name}: forward on {box_b.shape} (mirror), lossless took {ta:.1f}s")
    pb, tb = teacher_forward(net, spec, box_b, device)
    del net
    res: dict[str, Any] = {
        "model": name, "shape": list(pa.shape), "activation": spec.activation,
        "normalizer": spec.normalizer.mode, "voxel_um": spec.voxel_um,
        "forward_s": [round(ta, 1), round(tb, 1)],
        "all_channels": delta_stats(pa, pb),
    }
    if mask is not None:
        res["all_channels_in_ct"] = delta_stats(pa, pb, mask)
        res["ct_mask_frac"] = float(np.asarray(mask, bool).mean())
    if spec.activation == "softmax":
        res["argmax_change"] = argmax_change(pa, pb)
        if mask is not None:
            res["argmax_change_in_ct"] = argmax_change(pa, pb, mask)
    # the channel a consumer thresholds
    if name in ("recto", "m7"):
        fg = 1 if spec.fg_channel is None else int(spec.fg_channel)
        res["fg_channel"] = int(fg)
        res["fg"] = delta_stats(pa[fg], pb[fg])
        res["dice@0.5"] = dice(pa[fg] >= 0.5, pb[fg] >= 0.5)
        res["fg_frac"] = [float((pa[fg] >= 0.5).mean()), float((pb[fg] >= 0.5).mean())]
    elif name == "ink":
        res["fg"] = delta_stats(pa[0], pb[0])
        res["dice@0.5"] = dice(pa[0] >= 0.5, pb[0] >= 0.5)
        res["fg_frac"] = [float((pa[0] >= 0.5).mean()), float((pb[0] >= 0.5).mean())]
    elif name == "fiber":
        # 4-way softmax {background, vertical, horizontal, ink}; the student consumes vt/hz
        fa, fb = pa[1] + pa[2], pb[1] + pb[2]
        res["fg"] = delta_stats(fa, fb)
        res["dice@0.5"] = dice(fa >= 0.5, fb >= 0.5)
        res["fg_frac"] = [float((fa >= 0.5).mean()), float((fb >= 0.5).mean())]
        res["per_class"] = {k: delta_stats(pa[i], pb[i])
                            for i, k in enumerate(("bg", "vt", "hz", "ink"))}
    del pa, pb
    return res


def run_lasagna(box_a: np.ndarray, box_b: np.ndarray, models_dir: str, device: torch.device,
                core: tuple[slice, slice, slice] | None = None) -> dict[str, Any]:
    """lasagna at 9.6 um: 8 channels ``cos, grad_mag, dir0_z, dir1_z, ...``."""
    from tsm.labels import decode_lasagna_normal

    log("lasagna: loading")
    net, spec = load_teacher("lasagna", models_dir, device=device)
    log(f"lasagna: forward on {box_a.shape} (lossless)")
    pa, ta = teacher_forward(net, spec, box_a, device)
    log(f"lasagna: forward on {box_b.shape} (mirror), lossless took {ta:.1f}s")
    pb, tb = teacher_forward(net, spec, box_b, device)
    del net
    res: dict[str, Any] = {
        "model": "lasagna", "shape": list(pa.shape), "activation": spec.activation,
        "normalizer": spec.normalizer.mode, "voxel_um": spec.voxel_um,
        "forward_s": [round(ta, 1), round(tb, 1)],
        "all_channels": delta_stats(pa, pb),
        "cos": delta_stats(pa[0], pb[0]),
        "grad_mag": delta_stats(pa[1], pb[1]),
        "dice@0.5": dice(pa[0] >= 0.5, pb[0] >= 0.5),
        "fg_frac": [float((pa[0] >= 0.5).mean()), float((pb[0] >= 0.5).mean())],
    }
    # the normal decode is heavy (float64, ~30 temporaries): restrict it to the core that
    # corresponds to the level-0 box under test.
    sl = (slice(None),) + (core or (slice(None), slice(None), slice(None)))
    ca, cb = pa[sl], pb[sl]
    na = decode_lasagna_normal(ca)
    nb = decode_lasagna_normal(cb)
    ang = line_angle_deg(na, nb)
    res["core_shape"] = list(ca.shape[1:])
    res["core_cos"] = delta_stats(ca[0], cb[0])
    res["normal_angle_deg"] = angle_stats(ang)
    surf = (ca[0] >= 0.5) & (cb[0] >= 0.5)
    res["normal_angle_deg_on_surface"] = angle_stats(ang, surf)
    res["surface_frac"] = float(surf.mean())
    del pa, pb, ca, cb, na, nb
    return res


@torch.inference_mode()
def student_forward(net, info: dict[str, Any], box: np.ndarray, origin, device) -> tuple[np.ndarray, float]:
    from tsm.infer import activate_heads
    from tsm.student import normalize_ct

    x = normalize_ct(torch.from_numpy(np.ascontiguousarray(box))[None, None]).to(device)
    t0 = time.time()
    raw = net.raw(x, [tuple(int(v) for v in origin)])
    phys = activate_heads(raw, info["surface_mode"], info.get("fiber_mode", "class"))
    dt = time.time() - t0
    return phys[0].float().cpu().numpy(), dt


def run_student(path: str, box_a: np.ndarray, box_b: np.ndarray, origin, device: torch.device,
                clip: float = 20.0) -> dict[str, Any]:
    from tsm.infer import StudentNet, load_student

    log(f"student: loading {path}")
    model, info = load_student(path, device="cpu")
    net = StudentNet(model, 2.4, clip, surface_mode=info["surface_mode"],
                     input_radial=info.get("input_radial", False), axis=info.get("axis_path"),
                     input_axis=info.get("input_axis", False),
                     axis_tangent=info.get("axis_tangent", False),
                     fiber_mode=info.get("fiber_mode", "class"),
                     gap_class=info.get("gap_class", False), lsd=info.get("lsd", False)).eval().to(device)
    log(f"student: forward on {box_a.shape} at {tuple(origin)} (lossless)")
    pa, ta = student_forward(net, info, box_a, origin, device)
    log(f"student: forward on {box_b.shape} (mirror), lossless took {ta:.1f}s")
    pb, tb = student_forward(net, info, box_b, origin, device)
    del net, model
    mode = info["surface_mode"]
    res: dict[str, Any] = {
        "model": "student", "checkpoint": path, "step": info.get("step"),
        "surface_mode": mode, "shape": list(pa.shape), "origin_zyx": [int(v) for v in origin],
        "forward_s": [round(ta, 1), round(tb, 1)],
    }
    if mode == "faces":
        res["sdf_in"] = delta_stats(pa[0], pb[0])     # voxels
        res["sdf_out"] = delta_stats(pa[1], pb[1])
        res["sdf"] = delta_stats(pa[0:2], pb[0:2])
        da, db = face_dist_from_sdf(pa[0:2]), face_dist_from_sdf(pb[0:2])
        body_a = (pa[0] > 0) & (pa[1] < 0)
        body_b = (pb[0] > 0) & (pb[1] < 0)
        res["body_dice"] = dice(body_a, body_b)
        o = 1
    elif mode == "sides":
        res["d_face"] = delta_stats(pa[0], pb[0])
        res["sdf"] = res["d_face"]
        da, db = pa[0], pb[0]
        res["body_dice"] = dice(pa[1] >= 0.5, pb[1] >= 0.5)
        res["body"] = delta_stats(pa[1], pb[1])
        o = 1
    else:
        res["sdf"] = delta_stats(pa[0], pb[0])
        da, db = np.abs(pa[0]), np.abs(pb[0])
        res["body_dice"] = dice(pa[0] <= 0, pb[0] <= 0)
        o = 0
    res["d_face"] = delta_stats(da, db)
    res["zero_set_dice@1"] = dice(da <= 1.0, db <= 1.0)
    res["zero_set_frac"] = [float((da <= 1.0).mean()), float((db <= 1.0).mean())]
    res["valid"] = delta_stats(pa[1 + o], pb[1 + o])
    res["ink"] = delta_stats(pa[2 + o], pb[2 + o])
    res["ink_dice@0.5"] = dice(pa[2 + o] >= 0.5, pb[2 + o] >= 0.5)
    # a near-empty mask makes that Dice meaningless: record how big it is
    res["ink_frac"] = [float((pa[2 + o] >= 0.5).mean()), float((pb[2 + o] >= 0.5).mean())]
    res["normal_angle_deg"] = angle_stats(line_angle_deg(pa[6 + o:9 + o], pb[6 + o:9 + o]))
    if pa.shape[0] >= 13 + o:  # the fibre head (vt, hz) sits after `spare`
        res["fiber"] = delta_stats(pa[11 + o:13 + o], pb[11 + o:13 + o])
    res["all_channels"] = delta_stats(pa, pb)
    del pa, pb
    return res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _row(name: str, mean: float, p99: float, mx: float, dc: Any, am: Any, t: Any) -> str:
    fd = lambda v: "-" if v is None else f"{float(v):.4f}"  # noqa: E731
    ft = "-" if t is None else f"{t[0]:.0f}/{t[1]:.0f}"
    return f"{name:<12} {mean:>9.5f} {p99:>9.5f} {mx:>9.4f} {fd(dc):>9} {fd(am):>9} {ft:>10}"


def table(results: list[dict[str, Any]]) -> str:
    head = (f"{'model':<12} {'mean|d|':>9} {'p99|d|':>9} {'max|d|':>9} "
            f"{'dice@0.5':>9} {'argmaxdt':>9} {'fwd s l/m':>10}")
    lines = [head, "-" * len(head)]
    for r in results:
        name = r["model"]
        if name == "student":
            key = r.get("d_face") or r["sdf"]
            lines.append(_row("student d_face", key["mean"], key["p99"], key["max"],
                              r.get("zero_set_dice@1"), None, r.get("forward_s")))
            a = r["all_channels"]
            lines.append(_row("  all heads", a["mean"], a["p99"], a["max"], r.get("body_dice"), None, None))
            continue
        base = r.get("fg") or r.get("cos") or r["all_channels"]
        lines.append(_row(name, base["mean"], base["p99"], base["max"],
                          r.get("dice@0.5"), r.get("argmax_change"), r.get("forward_s")))
        a = r["all_channels"]
        lines.append(_row("  all ch", a["mean"], a["p99"], a["max"], None, None, None))
        if "normal_angle_deg" in r:
            ang = r["normal_angle_deg"]
            lines.append(f"{'  normal deg':<12} mean {ang['mean']:.3f}  p99 {ang['p99']:.3f}  "
                         f"max {ang['max']:.3f}  (on surface: mean "
                         f"{r['normal_angle_deg_on_surface']['mean']:.3f})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _triple(s: str) -> tuple[int, int, int]:
    v = [int(t) for t in str(s).replace(" ", "").split(",")]
    if len(v) == 1:
        v = v * 3
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected 1 or 3 comma-separated ints, got {s!r}")
    return (v[0], v[1], v[2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/paris4_eval_long30k.json")
    ap.add_argument("--origin", type=_triple, default=(34432, 15360, 18688),
                    help="level-0 box origin z,y,x")
    ap.add_argument("--size", type=_triple, default=(256, 256, 256), help="level-0 box size z,y,x")
    ap.add_argument("--models", default="recto,fiber,lasagna",
                    help=f"comma list of {sorted(TEACHERS)} (or 'none')")
    ap.add_argument("--pair", default=None, help="saved (2,Z,Y,X) uint8 npy: [0] lossless, [1] mirror")
    ap.add_argument("--pair-l2", default=None, help="same, for the level-2 lasagna box")
    ap.add_argument("--lasagna-level", type=int, default=2)
    ap.add_argument("--lasagna-patch", type=int, default=192,
                    help="level-2 window for lasagna, centred on the level-0 box (0 = the bare "
                         "downscaled box, which the 7-stage encoder may refuse)")
    ap.add_argument("--student", default=None, help="student checkpoint (latest.pt)")
    ap.add_argument("--student-patch", type=int, default=128,
                    help="level-0 window for the student, at the box origin")
    ap.add_argument("--clip", type=float, default=20.0, help="student sdf clip (voxels)")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--device", default=None, help=f"cpu (default) | cuda (needs {ALLOW_LOCAL_GPU_ENV})")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--baseline-noise", type=float, default=0.0,
                    help="reference run: replace the mirror box with the LOSSLESS box plus white "
                         "gaussian noise of this many grey levels.  The volcomp numbers only mean "
                         "something next to the model's ordinary sensitivity to input noise; "
                         "`--baseline-noise 3.9` matches the measured volcomp rms error")
    ap.add_argument("--baseline-seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="always read the lossless box from S3")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)

    device = pick_device(a.device)
    if device.type == "cpu":
        torch.set_num_threads(int(a.threads))
    log(f"device={device} threads={torch.get_num_threads()}")
    cfg = load_config(a.config)
    names = [] if a.models.strip().lower() in ("", "none") else [s for s in a.models.split(",") if s.strip()]
    for n in names:
        if n not in TEACHERS:
            raise SystemExit(f"unknown teacher {n!r}; known: {sorted(TEACHERS)}")

    report: dict[str, Any] = {
        "config": os.path.abspath(a.config), "device": str(device),
        "volume": {"mirror_url": cfg.volume.url, "lossless_url": cfg.volume.alt_url},
        "box": {"origin_zyx": list(a.origin), "size_zyx": list(a.size), "level": cfg.volume.level},
        "models": names, "results": [],
    }

    # ---- the level-0 box ------------------------------------------------- #
    l0_names = [n for n in names if n in L0_TEACHERS]
    need_l0 = bool(l0_names) or bool(a.student)
    box_a = box_b = None
    if need_l0:
        if a.pair:
            pair = np.load(a.pair)
            if pair.shape[0] != 2 or pair.dtype != np.uint8:
                raise SystemExit(f"{a.pair}: expected a (2, Z, Y, X) uint8 array, got {pair.shape} {pair.dtype}")
            box_a, box_b = np.ascontiguousarray(pair[0]), np.ascontiguousarray(pair[1])
            del pair
            if tuple(box_a.shape) != tuple(a.size):
                log(f"WARNING {a.pair} holds {box_a.shape}, --size says {tuple(a.size)}; using the file")
            report["box"]["source"] = os.path.abspath(a.pair)
            report["box"]["size_zyx"] = list(box_a.shape)
        else:
            box_a, box_b, info = read_pair(cfg, a.origin, a.size, cfg.volume.level, cfg.budget,
                                           use_cache=not a.no_cache)
            report["box"].update(info)
        if a.baseline_noise > 0:
            rng = np.random.default_rng(int(a.baseline_seed))
            noised = box_a.astype(np.float32) + rng.normal(0.0, float(a.baseline_noise), box_a.shape)
            box_b = np.ascontiguousarray(np.clip(np.rint(noised), 0, 255).astype(np.uint8))
            del noised
            report["variant"] = f"baseline: lossless vs lossless + N(0, {a.baseline_noise}) grey levels"
            log(f"BASELINE run: the second box is the lossless box + N(0, {a.baseline_noise}), "
                f"not the mirror")
        report["ct"] = ct_stats(box_a, box_b)
        c = report["ct"]
        log(f"CT box {box_a.shape}: lossless mean {c['lossless_mean']:.1f} std {c['lossless_std']:.1f}; "
            f"|d| mean {c['mean']:.2f} p90 {c['p90']:.0f} p99 {c['p99']:.0f} max {c['max']:.0f} "
            f"PSNR {c['psnr_db']:.1f} dB")

    ct_mask = (box_a > 0) if box_a is not None else None

    for n in l0_names:
        report["results"].append(run_teacher(n, box_a, box_b, a.models_dir, device, ct_mask))
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)

    # ---- lasagna at level 2 ---------------------------------------------- #
    if "lasagna" in names:
        lv = int(a.lasagna_level)
        scale = 1 << lv
        core_size = tuple(max(1, s // scale) for s in a.size)
        core0 = tuple(o // scale for o in a.origin)
        if a.lasagna_patch:
            p = int(a.lasagna_patch)
            org2 = tuple(int(c + cs // 2 - p // 2) for c, cs in zip(core0, core_size))
            size2 = (p, p, p)
        else:
            org2, size2 = core0, core_size
        if a.pair_l2:
            pr = np.load(a.pair_l2)
            la, lb = np.ascontiguousarray(pr[0]), np.ascontiguousarray(pr[1])
            del pr
            l2info = {"source": os.path.abspath(a.pair_l2)}
        else:
            la, lb, l2info = read_pair(cfg, org2, size2, lv, cfg.budget, use_cache=False)
        core = tuple(slice(c - o, c - o + s) for c, o, s in zip(core0, org2, core_size))
        if any(sl.start < 0 or sl.stop > d for sl, d in zip(core, la.shape)):
            core = None
        report["lasagna_box"] = {**l2info, "level": lv, "origin_zyx": list(org2),
                                 "size_zyx": list(la.shape), "core_origin_zyx": list(core0),
                                 "core_size_zyx": list(core_size)}
        report["lasagna_ct"] = ct_stats(la, lb)
        c = report["lasagna_ct"]
        log(f"L{lv} CT box {la.shape}: |d| mean {c['mean']:.2f} p99 {c['p99']:.0f} max {c['max']:.0f} "
            f"PSNR {c['psnr_db']:.1f} dB")
        report["results"].append(run_lasagna(la, lb, a.models_dir, device, core))
        del la, lb
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)

    # ---- the student ------------------------------------------------------ #
    if a.student:
        p = int(a.student_patch) or box_a.shape[0]
        if any(p > s for s in box_a.shape):
            raise SystemExit(f"--student-patch {p} does not fit in the box {box_a.shape}")
        sa = np.ascontiguousarray(box_a[:p, :p, :p])
        sb = np.ascontiguousarray(box_b[:p, :p, :p])
        report["results"].append(run_student(a.student, sa, sb, a.origin, device, a.clip))
        del sa, sb

    print()
    print(table(report["results"]))
    print()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        log(f"wrote {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
