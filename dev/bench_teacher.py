"""Benchmark teacher inference configurations on one real input box.

    uv run python dev/bench_teacher.py --teacher recto --configs trt,trt_nostream,trt_p320
    uv run python dev/bench_teacher.py --teacher recto --table

The box (default 384x512x512 for every teacher, read at the teacher's level from
the region start of configs/paris4_roi.json) is cached as .npy in the scratchpad.
The core (box minus a fixed 128 halo = 128x256x256) is the same for every
configuration, so patch sizes / TTA / precisions are compared on identical voxels.
For every configuration a fresh teacher is loaded, ``predict_box`` runs once for
warmup (compile / cudnn autotune / TRT build happen there) and ``TSM_BENCH_REPEAT``
times timed.  Outputs are compared with the cached ``baseline`` (torch bf16, native
patch, no TTA) and ``fp32_ref`` (torch fp32) outputs: max/mean |dprob|, Dice@0.5 of
the thresholded band and medial-surface distances (median / p90 / frac > 5 vox, both
directions).  Results accumulate in a JSON file; ``--table`` prints the markdown table.
Each invocation should stay under ~3 minutes: pass a few configs at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from tsm.config import RegionCfg, load_config
from tsm.limits import Budget, cuda_guard, free_cuda
from tsm.sliding import WindowSpec, predict_box, prepare_net
from tsm.teachers import TEACHERS, load_teacher

SCRATCH = os.environ.get(
    "TSM_SCRATCH", "/tmp/claude-1000/-home-forrest-tsm/193809ef-c338-4699-b345-e8f9520c6118/scratchpad"
)
BASE = dict(channels_last=False, cudnn_benchmark=False, compile=None, batch=1, empty_window_max=-1,
            backend="torch")
TRT = dict(BASE, backend="trt", cudnn_benchmark=True)
CONFIGS: dict[str, dict] = {
    "baseline": dict(BASE),  # torch bf16 autocast, native patch (the pre-TRT production path)
    "fp32_ref": dict(BASE, dtype="fp32"),  # torch fp32, no autocast: the numerical reference
    "fp16_torch": dict(BASE, dtype="fp16", cudnn_benchmark=True),  # torch fp16 autocast: precision-noise yardstick
    "cudnn": dict(BASE, cudnn_benchmark=True),
    "cl_cudnn": dict(BASE, channels_last=True, cudnn_benchmark=True),
    "compile_maxauto": dict(BASE, channels_last=True, cudnn_benchmark=True, compile="max-autotune-no-cudagraphs"),
    # TensorRT fp16 (production default in configs/paris4.json); engine on its own CUDA stream
    "trt": dict(TRT),
    "trt_nostream": dict(TRT, trt_stream=False),  # engine enqueued on the torch stream (old behaviour)
    "trt_b2": dict(TRT, batch=2),
    "trt_fp32": dict(TRT, trt_precision="fp32"),
    # bigger windows (fully convolutional nets; instance norm -> mild window-size dependence)
    "trt_p256": dict(TRT, patch=256),
    "trt_p320": dict(TRT, patch=320, workspace_gb=12),  # recto needs a 6.7 GB tensor at 320^3
    "trt_p384": dict(TRT, patch=384, workspace_gb=13),
    "torch_p320": dict(BASE, cudnn_benchmark=True, patch=320),
    # INT8 (Model Optimizer PTQ, calibrated on real windows of configs/paris4_small.json)
    "trt_int8": dict(TRT, trt_precision="int8"),
    "trt_int8_mse": dict(TRT, trt_precision="int8", calib_algorithm="mse"),
    "trt_int8_p320": dict(TRT, trt_precision="int8", patch=320),
    # test-time augmentation (same engine; flips / rot90 are tensor ops)
    "trt_flip8": dict(TRT, tta="flip8"),
    "trt_flip4rot2": dict(TRT, tta="flip4rot2"),
    "trt_flip8_rot4": dict(TRT, tta="flip8_rot4"),
    "torch_flip8": dict(BASE, cudnn_benchmark=True, tta="flip8"),
    "trt_int8_flip8": dict(TRT, trt_precision="int8", tta="flip8"),
}
DEFAULT_BOX = (384, 512, 512)
HALO = 128  # fixed: core = box - 2*halo is identical for every patch size
REPEAT = int(os.environ.get("TSM_BENCH_REPEAT", "2"))  # timed tiles after warmup; tile_s = min, tile_s_mean also kept
# paris4 slab (256 x 6144 x 6144, out_tile 384, halo 128): windows per teacher at native patch
SLAB_CORE_VOXELS = {"recto": 256 * 6144 * 6144, "ink": 256 * 6144 * 6144, "m7": 256 * 6144 * 6144,
                    "lasagna": 64 * 1536 * 1536, "m7_l2": 64 * 1536 * 1536}


def parse_teacher(entry: str):
    """'m7@l2' -> (TeacherSpec, level_shift 2, out name 'm7_l2')."""
    base, _, shift = str(entry).partition("@l")
    k = int(shift) if shift else 0
    return TEACHERS[base], k, (f"{base}_l{k}" if k else base)


def _level_for(tspec, cfg, level_shift: int = 0):
    import math

    k = int(round(math.log2(tspec.voxel_um / cfg.volume.voxel_um))) + int(level_shift)
    return cfg.volume.level + k, 2 ** k


def _open(cfg, level, voxel_um, start, shape, probe=64):
    from tsm.cli import _open_level

    return _open_level(cfg, level, voxel_um, RegionCfg(start, shape), probe)


def load_box(tspec, cfg_path: str, shape: tuple[int, int, int], level_shift: int = 0) -> tuple[np.ndarray, dict]:
    """Real box from the volume (cached).  Returns (box_u8, read_stats)."""
    cfg = load_config(cfg_path)
    level, factor = _level_for(tspec, cfg, level_shift)
    z0, y0, x0 = (int(v) // factor for v in cfg.region.start_zyx)
    cache = os.path.join(SCRATCH, f"bench_box_{tspec.name}_L{level}_{z0}_{y0}_{x0}_{'x'.join(map(str, shape))}.npy")
    stats = {"cached": os.path.exists(cache)}
    if os.path.exists(cache):
        return np.load(cache), stats
    reader = _open(cfg, level, tspec.voxel_um * factor, (z0, y0, x0), shape)
    t = time.perf_counter()
    box = reader.read(z0, z0 + shape[0], y0, y0 + shape[1], x0, x0 + shape[2])
    dt = time.perf_counter() - t
    stats.update({"read_s": dt, "mib_s": box.nbytes / (1 << 20) / dt, "source": reader.url})
    print(f"[bench] read {shape} from {reader.url} in {dt:.1f}s ({stats['mib_s']:.1f} MiB/s) "
          f"mean={box.mean():.1f} zero_frac={(box == 0).mean():.3f}", flush=True)
    np.save(cache, box)
    return box, stats


def calibration_windows(tspec, patch: int, n: int = 32, cfg_path: str = "configs/paris4_small.json",
                        level_shift: int = 0) -> np.ndarray:
    """Real windows for int8 PTQ from the start of the small-slab region (disjoint from the ROI box)."""
    from tsm.trt import collect_calibration_windows

    cfg = load_config(cfg_path)
    level, factor = _level_for(tspec, cfg, level_shift)
    start = tuple(int(v) // factor for v in cfg.region.start_zyx)
    cache = os.path.join(SCRATCH, f"calib_{tspec.name}_L{level}_p{patch}_{'_'.join(map(str, start))}.npy")
    reader = None if os.path.exists(cache) else _open(cfg, level, tspec.voxel_um * factor, start, (patch,) * 3)
    return collect_calibration_windows(reader, start, patch, n=n, cache_path=cache, empty_max=0)


def make_spec(tspec, c: dict) -> WindowSpec:
    p = int(c.get("patch") or tspec.patch[0])
    return WindowSpec(
        patch=p, out_tile=p, halo=HALO, batch=int(c.get("batch", 1)),
        dtype={"fp32": torch.float32, "fp16": torch.float16}.get(c.get("dtype"), torch.bfloat16),
        empty_window_max=int(c.get("empty_window_max", -1)), channels_last=bool(c.get("channels_last", False)),
        cudnn_benchmark=bool(c.get("cudnn_benchmark", False)), compile=c.get("compile"),
        backend=str(c.get("backend", "torch")), prefetch=False, tta=c.get("tta", "none"),
        tta_field="lasagna" if tspec.name == "lasagna" else "scalar",
    )


def run_config(entry: str, cname: str, box: np.ndarray, results_path: str, models_dir: str | None) -> dict:
    c = CONFIGS[cname]
    tspec0, level_shift, name = parse_teacher(entry)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    free_cuda()
    t = time.perf_counter()
    net, tspec = load_teacher(tspec0.name, models_dir or os.path.expanduser("~/.cache/tsm-models"), device=device,
                              dtype=torch.float32)
    load_s = time.perf_counter() - t
    spec = make_spec(tspec, c)
    if tspec.activation == "softmax":
        c_out = 2
    elif tspec.target in net.spec.task_heads:
        c_out = int(net.spec.task_heads[tspec.target])
    elif tspec.target in net.spec.task_decoders:
        c_out = int(net.spec.task_decoders[tspec.target].out_channels)
    else:
        c_out = int(net.spec.decoder.out_channels)
    core_shape = tuple(s - 2 * HALO for s in box.shape)
    row: dict = {"teacher": name, "config": cname, "batch": spec.batch, "patch": spec.patch, "tta": spec.tta,
                 "passes": spec.n_passes, "load_s": round(load_s, 1), "box": list(box.shape),
                 "core": list(core_shape), "n_windows": None, "precision": c.get("trt_precision", "auto"),
                 "stream": bool(c.get("trt_stream", True))}
    build_s = 0.0
    if spec.backend == "trt":
        from tsm.trt import TRTTeacher

        precision = c.get("trt_precision", "auto")
        calib = calibration_windows(tspec, int(tspec.patch[0]), level_shift=level_shift) if precision == "int8" else None
        free_before = torch.cuda.mem_get_info()[0]
        try:
            trt_net = TRTTeacher.build_or_load(net, tspec, batch=spec.batch, precision=precision, patch=spec.patch,
                                               use_stream=bool(c.get("trt_stream", True)), calib=calib,
                                               calib_algorithm=str(c.get("calib_algorithm", "max")),
                                               workspace_gb=float(c.get("workspace_gb", 6.0)))
        except Exception as exc:
            row.update({"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            _append(results_path, row)
            print(json.dumps(row), flush=True)
            return row
        del calib
        build_s = trt_net.build_seconds
        row["precision"] = trt_net.precision
        row["engine_mb"] = round(os.path.getsize(trt_net.engine_path) / (1 << 20))
        row["trt_device_mb"] = round(trt_net.device_memory_mb)
        del net
        free_cuda()
        net = trt_net
        row["trt_alloc_mb"] = round((free_before - torch.cuda.mem_get_info()[0]) / (1 << 20))
    net = prepare_net(net, spec, device)
    args = (net, tspec.normalizer, spec, c_out, tspec.activation, tspec.fg_channel, device)
    kw = dict(select=tspec.select, core_offset=(HALO,) * 3, core_shape=core_shape)
    try:
        torch.cuda.synchronize()
        t = time.perf_counter()
        q, n_win = predict_box(box, *args, **kw)
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - t
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(REPEAT):
            t = time.perf_counter()
            q, n_win = predict_box(box, *args, **kw)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        tile_s = min(times)
        row["tile_s_mean"] = round(sum(times) / len(times), 2)
    except torch.OutOfMemoryError as exc:
        row.update({"error": f"OOM: {str(exc)[:120]}"})
        _append(results_path, row)
        print(json.dumps(row), flush=True)
        return row
    peak_mb = torch.cuda.max_memory_allocated() / (1 << 20)
    row.update({"n_windows": n_win, "warmup_s": round(warm_s, 1), "build_s": round(build_s, 1),
                "tile_s": round(tile_s, 2), "windows_per_s": round(n_win / tile_s, 3),
                "passes_per_s": round(n_win * spec.n_passes / tile_s, 3),
                "mvox_per_s": round(float(np.prod(core_shape)) / tile_s / 1e6, 2),
                "peak_vram_mb": round(peak_mb), "gpu_used_mb_other": _gpu_used_other_mb()})
    np.save(_out_path(name, box.shape, cname), q)
    for ref in ("baseline", "fp32_ref", "trt"):
        stats = compare(name, box.shape, cname, ref)
        if stats:
            suffix = {"baseline": "", "fp32_ref": "_fp32", "trt": "_trt"}[ref]
            row.update({k + suffix: v for k, v in stats.items()})
    _append(results_path, row)
    print(json.dumps(row), flush=True)
    del net
    torch._dynamo.reset()
    free_cuda()
    return row


def _gpu_used_other_mb() -> int:
    """Memory used on the GPU by other processes (the shared teacher run) -- timings are noisy when > 0."""
    try:
        free, total = torch.cuda.mem_get_info()
        return round((total - free - torch.cuda.memory_reserved()) / (1 << 20))
    except Exception:
        return -1


def _out_path(name: str, shape, cname: str) -> str:
    return os.path.join(SCRATCH, f"bench_out_{name}_{'x'.join(map(str, shape))}_{cname}.npy")


def surface_metrics(qa: np.ndarray, qb: np.ndarray, thr: int = 128, far: float = 5.0) -> dict:
    """Dice@0.5 of the thresholded band (channel 0) and medial-surface distances a->b / b->a."""
    from edt import edt as _edt

    from tsm.labels import medial_surface

    a, b = qa[0] >= thr, qb[0] >= thr
    dice = 2 * np.logical_and(a, b).sum() / max(1, a.sum() + b.sum())
    out = {"dice": round(float(dice), 5)}
    if not a.any() or not b.any():
        return out
    ma, mb = medial_surface(a), medial_surface(b)
    for tag, src, dst in (("ab", ma, mb), ("ba", mb, ma)):
        d = _edt(np.ascontiguousarray(~dst, dtype=np.uint8), black_border=False)[src]
        if d.size == 0:
            continue
        out.update({f"med_{tag}": round(float(np.median(d)), 2), f"p90_{tag}": round(float(np.percentile(d, 90)), 2),
                    f"far_{tag}": round(float((d > far).mean()), 4)})
    out["n_medial"] = int(ma.sum())
    return out


def normal_metrics(qa: np.ndarray, qb: np.ndarray, stride: int = 4) -> dict:
    """lasagna: angle between decoded normals (line field) where both cos channels >= 0.5."""
    from tsm.labels import decode_lasagna_normal

    sl = (slice(None), slice(None, None, stride), slice(None, None, stride), slice(None, None, stride))
    a, b = qa[sl].astype(np.float32) / 255.0, qb[sl].astype(np.float32) / 255.0
    m = (a[0] >= 0.5) & (b[0] >= 0.5)
    if m.sum() < 10:
        return {}
    na, nb = decode_lasagna_normal(a), decode_lasagna_normal(b)
    cos = np.abs((na * nb).sum(0)[m]).clip(0, 1)
    ang = np.degrees(np.arccos(cos))
    return {"ang_med": round(float(np.median(ang)), 2), "ang_p90": round(float(np.percentile(ang, 90)), 2),
            "ang_gt10": round(float((ang > 10).mean()), 4)}


def compare(name: str, shape, cname: str, ref: str) -> dict | None:
    """max/mean |dprob|, band Dice, medial-surface distances (and normal angles for lasagna) of ``cname`` vs ``ref``."""
    pa, pb = _out_path(name, shape, cname), _out_path(name, shape, ref)
    if cname == ref or not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    q, qb = np.load(pa), np.load(pb)
    if q.shape != qb.shape:
        return None
    d = np.abs(q.astype(np.int16) - qb.astype(np.int16)) / 255.0
    out = {"max_dprob": round(float(d.max()), 4), "mean_dprob": round(float(d.mean()), 5),
           "frac_gt002": round(float((d > 0.02).mean()), 6)}
    out.update(surface_metrics(q, qb))
    if name.startswith("lasagna") and q.shape[0] == 8:
        out.update(normal_metrics(q, qb))
    return out


def _append(path: str, row: dict) -> None:
    rows = _load(path)
    rows = [r for r in rows if not (r["teacher"] == row["teacher"] and r["config"] == row["config"])]
    rows.append(row)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)


def _load(path: str) -> list[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


SPEED_COLS = ["teacher", "config", "patch", "passes", "precision", "n_windows", "build_s", "tile_s", "tile_s_mean",
              "windows_per_s", "passes_per_s", "mvox_per_s", "slab_h", "peak_vram_mb", "trt_device_mb",
              "gpu_used_mb_other", "error"]
AGREE_COLS = ["teacher", "config", "ref", "max_dprob", "mean_dprob", "dice", "med_ab", "p90_ab", "far_ab", "med_ba",
              "p90_ba", "far_ba", "ang_med", "ang_p90", "ang_gt10"]


def table(results_path: str, teacher: str | None) -> str:
    rows = [r for r in _load(results_path) if teacher is None or r["teacher"] == teacher]
    out = ["| " + " | ".join(SPEED_COLS) + " |", "|" + "|".join("---" for _ in SPEED_COLS) + "|"]
    for r in rows:
        if r.get("mvox_per_s"):
            vox = SLAB_CORE_VOXELS.get(r["teacher"], SLAB_CORE_VOXELS["recto"])
            r = dict(r, slab_h=round(vox / (r["mvox_per_s"] * 1e6) / 3600, 2))
        out.append("| " + " | ".join(str(r.get(c, "")) for c in SPEED_COLS) + " |")
    out += ["", "| " + " | ".join(AGREE_COLS) + " |", "|" + "|".join("---" for _ in AGREE_COLS) + "|"]
    for r in rows:
        for ref, suffix in (("baseline", ""), ("fp32_ref", "_fp32"), ("trt", "_trt")):
            if f"dice{suffix}" not in r:
                continue
            vals = {"teacher": r["teacher"], "config": r["config"], "ref": ref}
            vals.update({c: r.get(c + suffix, "") for c in AGREE_COLS[3:]})
            out.append("| " + " | ".join(str(vals.get(c, "")) for c in AGREE_COLS) + " |")
    return "\n".join(out)


def refresh(results_path: str) -> None:
    rows = _load(results_path)
    for r in rows:  # refresh deltas from the saved outputs (refs may have been run later)
        for ref, suffix in (("baseline", ""), ("fp32_ref", "_fp32"), ("trt", "_trt")):
            st = compare(r["teacher"], tuple(r["box"]), r["config"], ref)
            if st:
                r.update({k + suffix: v for k, v in st.items()})
    with open(results_path, "w") as fh:
        json.dump(rows, fh, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="recto", help="teacher name ('m7@l2' = 2 levels coarser); 'all' with --table")
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--box", default=None, help="Z,Y,X (default 384,512,512)")
    ap.add_argument("--config-json", default="configs/paris4_roi.json")
    ap.add_argument("--results", default=os.path.join(SCRATCH, "bench_results_v2.json"))
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="with --table: recompute agreement from saved outputs")
    ap.add_argument("--read-only", action="store_true", help="only read/cache the box and report MiB/s")
    ap.add_argument("--vram-frac", type=float, default=0.8, help="cuda_guard fraction (0.8 = production)")
    a = ap.parse_args()
    if a.table:
        if a.refresh:
            refresh(a.results)
        print(table(a.results, None if a.teacher == "all" else a.teacher))
        return 0
    tspec, level_shift, _ = parse_teacher(a.teacher)
    shape = tuple(int(v) for v in a.box.split(",")) if a.box else DEFAULT_BOX
    box, stats = load_box(tspec, a.config_json, shape, level_shift)  # type: ignore[arg-type]
    print(f"[bench] box {box.shape} mean={box.mean():.1f} {stats}", flush=True)
    if a.read_only:
        return 0
    cuda_guard(Budget(vram_frac=a.vram_frac))
    for cname in a.configs.split(","):
        run_config(a.teacher, cname, box, a.results, a.models_dir)
    print(table(a.results, a.teacher))
    return 0


if __name__ == "__main__":
    sys.exit(main())
