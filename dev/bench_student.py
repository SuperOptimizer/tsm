"""Benchmark student inference: torch bf16 vs TensorRT fp16, 50 % overlap vs receptive-field halo.

GPU ONLY -- nothing in this file runs on the CPU (engine builds, cuda timings, VRAM).
Run it when the box is free; the exact commands are at the bottom of this docstring.

One real input box (default 320x448x448, read once at 2.4 um from the config region and
cached as .npy in the scratchpad) is predicted with ``tsm.sliding.predict_box`` for every
configuration.  The scored core is the box minus a fixed ``HALO`` (128) on every side, so
every configuration is compared on identical voxels.

Configurations vary
  * backend: ``torch`` (bf16 autocast, channels_last / cudnn.benchmark as in production)
    vs ``trt`` (``tsm.trt.TRTStudent``, fp16; bf16 has no TensorRT 11 tactic for the
    decoder's 3D ConvTranspose),
  * patch: 128 / 192 / 256 / 320,
  * tiling: ``ov`` = 50 % overlap + gaussian blending (the production default, 8 windows
    per voxel) vs ``rf`` = receptive-field halo (``halo = rf_radius + 2``,
    ``step = patch - 2*halo``, uniform core weights).  ``rf`` needs ``patch > 2*halo``:
    with the canonical student (rf radius 122) that is patch >= 256, and it only *beats*
    50 % overlap in window count once ``patch > 4*halo`` (>= 512).  The 256/320 rows are
    there to measure exactly that trade-off (fewer, bigger windows but more of them).

Agreement is measured against ``torch_p128_ov`` (the current production path): SDF MAE in
voxels over the covered core, zero-crossing (``surface1``) Dice, and max/mean |dprob| over
the probability channels.

``body_stride=2`` (plan A1) is timed with a *randomly initialised* net: it is a different
architecture, so its outputs cannot be compared with the trained stride-1 weights -- only
its speed and VRAM are reported (``*_bs2`` rows, ``sdf_mae``/``dice`` omitted).

Commands (run later, GPU free):

    # 1. torch baseline + patch sweep (also writes the reference output for the agreement columns)
    uv run python dev/bench_student.py --configs torch_p128_ov,torch_p192_ov,torch_p256_ov,torch_p320_ov
    # 2. TensorRT fp16 (first call per patch builds and caches the engine; expect minutes)
    uv run python dev/bench_student.py --configs trt_p128_ov,trt_p192_ov,trt_p256_ov,trt_p320_ov
    # 3. receptive-field halo tiling (patch must be > 2*(rf+2) = 248: 256 and 320 only)
    uv run python dev/bench_student.py --configs torch_p256_rf,torch_p320_rf,trt_p256_rf,trt_p320_rf
    # 4. half-resolution body (random weights, speed only)
    uv run python dev/bench_student.py --configs torch_p128_ov_bs2,torch_p256_ov_bs2,trt_p256_ov_bs2
    # 5. the table
    uv run python dev/bench_student.py --table

Options: ``--config <run config>`` (default configs/paris4.json) for the volume/region,
``--checkpoint`` (default ``<out_dir>/train/latest.pt``), ``--box 320,448,448``,
``TSM_BENCH_REPEAT`` timed repeats (default 2).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from tsm.config import RegionCfg, load_config
from tsm.data import CLIP, decode_sdf
from tsm.infer import StudentNet, extract_surface, load_student, n_head_ch, rf_tiling
from tsm.limits import free_cuda
from tsm.sliding import WindowSpec, predict_box, prepare_net
from tsm.student import build_model, conv_macs, in_channels, normalize_ct

SCRATCH = os.environ.get(
    "TSM_SCRATCH", "/tmp/claude-1000/-home-forrest-tsm/193809ef-c338-4699-b345-e8f9520c6118/scratchpad"
)
DEFAULT_BOX = (320, 448, 448)
HALO = 128  # fixed scoring halo: core = box - 2*HALO is identical for every configuration
REPEAT = int(os.environ.get("TSM_BENCH_REPEAT", "2"))
REFERENCE = "torch_p128_ov"
PATCHES = (128, 192, 256, 320)


def configs() -> dict[str, dict]:
    """``<backend>_p<patch>_<ov|rf>[_bs2]`` -> spec knobs."""
    out: dict[str, dict] = {}
    for backend in ("torch", "trt"):
        for p in PATCHES:
            for tiling in ("ov", "rf"):
                for bs2 in (False, True):
                    name = f"{backend}_p{p}_{tiling}" + ("_bs2" if bs2 else "")
                    out[name] = {"backend": backend, "patch": p, "tiling": tiling, "body_stride": 2 if bs2 else 1}
    return out


CONFIGS = configs()


# --------------------------------------------------------------------------- #
# input box
# --------------------------------------------------------------------------- #
def fingerprint(path: str | None) -> str:
    """``<size>-<mtime_ns>`` of a file, ``none`` when missing: identifies a checkpoint that was
    overwritten in place (``latest.pt``), which the path alone cannot."""
    if not path:
        return "none"
    try:
        st = os.stat(os.path.expanduser(path))
    except OSError:
        return "none"
    return f"{st.st_size}-{st.st_mtime_ns}"


def manifest_diff(want: dict, have: dict | None) -> list[str]:
    """Keys of ``want`` that ``have`` does not reproduce (missing manifest -> all of them)."""
    if not have:
        return sorted(want)
    return sorted(k for k in want if json.dumps(have.get(k), sort_keys=True, default=str)
                  != json.dumps(want[k], sort_keys=True, default=str))


def load_box(cfg_path: str, shape: tuple[int, int, int]) -> np.ndarray:
    from tsm.cli import _open_level

    cfg = load_config(cfg_path)
    z0, y0, x0 = (int(v) for v in cfg.region.start_zyx)
    cache = os.path.join(SCRATCH, f"bench_student_box_{z0}_{y0}_{x0}_{'x'.join(map(str, shape))}.npy")
    # the file name carries only the region, so the volume identity goes into a manifest beside it:
    # a box cached from another volume/level must not be reused (R22)
    want = {"url": cfg.volume.url, "alt_url": cfg.volume.alt_url, "level": int(cfg.volume.level),
            "voxel_um": float(cfg.volume.voxel_um), "start_zyx": [z0, y0, x0], "shape": list(shape)}
    if os.path.exists(cache):
        try:
            have = json.load(open(cache + ".json"))
        except (OSError, ValueError):
            have = None
        diff = manifest_diff(want, have)
        if not diff:
            return np.load(cache)
        print(f"[bench] cached box {cache} was built with different {diff}: re-reading", flush=True)
    reader = _open_level(cfg, cfg.volume.level, cfg.volume.voxel_um, RegionCfg((z0, y0, x0), shape), 64)
    t = time.perf_counter()
    box = reader.read(z0, z0 + shape[0], y0, y0 + shape[1], x0, x0 + shape[2])
    dt = time.perf_counter() - t
    print(f"[bench] read {shape} from {reader.url} in {dt:.1f}s mean={box.mean():.1f} "
          f"zero_frac={(box == 0).mean():.3f}", flush=True)
    os.makedirs(SCRATCH, exist_ok=True)
    np.save(cache, box)
    with open(cache + ".json", "w") as fh:
        json.dump({**want, "source": reader.url}, fh, indent=1, sort_keys=True, default=str)
    return box


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #
def make_spec(c: dict, rf_radius: int) -> WindowSpec:
    p = int(c["patch"])
    if c["tiling"] == "rf":
        halo, step = rf_tiling(p, rf_radius)  # raises when 2*halo >= patch
        return WindowSpec(patch=p, step=step, halo=halo, weight="uniform", out_tile=p, batch=1,
                          dtype=torch.float32 if c["backend"] == "trt" else torch.bfloat16,
                          empty_window_max=-1, channels_last=False, cudnn_benchmark=True, prefetch=False,
                          backend=c["backend"], norm_scope="window")
    return WindowSpec(patch=p, step=p // 2, halo=HALO, weight="gaussian", out_tile=p, batch=1,
                      dtype=torch.float32 if c["backend"] == "trt" else torch.bfloat16,
                      empty_window_max=-1, channels_last=False, cudnn_benchmark=True, prefetch=False,
                      backend=c["backend"], norm_scope="window")


def build_student(cname: str, ckpt: str, device: torch.device, voxel_um: float, clip: float,
                  axis_path: str | None) -> tuple[StudentNet, dict, torch.nn.Module]:
    c = CONFIGS[cname]
    if c["body_stride"] == 2:  # a different net: random weights, speed only
        model, info = load_student(ckpt, "cpu")
        radial = bool(info.get("input_radial", False))
        model = build_model(widths=tuple(info["widths"]),
                            in_ch=in_channels(radial, bool(info.get("input_axis", False))),
                            surface_mode=info.get("surface_mode", "medial"), body_stride=2,
                            fiber=bool(info.get("fiber", False)),
                            fiber_mode=str(info.get("fiber_mode", "class")),
                            gap_class=bool(info.get("gap_class", False))).eval().to(device)
        info = {**info, "body_stride": 2, "random_weights": True,
                "rf_radius": int(model.receptive_field_radius())}
    else:
        model, info = load_student(ckpt, device)
    net = StudentNet(model, voxel_um, clip, tta="none", surface_mode=info.get("surface_mode", "medial"),
                     input_radial=bool(info.get("input_radial", False)),
                     axis=axis_path or info.get("axis_path"),
                     input_axis=bool(info.get("input_axis", False)),
                     axis_tangent=bool(info.get("axis_tangent", False)),
                     gap_class=bool(info.get("gap_class", False)),
                     lsd=bool(info.get("lsd", False)),
                     fiber_mode=str(info.get("fiber_mode", "class"))).to(device)
    return net, info, model


def run_config(cname: str, box: np.ndarray, ckpt: str, cfg_path: str, results_path: str) -> dict:
    c = CONFIGS[cname]
    cfg = load_config(cfg_path)
    device = torch.device("cuda")
    free_cuda()
    voxel_um, clip = float(cfg.volume.voxel_um), CLIP
    ckpt_fp = fingerprint(ckpt)
    net, info, model = build_student(cname, ckpt, device, voxel_um, clip, None)
    smode = info.get("surface_mode", "medial")
    nhead = n_head_ch(smode, bool(info.get("fiber", False)), str(info.get("fiber_mode", "class")),
                      bool(info.get("gap_class", False)), bool(info.get("lsd", False)))
    rf = int(info.get("rf_radius", model.receptive_field_radius()))
    row: dict = {"config": cname, "backend": c["backend"], "patch": c["patch"], "tiling": c["tiling"],
                 "body_stride": c["body_stride"], "rf_radius": rf, "surface_mode": smode,
                 "params_m": round(model.num_params() / 1e6, 2), "box": list(box.shape),
                 "checkpoint": os.path.abspath(os.path.expanduser(ckpt)), "checkpoint_fingerprint": ckpt_fp,
                 "random_weights": bool(info.get("random_weights", False))}
    try:
        spec = make_spec(c, rf)
    except ValueError as exc:  # rf halo does not fit into the patch
        row["error"] = str(exc)[:200]
        _append(results_path, row)
        print(json.dumps(row), flush=True)
        return row
    row.update({"step": spec.step, "halo": spec.halo, "weight": spec.weight,
                "windows_per_voxel": round((spec.patch / spec.step) ** 3, 2)})
    # MACs scale exactly with the voxel count (fully convolutional, patch divisible by the
    # divisor), so count them on a small window: a full 256^3 / 320^3 fp32 forward here alone
    # is several GiB and OOMed the 16 GB card before the engine was even built.
    p0 = model.divisor * max(1, -(-64 // model.divisor))
    row["gmac_per_window"] = round(
        conv_macs(model, torch.zeros(1, model.in_ch, *(p0,) * 3, device=device))
        * (spec.patch / p0) ** 3 / 1e9, 1)
    free_cuda()
    build_s = 0.0
    if spec.backend == "trt":
        from tsm.trt import TRTStudent

        try:
            engine = TRTStudent.build_or_load(model, patch=spec.patch, batch=spec.batch, precision="fp16",
                                              device=device, workspace_gb=float(os.environ.get("TSM_TRT_WS", "10")))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            _append(results_path, row)
            print(json.dumps(row), flush=True)
            return row
        build_s = engine.build_seconds
        row.update({"engine_mb": round(os.path.getsize(engine.engine_path) / (1 << 20)),
                    "trt_device_mb": round(engine.device_memory_mb)})
        net = StudentNet(engine, voxel_um, clip, tta="none", surface_mode=smode,
                         input_radial=bool(info.get("input_radial", False)), axis=info.get("axis_path"),
                         input_axis=bool(info.get("input_axis", False)),
                         axis_tangent=bool(info.get("axis_tangent", False)),
                         gap_class=bool(info.get("gap_class", False)),
                         lsd=bool(info.get("lsd", False))).to(device)
        del model
        free_cuda()
    net = prepare_net(net, spec, device)
    core_shape = tuple(s - 2 * HALO for s in box.shape)
    args = (net, normalize_ct, spec, nhead, "none", None, device)
    kw = dict(core_offset=(HALO,) * 3, core_shape=core_shape,
              box_origin_zyx=tuple(int(v) for v in cfg.region.start_zyx))
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
    except torch.OutOfMemoryError as exc:
        row["error"] = f"OOM: {str(exc)[:160]}"
        _append(results_path, row)
        print(json.dumps(row), flush=True)
        return row
    tile_s = min(times)
    row.update({
        "n_windows": n_win, "warmup_s": round(warm_s, 1), "build_s": round(build_s, 1),
        "tile_s": round(tile_s, 2), "tile_s_mean": round(sum(times) / len(times), 2),
        "windows_per_s": round(n_win / tile_s, 2),
        "mvox_per_s": round(float(np.prod(core_shape)) / tile_s / 1e6, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1 << 20)),
    })
    np.save(_out_path(box.shape, cname, ckpt_fp), q)
    if not row["random_weights"]:
        row.update(compare(box.shape, cname, REFERENCE, clip, ckpt_fp))
    _append(results_path, row)
    print(json.dumps(row), flush=True)
    del net
    free_cuda()
    return row


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #
def _out_path(shape, cname: str, ckpt_fp: str = "none") -> str:
    """Cached prediction file.  The checkpoint fingerprint is part of the name so that outputs of
    two different checkpoints (or of a retrained ``latest.pt``) are never compared with each other."""
    return os.path.join(SCRATCH, f"bench_student_out_{'x'.join(map(str, shape))}_ck{ckpt_fp}_{cname}.npy")


def compare(shape, cname: str, ref: str, clip: float, ckpt_fp: str = "none") -> dict:
    """SDF MAE (voxels), zero-crossing Dice and probability-channel differences vs ``ref``
    (both outputs of the same checkpoint -- the fingerprint is part of the file name)."""
    pa, pb = _out_path(shape, cname, ckpt_fp), _out_path(shape, ref, ckpt_fp)
    if cname == ref or not (os.path.exists(pa) and os.path.exists(pb)):
        return {}
    a, b = np.load(pa), np.load(pb)
    if a.shape != b.shape:
        return {}
    cov = (a[0] != 0) & (b[0] != 0)
    if not cov.any():
        return {"error": "no common coverage"}
    da, db = decode_sdf(a[0], clip), decode_sdf(b[0], clip)
    band = cov & (np.abs(db) < clip)
    sa = extract_surface(a[0], a[1], clip)
    sb = extract_surface(b[0], b[1], clip)
    dice = 2 * np.logical_and(sa, sb).sum() / max(1, sa.sum() + sb.sum())
    d = np.abs(a[1:].astype(np.int16) - b[1:].astype(np.int16)) / 255.0
    return {"ref": ref, "covered": round(float(cov.mean()), 4),
            "sdf_mae": round(float(np.abs(da - db)[cov].mean()), 4),
            "sdf_mae_band": round(float(np.abs(da - db)[band].mean()), 4) if band.any() else None,
            "sdf_max": round(float(np.abs(da - db)[cov].max()), 3),
            "zero_dice": round(float(dice), 5), "n_surface": int(sa.sum()), "n_surface_ref": int(sb.sum()),
            "max_dprob": round(float(d.max()), 4), "mean_dprob": round(float(d.mean()), 5)}


# --------------------------------------------------------------------------- #
# results / table
# --------------------------------------------------------------------------- #
def _append(path: str, row: dict) -> None:
    rows = [r for r in _load(path) if r["config"] != row["config"]]
    rows.append(row)
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1)


def _load(path: str) -> list[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


COLS = ["config", "patch", "step", "halo", "weight", "windows_per_voxel", "body_stride", "gmac_per_window",
        "n_windows", "build_s", "tile_s", "windows_per_s", "mvox_per_s", "peak_vram_mb", "trt_device_mb",
        "sdf_mae", "sdf_mae_band", "zero_dice", "max_dprob", "error"]


def table(results_path: str) -> str:
    rows = _load(results_path)
    out = ["| " + " | ".join(COLS) + " |", "|" + "|".join("---" for _ in COLS) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/paris4.json")
    ap.add_argument("--checkpoint", default=None, help="default <out_dir>/train/latest.pt")
    ap.add_argument("--configs", default="", help="comma separated names from CONFIGS")
    ap.add_argument("--box", default=",".join(str(v) for v in DEFAULT_BOX))
    ap.add_argument("--results", default=os.path.join(SCRATCH, "bench_student.json"))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(sorted(CONFIGS)))
        return
    if a.table:
        print(table(a.results))
        return
    if not torch.cuda.is_available():
        raise SystemExit("dev/bench_student.py needs a GPU")
    cfg = load_config(a.config)
    ckpt = a.checkpoint or os.path.join(cfg.out_dir, "train", "latest.pt")
    shape = tuple(int(v) for v in a.box.split(","))
    box = load_box(a.config, shape)
    names = [n.strip() for n in a.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; --list to see them all")
    for n in names:
        try:  # one OOM must not abort the sweep: record it and carry on
            run_config(n, box, ckpt, a.config, a.results)
        except torch.OutOfMemoryError as exc:
            row = {"config": n, **CONFIGS[n], "error": f"OOM: {str(exc)[:160]}"}
            _append(a.results, row)
            print(json.dumps(row), flush=True)
            free_cuda()
    print(table(a.results))


if __name__ == "__main__":
    main()
