"""tsm <stage> config.json [--dry-run] [--force]"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from tsm.config import RegionCfg, RunCfg, load_config
from tsm.limits import MemWatchdog, estimate_and_assert, peak_rss_mb
from tsm.volume import BrickWriter, VolumeReader, codec_names, open_cached_reader

PROBE_BRICK = 256


def _open(label: str, url: str, cfg: RunCfg) -> VolumeReader | None:
    try:
        r = VolumeReader(url, cfg.volume.level, cfg.volume.voxel_um, cfg.budget)
    except Exception as exc:
        print(f"[tsm] {label}: UNAVAILABLE ({type(exc).__name__}: {exc})")
        return None
    print(f"[tsm] {label}: {url}")
    print(f"[tsm]   shape={r.shape} chunks={r.chunks} dtype={r.dtype.name} codecs={codec_names(r.array)}")
    return r


def _timed_read(label: str, r: VolumeReader, z0: int, y0: int, x0: int) -> np.ndarray | None:
    n = PROBE_BRICK
    t = time.perf_counter()
    try:
        a = r.read(z0, z0 + n, y0, y0 + n, x0, x0 + n)
    except Exception as exc:
        print(f"[tsm] {label}: read FAILED ({type(exc).__name__}: {exc})")
        return None
    dt = time.perf_counter() - t
    mb = a.nbytes / (1 << 20)
    print(
        f"[tsm] {label}: read {n}^3 in {dt:.2f}s ({mb / dt:.1f} MiB/s), "
        f"mean={a.mean():.2f} nonzero={float((a > 0).mean()):.3f}"
    )
    if not a.any():
        print(f"[tsm] {label}: WARNING all-zero brick (chunks likely absent at this source)")
    return a


def stage_probe(cfg: RunCfg, args: argparse.Namespace) -> int:
    z0, y0, x0 = cfg.region.start_zyx
    estimate_and_assert(
        [
            ("probe_primary", (PROBE_BRICK,) * 3, np.uint8),
            ("probe_alt", (PROBE_BRICK,) * 3, np.uint8),
        ],
        cfg.budget,
    )
    if args.dry_run:
        print("[tsm] dry run: stopping before I/O")
        return 0
    primary = _open("primary", cfg.volume.url, cfg)
    alt = _open("alt", cfg.volume.alt_url, cfg) if cfg.volume.alt_url else None
    if primary is None and alt is None:
        print("[tsm] no readable volume source")
        return 1
    a = _timed_read("primary", primary, z0, y0, x0) if primary else None
    b = _timed_read("alt", alt, z0, y0, x0) if alt else None
    if a is not None and b is not None:
        if not a.any() or not b.any():
            print("[tsm] delta primary vs alt: SKIPPED (one source returned an empty brick)")
        else:
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            print(f"[tsm] delta primary vs alt: mean|d|={d.mean():.4f} max|d|={int(d.max())}")
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0 if (a is not None or b is not None) else 1


# --------------------------------------------------------------------------- #
# teacher stage
# --------------------------------------------------------------------------- #
LASAGNA_CHANNELS = ["cos", "grad_mag", "dir0_z", "dir1_z", "dir0_y", "dir1_y", "dir0_x", "dir1_x"]
TEACHER_CHANNELS = {
    "recto": ["recto"],
    "m7": ["surface"],
    "ink": ["ink"],
    "lasagna": LASAGNA_CHANNELS,
    # 4-way softmax, all four probabilities written (fg_channel is None): background,
    # vertical fiber, horizontal/angular fiber, ink.
    "fiber": ["fiber_bg", "fiber_vt", "fiber_hz", "fiber_ink"],
}
TEACHER_DEFAULTS = {
    "models": ["recto", "m7", "ink", "lasagna"],
    "out_tile": 192,
    # per-model values may be given as {"recto": ..., "m7@l2": ..., "m7": ...} (entry name, then base name)
    "tta": "none",  # none | flip8 | flip4rot2 | flip8_rot4 (bool accepted: true -> flip8)
    "patch": None,  # window size override (None = teacher native); multiple of 32
    "skip_existing": True,
    "halo": None,
    "step": None,
    "batch": 1,
    "dtype": "bf16",  # autocast dtype for the torch backend: bf16 | fp16 | fp32
    "empty_max": 0,
    "empty_window_max": 0,
    "channels_last": False,
    "cudnn_benchmark": True,
    "compile": None,
    "prefetch": True,
    "backend": "torch",
    "trt_precision": "auto",  # auto (fp16) | fp16 | bf16 | fp32 | int8 (Model Optimizer PTQ, see tsm.trt)
    "trt_stream": True,  # run the engine on its own CUDA stream (pipelined with normalisation / blending)
    "trt_calib_config": "configs/paris4_small.json",  # region whose CT calibrates int8 (its start, same level)
    "trt_calib_windows": 32,
    "trt_calib_algorithm": "max",  # max | mse
    "models_dir": None,
    "probe_brick": 64,
}


def _teacher_opts(cfg: RunCfg) -> dict:
    raw = cfg.extra.get("teacher", {})
    if not isinstance(raw, dict):
        raise ValueError("extra.teacher must be an object")
    unknown = sorted(set(raw) - set(TEACHER_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.teacher keys: {unknown}")
    opts = dict(TEACHER_DEFAULTS)
    opts.update(raw)
    for v in (opts["patch"].values() if isinstance(opts["patch"], dict) else [opts["patch"]]):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 32):
            raise ValueError(f"teacher.patch must be a positive multiple of 32, got {v!r}")
    for m in opts["models"]:
        base = str(m).partition("@l")[0]  # "m7@l2" = m7 run 2 levels coarser
        if base not in TEACHER_CHANNELS:
            raise ValueError(f"unknown teacher {m!r}; known: {sorted(TEACHER_CHANNELS)}")
    return opts


def _per_model(opts: dict, key: str, entry: str, default=None):
    """``opts[key]`` for one model: a dict is looked up by entry ("m7@l2"), then base name ("m7")."""
    v = opts.get(key, default)
    if isinstance(v, dict):
        base = str(entry).partition("@l")[0]
        v = v.get(str(entry), v.get(base, default))
    return default if v is None else v


def _level_for(teacher_um: float, vol_um: float, base_level: int) -> tuple[int, int]:
    """(zarr level, downscale factor) so that level voxel pitch == teacher pitch."""
    ratio = teacher_um / vol_um
    k = int(round(math.log2(ratio)))
    # relative tolerance: scan pitches differ slightly between scrolls (2.400 vs 2.399 um)
    # and the teachers are trained on voxels, not micrometres, so <=0.5 % is the same level.
    if abs(2 ** k - ratio) > 5e-3 * ratio:
        raise ValueError(f"teacher pitch {teacher_um} um is not a power-of-two multiple of {vol_um} um")
    return base_level + k, 2 ** k


def _scaled_region(cfg: RunCfg, factor: int) -> RegionCfg:
    start, size = cfg.region.start_zyx, cfg.region.size_zyx
    if factor == 1:
        return cfg.region
    for v in (*start, *size):
        if v % factor:
            raise ValueError(f"region {start}+{size} must be aligned to {factor} for this teacher")
    return RegionCfg(tuple(v // factor for v in start), tuple(v // factor for v in size))  # type: ignore[arg-type]


def _open_level(cfg: RunCfg, level: int, voxel_um: float, region: RegionCfg, probe: int,
                use_cache: bool = True) -> VolumeReader:
    """Local cache first (see ``tsm cache``), then the primary URL, then alt_url."""
    z0, y0, x0 = region.start_zyx
    n = int(probe)
    last_exc: Exception | None = None
    if use_cache:
        cached = open_cached_reader(cfg.volume.cache_root, cfg.volume.url, level, voxel_um,
                                    region, cfg.budget)
        if cached is not None:
            return cached
    for label, url in (("primary", cfg.volume.url), ("alt", cfg.volume.alt_url)):
        if not url:
            continue
        try:
            r = VolumeReader(url, level, voxel_um, cfg.budget)
            t = time.perf_counter()
            a = r.read(z0, z0 + n, y0, y0 + n, x0, x0 + n)
            dt = time.perf_counter() - t
        except Exception as exc:
            last_exc = exc
            print(f"[tsm] {label} level {level}: UNAVAILABLE ({type(exc).__name__}: {exc})", flush=True)
            continue
        if a.any():
            print(f"[tsm] {label} level {level}: {url} shape={r.shape} chunks={r.chunks} "
                  f"probe {n}^3 in {dt:.2f}s mean={a.mean():.1f}", flush=True)
            return r
        print(f"[tsm] {label} level {level}: probe brick all zero, trying next source", flush=True)
    raise RuntimeError(f"no volume source has data at level {level} ({last_exc})")


def open_ct(cfg: RunCfg, level_shift: int = 0, probe: int = 64) -> VolumeReader:
    """CT reader ``2**level_shift`` coarser than the config level (primary, else alt_url)."""
    factor = 2 ** int(level_shift)
    return _open_level(cfg, cfg.volume.level + int(level_shift), cfg.volume.voxel_um * factor,
                       _scaled_region(cfg, factor), probe)


def _preview_png(out_dir: str, cfg: RunCfg, reader: VolumeReader | None, models: list[str]) -> list[str]:
    """Middle-z slice of CT vs each 2.4 um teacher output as PNGs (needs PIL)."""
    import os

    try:
        from PIL import Image
    except ImportError:
        print("[tsm] preview skipped: PIL not installed (uv pip install pillow)")
        return []
    import zarr

    paths: list[str] = []
    z0, y0, x0 = cfg.region.start_zyx
    dz, dy, dx = cfg.region.size_zyx
    zm = z0 + dz // 2
    panels: list[tuple[str, np.ndarray]] = []
    if reader is not None:
        ct = reader.read(zm, zm + 1, y0, y0 + dy, x0, x0 + dx)[0]
        panels.append(("ct", ct))
    for m in models:
        path = os.path.join(out_dir, f"{m}.zarr")
        if not os.path.exists(os.path.join(path, "zarr.json")):
            continue
        arr = zarr.open_array(store=path, mode="r")
        attrs = dict(arr.attrs)
        scale = int(round(float(attrs.get("scale", 1.0))))
        lz = (zm - int(attrs["origin_zyx"][0]) * scale) // scale
        sl = np.asarray(arr[0, lz])
        if scale != 1:
            sl = np.repeat(np.repeat(sl, scale, axis=0), scale, axis=1)[:dy, :dx]
        panels.append((m, sl))
    for name, img in panels:
        p = os.path.join(out_dir, f"preview_{name}.png")
        Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(p)
        paths.append(p)
    if len(panels) > 1:
        h = max(a.shape[0] for _, a in panels)
        w = sum(a.shape[1] for _, a in panels) + 4 * (len(panels) - 1)
        canvas = np.zeros((h, w), dtype=np.uint8)
        xx = 0
        for _, a in panels:
            canvas[: a.shape[0], xx : xx + a.shape[1]] = a
            xx += a.shape[1] + 4
        p = os.path.join(out_dir, "preview_side_by_side.png")
        Image.fromarray(canvas).save(p)
        paths.append(p)
    return paths


def _calibration_windows(cfg: RunCfg, opts: dict, tspec, level: int, voxel_um: float, factor: int,
                         patch: int, trt_dir: str) -> np.ndarray:
    """Real uint8 windows for int8 PTQ: read at ``level`` from the start of ``trt_calib_config``'s
    region (default configs/paris4_small.json, disjoint from the ROI), cached under ``trt_dir``."""
    import os

    from tsm.trt import collect_calibration_windows

    calib_cfg = load_config(str(opts.get("trt_calib_config") or "configs/paris4_small.json"))
    start = tuple(int(v) // factor for v in calib_cfg.region.start_zyx)
    n = int(opts.get("trt_calib_windows", 32))
    cache = os.path.join(trt_dir, f"calib_{tspec.name}_L{level}_p{patch}_{'_'.join(map(str, start))}.npy")
    if not os.path.exists(cache):
        reader = _open_level(calib_cfg, level, voxel_um, RegionCfg(start, (patch, patch, patch)),
                             int(opts.get("probe_brick", 64)))
    else:
        reader = None
    return collect_calibration_windows(reader, start, patch, n=n, cache_path=cache,
                                       empty_max=int(opts.get("empty_window_max", 0)))


def run_teacher_model(
    cfg: RunCfg,
    name: str,
    out_dir: str,
    opts: dict,
    *,
    dry_run: bool = False,
    force: bool = False,
    level_shift: int = 0,
    out_name: str | None = None,
) -> tuple[dict | None, "VolumeReader | None"]:
    """Run one teacher over the config region and write ``<out_dir>/<out_name>.zarr``.

    ``level_shift`` runs the teacher ``2**level_shift`` coarser than its native
    pitch (the audit uses it to test m7 at level 2).  Returns (summary, reader);
    both are None on a dry run.
    """
    import json
    import os

    import torch

    from tsm.limits import cuda_guard, free_cuda
    from tsm.sliding import WindowSpec, plan_bytes, run_sliding
    from tsm.teachers import DEFAULT_MODELS_DIR, TEACHERS, encoder_strides, load_teacher

    out_name = out_name or name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_guard(cfg.budget)
    models_dir = opts.get("models_dir") or DEFAULT_MODELS_DIR
    skip_existing = bool(opts.get("skip_existing", True)) and not force
    tspec = TEACHERS[name]
    voxel_um = tspec.voxel_um * (2 ** int(level_shift))
    level, factor = _level_for(voxel_um, cfg.volume.voxel_um, cfg.volume.level)
    region = _scaled_region(cfg, factor)
    channels = TEACHER_CHANNELS[name]
    # a softmax teacher with an fg_channel writes one channel out of two logits; a softmax
    # teacher without one (fiber) writes all of its channels
    c_out = 2 if (tspec.activation == "softmax" and tspec.fg_channel is not None) else len(channels)
    entry = f"{name}@l{level_shift}" if level_shift else name
    patch = int(_per_model(opts, "patch", entry) or tspec.patch[0])
    wspec = WindowSpec(
        patch=patch, step=opts.get("step"), out_tile=int(opts.get("out_tile", 192)),
        halo=opts.get("halo"), batch=int(opts.get("batch", 1)), tta=_per_model(opts, "tta", entry, "none"),
        tta_field="lasagna" if name == "lasagna" else "scalar",
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(opts.get("dtype", "bf16"))],
        empty_max=int(opts.get("empty_max", 0)),
        norm_scope=str(opts.get("norm_scope", "window")),
        empty_window_max=int(opts.get("empty_window_max", 0)),
        channels_last=bool(opts.get("channels_last", False)),
        cudnn_benchmark=bool(opts.get("cudnn_benchmark", True)),
        compile=opts.get("compile"), prefetch=bool(opts.get("prefetch", True)),
        backend=str(opts.get("backend", "torch")),
    )
    native = "" if patch == int(tspec.patch[0]) else f" (native {tspec.patch[0]})"
    print(f"\n[tsm] === teacher {name} -> {out_name}: level {level} (x{factor}) region start={region.start_zyx} "
          f"size={region.size_zyx} patch={wspec.patch}{native} step={wspec.step} halo={wspec.halo} "
          f"out_tile={wspec.out_tile} tta={wspec.tta} backend={wspec.backend}"
          + (f" trt_precision={_per_model(opts, 'trt_precision', entry, 'auto')}" if wspec.backend == "trt" else ""),
          flush=True)
    rows, tiles = plan_bytes(wspec, region, c_out, tspec.activation, tspec.fg_channel)
    estimate_and_assert(rows, cfg.budget)
    n_win = sum(tp.n_windows(wspec.patch, wspec.step) for tp in tiles)
    print(f"[tsm] {name}: {len(tiles)} tiles, {n_win} windows of {wspec.patch}^3 x {wspec.n_passes} TTA passes "
          f"= {n_win * wspec.n_passes} engine passes", flush=True)
    if dry_run:
        return None, None

    reader = _open_level(cfg, level, voxel_um, region, int(opts.get("probe_brick", 64)))
    writer = BrickWriter(
        os.path.join(out_dir, f"{out_name}.zarr"), channels, region.size_zyx, chunk=128,
        origin_zyx=region.start_zyx, voxel_um=voxel_um, scale=float(factor),
    )
    if not skip_existing:
        writer._done = {}
    t = time.perf_counter()
    net, tspec = load_teacher(name, models_dir, device=device, dtype=torch.float32)
    print(f"[tsm] {name}: loaded in {time.perf_counter() - t:.1f}s, activation={tspec.activation}, "
          f"peak RSS {peak_rss_mb():.0f} MiB", flush=True)
    stride = max(encoder_strides(net))
    if patch % stride:
        raise ValueError(f"{name}: patch {patch} is not a multiple of the encoder stride {stride}")
    if wspec.backend == "trt":
        if device != "cuda":
            raise RuntimeError("backend 'trt' needs a CUDA device")
        from tsm.trt import TRTTeacher, engine_cache_key, net_fingerprint, teacher_cache_name

        precision = str(_per_model(opts, "trt_precision", entry, "auto"))
        trt_dir = os.path.join(os.fspath(models_dir), "trt")
        calib = None
        if precision == "int8":
            calib_algo = str(opts.get("trt_calib_algorithm", "max"))
            fp = net_fingerprint(net)
            key = engine_cache_key(teacher_cache_name(tspec, fp=fp, precision=precision,
                                                      calib_algorithm=calib_algo),
                                   patch, wspec.batch, precision)
            onnx_path = TRTTeacher.onnx_path(tspec, wspec.batch, precision, trt_dir, patch, fp=fp,
                                             calib_algorithm=calib_algo)
            if not (os.path.exists(os.path.join(trt_dir, key + ".plan")) or os.path.exists(onnx_path)):
                calib = _calibration_windows(cfg, opts, tspec, level, voxel_um, factor, int(tspec.patch[0]), trt_dir)
        t = time.perf_counter()
        trt_net = TRTTeacher.build_or_load(
            net, tspec, batch=wspec.batch, precision=precision, cache_dir=trt_dir, patch=patch,
            use_stream=bool(opts.get("trt_stream", True)), calib=calib,
            calib_algorithm=str(opts.get("trt_calib_algorithm", "max")),
        )
        del calib
        del net
        free_cuda()
        net = trt_net
        print(f"[tsm] {name}: TensorRT engine ready in {time.perf_counter() - t:.1f}s ({net.engine_path})",
              flush=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        summary = run_sliding(
            reader, net, tspec.normalizer, wspec, region, writer, c_out, tspec.activation, cfg.budget,
            device=device, voxel_scale=float(factor), fg_channel=tspec.fg_channel, select=tspec.select,
        )
    finally:
        del net
        free_cuda()
    summary.update({
        "model": name, "level": level, "channels": channels, "peak_rss_mb": peak_rss_mb(),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / (1 << 20)) if device == "cuda" else 0.0,
        "source": reader.url, "region_start_zyx": list(region.start_zyx),
        "region_size_zyx": list(region.size_zyx),
    })
    with open(os.path.join(out_dir, f"{out_name}.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[tsm] {name}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                          for k, v in summary.items()}), flush=True)
    return summary, reader


def stage_teacher(cfg: RunCfg, args: argparse.Namespace) -> int:
    import os

    opts = _teacher_opts(cfg)
    out_dir = os.path.join(cfg.out_dir, "teachers")
    os.makedirs(out_dir, exist_ok=True)
    reader0: VolumeReader | None = None

    for entry in opts["models"]:
        # "m7@l2" = run teacher m7 two levels coarser than its native pitch (level_shift=2), output m7_l2.zarr
        name, _, shift = str(entry).partition("@l")
        level_shift = int(shift) if shift else 0
        out_name = f"{name}_l{level_shift}" if level_shift else name
        _, reader = run_teacher_model(cfg, name, out_dir, opts, dry_run=args.dry_run, force=args.force,
                                      level_shift=level_shift, out_name=out_name)
        if reader is not None and reader0 is None and reader.level == cfg.volume.level:
            reader0 = reader

    if args.dry_run:
        print("[tsm] dry run: stopping before I/O")
        return 0
    try:
        paths = _preview_png(out_dir, cfg, reader0, [m for m in opts["models"] if m != "lasagna" and "@" not in str(m)])
        for p in paths:
            print(f"[tsm] preview {p}")
    except Exception as exc:  # preview is best-effort
        print(f"[tsm] preview failed: {type(exc).__name__}: {exc}")
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


def stage_audit(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.labels import run_audit

    run_audit(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


def stage_labels(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.labels import run_labels

    run_labels(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


def stage_train(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.train import run_train

    run_train(cfg, dry_run=args.dry_run, resume=args.resume, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


def _not_implemented(name: str):
    def run(cfg: RunCfg, args: argparse.Namespace) -> int:
        print(f"[tsm] stage {name!r} not implemented")
        return 2

    return run


def stage_infer(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.infer import run_infer

    run_infer(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


def stage_export(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.infer import run_export

    run_export(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


STAGES = {
    "probe": stage_probe,
    "audit": stage_audit,
    "teacher": stage_teacher,
    "labels": stage_labels,
    "train": stage_train,
    "infer": stage_infer,
    "export": stage_export,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tsm")
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", help="train: continue from <out_dir>/train/latest.pt")
    # the view / serve stages take their own options (--out, --box, --checkpoint, --port ...);
    # every other stage rejects an unknown flag exactly as before
    args, rest = p.parse_known_args(argv)
    if rest and args.stage not in ("view", "serve"):
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.stage == "serve" and args.dry_run:
        rest = [*rest, "--dry-run"]
    args.rest = rest
    cfg = load_config(args.config)
    print(f"[tsm] stage={args.stage} config={args.config} out_dir={cfg.out_dir}")
    print(f"[tsm] region start={cfg.region.start_zyx} size={cfg.region.size_zyx}")
    with MemWatchdog(cfg.budget):
        code = STAGES[args.stage](cfg, args)
    return code


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- #
# ablation stage (tsm.ablate): short runs with named extra.train overrides, scored on held-out crops
# --------------------------------------------------------------------------- #
def stage_ablate(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.ablate import run_ablation

    run_ablation(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


STAGES["ablate"] = stage_ablate


# --------------------------------------------------------------------------- #
# dino stage (tsm.dino): cache the DINO token grid + ink-likeness over the region
# --------------------------------------------------------------------------- #
def stage_dino(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.dino import run_dino

    run_dino(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


STAGES["dino"] = stage_dino


# --------------------------------------------------------------------------- #
# cache stage (tsm.cache): copy the region's CT to a local zarr (levels L, L+1, L+2)
# --------------------------------------------------------------------------- #
def stage_cache(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.cache import run_cache

    run_cache(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


STAGES["cache"] = stage_cache


# --------------------------------------------------------------------------- #
# feats stage (tsm.feats): cache the teacher encoder features over the region
# --------------------------------------------------------------------------- #
def stage_feats(cfg: RunCfg, args: argparse.Namespace) -> int:
    from tsm.feats import run_feats

    run_feats(cfg, dry_run=args.dry_run, force=args.force)
    print(f"[tsm] peak RSS {peak_rss_mb():.1f} MiB")
    return 0


STAGES["feats"] = stage_feats


# --------------------------------------------------------------------------- #
# view / serve stages (tsm.view): model-view packets for render3d's --view mode
# --------------------------------------------------------------------------- #
def stage_view(cfg: RunCfg, args: argparse.Namespace) -> int:
    """``tsm view <config> --out DIR --box z,y,x,dz,dy,dx ...`` = dev/view_export.py's main.

    The stage arguments are the exporter's own, forwarded verbatim (``args.rest``), so the
    two entry points cannot drift apart."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "dev", "view_export.py")
    spec = importlib.util.spec_from_file_location("tsm_view_export", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"view exporter not found at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main([args.config, *args.rest]))


def stage_serve(cfg: RunCfg, args: argparse.Namespace) -> int:
    """``tsm serve <config> --checkpoint X [--port 9760] ...`` (tsm.serve)."""
    from tsm.serve import main as serve_main

    return int(serve_main(list(args.rest), cfg=cfg))


STAGES["view"] = stage_view
STAGES["serve"] = stage_serve
