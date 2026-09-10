"""Measure the orientation bias of a teacher (or the student) on one real box.

    uv run python dev/orientation_bias.py --teacher recto --group oct48 --out DIR
    uv run python dev/orientation_bias.py --teacher lasagna --group oct48 --random-rotations 4 --max-deg 15
    uv run python dev/orientation_bias.py --teacher student --checkpoint train/latest.pt --group flip8

For every element ``g`` of the chosen group the model runs once on ``apply(box, g)``, the
output is brought back into the box frame with the per-channel rules of
``tsm.equivariance.invert_prediction`` (scalars: inverse spatial transform; normals: inverse
spatial transform then ``g^T``; lasagna dir pairs: decode -> rotate -> re-encode) and compared
with the identity prediction and with the group-averaged prediction.

Design: **one window, no sliding.**  The box is exactly one patch (256^3 for recto / ink at
2.4 um, 192^3 for m7 / lasagna at level 2 = 9.6 um), so every pass sees the same voxels.  A
signed permutation of a cube commutes with the per-patch instance normalisation (same set of
values -> same mean / std), with zero padding of the convolutions (the padded cube is the
permuted padded cube) and with the gaussian blend that a sliding run would add (none here):
whatever disagreement remains is the network's own anisotropy (learned filters / anisotropic
training data), not a pipeline artefact.  Arbitrary small rotations (``--random-rotations``)
resample trilinearly and expose the box corners (filled with 0 = air, like the masked volume),
so their disagreement includes an interpolation / corner floor; statistics are restricted to
the voxels that stay inside the box in both directions (``rotation_valid_mask``).

The box is read from the config volume (region centre of ``configs/paris4_small.json`` at the
model's level) and cached as ``.npy`` in the scratchpad; every inverted prediction is cached
as float16 ``.npy`` under ``<out>/preds`` so a run interrupted by ``--time-budget`` resumes
where it stopped (the GPU is shared: keep each invocation short).

Outputs in ``--out``: ``bias.json`` (all metrics, per element + summary), ``bias.md`` (table),
``montage.png`` (mid z slice: CT, identity, worst element inverted, |difference|, medial /
zero-crossing overlays).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tsm import equivariance as E  # noqa: E402
from tsm.config import RegionCfg, load_config  # noqa: E402
from tsm.teachers import TEACHERS, apply_activation, load_teacher  # noqa: E402

SCRATCH = os.environ.get(
    "TSM_SCRATCH", "/tmp/claude-1000/-home-forrest-tsm/193809ef-c338-4699-b345-e8f9520c6118/scratchpad"
)
DEFAULT_LEVEL_SHIFT = {"recto": 0, "ink": 0, "m7": 2, "lasagna": 0, "student": 0}  # lasagna's own pitch is 9.6 um
KIND = {"recto": "surface", "m7": "surface", "ink": "ink", "lasagna": "lasagna", "student": "student"}
RULE = {"surface": "scalar", "ink": "scalar", "lasagna": "lasagna", "student": "student"}


def log(msg: str) -> None:
    print(f"[bias] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# experiment manifests: never reuse a cached prediction from another experiment
# --------------------------------------------------------------------------- #
def file_fingerprint(path: str | None) -> str | None:
    """``<size>-<mtime_ns>`` of a file (cheap, no hashing) -- ``None`` when there is no such file.

    Enough to notice that a checkpoint was replaced or retrained between two runs; the path alone
    is not, because ``latest.pt`` is overwritten in place."""
    if not path:
        return None
    try:
        st = os.stat(os.path.expanduser(path))
    except OSError:
        return None
    return f"{st.st_size}-{st.st_mtime_ns}"


def manifest_diff(want: dict[str, Any], have: dict[str, Any] | None) -> list[str]:
    """Keys of ``want`` that ``have`` does not reproduce exactly (JSON-normalised); [] = reusable.

    Missing manifest -> everything differs.  Extra keys in ``have`` are ignored so that an older
    manifest with fewer recorded properties is simply rejected on the keys it lacks."""
    if not have:
        return sorted(want)
    def norm(v: Any) -> Any:
        return json.loads(json.dumps(v, sort_keys=True, default=str))
    return sorted(k for k in want if norm(have.get(k)) != norm(want[k]))


def read_manifest(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# box
# --------------------------------------------------------------------------- #
def parse_teacher(entry: str) -> tuple[str, int | None]:
    base, _, shift = str(entry).partition("@l")
    return base, (int(shift) if shift else None)


def level_of(cfg: Any, voxel_um: float, level_shift: int) -> tuple[int, int]:
    k = int(round(math.log2(voxel_um / cfg.volume.voxel_um))) + int(level_shift)
    return cfg.volume.level + k, 2 ** k


def load_box(cfg_path: str, name: str, voxel_um: float, level_shift: int, size: int,
             offset: tuple[int, int, int] | None, box_npy: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    if box_npy:
        box = np.load(box_npy)
        c = [(s - size) // 2 for s in box.shape]
        box = np.ascontiguousarray(box[c[0]:c[0] + size, c[1]:c[1] + size, c[2]:c[2] + size])
        return box, {"source": os.path.abspath(box_npy), "crop_start": c, "shape": list(box.shape)}
    cfg = load_config(cfg_path)
    level, factor = level_of(cfg, voxel_um, level_shift)
    centre = [(s + n / 2) / factor for s, n in zip(cfg.region.start_zyx, cfg.region.size_zyx)]
    start = [int(round(c - size / 2)) for c in centre]
    if offset is not None:
        start = [int(s // factor + o) for s, o in zip(cfg.region.start_zyx, offset)]
    cache = os.path.join(SCRATCH, f"orientation_box_L{level}_{start[0]}_{start[1]}_{start[2]}_{size}.npy")
    info: dict[str, Any] = {"level": level, "factor": factor, "start_zyx": start, "size": size,
                            "voxel_um": cfg.volume.voxel_um * factor, "cache": cache, "config": os.path.abspath(cfg_path)}
    # the file name carries level/start/size only, so the volume identity goes into a manifest:
    # a cache written from another volume (or another config's URL) must not be reused (R22)
    want = {"url": cfg.volume.url, "alt_url": cfg.volume.alt_url, "level": level, "start_zyx": start,
            "size": size, "voxel_um": cfg.volume.voxel_um * factor}
    if os.path.exists(cache):
        diff = manifest_diff(want, read_manifest(cache + ".json"))
        if diff:
            log(f"cached box {cache} was built with different {diff}: re-reading")
        else:
            info["cached"] = True
            return np.load(cache), info
    from tsm.cli import _open_level

    reader = _open_level(cfg, level, cfg.volume.voxel_um * factor, RegionCfg(tuple(start), (size,) * 3), 64)
    t = time.perf_counter()
    box = reader.read(start[0], start[0] + size, start[1], start[1] + size, start[2], start[2] + size)
    info.update({"cached": False, "read_s": round(time.perf_counter() - t, 1), "source": reader.url})
    write_manifest(cache + ".json", {**want, "source": reader.url})
    log(f"read {box.shape} at level {level} from {reader.url} in {info['read_s']}s mean={box.mean():.1f} "
        f"zero_frac={(box == 0).mean():.3f}")
    np.save(cache, box)
    return box, info


# --------------------------------------------------------------------------- #
# model runners: float CT in [0, 255] (Z, Y, X) on the device -> (C, Z, Y, X) float32 on the CPU
# --------------------------------------------------------------------------- #
def make_runner(name: str, device: torch.device, dtype: torch.dtype, models_dir: str, checkpoint: str | None,
                clip: float) -> tuple[Callable[[torch.Tensor], np.ndarray], dict[str, Any]]:
    use_amp = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    if name == "student":
        from tsm.infer import StudentNet, activate_heads, load_student
        from tsm.student import normalize_ct

        if not checkpoint:
            raise SystemExit("--checkpoint is required for --teacher student")
        net, ck = load_student(checkpoint, device=device)
        voxel_um = 2.4
        wrapper = StudentNet(net, voxel_um, clip, tta="none")

        def run(x: torch.Tensor) -> np.ndarray:
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                xin = normalize_ct(x / 255.0)[None, None]
                return activate_heads(wrapper.raw(xin))[0].float().cpu().numpy()

        return run, {"model": "student", "checkpoint": ck, "voxel_um": voxel_um, "channels":
                     ["sdf", "valid", "ink", "sin", "cos", "density", "nx", "ny", "nz", "conf", "spare"]}

    net, spec = load_teacher(name, models_dir, device=device, dtype=torch.float32)

    def run(x: torch.Tensor) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            xin = spec.normalizer(x)[None, None]
            out = spec.select(net(xin)).float()
            prob = apply_activation(out, spec.activation)
            if spec.activation in ("softmax", "sigmoid") and spec.fg_channel is not None:
                prob = prob[:, spec.fg_channel:spec.fg_channel + 1]
            return prob[0].cpu().numpy()

    return run, {"model": name, "file": spec.file, "patch": list(spec.patch), "activation": spec.activation,
                 "normalizer": spec.normalizer.mode, "voxel_um": spec.voxel_um}


# --------------------------------------------------------------------------- #
# transforms to evaluate
# --------------------------------------------------------------------------- #
def build_transforms(group: str, n_rot: int, max_deg: float, seed: int, shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    out = []
    for g in E.group_elements(group):
        t = E.classify(g)
        t.update({"matrix": g.tolist(), "type": "oct", "mask": None})
        out.append(t)
    rng = np.random.default_rng(seed)
    for i in range(n_rot):
        R = E.random_rotation_matrix(rng, max_deg)
        ang = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2))))
        out.append({"name": f"rot{i}_{ang:.1f}deg", "identity": False, "det": 1, "pure_flip": False, "flips_z": False,
                    "in_plane": False, "swaps_z": False, "n_flips": 0, "perm": None, "sign": None,
                    "matrix": R.tolist(), "type": "rotation", "angle_deg": ang,
                    "mask": E.rotation_valid_mask(shape, R, margin=3)})
    return out


def pred_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, "preds", f"{name}.npy")


def experiment_manifest(args: Any, name: str, kind: str, level_shift: int, box_info: dict[str, Any],
                        transforms: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything that changes a cached prediction: source box, model, precision and transforms.

    Compared with :func:`manifest_diff` against the manifest stored beside the cached ``.npy``
    files, so predictions from a different checkpoint / offset / box / transform set are never
    mixed into a new report."""
    ck = os.path.abspath(os.path.expanduser(args.checkpoint)) if getattr(args, "checkpoint", None) else None
    return {
        "teacher": name, "kind": kind, "level_shift": int(level_shift), "clip": float(args.clip),
        "checkpoint": ck, "checkpoint_fingerprint": file_fingerprint(ck),
        "config": os.path.abspath(args.config), "dtype": args.dtype,
        "box_npy": os.path.abspath(args.box_npy) if getattr(args, "box_npy", None) else None,
        "box": {k: box_info.get(k) for k in ("level", "factor", "start_zyx", "size", "voxel_um", "shape", "crop_start")},
        "group": args.group, "random_rotations": int(getattr(args, "random_rotations", 0)),
        "max_deg": float(getattr(args, "max_deg", 0.0)), "seed": int(args.seed),
        "transforms": {t["name"]: np.round(np.asarray(t["matrix"], np.float64), 9).tolist() for t in transforms},
    }


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pname = rows[0]["primary_name"]
    non_id = [r for r in rows if not r["identity"] and r["type"] == "oct"]
    prim = {r["name"]: r["primary_vs_identity"] for r in non_id}
    ranked = sorted(non_id, key=lambda r: -r["primary_vs_identity"])
    classes = {
        "pure_flips": [r for r in non_id if r["pure_flip"]],
        "z_flip_only": [r for r in non_id if r["pure_flip"] and r["sign"] == [-1, 1, 1]],
        "in_plane_flips": [r for r in non_id if r["pure_flip"] and r["sign"][0] == 1],
        "in_plane_with_yx_swap": [r for r in non_id if r["in_plane"] and not r["pure_flip"]],
        "rot90_about_z": [r for r in non_id if r["in_plane"] and r["det"] == 1 and r["sign"][0] == 1],
        "swaps_z": [r for r in non_id if r["swaps_z"]],
        "proper_rotations": [r for r in non_id if r["det"] == 1],
        "reflections": [r for r in non_id if r["det"] == -1],
    }
    by_class = {k: {"n": len(v), "mean": _mean([r["primary_vs_identity"] for r in v]),
                    "max": (max(r["primary_vs_identity"] for r in v) if v else None)} for k, v in classes.items()}
    xs = [r["primary_vs_identity"] for r in non_id]
    corr = {
        "swaps_z": _corr(xs, [1.0 if r["swaps_z"] else 0.0 for r in non_id]),
        "flips_z_axis": _corr(xs, [1.0 if r["sign"][0] < 0 else 0.0 for r in non_id]),
        "n_flips": _corr(xs, [float(r["n_flips"]) for r in non_id]),
        "reflection": _corr(xs, [1.0 if r["det"] < 0 else 0.0 for r in non_id]),
    }
    rot = [r for r in rows if r["type"] == "rotation"]
    verdict = []
    sz, pf = by_class["swaps_z"]["mean"], by_class["pure_flips"]["mean"]
    ip = by_class["in_plane_with_yx_swap"]["mean"]
    if sz is not None and pf is not None:
        if sz > 1.5 * max(pf, 1e-9) and (ip is None or sz > 1.5 * max(ip, 1e-9)):
            verdict.append("z<->x/y axis swaps disagree most: anisotropy along the scan axis dominates")
        elif pf > 1.5 * max(sz, 1e-9):
            verdict.append("pure flips disagree more than axis swaps: chirality / direction bias rather than anisotropy")
        else:
            verdict.append("axis swaps and pure flips disagree similarly")
    zf = by_class["z_flip_only"]["mean"]
    if zf is not None and pf is not None and by_class["in_plane_flips"]["mean"] is not None:
        verdict.append(f"z-flip alone {zf:.4g} vs in-plane flips {by_class['in_plane_flips']['mean']:.4g} ({pname})")
    return {
        "primary_name": pname,
        "vs_identity": {"mean": _mean(xs), "max": (max(xs) if xs else None), "min": (min(xs) if xs else None)},
        "vs_group_mean": {"mean": _mean([r["primary_vs_mean"] for r in rows if r["type"] == "oct"]),
                          "max": (max(r["primary_vs_mean"] for r in rows if r["type"] == "oct") if rows else None),
                          "identity": next((r["primary_vs_mean"] for r in rows if r["identity"]), None)},
        "worst": [{"name": r["name"], pname: r["primary_vs_identity"]} for r in ranked[:5]],
        "best": [{"name": r["name"], pname: r["primary_vs_identity"]} for r in ranked[-3:]],
        "by_class": by_class,
        "correlation_with_primary": corr,
        "rotations": {"n": len(rot), "mean": _mean([r["primary_vs_identity"] for r in rot]),
                      "max": (max(r["primary_vs_identity"] for r in rot) if rot else None)},
        "verdict": verdict,
        "per_element_primary": prim,
    }


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def key_columns(kind: str, m: dict[str, Any]) -> dict[str, Any]:
    if kind == "surface":
        s = m["medial"]["symmetric"]
        return {"dice": m["dice"], "mean_abs_dp": m["mean_abs"], "med_med": s["median"], "med_p90": s["p90"],
                "frac>3": s["frac_gt3"], "frac>5": s["frac_gt5"]}
    if kind == "ink":
        return {"dice": m["dice"], "mean_abs_dp": m["mean_abs"], "auprc": m["auprc"]}
    if kind == "lasagna":
        a = m["normal_angle"]
        return {"cos_mae": m["cos_mae"], "grad_mae": m["grad_mag_mae"], "ang_mean": a["mean"], "ang_med": a["median"],
                "ang_p90": a["p90"], "frac>10": a["frac_gt10"]}
    z = m["zero_crossing"]["symmetric"]
    return {"sdf_mae": m["sdf_mae_band"], "zc_med": z["median"], "zc_p90": z["p90"], "phase_deg": m["phase_err_deg"]["mean"],
            "ang_mean": m["normal_angle"]["mean"], "ink_dice": m["ink_dice"], "valid_abs_d": m["valid_mean_abs"]}


def write_markdown(path: str, res: dict[str, Any]) -> None:
    kind = res["kind"]
    rows = res["transforms"]
    s = res["summary"]
    lines = [f"# Orientation bias: {res['model']['model']} ({kind}), group {res['group']}", "",
             f"Box {res['box']['shape']} at level {res['box'].get('level', '-')} start {res['box'].get('start_zyx', '-')} "
             f"({res['box'].get('voxel_um', '-')} um); single window, no sliding; dtype {res['dtype']}, device {res['device']}.", "",
             f"Primary disagreement `{s['primary_name']}` vs identity: mean {_fmt(s['vs_identity']['mean'])}, "
             f"max {_fmt(s['vs_identity']['max'])}; vs group mean: mean {_fmt(s['vs_group_mean']['mean'])}, "
             f"identity {_fmt(s['vs_group_mean']['identity'])}.", ""]
    lines.append("## By class (vs identity)")
    lines.append("| class | n | mean | max |")
    lines.append("|---|---|---|---|")
    for k, v in s["by_class"].items():
        lines.append(f"| {k} | {v['n']} | {_fmt(v['mean'])} | {_fmt(v['max'])} |")
    lines.append("")
    lines.append("Correlation of the primary metric with: " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in s["correlation_with_primary"].items()))
    lines.append("")
    for v in s["verdict"]:
        lines.append(f"- {v}")
    lines.append("")
    lines.append("## Per transform")
    cols = list(key_columns(kind, rows[0]["vs_identity"]).keys())
    lines.append("| name | det | swaps_z | flips | " + " | ".join(cols) + f" | {s['primary_name']} vs mean |")
    lines.append("|---|---|---|---|" + "---|" * len(cols) + "---|")
    for r in sorted(rows, key=lambda r: (r["type"] != "oct", -r["primary_vs_identity"])):
        kc = key_columns(kind, r["vs_identity"])
        flips = "".join("zyx"[i] for i, sg in enumerate(r["sign"] or []) if sg < 0) or "-"
        lines.append(f"| {r['name']} | {r['det']} | {int(r['swaps_z'])} | {flips} | " + " | ".join(_fmt(v) for v in kc.values())
                     + f" | {_fmt(r['primary_vs_mean'])} |")
    lines.append("")
    lines.append(f"Montage: `{res['files']['montage']}` (worst element `{s['worst'][0]['name'] if s['worst'] else '-'}`).")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# montage
# --------------------------------------------------------------------------- #
def _gray(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


def _display(kind: str, pred: np.ndarray, z: int, clip: float) -> tuple[np.ndarray, np.ndarray, str]:
    """(gray slice u8, surface mask slice, label) of one prediction's mid slice."""
    if kind in ("surface", "ink"):
        p = pred[0, z]
        return _gray(p, 0, 1), p >= 0.5, "prob"
    if kind == "lasagna":
        p = pred[0, z]
        return _gray(p, 0, 1), p > 0.9, "cos"
    p = pred[0, z]
    surf = E._thin_zero_crossing(pred[0], pred[1] > 0.5)[z]
    return _gray(p, -clip, clip), surf, "sdf"


def _medial_slice(kind: str, pred: np.ndarray, z: int, clip: float) -> np.ndarray:
    if kind == "surface":
        from tsm.labels import _drop_small_components, medial_surface

        zs = slice(max(0, z - 6), z + 7)
        sub = medial_surface(_drop_small_components(pred[0, zs] >= 0.5, 100))
        return sub[min(z, 6)]
    return _display(kind, pred, z, clip)[1]


def write_montage(path: str, kind: str, ct: np.ndarray, ident: np.ndarray, worst: np.ndarray, worst_name: str,
                  clip: float) -> str:
    from PIL import Image, ImageDraw

    from tsm.labels import _overlay, _to_rgb

    z = ct.shape[0] // 2
    ct_u8 = ct[z].astype(np.uint8)
    gi, _, lab = _display(kind, ident, z, clip)
    gw, _, _ = _display(kind, worst, z, clip)
    diff = np.abs(ident[0, z].astype(np.float32) - worst[0, z].astype(np.float32))
    dmax = 1.0 if kind != "student" else clip
    gd = _gray(diff, 0, dmax * 0.5)
    mi, mw = _medial_slice(kind, ident, z, clip), _medial_slice(kind, worst, z, clip)
    over = _overlay(_overlay(_to_rgb(ct_u8), mi, (255, 40, 40), 0.9), mw, (40, 220, 255), 0.6)
    over[mi & mw] = (255, 255, 60)
    tiles = [(_to_rgb(ct_u8), "CT"), (_to_rgb(gi), f"identity {lab}"), (_to_rgb(gw), f"{worst_name} inverted"),
             (_to_rgb(gd), f"|diff| (0..{dmax * 0.5:g})"), (over, "medial: id red / worst cyan / both yellow")]
    h, w = ct_u8.shape
    pad = 18
    img = Image.new("RGB", (len(tiles) * (w + 4), h + pad), (30, 30, 30))
    dr = ImageDraw.Draw(img)
    for i, (t, name) in enumerate(tiles):
        img.paste(Image.fromarray(np.ascontiguousarray(t)), (i * (w + 4), pad))
        dr.text((i * (w + 4) + 2, 3), name, fill=(255, 255, 255))
    img.save(path)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def gpu_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="recto | m7 | ink | lasagna | student (or e.g. m7@l2)")
    ap.add_argument("--config", default="configs/paris4_small.json")
    ap.add_argument("--box", type=int, default=None, help="box edge (default: the model's patch; student 128)")
    ap.add_argument("--offset", default=None, help="z,y,x offset from the region start at the model level (default: centred)")
    ap.add_argument("--box-npy", default=None, help="use the centre of this cached .npy box instead of reading the volume")
    ap.add_argument("--group", default="oct48", choices=sorted(E.GROUPS))
    ap.add_argument("--random-rotations", type=int, default=0)
    ap.add_argument("--max-deg", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default <scratch>/orientation_bias/<teacher>_<group>")
    ap.add_argument("--checkpoint", default=None, help="student checkpoint (--teacher student)")
    ap.add_argument("--models-dir", default=os.path.expanduser("~/.cache/tsm-models"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--dtype", default=None, choices=["fp32", "bf16", "fp16"],
                    help="autocast dtype on the GPU (default: bf16 for recto / ink / m7 -- fp32 at 256^3 oversubscribes "
                         "a shared 16 GB card -- fp32 for lasagna / student); fp32 on the CPU")
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--level-shift", type=int, default=None, help="levels coarser than the model's own pitch (default m7: 2)")
    ap.add_argument("--time-budget", type=float, default=170.0, help="seconds of model runs per invocation (resume by re-running)")
    ap.add_argument("--gpu-max-used-gb", type=float, default=4.0, help="refuse the GPU when more than this is already in use")
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recompute", action="store_true", help="ignore cached predictions")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    name, shift = parse_teacher(args.teacher)
    if name not in KIND:
        raise SystemExit(f"unknown teacher {name!r}; known: {sorted(KIND)}")
    kind = KIND[name]
    level_shift = args.level_shift if args.level_shift is not None else (shift if shift is not None else DEFAULT_LEVEL_SHIFT[name])
    out_dir = args.out or os.path.join(SCRATCH, "orientation_bias", f"{name}_{args.group}")
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        used = gpu_used_gb()
        if used > args.gpu_max_used_gb and not args.force_gpu:
            log(f"GPU has {used:.1f} GB in use (> {args.gpu_max_used_gb} GB): refusing to run; use --device cpu or --force-gpu")
            return 2
        log(f"GPU in use before start: {used:.1f} GB")
    if args.dtype is None:
        args.dtype = "bf16" if (device.type == "cuda" and name in ("recto", "ink", "m7")) else "fp32"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    voxel_um = 2.4 if name == "student" else TEACHERS[name].voxel_um
    size = args.box or (128 if name == "student" else int(TEACHERS[name].patch[0]))
    offset = tuple(int(v) for v in args.offset.split(",")) if args.offset else None
    box, box_info = load_box(args.config, name, voxel_um, level_shift, size, offset, args.box_npy)
    box_info["shape"] = list(box.shape)
    box_info["mean"] = float(box.mean())
    log(f"box {box.shape} mean {box.mean():.1f} ({'cached' if box_info.get('cached') else 'read'})")

    transforms = build_transforms(args.group, args.random_rotations, args.max_deg, args.seed, tuple(box.shape))
    # R22: the output directory name only carries teacher + group, so everything else that changes
    # a prediction goes into a manifest and is compared before a cached .npy is reused.
    manifest_path = os.path.join(out_dir, "preds", "manifest.json")
    manifest = experiment_manifest(args, name, kind, level_shift, box_info, transforms)
    stale = manifest_diff(manifest, read_manifest(manifest_path))
    have_any = any(os.path.exists(pred_path(out_dir, t["name"])) for t in transforms)
    if stale and have_any:
        log(f"cached predictions in {out_dir} come from a different experiment ({stale}): recomputing")
    reuse = not stale and not args.recompute
    todo = [t for t in transforms if not reuse or not os.path.exists(pred_path(out_dir, t["name"]))]
    log(f"{len(transforms)} transforms, {len(transforms) - len(todo)} cached predictions")

    run_info_path = os.path.join(out_dir, "preds", "run_info.json")
    run_info: dict[str, Any] = {"model": {"model": name}, "dtype": args.dtype, "device": device.type}
    if os.path.exists(run_info_path):
        with open(run_info_path) as fh:
            run_info = json.load(fh)  # the invocation that computed the cached predictions
    if todo:
        run, model_info = make_runner(name, device, dtype, args.models_dir, args.checkpoint, args.clip)
        run_info = {"model": model_info, "dtype": args.dtype, "device": device.type}
        with open(run_info_path, "w") as fh:
            json.dump(run_info, fh, indent=1)
        write_manifest(manifest_path, manifest)
        box_t = torch.from_numpy(box).to(device).float()
        t_start = time.perf_counter()
        for t in todo:
            t0 = time.perf_counter()
            g = np.asarray(t["matrix"])
            xin = E.apply(box_t, g.astype(np.int64)) if t["type"] == "oct" else E.apply_rotation(box_t, g)
            pred = run(xin)
            del xin
            inv = E.invert_prediction(pred, g, RULE[kind])
            np.save(pred_path(out_dir, t["name"]), np.asarray(inv, dtype=np.float16))
            log(f"{t['name']:>16s}  {time.perf_counter() - t0:5.1f}s  out {tuple(pred.shape)}")
            if time.perf_counter() - t_start > args.time_budget and t is not todo[-1]:
                left = len(todo) - todo.index(t) - 1
                log(f"time budget {args.time_budget}s reached: {left} transforms left; re-run the same command to resume")
                return 3
        del box_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # metrics
    ident = np.load(pred_path(out_dir, transforms[0]["name"])).astype(np.float32)
    assert transforms[0]["identity"]
    oct_names = [t["name"] for t in transforms if t["type"] == "oct"]
    mean = np.zeros_like(ident)
    for n in oct_names:
        mean += np.load(pred_path(out_dir, n)).astype(np.float32)
    mean /= len(oct_names)
    mkw = {"clip": args.clip} if kind == "student" else {}
    rows: list[dict[str, Any]] = []
    for t in transforms:
        p = np.load(pred_path(out_dir, t["name"])).astype(np.float32)
        m_id = E.metrics(ident, p, kind, t["mask"], **mkw)
        m_mean = E.metrics(mean, p, kind, t["mask"], **mkw)
        pn, pv = E.primary_metric(kind, m_id)
        row = {k: v for k, v in t.items() if k != "mask"}
        row.update({"vs_identity": m_id, "vs_mean": m_mean, "primary_name": pn, "primary_vs_identity": pv,
                    "primary_vs_mean": E.primary_metric(kind, m_mean)[1],
                    "mask_frac": (float(t["mask"].mean()) if t["mask"] is not None else 1.0)})
        rows.append(row)
        log(f"{t['name']:>16s}  {pn} vs id {pv:.5f}  vs mean {row['primary_vs_mean']:.5f}")
    summary = summarize(kind, rows)
    worst_name = summary["worst"][0]["name"] if summary["worst"] else transforms[-1]["name"]
    worst = np.load(pred_path(out_dir, worst_name)).astype(np.float32)
    montage = write_montage(os.path.join(out_dir, "montage.png"), kind, box, ident, worst, worst_name, args.clip)
    res = {
        "model": run_info["model"], "kind": kind, "group": args.group, "random_rotations": args.random_rotations,
        "max_deg": args.max_deg, "seed": args.seed, "dtype": run_info["dtype"], "device": run_info["device"], "box": box_info,
        "level_shift": level_shift, "single_window": True,
        "note": "box == patch: one forward pass per transform; instance norm / padding commute with the group, "
                "so the disagreement is the network's own anisotropy. Rotations add an interpolation floor.",
        "transforms": rows, "summary": summary,
        "files": {"json": os.path.join(out_dir, "bias.json"), "md": os.path.join(out_dir, "bias.md"), "montage": montage},
    }
    with open(res["files"]["json"], "w") as fh:
        json.dump(res, fh, indent=1)
    write_markdown(res["files"]["md"], res)
    log(f"primary {summary['primary_name']}: mean {summary['vs_identity']['mean']:.5f} max {summary['vs_identity']['max']:.5f} "
        f"worst {worst_name}")
    for v in summary["verdict"]:
        log(v)
    log(f"wrote {res['files']['json']}, {res['files']['md']}, {montage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
