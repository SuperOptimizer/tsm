"""Which orientation transforms keep the recto teacher's band on the SAME sheet face?

The recto band is one-sided: on the small slab it hugs the **in** face (the face toward the
umbilicus) -- in-face coverage 0.72 vs 0.14 at k = 1 voxel (docs/label_store.md).  A TTA set is
only safe for a two-face student if every element of it leaves the band on that same face; an
element that moves it to the *out* face averages the two faces together and smears the target.

For every transform ``g`` (the 48 octahedral elements plus random rotations) the teacher runs on
``apply(box, g)``, the prediction is brought back with ``tsm.equivariance.invert_prediction`` and
scored against the CT-derived two-face labels (``dev/tta_side_faces.py`` -> ``fine.zarr`` with
``sdf_in / sdf_out / faces_valid``):

* in-face / out-face coverage at k = 1, 3 voxels (fraction of face voxels with a band voxel within k)
* **side score** = in / (in + out) at k = 1
* fraction of band voxels closer to face_in than to face_out
* medial-surface distances and Dice against the identity prediction (``equivariance.metrics``)

``g`` is SIDE-PRESERVING when its side score is >= 0.9 x the identity's and >= 0.6; otherwise it is
side-flipping / ambiguous.  The script then averages the inverted predictions over the
side-preserving set (S_pres, the "recto band"), over the side-flipping set (S_flip, read as a
**verso detector**) and over all 48, and scores those averaged bands the same way.

    uv run python dev/tta_side.py --teacher recto --pred-dirs ~/tsm-output/orientation_bias/recto_oct48
    uv run python dev/tta_side.py --teacher m7 --box 192 --offset 32,416,416
    uv run python dev/tta_side.py --teacher m7@l2 --box 192

Model runs reuse ``dev/orientation_bias.py`` (one 256^3 / 192^3 box == one patch, one forward per
element, bf16 on the GPU) and its float16 prediction cache, so a run whose predictions are already
cached needs no GPU at all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import orientation_bias as OB  # noqa: E402
from tsm import equivariance as E  # noqa: E402
from tsm.labels import _edt, decode_sdf_u8  # noqa: E402

OUT_ROOT = os.path.expanduser("~/tsm-output/tta_side")
FACES_STORE = os.path.join(OUT_ROOT, "faces", "fine.zarr")
K_LIST = (1, 3)


def log(msg: str) -> None:
    print(f"[tta_side] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# faces
# --------------------------------------------------------------------------- #
def load_faces(path: str, clip: float = 20.0) -> dict[str, Any]:
    """The two-face labels of a store: face_in / face_out zero sets, decoded sdfs, valid mask."""
    import zarr

    arr = zarr.open_array(store=os.path.expanduser(path), mode="r")
    attrs = dict(arr.attrs)
    ch = list(attrs.get("channels", []))
    origin = [int(v) for v in attrs.get("origin_zyx", (0, 0, 0))]
    idx = {n: i for i, n in enumerate(ch)}
    si = np.asarray(arr[idx["sdf_in"]])
    so = np.asarray(arr[idx["sdf_out"]])
    valid = np.asarray(arr[idx["faces_valid"]])
    sup = valid == 1
    return {
        "path": os.path.abspath(os.path.expanduser(path)), "origin_zyx": origin,
        "shape": list(si.shape), "voxel_um": float(attrs.get("voxel_um", 2.4)),
        "sdf_in": decode_sdf_u8(si, clip), "sdf_out": decode_sdf_u8(so, clip),
        "face_in": (si == 128) & sup, "face_out": (so == 128) & sup, "supervise": sup,
    }


def face_window(faces: dict[str, Any], start_zyx: list[int], size: int, factor: int) -> dict[str, Any]:
    """Intersect a model box (level-0 start ``start_zyx * factor``, ``size`` voxels of that level)
    with the faces box; return the level-0 slices into both."""
    b0 = [int(s) * int(factor) for s in start_zyx]
    b1 = [v + size * int(factor) for v in b0]
    f0 = list(faces["origin_zyx"])
    f1 = [f0[a] + int(faces["shape"][a]) for a in range(3)]
    lo = [max(b0[a], f0[a]) for a in range(3)]
    hi = [min(b1[a], f1[a]) for a in range(3)]
    if any(hi[a] <= lo[a] for a in range(3)):
        raise SystemExit(f"model box {b0}..{b1} does not overlap the faces box {f0}..{f1}")
    return {
        "level0_lo": lo, "level0_hi": hi,
        "faces_sl": tuple(slice(lo[a] - f0[a], hi[a] - f0[a]) for a in range(3)),
        "box_lo0": lo, "box_b0": b0, "factor": int(factor),
    }


def pred_on_window(pred: np.ndarray, win: dict[str, Any]) -> np.ndarray:
    """One prediction channel (model-level grid) resampled (nearest, x factor) onto the level-0
    window of the faces box."""
    p = np.asarray(pred, dtype=np.float32)
    p = p[0] if p.ndim == 4 else p
    k = win["factor"]
    if k > 1:
        p = np.repeat(np.repeat(np.repeat(p, k, 0), k, 1), k, 2)
    sl = tuple(slice(win["level0_lo"][a] - win["box_b0"][a], win["level0_hi"][a] - win["box_b0"][a]) for a in range(3))
    return np.ascontiguousarray(p[sl])


def mask_on_window(mask: Any, win: dict[str, Any]) -> np.ndarray:
    """A model-grid boolean mask (e.g. ``E.rotation_valid_mask``) resampled onto the faces window.

    Same nearest-neighbour resampling / cropping as :func:`pred_on_window`, so the rotation's valid
    support can be applied to exactly the voxels the metrics score."""
    return pred_on_window(np.asarray(mask, np.float32), win) > 0.5


# --------------------------------------------------------------------------- #
# side metrics
# --------------------------------------------------------------------------- #
def side_metrics(prob: np.ndarray, f: dict[str, Any], thr: float = 0.5,
                 support: np.ndarray | None = None) -> dict[str, Any]:
    """Face coverage / side score of one probability volume already on the faces grid.

    ``support`` (optional) is the valid-support mask of an arbitrary rotation: voxels outside it were
    filled by interpolation / padding rather than predicted, and scoring them would report the
    resampling as model orientation bias (R23).  The band, both faces and the supervised set are
    all restricted to it."""
    prob = np.asarray(prob, np.float32)
    band = prob >= thr
    if support is not None:
        band = band & support
    out: dict[str, Any] = {"band_frac": float(band.mean()), "n_band": int(band.sum())}
    fin, fout = f["face_in"], f["face_out"]
    if support is not None:
        fin, fout = fin & support, fout & support
        out["support_frac"] = float(np.asarray(support, bool).mean())
    out["n_face_in"], out["n_face_out"] = int(fin.sum()), int(fout.sum())
    if not band.any():
        for k in K_LIST:
            out[f"in_cov{k}"] = 0.0
            out[f"out_cov{k}"] = 0.0
            out[f"side{k}"] = 0.5
        out["frac_closer_in"] = None
        return out
    d = _edt(~band)
    for k in K_LIST:
        ci = float((d[fin] <= k).mean()) if fin.any() else 0.0
        co = float((d[fout] <= k).mean()) if fout.any() else 0.0
        out[f"in_cov{k}"] = ci
        out[f"out_cov{k}"] = co
        out[f"side{k}"] = (ci / (ci + co)) if (ci + co) > 0 else 0.5
    sel = band & f["supervise"] if support is None else band & f["supervise"] & support
    if sel.any():
        di, do = np.abs(f["sdf_in"][sel]), np.abs(f["sdf_out"][sel])
        out["frac_closer_in"] = float((di < do).mean())
        out["frac_closer_out"] = float((do < di).mean())
        out["n_band_supervised"] = int(sel.sum())
    else:
        out["frac_closer_in"] = out["frac_closer_out"] = None
    return out


def medial_vs_faces(prob: np.ndarray, f: dict[str, Any], thr: float = 0.5) -> dict[str, Any]:
    """Distance of the band's medial surface to the in / out face zero sets."""
    from tsm.labels import _drop_small_components, medial_surface

    med = medial_surface(_drop_small_components(prob >= thr, 100))
    return {"n_medial": int(med.sum()),
            "to_face_in": E.surface_distances(med, f["face_in"])["a_to_b"],
            "to_face_out": E.surface_distances(med, f["face_out"])["a_to_b"]}


def classify_side(side: float, id_side: float, rel: float = 0.9, floor: float = 0.6) -> str:
    return "preserving" if (side >= rel * id_side and side >= floor) else "flipping"


# --------------------------------------------------------------------------- #
# the direction rule for S_flip coherence
# --------------------------------------------------------------------------- #
def _lattice_dirs() -> list[np.ndarray]:
    ds = []
    for z in (-1, 0, 1):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                if (z, y, x) != (0, 0, 0):
                    v = np.array([z, y, x], dtype=np.float64)
                    ds.append(v / np.linalg.norm(v))
    return ds


def direction_rule(rows: list[dict[str, Any]], extra: int = 400, seed: int = 0) -> dict[str, Any]:
    """Best 'sign of d . g d' rule separating preserving from flipping octahedral elements."""
    oct_rows = [r for r in rows if r["type"] == "oct"]
    y = np.array([0.0 if r["side_class"] == "flipping" else 1.0 for r in oct_rows])  # identity counts as preserving
    gs = [np.asarray(r["matrix"], dtype=np.float64) for r in oct_rows]
    cands = list(_lattice_dirs())
    rng = np.random.default_rng(seed)
    for _ in range(extra):  # small perturbations of the lattice directions
        v = cands[rng.integers(len(_lattice_dirs()))] + rng.normal(scale=0.25, size=3)
        cands.append(v / np.linalg.norm(v))
    best = {"acc": -1.0}
    for d in cands:
        pred = np.array([1.0 if float(d @ (g @ d)) > 0 else 0.0 for g in gs])
        for flip in (False, True):
            p = 1.0 - pred if flip else pred
            acc = float((p == y).mean())
            if acc > best["acc"]:
                best = {"acc": acc, "d": [float(v) for v in d], "invert": flip,
                        "n": len(y), "errors": [oct_rows[i]["name"] for i in np.nonzero(p != y)[0]]}
    base = {
        "det_positive": float((np.array([1.0 if r["det"] > 0 else 0.0 for r in oct_rows]) == y).mean()),
        "no_z_swap": float((np.array([0.0 if r["swaps_z"] else 1.0 for r in oct_rows]) == y).mean()),
        "majority": float(max(y.mean(), 1 - y.mean())),
    }
    best["baselines"] = base
    return best


# --------------------------------------------------------------------------- #
# curvature split
# --------------------------------------------------------------------------- #
def curvature_split(f: dict[str, Any], axis_yx: tuple[float, float] | None, origin: list[int]) -> dict[str, np.ndarray]:
    """Face-voxel subsets: flat vs curved (|Laplacian of sdf_in|) and near vs far from the axis."""
    from scipy import ndimage as ndi

    lap = np.abs(ndi.laplace(f["sdf_in"].astype(np.float32)))
    fin, fout = f["face_in"], f["face_out"]
    both = fin | fout
    if not both.any():
        return {}
    t = float(np.median(lap[both]))
    out = {"flat": lap <= t, "curved": lap > t, "_lap_median": t}
    if axis_yx is not None:
        zz, yy, xx = np.indices(f["face_in"].shape)
        r = np.sqrt((yy + origin[1] - axis_yx[0]) ** 2 + (xx + origin[2] - axis_yx[1]) ** 2)
        rt = float(np.median(r[both]))
        out["near_axis"] = r <= rt
        out["far_axis"] = r > rt
        out["_r_median"] = rt
    return out


def side_on_subset(prob: np.ndarray, f: dict[str, Any], sub: np.ndarray, k: int = 1, thr: float = 0.5) -> dict[str, Any]:
    band = prob >= thr
    if not band.any():
        return {"in_cov": 0.0, "out_cov": 0.0, "side": 0.5, "n_in": 0, "n_out": 0}
    d = _edt(~band)
    fin, fout = f["face_in"] & sub, f["face_out"] & sub
    ci = float((d[fin] <= k).mean()) if fin.any() else 0.0
    co = float((d[fout] <= k).mean()) if fout.any() else 0.0
    return {"in_cov": ci, "out_cov": co, "side": (ci / (ci + co)) if ci + co else 0.5,
            "n_in": int(fin.sum()), "n_out": int(fout.sum())}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _f(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def write_markdown(path: str, res: dict[str, Any]) -> None:
    rows = res["transforms"]
    L = [f"# Side-preserving TTA: {res['model']} on {res['box']['shape']} at level {res['box'].get('level')}", "",
         f"Faces: `{res['faces']['path']}` origin {res['faces']['origin_zyx']} shape {res['faces']['shape']}; "
         f"evaluation window (level 0) {res['window']['level0_lo']}..{res['window']['level0_hi']}; "
         f"face_in {rows[0]['side']['n_face_in']} voxels, face_out {rows[0]['side']['n_face_out']}.", "",
         f"Identity side score (k=1) **{_f(res['identity_side'])}**; classification threshold "
         f"max(0.9 x identity, 0.6) = **{_f(res['side_threshold'])}**.", "",
         f"S_pres: {len(res['s_pres'])} of 48 octahedral elements; S_flip: {len(res['s_flip'])}.", "",
         "## Per transform (sorted by side score k=1)",
         "| name | det | swaps_z | flips | side1 | side3 | in_cov1 | out_cov1 | in_cov3 | out_cov3 | closer_in | dice_id | med_med | class |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["type"] != "oct", -r["side"]["side1"])):
        s, m = r["side"], r["vs_identity"]
        flips = "".join("zyx"[i] for i, sg in enumerate(r["sign"] or []) if sg < 0) or "-"
        L.append(f"| {r['name']} | {r['det']} | {int(r['swaps_z'])} | {flips} | {_f(s['side1'])} | {_f(s['side3'])} | "
                 f"{_f(s['in_cov1'])} | {_f(s['out_cov1'])} | {_f(s['in_cov3'])} | {_f(s['out_cov3'])} | "
                 f"{_f(s['frac_closer_in'])} | {_f(m['dice'])} | {_f(m['medial']['symmetric']['median'])} | {r['side_class']} |")
    L += ["", "## By class"]
    L += ["| class | n | mean side1 | min | max | n preserving |", "|---|---|---|---|---|---|"]
    for k, v in res["by_class"].items():
        L.append(f"| {k} | {v['n']} | {_f(v['mean'])} | {_f(v['min'])} | {_f(v['max'])} | {v['n_preserving']} |")
    L += ["", "## Averaged bands"]
    L += ["| set | n | side1 | side3 | in_cov1 | out_cov1 | closer_in | closer_out | dice vs identity | medial med vs identity |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["averages"].items():
        s = v["side"]
        L.append(f"| {k} | {v['n']} | {_f(s['side1'])} | {_f(s['side3'])} | {_f(s['in_cov1'])} | {_f(s['out_cov1'])} | "
                 f"{_f(s['frac_closer_in'])} | {_f(s['frac_closer_out'])} | {_f(v['vs_identity']['dice'])} | "
                 f"{_f(v['vs_identity']['medial']['symmetric']['median'])} |")
    if res.get("average_pairs"):
        L += ["", "| pair | dice | medial median |", "|---|---|---|"]
        for k, v in res["average_pairs"].items():
            L.append(f"| {k} | {_f(v['dice'])} | {_f(v['medial']['symmetric']['median'])} |")
    if res.get("medial_vs_faces"):
        L += ["", "## Averaged band medial vs the CT faces (voxels)",
              "| set | n medial | to face_in median | p90 | to face_out median | p90 |", "|---|---|---|---|---|---|"]
        for k, v in res["medial_vs_faces"].items():
            L.append(f"| {k} | {v['n_medial']} | {_f(v['to_face_in']['median'])} | {_f(v['to_face_in']['p90'])} | "
                     f"{_f(v['to_face_out']['median'])} | {_f(v['to_face_out']['p90'])} |")
    if res.get("direction_rule"):
        d = res["direction_rule"]
        L += ["", "## Is S_flip coherent?",
              f"Best rule `sign(d . g d)`{' (inverted)' if d['invert'] else ''} with "
              f"d = [{', '.join(f'{v:.3f}' for v in d['d'])}] (z, y, x): accuracy **{_f(d['acc'])}** on {d['n']} elements; "
              f"errors: {', '.join(d['errors']) if d['errors'] else 'none'}.",
              f"Baselines: det=+1 {_f(d['baselines']['det_positive'])}, no z-swap {_f(d['baselines']['no_z_swap'])}, "
              f"majority {_f(d['baselines']['majority'])}."]
    if res.get("curvature"):
        L += ["", "## Flip completeness by local geometry (side score k=1)",
              "| subset | n face_in | n face_out | " + " | ".join(res["curvature"]["sets"]) + " |",
              "|---|---|---|" + "---|" * len(res["curvature"]["sets"])]
        for sub, v in res["curvature"]["rows"].items():
            first = next(iter(v.values()))
            L.append(f"| {sub} | {first['n_in']} | {first['n_out']} | "
                     + " | ".join(_f(v[s]["side"]) for s in res["curvature"]["sets"]) + " |")
    if res.get("montage"):
        L += ["", f"Montage: `{res['montage']}`"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# montage
# --------------------------------------------------------------------------- #
def write_montage(path: str, ct: np.ndarray, f: dict[str, Any], panels: list[tuple[str, np.ndarray]]) -> str:
    from PIL import Image, ImageDraw

    from tsm.labels import _overlay, _to_rgb

    z = ct.shape[0] // 2
    base = _to_rgb(ct[z].astype(np.uint8))
    tiles = [(_overlay(_overlay(base.copy(), f["face_in"][z], (255, 40, 40), 0.8),
                       f["face_out"][z], (60, 120, 255), 0.8), "CT + face_in red / face_out blue")]
    for name, p in panels:
        img = _overlay(_overlay(_overlay(base.copy(), f["face_in"][z], (255, 40, 40), 0.8),
                                f["face_out"][z], (60, 120, 255), 0.8),
                       p[z] >= 0.5, (40, 255, 60), 0.55)
        tiles.append((img, name))
    h, w = base.shape[:2]
    pad = 18
    img = Image.new("RGB", (len(tiles) * (w + 4), h + pad), (25, 25, 25))
    dr = ImageDraw.Draw(img)
    for i, (t, name) in enumerate(tiles):
        img.paste(Image.fromarray(np.ascontiguousarray(t)), (i * (w + 4), pad))
        dr.text((i * (w + 4) + 2, 3), name, fill=(255, 255, 255))
    img.save(path)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def find_pred(dirs: list[str], name: str) -> str | None:
    for d in dirs:
        p = os.path.join(os.path.expanduser(d), "preds", f"{name}.npy")
        if os.path.exists(p):
            return p
    return None


def side_manifest(args: Any, name: str, level_shift: int, box_info: dict[str, Any],
                  fw: dict[str, Any], transforms: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment identity of a cached prediction set: box, model, precision, faces store, transforms."""
    fs = os.path.abspath(os.path.expanduser(args.faces_store))
    return {
        "script": "tta_side", "teacher": args.teacher, "model": name, "level_shift": int(level_shift),
        "config": os.path.abspath(args.config), "dtype": args.dtype, "models_dir": os.path.abspath(args.models_dir),
        "box": {k: box_info.get(k) for k in ("level", "factor", "start_zyx", "size", "voxel_um", "shape")},
        "faces_store": fs, "faces_fingerprint": OB.file_fingerprint(fs),
        "faces_origin_zyx": list(fw["origin_zyx"]), "faces_shape": list(fw["shape"]),
        "group": args.group, "rotations": int(args.rotations), "max_degs": str(args.max_degs),
        "seed": int(args.seed),
        "transforms": {t["name"]: np.round(np.asarray(t["matrix"], np.float64), 9).tolist() for t in transforms},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="recto")
    ap.add_argument("--config", default="configs/paris4_small.json")
    ap.add_argument("--box", type=int, default=None)
    ap.add_argument("--offset", default=None)
    ap.add_argument("--group", default="oct48", choices=sorted(E.GROUPS))
    ap.add_argument("--faces-store", default=FACES_STORE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred-dirs", nargs="*", default=[], help="extra dirs holding a preds/ cache")
    ap.add_argument("--rotations", type=int, default=0, help="random rotations per max-deg bucket")
    ap.add_argument("--max-degs", default="30,90")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--models-dir", default=os.path.expanduser("~/.cache/tsm-models"))
    ap.add_argument("--time-budget", type=float, default=900.0)
    ap.add_argument("--gpu-max-used-gb", type=float, default=4.0)
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-run", action="store_true", help="fail instead of running the model for a missing prediction")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    name, shift = OB.parse_teacher(args.teacher)
    kind = OB.KIND[name]
    level_shift = shift if shift is not None else OB.DEFAULT_LEVEL_SHIFT[name]
    tag = args.teacher.replace("@", "_")
    out_dir = args.out or os.path.join(OUT_ROOT, tag)
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)

    from tsm.teachers import TEACHERS

    voxel_um = TEACHERS[name].voxel_um
    size = args.box or int(TEACHERS[name].patch[0])
    offset = tuple(int(v) for v in args.offset.split(",")) if args.offset else None
    box, box_info = OB.load_box(args.config, name, voxel_um, level_shift, size, offset, None)
    box_info["shape"] = list(box.shape)
    log(f"box {box.shape} start {box_info['start_zyx']} level {box_info['level']} factor {box_info['factor']}")

    faces = load_faces(args.faces_store)
    win = face_window(faces, box_info["start_zyx"], size, box_info["factor"])
    sl = win["faces_sl"]
    fw = {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in faces.items()}
    fw["origin_zyx"], fw["shape"], fw["path"] = faces["origin_zyx"], faces["shape"], faces["path"]
    log(f"window {win['level0_lo']}..{win['level0_hi']} face_in {int(fw['face_in'].sum())} "
        f"face_out {int(fw['face_out'].sum())} supervised {float(fw['supervise'].mean()):.3f}")

    # ---- transforms ------------------------------------------------------- #
    transforms = OB.build_transforms(args.group, 0, 0.0, args.seed, tuple(box.shape))
    if args.rotations:
        rng = np.random.default_rng(args.seed)
        for md in [float(v) for v in args.max_degs.split(",")]:
            for i in range(args.rotations):
                R = E.random_rotation_matrix(rng, md)
                ang = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2))))
                transforms.append({"name": f"rot{int(md)}_{i}_{ang:.1f}deg", "identity": False, "det": 1,
                                   "pure_flip": False, "flips_z": False, "in_plane": False, "swaps_z": False,
                                   "n_flips": 0, "perm": None, "sign": None, "matrix": R.tolist(),
                                   "type": "rotation", "angle_deg": ang, "max_deg": md,
                                   "mask": E.rotation_valid_mask(tuple(box.shape), R, margin=3)})

    # R22: only reuse a preds/ cache whose manifest describes exactly this experiment
    manifest = side_manifest(args, name, level_shift, box_info, fw, transforms)
    manifest_path = os.path.join(out_dir, "preds", "manifest.json")
    dirs = []
    for d in [out_dir] + list(args.pred_dirs):
        diff = OB.manifest_diff(manifest, OB.read_manifest(os.path.join(os.path.expanduser(d), "preds", "manifest.json")))
        if not diff:
            dirs.append(d)
        elif any(find_pred([d], t["name"]) for t in transforms):
            log(f"ignoring the prediction cache in {d}: different {diff}")
    todo = [t for t in transforms if find_pred(dirs, t["name"]) is None]
    log(f"{len(transforms)} transforms, {len(transforms) - len(todo)} cached")
    if todo:
        dirs = [out_dir] + [d for d in dirs if d != out_dir]  # the new predictions land in out_dir
        if args.no_run:
            raise SystemExit(f"{len(todo)} predictions missing and --no-run given: {[t['name'] for t in todo][:5]}")
        device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                              (args.device if args.device != "auto" else "cpu"))
        if device.type == "cuda":
            used = OB.gpu_used_gb()
            if used > args.gpu_max_used_gb and not args.force_gpu:
                raise SystemExit(f"GPU has {used:.1f} GB in use (> {args.gpu_max_used_gb}); "
                                 f"use --device cpu or --force-gpu")
            log(f"GPU in use before start: {used:.1f} GB")
        dt = args.dtype or ("bf16" if device.type == "cuda" else "fp32")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
        run, _ = OB.make_runner(name, device, dtype, args.models_dir, None, 20.0)
        OB.write_manifest(manifest_path, manifest)  # these preds belong to this experiment
        box_t = torch.from_numpy(box).to(device).float()
        t0 = time.perf_counter()
        for t in todo:
            g = np.asarray(t["matrix"])
            xin = E.apply(box_t, g.astype(np.int64)) if t["type"] == "oct" else E.apply_rotation(box_t, g)
            pred = run(xin)
            del xin
            inv = E.invert_prediction(pred, g, OB.RULE[kind])
            np.save(os.path.join(out_dir, "preds", f"{t['name']}.npy"), np.asarray(inv, dtype=np.float16))
            log(f"{t['name']:>18s} {time.perf_counter() - t0:6.1f}s")
            if time.perf_counter() - t0 > args.time_budget and t is not todo[-1]:
                log("time budget reached; re-run to resume")
                return 3
        del box_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- per-transform metrics -------------------------------------------- #
    def load(nm: str) -> np.ndarray:
        return np.load(find_pred(dirs, nm)).astype(np.float32)

    ident_full = load(transforms[0]["name"])
    ident = pred_on_window(ident_full, win)
    id_side = side_metrics(ident, fw)
    id_s1 = id_side["side1"]
    thr = max(0.9 * id_s1, 0.6)
    log(f"identity side1 {id_s1:.4f} in {id_side['in_cov1']:.4f} out {id_side['out_cov1']:.4f} -> threshold {thr:.4f}")

    rows: list[dict[str, Any]] = []
    for t in transforms:
        p = pred_on_window(load(t["name"]), win)
        # arbitrary rotations only predict inside their valid support; everything outside is
        # interpolation/padding and must not be scored (R23)
        sup = mask_on_window(t["mask"], win) if t.get("mask") is not None else None
        s = side_metrics(p, fw, support=sup)
        m = E.metrics(ident, p, "surface", sup)
        row = {k: v for k, v in t.items() if k != "mask"}
        row.update({"side": s, "vs_identity": m,
                    "side_class": "identity" if t["identity"] else classify_side(s["side1"], id_s1)})
        rows.append(row)
        log(f"{t['name']:>18s} side1 {s['side1']:.4f} in {s['in_cov1']:.4f} out {s['out_cov1']:.4f} "
            f"dice {m['dice']:.4f} {row['side_class']}")

    oct_rows = [r for r in rows if r["type"] == "oct"]
    s_pres = [r["name"] for r in oct_rows if r["side_class"] in ("preserving", "identity")]
    s_flip = [r["name"] for r in oct_rows if r["side_class"] == "flipping"]

    def cls(sel: list[dict[str, Any]]) -> dict[str, Any]:
        v = [r["side"]["side1"] for r in sel]
        return {"n": len(sel), "mean": (float(np.mean(v)) if v else None), "min": (min(v) if v else None),
                "max": (max(v) if v else None),
                "n_preserving": sum(1 for r in sel if r["side_class"] in ("preserving", "identity"))}

    non_id = [r for r in oct_rows if not r["identity"]]
    by_class = {
        "identity": cls([r for r in oct_rows if r["identity"]]),
        "pure_flips": cls([r for r in non_id if r["pure_flip"]]),
        "z_flip_only": cls([r for r in non_id if r["pure_flip"] and r["sign"] == [-1, 1, 1]]),
        "in_plane_flips": cls([r for r in non_id if r["pure_flip"] and r["sign"][0] == 1]),
        "in_plane_rot90_about_z": cls([r for r in non_id if r["in_plane"] and r["det"] == 1 and r["sign"][0] == 1]),
        "in_plane_any": cls([r for r in non_id if r["in_plane"]]),
        "swaps_z": cls([r for r in non_id if r["swaps_z"]]),
        "det_plus1": cls([r for r in non_id if r["det"] == 1]),
        "det_minus1": cls([r for r in non_id if r["det"] == -1]),
    }
    for md in sorted({r.get("max_deg") for r in rows if r["type"] == "rotation"} - {None}):
        by_class[f"rotations<={int(md)}deg"] = cls([r for r in rows if r.get("max_deg") == md])

    # ---- averaged bands ---------------------------------------------------- #
    def average(names: list[str]) -> np.ndarray:
        acc = np.zeros_like(ident)
        for n in names:
            acc += pred_on_window(load(n), win)
        return acc / max(len(names), 1)

    all48 = [r["name"] for r in oct_rows]
    sets = {"identity": [transforms[0]["name"]], "S_pres": s_pres, "S_flip": s_flip, "all48": all48}
    avg = {k: (ident if k == "identity" else average(v)) for k, v in sets.items() if v}
    averages = {}
    for k, a in avg.items():
        averages[k] = {"n": len(sets[k]), "members": sets[k], "side": side_metrics(a, fw),
                       "vs_identity": E.metrics(ident, a, "surface", None)}
        log(f"avg {k:>8s} n={len(sets[k]):2d} side1 {averages[k]['side']['side1']:.4f} "
            f"in {averages[k]['side']['in_cov1']:.4f} out {averages[k]['side']['out_cov1']:.4f}")
    pairs = {}
    keys = [k for k in ("S_pres", "S_flip", "all48") if k in avg]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs[f"{a} vs {b}"] = E.metrics(avg[a], avg[b], "surface", None)
    mvf = {k: medial_vs_faces(a, fw) for k, a in avg.items()}

    # ---- coherence of S_flip / curvature ----------------------------------- #
    rule = direction_rule(rows)
    log(f"direction rule acc {rule['acc']:.3f} d={np.round(rule['d'], 3).tolist()} invert={rule['invert']}")
    curv_sets = {k: v for k, v in curvature_split(fw, None, win["level0_lo"]).items() if not k.startswith("_")}
    curvature = None
    if curv_sets:
        keys2 = [k for k in ("identity", "S_pres", "S_flip", "all48") if k in avg]
        curvature = {"sets": keys2, "rows": {sub: {k: side_on_subset(avg[k], fw, m) for k in keys2}
                                            for sub, m in curv_sets.items()}}

    # ---- montage ----------------------------------------------------------- #
    montage = None
    if kind == "surface":
        best = max([r for r in oct_rows if not r["identity"]], key=lambda r: r["side"]["side1"])
        worst = min(oct_rows, key=lambda r: r["side"]["side1"])
        ct_win = pred_on_window(box[None].astype(np.float32), win)
        panels = [("identity band", ident),
                  (f"preserving {best['name']}", pred_on_window(load(best["name"]), win)),
                  (f"flipping {worst['name']}", pred_on_window(load(worst["name"]), win))]
        for k in ("S_pres", "S_flip", "all48"):
            if k in avg:
                panels.append((f"avg {k}", avg[k]))
        montage = write_montage(os.path.join(out_dir, "montage.png"), ct_win, fw, panels)

    res = {
        "model": args.teacher, "kind": kind, "group": args.group, "box": box_info, "level_shift": level_shift,
        "faces": {"path": faces["path"], "origin_zyx": faces["origin_zyx"], "shape": faces["shape"]},
        "window": {k: v for k, v in win.items() if k != "faces_sl"},
        "identity_side": id_s1, "side_threshold": thr, "rule": "side1 >= max(0.9*identity, 0.6)",
        "transforms": rows, "by_class": by_class, "s_pres": s_pres, "s_flip": s_flip,
        "averages": averages, "average_pairs": pairs, "medial_vs_faces": mvf,
        "direction_rule": rule, "curvature": curvature, "montage": montage,
    }
    with open(os.path.join(out_dir, "table.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    write_markdown(os.path.join(out_dir, "table.md"), res)
    log(f"S_pres ({len(s_pres)}): {' '.join(s_pres)}")
    log(f"S_flip ({len(s_flip)}): {' '.join(s_flip)}")
    log(f"wrote {out_dir}/table.md, table.json" + (f", {montage}" if montage else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
