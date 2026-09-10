"""Build the native-2.4 um winding field (tsm.winding_fine) over a crop and montage it.

    uv run python dev/winding_fine_crop.py configs/paris4_faces_rv_crop.json \
        --fine ~/tsm-output/slab_faces_rv/labels/fine.zarr \
        --coarse ~/tsm-output/slab_faces_tta/labels/coarse.zarr \
        --z 100 --out ~/tsm-output/winding_fine_crop

Reads the two-face channels of an existing fine store and the coarse (9.6 um) winding store,
runs :func:`tsm.winding_fine.winding_fine_box` brick-wise with a halo (nothing is written to
any store), and writes ``<out>/winding_fine_z<z>.png``:

    CT | coarse-upsampled phase (cos) | refined phase (cos) | refined valid | |dw| (heat)

and prints the fraction of the crop the refined field supervises, the median ``|dw|`` against
the coarse prior (in turns, over the voxels both cover) and the fraction beyond 0.25 turn.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.cli import open_ct  # noqa: E402
from tsm.config import load_config  # noqa: E402
from tsm.labels import DEFAULT_AXIS, load_axis, radial_field, read_box_padded  # noqa: E402
from tsm.winding_fine import WF_DEFAULTS, upsample_coarse_prior, winding_fine_box  # noqa: E402

RESID_BINS = 200   # signed dw histogram over [-0.5, 0.5) turns
PERIOD_MAX = 400   # sheet-spacing histogram, 1 voxel per bin
RATIO_BINS = 400   # coarse period / fine period, 0 .. RATIO_MAX
RATIO_MAX = 20.0


def _hist_pct(hist: np.ndarray, centres: np.ndarray, p: float) -> float | None:
    """Percentile of the values a histogram counts (bin centres)."""
    n = int(hist.sum())
    if n == 0:
        return None
    cum = np.cumsum(hist)
    j = int(np.searchsorted(cum, (n - 1) * p / 100.0, side="right"))
    return float(centres[min(j, len(centres) - 1)])


def _circ_mean(hist: np.ndarray, centres: np.ndarray) -> float:
    """Circular mean (turns, in [-0.5, 0.5)) of a histogram of signed phase residuals."""
    if hist.sum() == 0:
        return 0.0
    ang = 2 * np.pi * centres
    return float(np.arctan2((hist * np.sin(ang)).sum(), (hist * np.cos(ang)).sum()) / (2 * np.pi))


def _decode_signed(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.float32) - 127.5) / 127.5


def _gray(a: np.ndarray, gain: float = 1.0) -> np.ndarray:
    return np.stack([np.clip(a.astype(np.float32) * gain, 0, 255).astype(np.uint8)] * 3, -1)


def _cos_rgb(cos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """cos(2 pi w) in [-1, 1] -> blue (trough) .. black .. orange (crest); invalid = dark grey."""
    v = np.clip(cos, -1, 1)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    rgb = np.stack([255 * pos, 160 * pos + 60 * neg, 255 * neg], -1).astype(np.uint8)
    rgb[~valid] = (25, 25, 25)
    return rgb


def _heat(v: np.ndarray, vmax: float, valid: np.ndarray) -> np.ndarray:
    """0..vmax -> black-red-yellow-white; invalid = dark grey."""
    t = np.clip(np.asarray(v, np.float32) / max(vmax, 1e-6), 0, 1)
    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(3 * t - 2, 0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[~valid] = (25, 25, 25)
    return rgb


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0))
    return np.asarray(pil)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", required=True, help="output directory (png + json)")
    ap.add_argument("--fine", default=None, help="fine store with sdf_in/sdf_out (default <out_dir>/labels/fine.zarr)")
    ap.add_argument("--coarse", default=None, help="coarse winding store (default <out_dir>/labels/coarse.zarr)")
    ap.add_argument("--axis", default=None, help="umbilicus json (default extra.labels.axis_path)")
    ap.add_argument("--start", type=int, nargs=3, default=None, help="global z y x (default the config region)")
    ap.add_argument("--size", type=int, nargs=3, default=None, help="dz dy dx (default the config region)")
    ap.add_argument("--z", type=int, default=100, help="montage slice, local to the crop")
    ap.add_argument("--brick", type=int, nargs=3, default=[128, 256, 256])
    ap.add_argument("--halo", type=int, default=48)
    ap.add_argument("--clip", type=float, default=20.0, help="the SDF clip the store was built with")
    ap.add_argument("--snap-max", type=float, default=WF_DEFAULTS["snap_max"],
                    help="opt-in: gate the fine phase on the coarse prior (default: no gate)")
    ap.add_argument("--phase-offset", type=float, default=WF_DEFAULTS["phase_offset"])
    ap.add_argument("--min-period", type=float, default=WF_DEFAULTS["min_period"])
    ap.add_argument("--max-period", type=float, default=WF_DEFAULTS["max_period"])
    ap.add_argument("--ct-gain", type=float, default=1.4)
    a = ap.parse_args(argv)
    from PIL import Image

    cfg = load_config(a.config)
    lopts = cfg.extra.get("labels", {}) or {}
    fine_path = os.path.expanduser(a.fine or os.path.join(cfg.out_dir, "labels", "fine.zarr"))
    coarse_path = os.path.expanduser(a.coarse or os.path.join(cfg.out_dir, "labels", "coarse.zarr"))
    axis = load_axis(os.path.expanduser(a.axis or lopts.get("axis_path") or DEFAULT_AXIS))
    start = [int(v) for v in (a.start or cfg.region.start_zyx)]
    size = [int(v) for v in (a.size or cfg.region.size_zyx)]
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)

    fine = zarr.open_array(store=fine_path, mode="r")
    fch = {n: i for i, n in enumerate(fine.attrs["channels"])}
    fo = [int(v) for v in fine.attrs["origin_zyx"]]
    for need in ("sdf_in", "sdf_out", "faces_valid"):
        if need not in fch:
            raise SystemExit(f"{fine_path} has no {need!r} channel (channels={sorted(fch)})")
    coarse = zarr.open_array(store=coarse_path, mode="r")
    co = [int(v) for v in coarse.attrs["origin_zyx"]]
    k = int(round(float(coarse.attrs.get("scale", 4))))
    print(f"[wf] fine {fine_path} shape={tuple(fine.shape)} origin={fo}", flush=True)
    print(f"[wf] coarse {coarse_path} shape={tuple(coarse.shape)} origin={co} scale={k}", flush=True)
    print(f"[wf] crop start={start} size={size} brick={a.brick} halo={a.halo}", flush=True)

    zl = int(a.z)
    if not 0 <= zl < size[0]:
        raise SystemExit(f"--z {zl} outside the crop depth {size[0]}")
    panel = {n: np.zeros((size[1], size[2]), dt)
             for n, dt in (("cos_fine", np.float32), ("cos_coarse", np.float32),
                           ("resid", np.float32), ("ratio", np.float32),
                           ("valid", np.uint8), ("pvalid", np.uint8))}

    shist = np.zeros(RESID_BINS, np.int64)              # signed |dw| over [-0.5, 0.5)
    ph_fine = np.zeros(PERIOD_MAX + 1, np.int64)        # geometric sheet spacing (voxels)
    ph_coarse = np.zeros(PERIOD_MAX + 1, np.int64)      # 1 / coarse density, same voxels
    rhist = np.zeros(RATIO_BINS, np.int64)              # coarse period / fine period
    n_vox = n_valid = n_cand = 0
    t0 = time.perf_counter()
    bz, by, bx = (max(1, int(v)) for v in a.brick)
    h = int(a.halo)
    nb = sum(1 for _ in range(0, size[0], bz)) * sum(1 for _ in range(0, size[1], by)) \
        * sum(1 for _ in range(0, size[2], bx))
    i = 0
    for z0 in range(0, size[0], bz):
        for y0 in range(0, size[1], by):
            for x0 in range(0, size[2], bx):
                lo = (z0, y0, x0)
                hi = (min(z0 + bz, size[0]), min(y0 + by, size[1]), min(x0 + bx, size[2]))
                glo = [start[d] + lo[d] - h for d in range(3)]
                ghi = [start[d] + hi[d] + h for d in range(3)]
                shp = tuple(ghi[d] - glo[d] for d in range(3))
                sdf_in, sdf_out, fv = (
                    read_box_padded(fine, [g - fo[d] for d, g in enumerate(glo)],
                                    [g - fo[d] for d, g in enumerate(ghi)], channels=fch[c])[0]
                    for c in ("sdf_in", "sdf_out", "faces_valid"))
                prior = upsample_coarse_prior(coarse, [g - co[d] * k for d, g in enumerate(glo)],
                                              [g - co[d] * k for d, g in enumerate(ghi)], scale=k,
                                              normal=True, density=True)
                rad = radial_field(axis, glo, shp, scale=1)
                stats: dict = {}
                out = winding_fine_box(sdf_in, sdf_out, fv, prior, rad, float(a.clip),
                                       snap_max=(None if a.snap_max is None else float(a.snap_max)),
                                       min_period=float(a.min_period),
                                       max_period=float(a.max_period),
                                       phase_offset=float(a.phase_offset), stats=stats)
                core = tuple(slice(h, h + hi[d] - lo[d]) for d in range(3))
                sin_f = _decode_signed(out[0][core])
                cos_f = _decode_signed(out[1][core])
                valid = out[7][core] == 1
                pph = prior["phase"][core]
                pv = prior["valid"][core] == 1
                f = np.mod(np.arctan2(sin_f, cos_f) / (2 * np.pi), 1.0)
                diff = pph - f
                signed = diff - np.rint(diff)
                resid = np.abs(signed)
                cand = valid & pv
                shist += np.histogram(signed[cand], bins=RESID_BINS, range=(-0.5, 0.5))[0]
                # the two sheet spacings this crop implies: the geometry's (from the refined
                # density channel) and the coarse winding teacher's (1 / its density)
                pf = 1.0 / np.maximum(out[2][core].astype(np.float32) / 1000.0, 1e-6)
                pc = 1.0 / np.maximum(prior["density"][core], 1e-6)
                ph_fine += np.histogram(pf[cand], bins=PERIOD_MAX + 1, range=(-0.5, PERIOD_MAX + 0.5))[0]
                ph_coarse += np.histogram(pc[cand], bins=PERIOD_MAX + 1, range=(-0.5, PERIOD_MAX + 0.5))[0]
                ratio = pc / np.maximum(pf, 1e-6)
                rhist += np.histogram(ratio[cand], bins=RATIO_BINS, range=(0.0, RATIO_MAX))[0]
                n_vox += int(np.prod([hi[d] - lo[d] for d in range(3)]))
                n_valid += int(valid.sum())
                n_cand += int(cand.sum())
                del pf, pc, signed
                if lo[0] <= zl < hi[0]:
                    sl = (slice(y0, hi[1]), slice(x0, hi[2]))
                    zz = zl - lo[0]
                    panel["cos_fine"][sl] = cos_f[zz]
                    panel["cos_coarse"][sl] = np.cos(2 * np.pi * pph[zz])
                    panel["resid"][sl] = resid[zz]
                    panel["ratio"][sl] = ratio[zz]
                    panel["valid"][sl] = valid[zz]
                    panel["pvalid"][sl] = pv[zz]
                i += 1
                print(f"[wf] brick {i}/{nb} {lo}->{hi} valid {valid.mean():.3f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
                del sdf_in, sdf_out, fv, prior, rad, out, sin_f, cos_f, valid, pph, pv, f, resid, ratio

    # ---- numbers ----------------------------------------------------------
    rcen = (np.arange(RATIO_BINS) + 0.5) * (RATIO_MAX / RATIO_BINS)  # period-ratio bin centres
    scen = (np.arange(RESID_BINS) + 0.5) / RESID_BINS - 0.5       # signed residual bin centres
    pcen = np.arange(PERIOD_MAX + 1).astype(np.float64)           # period bin centres (voxels)
    fold = shist[::-1] + shist                                    # |dw|: fold the signed histogram
    ahist = fold[RESID_BINS // 2:]
    acen = scen[RESID_BINS // 2:]
    off = _circ_mean(shist, scen)
    # the same residual with the systematic phase-convention offset removed
    shift = int(round(off * RESID_BINS))
    cent = np.roll(shist, -shift)
    rfold = (cent[::-1] + cent)[RESID_BINS // 2:]

    res = {
        "fine_store": fine_path, "coarse_store": coarse_path,
        "start_zyx": start, "size_zyx": size, "z": zl,
        "opts": {"clip": a.clip, "snap_max": a.snap_max, "phase_offset": a.phase_offset,
                 "min_period": a.min_period, "max_period": a.max_period, "halo": h, "brick": a.brick},
        "frac_valid": n_valid / max(n_vox, 1),
        "n_compared": n_cand,
        "median_abs_dw_turns": _hist_pct(ahist, acen, 50), "p90_abs_dw_turns": _hist_pct(ahist, acen, 90),
        "frac_abs_dw_gt_0.25": float(ahist[len(acen) // 2:].sum() / max(ahist.sum(), 1)),
        # the fine phase has its integer turn on the IN face, the coarse teacher has its own
        # zero: this is that systematic offset, and the residual with it removed
        "phase_convention_offset_turns": off,
        "median_abs_dw_after_offset": _hist_pct(rfold, acen, 50),
        # the two notions of "one turn" this crop carries
        "sheet_spacing_voxels": {p: _hist_pct(ph_fine, pcen, p) for p in (10, 50, 90)},
        "coarse_spacing_voxels": {p: _hist_pct(ph_coarse, pcen, p) for p in (10, 50, 90)},
        # how badly the 9.6 um teacher aliases the sheets it is supposed to count
        "period_ratio": {p: _hist_pct(rhist, rcen, p) for p in (10, 50, 90)},
        "frac_period_ratio_gt_2": float(rhist[rcen > 2.0].sum() / max(rhist.sum(), 1)),
        "abs_dw_hist": {"edges": [float(v) for v in np.linspace(0.0, 0.5, RESID_BINS // 2 + 1)],
                        "counts": [int(v) for v in ahist]},
        "seconds": time.perf_counter() - t0,
    }
    med_f, med_c = res["sheet_spacing_voxels"][50], res["coarse_spacing_voxels"][50]
    res["spacing_ratio_coarse_over_fine"] = (None if not med_f else (med_c or 0.0) / med_f)
    print(f"[wf] fraction valid           {res['frac_valid']:.4f}")
    print(f"[wf] median |dw| vs coarse    {res['median_abs_dw_turns']} turns "
          f"(p90 {res['p90_abs_dw_turns']}, over {n_cand} voxels)")
    print(f"[wf] fraction |dw| > 0.25     {res['frac_abs_dw_gt_0.25']:.4f}")
    print(f"[wf] phase convention offset  {off:+.3f} turns -> median |dw| after it "
          f"{res['median_abs_dw_after_offset']}")
    print(f"[wf] sheet spacing (geometry) p10/50/90 {res['sheet_spacing_voxels']} voxels")
    print(f"[wf] sheet spacing (coarse)   p10/50/90 {res['coarse_spacing_voxels']} voxels "
          f"-> ratio {res['spacing_ratio_coarse_over_fine']}")
    print(f"[wf] period ratio p10/50/90   {res['period_ratio']}; "
          f"fraction aliased (> 2) {res['frac_period_ratio_gt_2']:.4f}")
    print("[wf] |dw| histogram (20 bins of 0.025 turns): "
          + " ".join(f"{v:.3f}" for v in (ahist.reshape(-1, 5).sum(1) / max(ahist.sum(), 1))))

    # ---- montage ----------------------------------------------------------
    ct = open_ct(cfg, level_shift=0).read(start[0] + zl, start[0] + zl + 1,
                                          start[1], start[1] + size[1],
                                          start[2], start[2] + size[2])[0]
    vf = panel["valid"].astype(bool)
    vp = panel["pvalid"].astype(bool)
    base = _gray(ct, a.ct_gain)
    vimg = base.copy()
    vimg[vf] = (60, 220, 90)
    ims = [
        _label(base, f"CT z={start[0] + zl}"),
        _label(_cos_rgb(panel["cos_coarse"], vp), "coarse phase cos (upsampled)"),
        _label(_cos_rgb(panel["cos_fine"], vf), "refined phase cos (2.4 um)"),
        _label(vimg, f"refined valid ({vf.mean():.3f} of the slice)"),
        _label(_heat(panel["resid"], 0.5, vf & vp), "|dw| vs coarse (0 black .. 0.5 white)"),
        _label(_heat(panel["ratio"], 8.0, vf & vp), "coarse period / fine period (0 .. 8)"),
    ]
    out_png = os.path.join(os.path.expanduser(a.out), f"winding_fine_z{zl}.png")
    Image.fromarray(np.concatenate(ims, 1)).save(out_png)
    res["png"] = out_png
    out_json = os.path.join(os.path.expanduser(a.out), f"winding_fine_z{zl}.json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[wf] wrote {out_png} and {out_json} in {res['seconds']:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
