"""Sanity statistics for the label stores: ``uv run python dev/labels_sanity.py configs/paris4_roi.json``.

Reads <out_dir>/labels/{fine,coarse}.zarr and the axis, prints (and writes
<out_dir>/labels/sanity.json): sdf_valid fractions, decoded-sdf histogram, fraction of
supervised voxels within 1 voxel of the zero crossing, |sin,cos| unit-circle norm,
fraction of coarse normals with n . r_hat > 0, coarse valid fractions.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import zarr

from tsm.config import load_config
from tsm.labels import LABEL_DEFAULTS, decode_sdf_u8, decode_signed_u8, load_axis, radial_field


def main(argv: list[str]) -> int:
    cfg = load_config(argv[1])
    opts = dict(LABEL_DEFAULTS)
    opts.update(cfg.extra.get("labels", {}))
    clip = float(opts["clip"])
    ldir = os.path.join(cfg.out_dir, "labels")
    fine = zarr.open_array(store=os.path.join(ldir, "fine.zarr"), mode="r")
    coarse = zarr.open_array(store=os.path.join(ldir, "coarse.zarr"), mode="r")
    out: dict = {"fine": {}, "coarse": {}}

    # ---- fine (slab by slab along z to bound memory) ---------------------------
    nz = fine.shape[1]
    vh = np.zeros(3, dtype=np.int64)
    sdf_h = np.zeros(int(2 * clip) + 1, dtype=np.int64)
    edges = np.arange(-clip - 0.5, clip + 1.0, 1.0)
    n_sup = near = zero = n_pos = 0
    ink_sum = 0.0
    ink_n = 0
    nodata_sdf_ok = True
    for z0 in range(0, nz, 32):
        blk = np.asarray(fine[:, z0:z0 + 32])
        sdf_u8, val, ink, ink_v = blk
        vh += np.bincount(val.ravel(), minlength=3)[:3]
        sup = val == 1
        d = decode_sdf_u8(sdf_u8[sup], clip)
        n_sup += int(sup.sum())
        sdf_h += np.histogram(d, bins=edges)[0]
        near += int((np.abs(d) <= 1.0).sum())
        zero += int((sdf_u8[sup] == 128).sum())
        n_pos += int((d > 0).sum())
        nodata_sdf_ok &= bool((sdf_u8[val == 0] == 0).all()) and bool((sdf_u8[val > 0] > 0).all())
        ink_sum += float(ink[ink_v == 1].sum())
        ink_n += int((ink_v == 1).sum())
    tot = int(vh.sum())
    out["fine"] = {
        "shape": list(fine.shape), "attrs": dict(fine.attrs),
        "sdf_valid_fraction": {"0_nodata": vh[0] / tot, "1_supervise": vh[1] / tot, "2_ignore": vh[2] / tot},
        "sdf_hist_supervised": {"edges": edges.tolist(), "counts": sdf_h.tolist()},
        "frac_within_1vox_of_zero": near / max(n_sup, 1), "frac_zero_set": zero / max(n_sup, 1),
        "frac_sdf_positive": n_pos / max(n_sup, 1),
        "sdf_u8_zero_iff_nodata": nodata_sdf_ok,
        "ink_valid_fraction": ink_n / tot, "ink_mean_prob": ink_sum / max(ink_n, 1) / 255.0,
    }

    # ---- coarse -------------------------------------------------------------
    c = np.asarray(coarse[:])
    k = int(round(float(coarse.attrs.get("scale", 4.0))))
    origin_c = [int(v) for v in coarse.attrs["origin_zyx"]]
    axis = load_axis(opts["axis_path"])
    rad = radial_field(axis, origin_c, c.shape[1:], scale=k)
    s, cs = decode_signed_u8(c[0]), decode_signed_u8(c[1])
    n = np.stack([decode_signed_u8(c[5]), decode_signed_u8(c[4]), decode_signed_u8(c[3])])
    val = c[7]
    sup = val == 1
    vhc = np.bincount(val.ravel(), minlength=3)[:3]
    nrm = np.sqrt((n * n).sum(0))
    dot = (n * rad).sum(0)
    out["coarse"] = {
        "shape": list(c.shape), "attrs": dict(coarse.attrs),
        "valid_fraction": {"0_nodata": vhc[0] / val.size, "1_supervise": vhc[1] / val.size, "2_ignore": vhc[2] / val.size},
        "mean_phase_unit_norm": float(np.sqrt(s * s + cs * cs)[sup].mean()),
        "mean_normal_norm": float(nrm[sup].mean()),
        "frac_n_dot_r_pos": float((dot[sup] > 0).mean()),
        "median_abs_n_dot_r": float(np.median(np.abs(dot[sup]))),
        "density_wraps_per_l2vox": {"median": float(np.median(c[2][sup]) / 1000.0), "p95": float(np.percentile(c[2][sup], 95) / 1000.0)},
        "conf_mean": float(c[6][sup].mean()),
        "sin_mean": float(s[sup].mean()), "cos_mean": float(cs[sup].mean()),
    }
    print(json.dumps(out, indent=2, default=float))
    with open(os.path.join(ldir, "sanity.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
