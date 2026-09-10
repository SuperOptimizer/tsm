"""Build a two-face label store (``fine.zarr`` with sdf_in / sdf_out / faces_valid / thickness)
for one box of the small-slab region, so ``dev/tta_side.py --faces-store`` has ground truth.

``~/tsm-output/small_faces`` only ever kept the summary JSON of a 32-slice slab (no zarr), so
this rebuilds the same labels with the same recipe (``tsm.labels.build_face_labels``,
body_threshold 60, close_radius 6) over a box deep enough for the 256^3 recto box and the
192^3 m7 box, and writes it to ``~/tsm-output/tta_side/faces/fine.zarr`` (never into
``~/tsm-output/small``).

    uv run python dev/tta_side_faces.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tsm.cli import open_ct  # noqa: E402
from tsm.config import load_config  # noqa: E402
from tsm.labels import (  # noqa: E402
    COARSE_CHANNELS,
    DEFAULT_AXIS,
    build_coarse_field,
    build_face_labels,
    fine_channels,
    load_axis,
    radial_field,
    read_box_padded,
    upsample_coarse_normal,
)
from tsm.volume import BrickWriter  # noqa: E402

OUT = os.path.expanduser("~/tsm-output/tta_side")
CFG = "/home/forrest/tsm/configs/paris4_small.json"
CLIP = 20.0
K = 4
# level-0 box (start, size) chosen to contain the recto 256^3 box (34176, 15488, 18816) and the
# m7 192^3 box (34208, 15520, 18848) with a halo.
BOX_START = (34176, 15424, 18752)
BOX_SIZE = (256, 384, 384)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config(CFG)
    z0, y0, x0 = cfg.region.start_zyx
    axis = load_axis(DEFAULT_AXIS)
    store = os.path.join(OUT, "faces", "fine.zarr")
    if os.path.exists(os.path.join(store, "zarr.json")):
        print("exists:", store)
        return 0

    lo = [BOX_START[a] - z0 if a == 0 else BOX_START[a] - (y0 if a == 1 else x0) for a in range(3)]
    lo = [BOX_START[0] - z0, BOX_START[1] - y0, BOX_START[2] - x0]
    hi = [lo[a] + BOX_SIZE[a] for a in range(3)]
    print("region-local box", lo, hi)

    ct = open_ct(cfg, 0, 64).read(BOX_START[0], BOX_START[0] + BOX_SIZE[0],
                                  BOX_START[1], BOX_START[1] + BOX_SIZE[1],
                                  BOX_START[2], BOX_START[2] + BOX_SIZE[2])
    print("ct", ct.shape, float(ct.mean()))

    # ---- coarse field (level 2) over the box footprint, halo 4 --------------
    halo = 4
    clo = [lo[a] // K - halo for a in range(3)]
    chi = [-(-hi[a] // K) + halo for a in range(3)]
    las = zarr.open_array(store=os.path.expanduser("~/tsm-output/small/teachers/lasagna.zarr"), mode="r")
    lb = read_box_padded(las, clo, chi)
    ctl2 = open_ct(cfg, 2, 64).read(z0 // K + clo[0], z0 // K + chi[0], y0 // K + clo[1], y0 // K + chi[1],
                                    x0 // K + clo[2], x0 // K + chi[2])
    radc = radial_field(axis, [z0 // K + clo[0], y0 // K + clo[1], x0 // K + clo[2]], lb.shape[1:], scale=K)
    coarse = build_coarse_field(lb, ctl2, radc, ct_mask_threshold=30.0, ct_sigma=1.0, ct_mask_dilate=2)
    del lb, ctl2, radc
    print("coarse", coarse.shape)

    # coarse array indices are box-local: fine local coords inside it start at (lo - clo*K)
    flo = [lo[a] - clo[a] * K for a in range(3)]
    fhi = [flo[a] + BOX_SIZE[a] for a in range(3)]
    n_up = upsample_coarse_normal(coarse, flo, fhi, scale=K)
    rad = radial_field(axis, BOX_START, BOX_SIZE, scale=1)
    del coarse

    recto = read_box_padded(zarr.open_array(store=os.path.expanduser("~/tsm-output/small/teachers/recto.zarr"),
                                            mode="r"), lo, hi, channels=0)[0]
    st: dict = {}
    si, so, valid, thick = build_face_labels(recto, ct, n_up, rad, CLIP, body_threshold=60.0,
                                             close_radius=6.0, stats=st)
    print("valid fractions", {k: float((valid == k).mean()) for k in (0, 1, 2)})

    ch = fine_channels(True)
    w = BrickWriter(store, ch, BOX_SIZE, chunk=128, origin_zyx=BOX_START, voxel_um=2.4, scale=1.0)
    data = {"sdf_in": si, "sdf_out": so, "faces_valid": valid, "thickness": thick}
    for c, name in enumerate(ch):
        w.write(c, BOX_START[0], BOX_START[1], BOX_START[2],
                data.get(name, np.zeros(BOX_SIZE, np.uint8)))
    meta = {"box_start_zyx": list(BOX_START), "box_size_zyx": list(BOX_SIZE), "clip": CLIP,
            "body_threshold": 60.0, "close_radius": 6.0, "config": CFG,
            "valid_fraction": {str(k): float((valid == k).mean()) for k in (0, 1, 2)},
            "face_in_voxels": st.get("face_in_voxels"), "face_out_voxels": st.get("face_out_voxels"),
            "face_ambiguous_voxels": st.get("face_ambiguous_voxels"),
            "note": "channels sdf/sdf_valid/ink/ink_valid are zero (not built); only the face channels are real"}
    with open(os.path.join(OUT, "faces", "faces.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote", store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
