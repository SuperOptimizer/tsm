"""Montage: CT | current (CT-only) face labels | the merged rectoverso+CT face labels.

    uv run python dev/faces_rv_montage.py configs/paris4_faces_rv_crop.json \
        --old ~/tsm-output/slab_faces_tta/labels/fine.zarr --z 100 --out out.png

Panels share one z slice of the config region: the CT, the face labels of ``--old`` and the
face labels of ``--new`` (default ``<out_dir>/labels/fine.zarr``).  In the label panels the
in-face zero set is red, the out-face zero set blue, ``faces_valid == 2`` (ignore) is dimmed,
and -- where the store carries ``faces_source`` -- a thin green outline marks the region the
upstream rectoverso builder supplied (``faces_source == 1``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.cli import open_ct  # noqa: E402
from tsm.config import load_config  # noqa: E402


def _panel(path, origin, zl, y0, x0, side, base):
    """CT-backed RGB panel of one face-label store, or the plain CT when it is missing."""
    img = base.copy()
    if not path or not os.path.exists(os.path.join(path, "zarr.json")):
        return img, {}
    a = zarr.open_array(store=path, mode="r")
    ch = {n: i for i, n in enumerate(a.attrs["channels"])}
    o = [int(v) for v in a.attrs["origin_zyx"]]
    dz, dy, dx = (origin[0] + zl - o[0], origin[1] + y0 - o[1], origin[2] + x0 - o[2])

    def rd(name):
        return np.asarray(a[ch[name], dz, dy:dy + side, dx:dx + side])

    sin, sout, val = rd("sdf_in"), rd("sdf_out"), rd("faces_valid")
    img[val == 2] = (img[val == 2] * 0.4).astype(np.uint8)
    img[val == 0] = (img[val == 0] * 0.2).astype(np.uint8)
    img[(sin != 0) & (np.abs(sin.astype(np.int16) - 128) <= 3)] = (255, 40, 40)
    img[(sout != 0) & (np.abs(sout.astype(np.int16) - 128) <= 3)] = (60, 120, 255)
    stats = {"valid1": float((val == 1).mean()), "valid2": float((val == 2).mean()),
             "valid0": float((val == 0).mean())}
    if "faces_source" in ch:
        src = rd("faces_source")
        rv = src == 1
        from scipy import ndimage as ndi
        outline = rv ^ ndi.binary_erosion(rv, np.ones((3, 3), bool))
        img[outline] = (60, 255, 120)
        stats.update({"src_rectoverso": float(rv.mean()), "src_ct": float((src == 2).mean()),
                      "src_none": float((src == 0).mean())})
    return img, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--new", default=None, help="default <out_dir>/labels/fine.zarr")
    ap.add_argument("--old", default=None, help="the CT-only faces store to compare against")
    ap.add_argument("--z", type=int, default=100, help="z, local to the config region")
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from PIL import Image, ImageDraw

    cfg = load_config(a.config)
    origin = [int(v) for v in cfg.region.start_zyx]
    new = os.path.expanduser(a.new or os.path.join(cfg.out_dir, "labels", "fine.zarr"))
    old = os.path.expanduser(a.old) if a.old else None

    z, y0, x0, s = a.z, a.y0, a.x0, a.side
    ct = open_ct(cfg, level_shift=0).read(origin[0] + z, origin[0] + z + 1,
                                          origin[1] + y0, origin[1] + y0 + s,
                                          origin[2] + x0, origin[2] + x0 + s)[0]
    base = np.stack([np.clip(ct.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)] * 3, -1)

    panels = [("CT z=%d" % (origin[0] + z), base, {})]
    for tag, p in (("current CT-only faces", old), ("merged rectoverso+CT faces", new)):
        img, st = _panel(p, origin, z, y0, x0, s, base)
        panels.append((f"{tag} (IN red / OUT blue / ignore dim / rectoverso outlined)", img, st))
        print(f"[montage] {tag}: {p} {st}", flush=True)

    ims = []
    for t, im, _ in panels:
        pil = Image.fromarray(im)
        d = ImageDraw.Draw(pil)
        d.rectangle([0, 0, 7 * len(t) + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), t, fill=(255, 255, 0))
        ims.append(np.asarray(pil))
    out = os.path.expanduser(a.out)
    Image.fromarray(np.concatenate(ims, 1)).save(out)
    print(f"[montage] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
