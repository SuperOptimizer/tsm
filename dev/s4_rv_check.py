"""Registration spot-check: one CT slice vs the resampled rectoverso labels.

    uv run python dev/s4_rv_check.py configs/s4.json --rv ~/tsm-output/s4/rectoverso \
        --zlocal 128 --y 1024 --x 2048 --side 1024 --out ~/tsm-output/s4_rv_check.png

Reads the CT slice straight from the volume (no cache), overlays the recto/verso
bands, and sweeps a (dy, dx) shift of the label mask reporting mean CT under it.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsm.config import load_config  # noqa: E402


def open_ct(url, level):
    if url.startswith("s3://"):
        import s3fs
        store = s3fs.S3Map(root=url.rstrip("/"), s3=s3fs.S3FileSystem(anon=True), check=False)
    else:
        import fsspec
        store = fsspec.get_mapper(url.rstrip("/"))
    return zarr.open_group(store=store, mode="r")[str(level)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--rv", required=True)
    ap.add_argument("--zlocal", type=int, default=128)
    ap.add_argument("--y", type=int, default=1024)
    ap.add_argument("--x", type=int, default=2048)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", type=int, default=16)
    a = ap.parse_args()

    cfg = load_config(a.config)
    z0, y0, x0 = cfg.region.start_zyx
    ct_arr = open_ct(cfg.volume.alt_url or cfg.volume.url, cfg.volume.level)
    m = a.sweep
    zc = z0 + a.zlocal
    ct = np.asarray(ct_arr[zc,
                           y0 + a.y - m: y0 + a.y + a.side + m,
                           x0 + a.x - m: x0 + a.x + a.side + m]).astype(np.float32)
    rv = zarr.open_array(store=str(Path(a.rv) / "rectoverso.zarr"), mode="r")
    lab = np.asarray(rv[0, a.zlocal, a.y:a.y + a.side, a.x:a.x + a.side])
    hz = np.asarray(rv[1, a.zlocal, a.y:a.y + a.side, a.x:a.x + a.side])
    mask = lab > 0
    print(f"[chk] z_abs={zc} y={y0 + a.y} x={x0 + a.x} side={a.side} "
          f"label frac={mask.mean():.4f} ct mean={ct.mean():.1f}")

    best, rows = None, []
    for dy in range(-m, m + 1, 2):
        row = []
        for dx in range(-m, m + 1, 2):
            sub = ct[m + dy:m + dy + a.side, m + dx:m + dx + a.side]
            v = float(sub[mask].mean())
            row.append(round(v, 2))
            if best is None or v > best[0]:
                best = (v, dy, dx)
        rows.append(row)
    base = float(ct[m:m + a.side, m:m + a.side].mean())
    v0 = float(ct[m:m + a.side, m:m + a.side][mask].mean())
    print(f"[chk] mean CT: whole crop {base:.2f} | under mask at (0,0) {v0:.2f} | "
          f"best {best[0]:.2f} at dy={best[1]} dx={best[2]}")
    json.dump({"z_abs": int(zc), "y": int(y0 + a.y), "x": int(x0 + a.x), "side": a.side,
               "label_fraction": float(mask.mean()), "ct_mean_crop": base,
               "ct_mean_under_mask": v0, "best": {"value": best[0], "dy": best[1], "dx": best[2]},
               "sweep_step": 2, "sweep_range": m,
               "grid": rows}, open(str(a.out) + ".json", "w"), indent=1)

    from PIL import Image, ImageDraw
    g = np.clip(ct[m:m + a.side, m:m + a.side] * 1.6, 0, 255).astype(np.uint8)
    base_rgb = np.stack([g] * 3, -1)
    o = base_rgb.astype(np.float32).copy()
    for k, c in {1: (255, 40, 40), 2: (60, 120, 255), 3: (255, 240, 40)}.items():
        sel = lab == k
        o[sel] = 0.45 * o[sel] + 0.55 * np.asarray(c, np.float32)
    o2 = base_rgb.astype(np.float32).copy()
    for k, c in {1: (40, 220, 80), 2: (230, 60, 230), 3: (140, 140, 140)}.items():
        sel = hz == k
        o2[sel] = 0.45 * o2[sel] + 0.55 * np.asarray(c, np.float32)
    ims = []
    for t, arr in (("CT", base_rgb), ("rectoverso (recto red / verso blue / both yellow)",
                                      o.astype(np.uint8)),
                   ("hzvt (hz green / vt magenta)", o2.astype(np.uint8))):
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 7 * len(t) + 8, 16], fill=(0, 0, 0))
        d.text((4, 3), t, fill=(255, 255, 0))
        ims.append(np.asarray(im))
    Image.fromarray(np.concatenate(ims, 1)).save(a.out)
    print("[chk] wrote", a.out)


if __name__ == "__main__":
    main()
