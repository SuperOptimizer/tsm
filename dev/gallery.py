"""Render a review gallery of teacher outputs for a small region.

usage: uv run python dev/gallery.py <out_dir> [n_slices]
Writes <out_dir>/gallery/*.png: per z-slice panels (CT, recto, m7, m7@9.6um, ink,
lasagna cos, lasagna normal RGB, overlay of recto medial + m7 on CT) and one y-z cut.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, zarr
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsm.labels import medial_surface, decode_lasagna_normal  # noqa: E402

out = Path(sys.argv[1]); n_sl = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T = out / "teachers"; G = out / "gallery"; G.mkdir(exist_ok=True)

def load(name):
    p = T / f"{name}.zarr"
    return np.asarray(zarr.open(str(p), mode="r")) if p.exists() else None

recto, m7, ink, las, m7l2 = (load(n) for n in ("recto", "m7", "ink", "lasagna", "m7_l2"))
attrs = zarr.open(str(T / "recto.zarr"), mode="r").attrs.asdict()
origin = attrs.get("origin_zyx"); Z, Y, X = recto.shape[1:]
cfg = json.load(open(out.parent / "v1" / "teachers" / "recto.summary.json")) if False else None
# CT from S3 (fallback url)
from tsm.volume import VolumeReader  # noqa: E402
url = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
rd = VolumeReader(url, level=0, voxel_um=2.4)
z0, y0, x0 = origin
ct = rd.read(z0, z0 + Z, y0, y0 + Y, x0, x0 + X)

def up4(a):  # level-2 (C,z,y,x) -> level-0 nearest, cropped to (Z,Y,X)
    return np.repeat(np.repeat(np.repeat(a, 4, 1), 4, 2), 4, 3)[:, :Z, :Y, :X]

las_up = up4(las) if las is not None else None
m7l2_up = up4(m7l2) if m7l2 is not None else None
normal = decode_lasagna_normal(las.astype(np.float32) / 255.0) if las is not None else None  # (3,z,y,x) xyz?
normal_up = up4(normal) if normal is not None else None

def g(a):  # uint8 gray -> RGB image array
    return np.stack([a] * 3, -1)
def rgb_normal(n):  # (3,H,W) in [-1,1] -> uint8 RGB
    return np.clip((np.abs(n) * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
def overlay(ctsl, layers):  # layers: list of (mask, color)
    im = g(np.clip(ctsl.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)).copy()
    for m, col in layers:
        im[m] = (0.35 * im[m] + 0.65 * np.array(col)).astype(np.uint8)
    return im
def label(im, text):
    pil = Image.fromarray(im); d = ImageDraw.Draw(pil); d.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0)); d.text((4, 2), text, fill=(255, 255, 0)); return np.asarray(pil)

# medial surfaces computed once on the whole (small) volume
rm = medial_surface(recto[0] >= 128); mm = medial_surface(m7[0] >= 128) if m7 is not None else None
zs = [int(Z * (i + 1) / (n_sl + 1)) for i in range(n_sl)]
for z in zs:
    c = ct[z]
    panels = [label(g(np.clip(c * 1.4, 0, 255).astype(np.uint8)), f"CT z={z0 + z}"),
              label(g(recto[0, z]), "recto 2.4um"), label(g(m7[0, z]), "m7 2.4um") if m7 is not None else None,
              label(g(m7l2_up[0, z]), "m7 9.6um (x4)") if m7l2_up is not None else None,
              label(g(ink[0, z]), "ink") if ink is not None else None,
              label(g(las_up[0, z]), "lasagna cos") if las_up is not None else None,
              label(rgb_normal(normal_up[:, z]), "lasagna |normal| rgb") if normal_up is not None else None,
              label(overlay(c, [(recto[0, z] >= 128, (120, 0, 0)), (rm[z], (255, 60, 60)), (mm[z] if mm is not None else np.zeros_like(rm[z]), (0, 255, 0))]), "CT + recto band/medial(red) + m7 medial(green)")]
    panels = [p for p in panels if p is not None]
    row1 = np.concatenate(panels[:4], 1); row2 = np.concatenate(panels[4:8], 1)
    if row2.shape[1] < row1.shape[1]:
        row2 = np.pad(row2, ((0, 0), (0, row1.shape[1] - row2.shape[1]), (0, 0)))
    Image.fromarray(np.concatenate([row1, row2], 0)).save(G / f"z{z0 + z}.png")
# y-z cut at mid x
xm = X // 2
cut = lambda a: a[:, :, xm]  # (Z,Y)
panels = [label(g(np.clip(cut(ct) * 1.4, 0, 255).astype(np.uint8)), f"CT yz cut x={x0 + xm}"), label(g(cut(recto[0])), "recto"),
          label(g(cut(m7[0])), "m7") if m7 is not None else None, label(g(cut(ink[0])), "ink") if ink is not None else None,
          label(overlay(cut(ct), [(cut(rm), (255, 60, 60)), (cut(mm) if mm is not None else np.zeros_like(cut(rm)), (0, 255, 0))]), "CT + medials")]
panels = [p for p in panels if p is not None]
Image.fromarray(np.concatenate(panels, 1)).save(G / "yz_cut.png")
# thin-surface stats
from tsm.labels import surface_stats  # noqa: E402
stats = {"recto_medial": surface_stats(rm, 100), "m7_medial": surface_stats(mm, 100) if mm is not None else None,
         "recto_fg": float((recto[0] >= 128).mean()), "m7_fg": float((m7[0] >= 128).mean()) if m7 is not None else None,
         "ink_fg": float((ink[0] >= 128).mean()) if ink is not None else None}
json.dump(stats, open(G / "stats.json", "w"), indent=1, default=float)
print("gallery written to", G, "\n", json.dumps(stats, indent=1, default=float)[:1500])
