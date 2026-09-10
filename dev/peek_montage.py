"""Quick montage: student pred.zarr vs teacher recto/lasagna on a sub-box of the eval region.
usage: uv run python dev/peek_montage.py <peek_out_dir> <eval_out_dir> <z_local>"""
import sys, json, numpy as np, zarr
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsm.labels import medial_surface
from tsm.infer import decode_pred
peek, ev, zl = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
pz = zarr.open(str(peek / "student" / "pred.zarr"), mode="r")
attrs = pz.attrs.asdict(); o = attrs.get("origin_zyx"); chans = attrs.get("channels")
print("pred", pz.shape, o, chans)
tz = zarr.open(str(ev / "teachers" / "recto.zarr"), mode="r"); to = tz.attrs.asdict()["origin_zyx"]
lz = zarr.open(str(ev / "teachers" / "lasagna.zarr"), mode="r"); lo = lz.attrs.asdict()["origin_zyx"]
Z, Y, X = pz.shape[1:]
dz, dy, dx = [o[i] - to[i] for i in range(3)]
recto = np.asarray(tz[0, dz:dz + Z, dy:dy + Y, dx:dx + X])
cz, cy, cx = [o[i] // 4 - lo[i] for i in range(3)]
las_cos = np.asarray(lz[0, cz:cz + Z // 4, cy:cy + Y // 4, cx:cx + X // 4])
las_cos = np.repeat(np.repeat(np.repeat(las_cos, 4, 0), 4, 1), 4, 2)
from tsm.cli import open_ct  # noqa
from tsm.config import load_config
cfg = load_config(str(peek.parent / "peek_cfg.json")) if (peek.parent / "peek_cfg.json").exists() else None
ct = None
try:
    import subprocess
    cfgp = "/home/forrest/tsm/configs/peek_v1b.json"
    cfg = load_config(cfgp); rd = open_ct(cfg); ct = rd.read(o[0], o[0] + Z, o[1], o[1] + Y, o[2], o[2] + X)
except Exception as e:
    print("ct read failed", e); ct = np.zeros((Z, Y, X), np.uint8)
ci = {n: i for i, n in enumerate(chans)}
sdf_u8 = np.asarray(pz[ci["sdf"], zl]); valid = np.asarray(pz[ci["valid"], zl]); cos_u8 = np.asarray(pz[ci["cos"], zl])
nx, ny, nz = (np.asarray(pz[ci[k], zl]).astype(np.float32) / 127.5 - 1 for k in ("nx", "ny", "nz"))
surf1 = np.asarray(pz[ci["surface1"], zl]) > 0 if "surface1" in ci else None
sdf = (sdf_u8.astype(np.float32) - 128) * 20 / 127
rm = medial_surface(recto >= 128)[zl]
def g(a): return np.stack([a] * 3, -1).astype(np.uint8)
ctp = np.clip(ct[zl].astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
# sdf colormap: blue negative, red positive, white near zero
s = np.clip(sdf / 20, -1, 1)
sdf_rgb = np.stack([np.clip(255 * (1 + np.minimum(s, 0)), 0, 255), np.clip(255 * (1 - np.abs(s)), 0, 255), np.clip(255 * (1 - np.maximum(s, 0)), 0, 255)], -1).astype(np.uint8)
sdf_rgb[np.abs(sdf) < 0.6] = (255, 255, 0)
ov = g(ctp).copy()
ov[rm] = (0, 255, 0)
if surf1 is not None: ov[surf1] = (255, 0, 0)
nrm = np.clip(np.abs(np.stack([nx, ny, nz], -1)) * 255, 0, 255).astype(np.uint8)
panels = [("CT", g(ctp)), ("recto teacher", g(recto[zl])), ("student sdf (yellow=0)", sdf_rgb), ("student surface(red) vs recto medial(green)", ov),
          ("lasagna cos", g(las_cos[zl])), ("student cos", g(cos_u8)), ("student |normal| rgb", nrm), ("student valid", g(valid))]
ims = []
for t, a in panels:
    im = Image.fromarray(a); d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 * len(t) + 6, 14], fill=(0, 0, 0)); d.text((3, 1), t, fill=(255, 255, 0)); ims.append(np.asarray(im))
row1 = np.concatenate(ims[:4], 1); row2 = np.concatenate(ims[4:], 1)
out = peek / f"peek_z{o[0] + zl}.png"; Image.fromarray(np.concatenate([row1, row2], 0)).save(out); print("wrote", out)
