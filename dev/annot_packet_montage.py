"""Mid-slice montage of one annot packet: CT | CT+faces | faces_source | rv_class."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

pdir = sys.argv[1]
out = sys.argv[2]
meta = json.load(open(os.path.join(pdir, "meta.json")))
S = [int(v) for v in meta["size_zyx"]]
rd = lambda n: np.fromfile(os.path.join(pdir, f"{n}.u8"), np.uint8).reshape(S)
z = S[0] // 2
ct = rd("ct")[z]
fin, fout, ig = rd("faces_in")[z], rd("faces_out")[z], rd("ignore")[z]
src, rvc = rd("source")[z], rd("rv_class")[z]

grey = np.stack([np.clip(ct.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)] * 3, -1)
faces = grey.copy()
faces[ig > 0] = (faces[ig > 0] * 0.45).astype(np.uint8)
faces[fout > 0] = (60, 120, 255)
faces[fin > 0] = (230, 40, 40)
SRC = {0: (20, 20, 20), 1: (60, 200, 90), 2: (240, 170, 40), 3: (230, 60, 230)}
RVC = {0: (20, 20, 20), 1: (230, 40, 40), 2: (60, 120, 255), 3: (240, 230, 60)}
src_rgb = np.zeros(ct.shape + (3,), np.uint8)
rvc_rgb = np.zeros(ct.shape + (3,), np.uint8)
for k, c in SRC.items():
    src_rgb[src == k] = c
for k, c in RVC.items():
    rvc_rgb[rvc == k] = c

panels = [
    (f"CT z={meta['origin_zyx'][0] + z}", grey),
    ("faces: IN red / OUT blue / ignore dim", faces),
    ("faces_source: 0 blk / 1 rv grn / 2 ct org / 3 human mag", src_rgb),
    ("rv_class: 1 recto red / 2 verso blue / 3 contact yel", rvc_rgb),
]
ims = []
for t, im in panels:
    pil = Image.fromarray(im)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 7 * len(t) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), t, fill=(255, 255, 0))
    ims.append(np.asarray(pil))
Image.fromarray(np.concatenate(ims, 1)).save(out)
print("wrote", out, "from", pdir, "score", meta["score"])
for n, a in (("faces_in", fin), ("faces_out", fout), ("ignore", ig)):
    print(f"  {n} slice frac {float((a > 0).mean()):.4f}")
print("  source fracs", {int(k): round(float((src == k).mean()), 4) for k in (0, 1, 2, 3)})
print("  rv_class fracs", {int(k): round(float((rvc == k).mean()), 4) for k in (0, 1, 2, 3)})
