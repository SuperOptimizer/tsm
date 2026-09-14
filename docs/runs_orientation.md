# Orientation-free surface runs (held-out Paris 4, 30k steps unless noted)

Unordered = Dice@2 of predicted sides vs recto ∪ verso human bands; body topology per 100 reference bodies (18).
Fibre: class accuracy vs human hz/vt bands; overlap = fraction of firing voxels where vt and hz both fire.

| run | recipe | unordered Dice | body merges | body missed | body spurious | ink AUPRC | fibre acc | fibre overlap | verdict |
|---|---|---|---|---|---|---|---|---|---|
| faces30k (eval_long30k) | two-face, radial only, class fibre | 0.636 | 5.6 | 22.2 | 0.0 | 0.686 | 0.57 | 0.90 | baseline |
| faces gc1 (eval_topo30k_gc1) | + gap 1.0 / clDice 1.0 | 0.669 | 5.6 | 11.1 | 0.0 | 0.668 | 0.65 | 0.90 | topology gain, Dice +0.03 |
| faces100k | 100k steps | 0.661 | 11.1 | 27.8 | 0.0 | 0.756 | 0.62 | 0.90 | overfits topology |
| body30k | signed body SDF | 0.507 | 5.6 | 66.7 | 5.6 | 0.608 | 0.50 | 0.95 | REJECT (range compression) |
| body30k_bgf | body + axis tangent + direction fibre | 0.516 | 0.0 | 72.2 | 5.6 | 0.635 | 0.54 | 0.12 | REJECT surface; fibre exclusivity fixed |
| faces30k_gf | two-face + axis tangent + direction fibre | 0.639 | 5.6 | 16.7 | 0.0 | 0.633 | 0.67 | 0.03 | surface neutral/+topology; fibre near gate |
| sides30k (skin extraction) | d_face + body mask + axis + direction fibre | 0.36 | 0.0 | 61.1 | 5.6 | 0.608 | 0.56 | 0.04 | REJECT as-is; body soft in crumples, d_face compressed |

Notes 2026-09-13: sides30k body>0.3 threshold raises band recall 0.20→0.32 (skin), 0.5 is the eval default; body components ≥2000 vox: 33 vs 18 reference.
Pending: sides30k_aux (gap+clDice), sides30k_gap (gap class, dilate, border, eikonal, CT gate), sides30k_lsd; Scroll 4 checks for all.
