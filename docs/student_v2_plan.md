# Student v2 plan: faster and more accurate than the teachers

Status: proposal (2026-09-02), not yet scheduled. Applies to the canonical 2.4 µm
model (~20M params, heads: surface [sdf, valid], ink, winding [sin, cos, density,
normal, conf]) and, by distillation, to the 4.8 / 9.6 µm models.

## Goals
- Inference 10–20× faster per voxel than a teacher (recto: ~3 windows/s of 256³ on
  the RTX 5080 in TensorRT FP16).
- Surface quality better than recto/m7 on medial-surface metrics and the anti-blob
  morphology gate, with sub-voxel surface localisation from the SDF.
- Everything truly 3D, no post-processing, no third-party augmentation libraries.

## A. Architecture changes (do before long training)
1. **Half-resolution body.** Stride-2 stem (2.4 → 4.8 µm) for the whole
   encoder–decoder; one narrow full-resolution block + 1×1×1 heads restore
   1-voxel SDF precision. ~8× fewer voxels in the body → expected 3–5× speed.
2. **Foldable graph.** BatchNorm3d instead of GroupNorm (fold into conv at
   export), LeakyReLU/ReLU instead of SiLU. Result: pure conv+act graph that
   TensorRT/INT8 fuse cleanly. Keep channels_last, widths multiples of 64 in
   the deep stages, narrow full-res layers.
3. **Coarse context branch.** Second input = 9.6 µm crop covering 4× the field
   of view around the fine crop (1/64 the voxels), encoded by a small branch and
   injected at the bottleneck (concat + 1×1×1). Gives whole-winding context the
   teachers never see (sheet identity, gaps, face orientation).
4. **Depthwise-separable convs only in the wide stages** (measure; skip if the
   gain is < 15%).
5. **Big inference windows, minimal overlap.** 384³+ windows in FP16; halo =
   receptive-field radius only (~1.3 windows/voxel instead of 8). Validate on
   medial-surface agreement like the teacher tiling check.
6. **INT8 engine + air skipping** via the same TensorRT path as the teachers.

## B. Label quality
1. Teacher labels with heavy TTA (flip8 × in-plane rot4 = 32 passes for recto and
   lasagna; flip8 for ink and m7@l2), averaged in logit space, vector channels
   transformed back correctly. Compare against no-TTA labels on medial agreement.
2. Fused multi-teacher targets with agreement → confidence, disagreement → ignore
   (already in `tsm labels`); m7@l2 as a coarse sheet-existence prior.
3. Sub-voxel SDF targets: keep the float EDT (do not round to the voxel grid
   before encoding); add a sharpness term near the zero set.

## C. Training recipe
1. Losses as implemented + feature distillation from recto/ink encoders
   (`feat_distill`, stages /8 and /16) + cross-field consistency
   (∇sdf ∥ winding normal; zero set on phase maxima) + socratic-style
   preservation terms when fine-tuning from a previous checkpoint (KL to the
   previous model where labels are silent; one-sided positive preservation;
   weight trust region).
2. Augmentation v2 (ours, GPU, truly 3D, vector-aware) with the ablation harness
   (`dev/ablate.py`) to decide the preset per head.
3. Long training, small batch (Kaggle finding: keep improving to thousands of
   epochs); EMA; cosine schedule; checkpoints kept as a ladder, not just last.
4. **Self-distillation rounds:** student re-labels with TTA → keep voxels where
   student and teachers agree → retrain from the previous checkpoint under a
   trust region → accept the round only if the anti-blob/morphology gate and
   held-out medial metrics improve. 2–3 rounds.
5. **Ensemble then distill:** 3 seeds → averaged labels → one model.
6. **More scrolls** once Paris 4 is solid (0139, 1667, 0343P, 0500P2, MANBp …);
   holdout by scroll, not by crop.

## D. Evaluation (release gate, independent of training loss)
- Surface: SDF MAE in the ±clip band, zero-crossing Dice, medial-surface
  distance (median/p90/frac>3 vox) vs teacher medial and vs traced tifxyz
  meshes where available, thickness/spacing distributions vs physical
  expectation, anti-blob gate (interior fraction, max thickness regressions),
  component counts.
- Ink: AUPRC vs soft teacher; winding: circular phase error, density MAE,
  normal angle error, conf AUROC.
- Speed: voxels/s and VRAM for torch bf16, TRT FP16, TRT INT8 on RTX 4000 and
  5000 class GPUs.

## E. Distillation to 4.8 / 9.6 µm
- Run the canonical model (with TTA) over the slab, average-pool its outputs to
  the target pitch (sin/cos and normals renormalised, density × factor, SDF in
  target voxels with the clip rescaled), train the coarse model with the same
  losses + feature distillation from the canonical model's encoder.
- The 9.6 µm model is what whole-scroll runs use; export lasagna/spiral formats
  from it via `tsm export`.

## Order of work
1. A1–A3 + B1 (architecture + TTA labels) → retrain baseline → compare.
2. C4 self-distillation round 1 → gate.
3. A5–A6 + INT8 → speed numbers.
4. C5 ensemble-distill, then E.

## F. DINO / dinovol (researched 2026-09-02; reimplement, do not import)
Released dinovol = 3D ViT (patch 8, embed 864, depth 24, heads 16, mixed RoPE,
4 register tokens, SwiGLU with mid-LayerNorm, no abs pos-embed; weights at
~/.cache/tsm-models/dinovol.pt, full hyperparameters in the .tsm-model
model.json). Runs only at 128³ windows (full attention; 256³ OOMs on 24 GB).
1. **DINO feature distillation (cached offline).** Pure-torch forward-only ViT
   reimplementation → run once over the slab at 128³ (token-level overlap
   stitching), L2-normalise tokens, PCA 864→64, store fp16 at the /8 grid
   (~2.4 GB per slab). Third `feat_distill` term: student /8 stage → 1×1×1
   projector → cosine loss vs cached tokens. No DINO forward in the train loop.
2. **Ink-likeness map**: cosine similarity of cached tokens to the expert
   reference embedding (~/.cache/tsm-production/models/auxiliary/*/avg_ref_embedding.npy)
   → agreement signal vs the ink teacher in `tsm labels` (confidence / ignore).
3. **Mean-teacher consistency on unlabeled crops**: EMA student as teacher,
   two Augment-v2 views of the same crop, cosine loss between bottleneck maps
   after undoing the transforms; optional KoLeo spread term (30 lines) against
   collapse. Uses unlabeled slabs / other scrolls without labels.
4. Augmentations to add from dinovol's CT pipeline: slice-illumination
   inhomogeneity, local gamma (low-res simulation already in v2).
Not worth it: retraining a ViT; a DINO-token decoder as a 4th teacher (diagnostic only).

### F.1 Implemented 2026-09-03 (`src/tsm/dino.py`, `tsm dino`)
Pure-torch `DinoVolViT` + `load_dinovol` (strict key parity with the 463-tensor
inventory, 215,859,168 params) and the `tsm dino` cache stage; `feat_distill`
accepts `"dino"` as a teacher (cached tokens, no forward in the train loop).
**Parity: bit-exact** (max|Δ| = 0 on patch/cls/reg tokens and on the RoPE
sin/cos) vs villa's `dinovol_2.model.dinov2_eva.Eva` at 48³ on CPU with the
released weights — `dev/parity_dino.py` (needs the villa clone + a `--target`
install of timm/einops/torchvision; nothing is imported at runtime).
Store layout, the per-window `robust` normalisation and the signed-permutation
augmentation rule are documented in `docs/label_store.md` ("DINO token cache").
Still open from F: the ink-likeness agreement signal in `tsm labels` (F.2 uses
`ink_likeness.zarr`, which the stage already writes), mean-teacher consistency
(F.3) and the two CT augmentations (F.4).

## G. Augmentation / TTA literature findings (2026-09-02 survey)
- **Function matching** (Beyer et al., CVPR 2022): teacher and student must see the
  identical augmented view. Spatial augs already satisfy this (labels transformed
  with the crop); for intensity augs add an ablation with the teacher run online on
  the augmented crop (surface head first).
- **BatchNorm + TTA conflict** (arXiv:2604.09697): geometric TTA hurt in 11/12 BN
  models. Open decision vs A2 (BN for export folding): keep GroupNorm unless folded
  frozen-BN + TTA is verified to still help; always include the identity view.
- **Mixup only in regression-safe form** (C-Mixup, NeurIPS 2022): label-similarity
  gated pairs; never naive mixup on SDF/normal/phase.
- **z-translation** augmentation: repeatedly cited as the most effective single aug
  by Vesuvius practitioners; add (phase targets move with the crop, no change).
- **TTA**: un-rotate vectors/angles before averaging (correctness); gains plateau
  early (SegTTA: well before 40 views) → 8–24 views; learned/weighted aggregation
  (Shanmugam 2021, BayTTA) beats uniform averaging cheaply.
- **Rotation range is a hyperparameter** (Diaz et al. 2024: <20° sometimes hurt);
  ablate uniform SO(3) vs ±30° vs octahedral-only; equivariant nets (escnn) only
  if augmentation fails to remove measured bias.
- **Physics augs**: generic noise recovers ~25% of the ring-artifact robustness gap
  that targeted simulation recovers (arXiv:2510.06584); a micro-CT artifact
  simulator (rings, beam hardening, Poisson projection noise) is a gap worth
  filling later; sheet-aware elastic deformation likewise novel.
- Copy-paste (TumorCP/CarveMix) for ink only with seam-harmonised blending; risky
  for thin sheets.

## H. Two-face surface head (user proposal, 2026-09-02)
Motivation: the recto teacher's recto/verso choice is orientation-defined
(measured: a 180° in-plane rotation moves its band to the other face). Predict
BOTH faces and decide names afterwards geometrically.
- Surface head → 3 channels: `sdf_face_in` (face toward the umbilicus),
  `sdf_face_out`, `valid`. Ordering by the outward normal / axis, never by the
  teacher. Recto = face_in for these scrolls (writing faces the axis); thickness
  = gap between the zero sets; sheet interior = between them.
- Labels (`tsm labels --faces`): sheet body from smoothed CT (papyrus vs air),
  per sheet component; recto face = component boundary adjacent to the recto
  teacher band; verso = opposite boundary of the same component; validity 2
  where sheets touch (no air gap) or the component is a delaminated layer
  (thickness far from the 40–60 vox prior). m7@2.4um lines as a check.
- TTA-averaged teacher bands feed the sheet-body/recto-side decision only.
- v1 baseline (medial-of-band SDF) is kept for comparison.

### H.1 Teacher-derived faces via m7 (measured 2026-09-02)
m7's face choice follows a z-axis convention: z-fixed transforms (flip8_rot4)
keep the inward face; z-swapping transforms yield the outward face. So the
recto and verso bands come from ONE teacher: avg over z-fixed set → recto
band, avg over z-swapping set → verso band (validated on the small slab:
medial 6.4 vox from face_out vs 19.6 from face_in). Use these as the
band-adjacency source for faces mode instead of the recto teacher, and never
average m7 over all 48. Results/tables: ~/tsm-output/tta_side.

### Measured 2026-09-03: v1 student inference speed
`tsm infer` (torch bf16, 128³ windows, step 64, out_tile 192): ~5.3 windows/s
≈ 11 Mvox/s — SLOWER per voxel than the recto teacher in TensorRT (≈54 Mvox/s)
because of 8× window overlap and no engine. A1/A5/A6 (half-res body, big
windows with receptive-field halo, TRT engine) are therefore required, not
optional, for the student to be the fast model.

### Implemented 2026-09-03: A1 / A5 / A6 (CPU work; GPU numbers pending)
- **A1 half-resolution body** — `TSMNet(body_stride=2, fullres_width=32, norm="group"|"batch")`
  (`extra.train.body_stride` / `fullres_width` / `norm`). Stride-2 stem (the stride sits on the
  first stem residual block's k3 conv; space-to-depth + 1×1 is interchangeable), the existing
  encoder–decoder at 4.8 µm, then one residual block at `fullres_width` on
  `[network input, trilinearly upsampled decoder output]` before the 1×1×1 heads, so the SDF keeps
  1-voxel precision. Deep supervision level 0 is that block, level l ≥ 1 the decoder level l−1.
  Measured (in_ch 5, canonical widths, 128³): **706.4 → 216.5 GMAC (3.26×)**, params 19.97 M → 20.03 M.
  The full-resolution block is 127.5 of those 216.5 GMAC — `fullres_width=16` would cut ~60 GMAC more.
  GroupNorm stays the default (the BN/TTA conflict, section G); `norm="batch"` exists to ablate.
- **A5 receptive field** — `TSMNet.receptive_field_radius()` (analytic support-interval walk of the
  kernel/stride schedule; `method="empirical"` back-propagates from single output voxels with the
  norms bypassed and matches it exactly at `body_stride=1`, and to within 1 at `body_stride=2`).
  **Radius = 122 voxels (body_stride 1), 248 (body_stride 2)** — the stride-2 stem doubles every jump.
  `extra.infer.halo = "rf"` sets `halo = radius + 2`, `step = patch − 2·halo` and switches the
  blend to uniform core weights (`WindowSpec.weight = "uniform"`, `sliding.uniform_core_weight`).
  **Consequence: at these radii A5 does not pay off yet.** It needs `patch > 2·halo` (≥ 256) to run
  at all and `patch > 4·halo` (≥ 512) to beat 50 % overlap: 384³ → 22.5 windows/voxel, 512³ → 7.3,
  768³ → 3.2, versus 8.0 for 50 % overlap. To make A5 the promised ~1.3 windows/voxel the *radius*
  has to come down (fewer stages / fewer blocks per stage / smaller kernels) or the halo has to be
  set from the **effective** receptive field (`receptive_field_radius("empirical", eps=…)`, or just
  a measured constant) — the plumbing takes any integer `halo` with `weight: "uniform"`.
- **A6 TensorRT student** — `tsm.trt.TRTStudent` / `export_student_onnx`, `extra.infer.backend =
  "trt"` (fp16; bf16 has no TensorRT 11 tactic for the decoder's ConvTranspose). N-channel input
  (5 with the radial channels, computed torch-side and concatenated before the engine call), the
  raw 11/12-channel head tensor out, split back into `{surface, ink, winding}` so `StudentNet`,
  the TTA and the blending are untouched. Engine cache key as for the teachers, with the name
  `student_i<in>o<out>_w<widths>_bs<body_stride>`. Verified on the CPU: ONNX export at 64³ for
  body_stride 1 and 2 and for the two-face head passes `onnx.checker`, upsampling is emitted as
  `Resize`-by-*scales* so `_redim_onnx` can re-dim the graph to the build patch. The engine build
  and every timing are GPU-only: `dev/bench_student.py` (torch bf16 vs TRT fp16 × patch
  128/192/256/320 × 50 %-overlap vs rf-halo, Mvox/s, VRAM, SDF MAE / zero-crossing Dice vs the
  50 %-overlap torch reference, plus a randomly initialised body_stride=2 net for speed only).

### F.2 Cached conv-teacher features — implemented 2026-09-03 (`src/tsm/feats.py`, `tsm feats`)
Cache recto/ink encoder features (/8: 256 ch, /16: 320 ch) once per slab like
the DINO tokens (fp16, PCA→64; ~2.4 GB/teacher/stage) via a `tsm feats` stage;
feat_distill reads them per crop. Removes the live-teacher cost (currently
+40 %/step, applied every 2nd step). Caveat: cached = clean-volume features
(no function matching under intensity aug; flips/rot90 only). Ablation on the
desk: cached vs live, all else fixed; optionally cache a few fixed orientations.

`tsm feats <config>` writes `<out_dir>/feats/<teacher>_s<stage>.zarr` (+ one
`pca_<teacher>_s<stage>.npz`), resumable per block; `feat_distill.source`
(`"live"` | `"cached"`, or `sources: {teacher: ...}` per teacher) picks where the
target comes from. Measured on the 256×6144² slab: 1024 windows per teacher,
2.25 GiB per /8 stage and 0.28 GiB per /16 stage, 5.06 GiB in total for
recto+ink at pca_dim 64. Window tiling uses a 256³ window with a 32-voxel
border discarded on every face (= 4 feature cells at /8) and the cores stitched
without blending — the encoder's receptive field at /8 is hundreds of voxels, so
a ramp blend of contaminated values (what `sliding.predict_box` does for
decoder outputs) is wrong here. Measured (`tests/test_feats.py`): with the norm
layer removed the core of every window matches a single whole-region forward to
~2 % of the feature scale while the discarded ring is >10× worse. **The residual
seam is InstanceNorm3d, not the halo**: the published ResEnc encoders normalise
over the whole window, so a window's features carry a per-channel affine offset
no border can remove; the cached grid is therefore the teacher's *own
inference-time* feature (`tsm teacher` runs the same 256³ windows), while the
live path currently normalises over the 128³ training crop. `configs/ablate_feats.json`
(3000 steps, seed 0, y-band holdout) measures whether that matters:
`live_every1 / live_every2 / cached_every1 / cached_plus_dino / none`.
