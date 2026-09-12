# TSM label store spec (v1, 2026-09-02)

Two zarr v3 uint8 stores written with `tsm.volume.BrickWriter` (shape (C,Z,Y,X), chunks (1,128,128,128), attrs: channels, voxel_um, origin_zyx, scale). All coordinates are in the voxel grid of that level. `0` in a *_valid channel means "no data / do not supervise", `1` = supervise, `2` = ignore (teacher uncertain).

## `<out_dir>/labels/fine.zarr` (2.4 um, level 0, same region as the teachers)
| ch | name | encoding |
|---|---|---|
| 0 | sdf | signed distance (voxels) to the recto medial surface, clipped to +-CLIP (CLIP=20 vox = 48 um): u8 = round(128 + sdf*127/CLIP); 0 reserved for no-data. Sign: positive on the outward side (away from the axis) of the surface, negative inward. Orientation from the coarse winding normal (upsampled) where valid, else the radial direction from the axis. |
| 1 | sdf_valid | 0 no data (outside region / air where CT mask is 0), 1 supervise, 2 ignore (recto prob in [0.2,0.8] and farther than 2 vox from the medial surface) |
| 2 | ink | ink teacher prob * 255 |
| 3 | ink_valid | 0/1 (1 inside region and CT mask) |

### Fibre-orientation labels (optional, 2026-09-06)

When a `fiber.zarr` teacher store exists next to the others (`extra.labels.teachers_dir`), `tsm labels`
appends three more channels to `fine.zarr`, **after** the two-face block, so a store built without them
(or without faces) is still a prefix of one built with them (`tsm.labels.fine_channels(faces, fiber)`).
No fibre teacher store => the channels are simply absent and nothing else changes.

| ch | name | encoding |
|---|---|---|
| -3 | fiber_vt | fibre teacher softmax channel 1 (**vertical fibre**) * 255, copied through like `ink` |
| -2 | fiber_hz | fibre teacher softmax channel 2 (**horizontal / angular fibre**) * 255, copied through like `ink` |
| -1 | fiber_valid | 0/1 — 1 where the CT data mask holds (the same mask `build_fine_labels` returns for `ink_valid`) and the teacher store covers the voxel |

The teacher is `scrollprize/fiber_ink_4class_selfdistill` (MIT, checkpoint
`p4_4class_ddp8_20260526_step029000.pth`, `ema` weights), registered as `fiber` in `tsm.teachers`:
villa `NetworkFromConfig` 3D UNet (`shared_encoder` + `shared_decoder` + `task_heads.labels`),
256³ patches, **4-way softmax** over `0 background · 1 vertical fibre · 2 horizontal/angular fibre ·
3 ink`, normalisation `percentile_minmax` (per-crop 1st–99th percentile clip rescaled to [0,1] —
the same rule as the `ink` teacher, read from villa `scripts/fiber_5class/dataset.py`).  `tsm teacher`
writes all four probabilities as `<out_dir>/teachers/fiber.zarr` with channels
`fiber_bg, fiber_vt, fiber_hz, fiber_ink` (uint8): the spec has `fg_channel = None`, so the sliding
engine accumulates all four logits and applies the softmax once after blending — the 2-class
logit-difference shortcut does not apply and TTA (`flip8`) averages the four raw logit fields.
Only `fiber_vt` / `fiber_hz` are carried into the label store; `fiber_bg` is redundant and `fiber_ink`
duplicates the dedicated ink teacher.

Recto medial surface = EDT ridge of (recto prob >= 0.5) after dropping 26-components < 100 voxels (tsm.labels.medial_surface). The "surface" for TSM is this medial surface (the recto mesh, see audit); the band's face is NOT used.

### Two-face surface labels (optional, `extra.labels.faces`, 2026-09-02)
`extra.labels.faces.enabled = true` appends four channels to `fine.zarr`; channels 0..3 are
unchanged, so a store built without them still loads (`tsm.labels.fine_channels(faces)`).

| ch | name | encoding |
|---|---|---|
| 4 | sdf_in | signed distance (voxels) to the sheet's **in** face (the face toward the umbilicus), same encoding/sign convention as `sdf` (positive on the outward side, `encode_sdf_u8`, 0 = no data) |
| 5 | sdf_out | the same for the **out** face (away from the axis) |
| 6 | faces_valid | 0 no data (outside the CT mask), 1 supervise, 2 ignore |
| 7 | thickness | local sheet thickness in voxels: `2 * EDT` at the nearest medial voxel of the sheet body, rounded and clipped to 255 (0 = no data) |

`tsm.labels.build_face_labels` (brick-wise with the same halo as `build_fine_labels`):
- **Sheet body (bundle level).**  A "sheet" here is a *bundle* of several thin fibre layers with
  air gaps of a few voxels between them; thresholding alone gives layer-level bodies (that is why
  the first version of this section reported a 11-13 voxel "sheet thickness").  The body is the
  gaussian-smoothed CT (`faces.body_sigma`, 2 vox) > `faces.body_threshold`, **closed by a ball of
  `faces.close_radius` voxels** (`labels.close_ball`, computed as two EDTs: dilation
  `d(x, body) <= r`, then erosion `d(x, complement) > r`; the input is edge-padded by `r` first so
  the box faces are not eaten), then 26-components smaller than `faces.min_component` (100) are
  dropped.  A ball closing bridges every gap narrower than `2r`, so `r` must be below half the
  inter-sheet air gap.  `faces.close_along_normal` swaps the ball for a **line element of
  half-length `r` oriented, per voxel, along `n_out`** (`labels.close_normal_line`): it bridges the
  same intra-bundle gaps but leaves in-plane structure (holes, lateral near-contacts between
  sheets) untouched instead of rounding it off.  It does *not* by itself protect against merging
  two sheets across the inter-sheet gap -- that gap lies along `n_out` too, so `r < gap/2` is
  required either way.  `faces.close_radius = 0` falls back to `faces.closing` (1) iterations of
  the 26-neighbourhood structure.
  `body_threshold = 60` was picked from the Paris 4 ROI at level 0: the smoothed-CT histogram is
  bimodal with the air/gap peak at 40-50 (16.9 M of 33.6 M voxels in the 40-50 bin) and papyrus
  spread over 60-150; the valley is at ~52-58.  60 gives a body fraction of 0.356 and 4 large
  26-components (= the 4 windings crossing the ROI); 55 and 70 give 0.376 / 0.316 and the same
  component structure, so the choice is not delicate.
- **Faces (local march; corrected 2026-09-02).**  `n_out` = the upsampled coarse normal where its
  norm >= 0.5, else the radial direction, evaluated **at the voxel itself** and with the sign
  forced to the store convention `n . r_hat >= 0` (`labels._normal_field`; the fraction of voxels
  re-oriented is reported -- measured ~0 on both test regions, i.e. the coarse field was already
  consistent).  A boundary voxel `p` (body voxel with a 6-neighbour outside the body) is
  classified by a **purely local march** (`labels._local_faces`): step `t = 1..faces.side_reach`
  (6) voxels from `p`; if the body is hit along `+n_out` the body lies outward of `p`, so `p` is
  on the **in** face; if it is hit along `-n_out`, `p` is on the **out** face; a hit on both sides
  or on neither (thin spurs, corners, a neighbouring sheet closer than `side_reach`) leaves `p`
  **ambiguous** -> `faces_valid = 2` in its neighbourhood.  `sdf_in` / `sdf_out` are the signed
  distances to those two surfaces, sign `sign((p - q) . n_out(q))` with `q` the nearest face
  voxel, clipped to +-`clip`.  The split is purely geometric: the recto teacher never names a face.

  *Why not the nearest medial voxel.*  The first implementation used `(p - m) . n_out(m)` with
  `m` the nearest voxel of the body's medial surface.  That rule breaks whenever a *neighbouring*
  sheet's medial ridge is closer than the sheet's own -- a thick bundle (half-thickness 15) beside
  a thin delaminated layer across an 8-voxel gap picks the neighbour's ridge (8 + 2 = 10 < 15) and
  flips the whole out face to "in", the visible symptom being two sheets facing the same gap both
  drawn as in faces.  Measured disagreement between the two rules on real data: **36.6 %** of
  boundary voxels on the small slab and **15.6 %** on the core ROI (recorded per brick as
  `fine.faces.local_vs_centroid_disagree_frac`).
  `tests/test_faces.py::test_local_side_beats_nearest_medial_on_concentric_shells` is that
  configuration as two concentric cylinders: the local rule is 100 % correct with zero ambiguity,
  the nearest-medial rule 74 %.

  *Validation (small slab, `configs/paris4_small.json`, ~1500 voxels from the axis, z 112-144).*
  Marching from every out-face voxel along `+n_out` across the air gap lands on an in face of the
  next sheet for **99.41 %** of the 245 754 rays that hit (wrong face 0.02 %); the reverse march
  from every in-face voxel lands on an out face for **99.51 %** (wrong face 0.02 %).  The coarse
  normal is the true sheet normal there: median `|cos(n_out, grad CT)| = 0.93` (radial: 0.67).
- **Closing-radius sweep (Paris 4 ROI, `body_threshold = 60`, ball closing).**  Component median
  thickness in voxels for the 4 large 26-components, the overall thickness percentiles and the
  supervised fraction:

  | `close_radius` | components | component median thickness (vox) | thickness p10/med/p90 (vox) | valid 1 / 2 |
  |---|---|---|---|---|
  | 0 (layer level) | 4 (+1 speck) | 16.1 / 12.6 / 16.0 / 11.8 | 2 / 11 / 26 | 0.368 / 0.632 |
  | 4 | 4 (+1 speck) | 22.0 / 14.1 / 18.4 / 12.3 | 2 / 13 / 36 | 0.356 / 0.644 |
  | **6 (default)** | 4 | **31.7 / 16.5 / 28.6 / 34.3** | 2 / 16 / 46 | 0.678 / 0.322 |
  | 8 | 4 | 38.1 / 19.3 / 31.7 / 35.4 | 2 / 19 / 52 | 0.696 / 0.304 |
  | 6, `close_along_normal` | 4 | 25.5 / 15.0 / 24.1 / 34.1 | 2 / 10 / 40 | 0.351 / 0.649 |

  The number of windings crossing the ROI, from the unwrapped coarse phase (`atan2(sin, cos)`
  along y / x / diagonal lines through the mid-z slice), is **2.86 wraps ~ 3 windings**, and the
  4 large components are the same 4 at every radius: no radius up to 8 merges neighbouring
  sheets.  `close_radius = 6` is the default: it is the smallest radius that brings the bundle
  thickness into the 30-45 voxel range physically expected (3 of the 4 components; the second is
  a genuinely thinner sheet at 16-19 voxels at every radius) and it keeps a factor-2 margin
  against the inter-sheet gap.  The oriented (`close_along_normal`) closing produces a thinner
  body and half the supervised fraction here, so the ball stays the default.
- **Which face carries the writing.**  Two diagnostics are recorded per component in
  `labels.summary.json` under `fine.faces`.  `recto_is_in` = the fraction of the *band-adjacent*
  boundary (boundary voxels within `faces.band_reach` (8) voxels of `recto prob >=
  recto_threshold`) that landed on the in face.  It is a blunt instrument: a band sitting in an
  air gap is within `band_reach` of the in face of one sheet *and* the out face of its neighbour,
  so it is pulled toward 0.5 by construction.  The sharp measurement is the **per-face band
  coverage**: the fraction of in-face voxels with a recto-band voxel within k, against the same
  for out-face voxels.

  On the **small slab** (unambiguous geometry, `close_radius = 6`, bundle median thickness 32 vox
  = 77 um) the band is clearly **one-sided on the IN face** -- the face toward the umbilicus:

  | k (vox) | in-face coverage | out-face coverage | in / (in + out) |
  |---|---|---|---|
  | 1 | 0.362 | 0.139 | **0.722** |
  | 2 | 0.406 | 0.161 | 0.716 |
  | 3 | 0.480 | 0.192 | 0.714 |
  | 5 | 0.575 | 0.248 | 0.699 |
  | 8 | 0.706 | 0.348 | 0.670 |

  Per component: 0.688 (the 22.6 M-voxel component that spans the slab and includes the messy
  merged regions), **0.971** and **1.000** for the two smaller, cleaner components.  `recto_is_in`
  on the same slab is 0.611 weighted (0.597 / 0.899 / 0.770 per component) -- the same lean,
  blunted as expected.  So **the recto teacher's band is face-selective and sits on the in face**;
  the earlier "0.5, band centred between the faces" reading was produced by the faulty
  nearest-medial side assignment and is withdrawn.
  The band is nonetheless *wide*: 21.6 % of the slab is recto foreground against a 27.0 % body
  fraction, only 40 % of band voxels lie inside the CT body, and inside a bundle their normalised
  position (0 = in face, 1 = out face) is p10/median/p90 = 0.20 / 0.50 / 0.74.  It marks the
  writing *side*, not a thin surface.
- On the **core ROI** the same coverage ratio is 0.39-0.48 (slightly favouring the out face) and
  `recto_is_in` is 0.47.  That region is at the scroll core with the umbilicus *inside* it, where
  the windings are flattened and the outward direction is weakest; it is not evidence about the
  writing face.
- **`faces_valid` = 2** where any of:
  (a) *sheets in contact* -- the `faces.gap_min` (3) voxels immediately beyond an out-face voxel
  along `n_out` are not all air (another sheet starts right there).  The ignore is propagated to
  every voxel whose nearest out-face voxel is such a contact voxel and is within `clip` of it;
  (a2) the coarse field exists but its normal is missing at the voxel -- the outward sense would
  rest on the radial fallback (the same rule `build_fine_labels` applies to `sdf_valid`);
  (a3) the local march could not classify the nearest boundary voxel (see above);
  (b) the voxel's nearest sheet component has a median thickness outside
  `[faces.min_thickness, faces.max_thickness]` (15 .. 90 voxels: delaminated single layers below,
  fused multi-sheet blobs above);
  (c) the recto ignore band (`labels.ignore_band` and farther than `labels.ignore_near` from
  *either* face) -- the same rule as `sdf_valid`.
- **Validity budget (measured).**  Small slab (`close_radius = 6`): `faces_valid` 0 / 1 / 2 =
  0.0001 / **0.698** / 0.301.  Core ROI: 0.0002 / **0.637** / 0.362 -- more of the core is ignored,
  as it should be: 12 % of its boundary voxels fall back to the radial normal (rule a2, the coarse
  field is invalid around the flattened windings at the umbilicus) and 14 % of them are ambiguous
  for the local march (rule a3), against 0 % and 12 % on the slab.  Gap-pair consistency in the
  core is 97.6 % / 97.9 % (against 99.4 % / 99.5 % on the slab).
#### Building the block into an existing store
`dev/add_winding_fine.py <config>` appends the eight channels to a finished fine store out of
place (`fine.zarr.new`, then two renames), brick-wise with the same halo, brick grid and
worker pool `tsm labels` uses; `done.json` is the progress file, so it resumes.  Readers are
safe (nothing is modified in place; a training run keeps its inode until it reopens); it
refuses to run while a `tsm labels` process holds the config or `done.json` is fresher than
`--stale-minutes`.  The result is identical to the inline build except within ~3 voxels of a
brick boundary, where the inline build sees its own in-memory halo and the append sees the
stored one (0.4 % of the region, supervised fraction moves by 0.001).
`configs/paris4_faces_rvfw.json` is `paris4_faces_rvf.json` + `labels.winding_fine: true` +
`train.winding_source: "merge"`, out_dir `slab_faces_rvfw`, sharing
`slab_faces_rv/labels/fine.zarr`.

### Face source: upstream recto/verso bands (`extra.labels.faces.source`, 2026-09-06)
`faces.source` selects where the two faces come from: `"ct"` (default, everything above),
`"rectoverso"` (`tsm.rvfaces.build_rv_face_labels`) or `"merge"`.  The non-CT sources need
`faces.rectoverso_store` = the `rectoverso.zarr` written by `dev/rectoverso_slab.py` (the
upstream Scroll-1 recto/verso mesh labels resampled onto our grid by `tsm.xframe`; ch0 classes
0 bg / 1 recto / 2 verso / 3 contact, ch1 hz/vt).

- **Surfaces.**  recto = the face toward the umbilicus = our **in** face, verso = **out**; a
  contact voxel is on *both* (`recto_band = {1,3}`, `verso_band = {2,3}`).  The bands are 3-7
  voxels thick on our grid, so each is thinned to its ~1-voxel medial sheet by the 3-D EDT ridge
  (`labels.medial_surface`) after dropping components < `faces.rv_min_component` (32).
- **Sign, from geometry only.**  `sign((p - q) . n_out(q))` with `q` the nearest voxel of that
  face and `n_out` the outward direction there: the radial direction from the axis, refined by a
  local PCA of the band voxels in a `faces.rv_normal_window`³ (7) box (the eigenvector of the
  smallest eigenvalue is the sheet normal), oriented to agree with the radial direction, and
  falling back to it where the window holds < `rv_min_normal_count` (6) band voxels or the sheet
  is not planar (`lambda0 > rv_planarity_max * lambda1`).  The CT is never consulted.
- **`faces_valid` = 1** where the voxel is within `clip` of **both** an in-face and an out-face
  voxel, 2 elsewhere, 0 outside the CT data mask.  A brick with no band at all (or only one of
  the two faces) is entirely ignore.
- **`thickness`** = `sdf_in - sdf_out` (unclipped distances) on the sheet interior.  Both SDFs are
  distances to the *nearest* voxel of their own face set, exactly as in the CT builder, so the
  interior is resolved only out to the point equidistant from a sheet's own face and the next
  sheet's; at a contact plane the two SDFs share a sign on either side of it.
- **`"merge"`** runs both builders, takes the rectoverso result wherever it is valid and the CT
  result where rectoverso is ignore but the CT builder supervises, and appends three uint8
  channels **after** the fiber block (`tsm.rvfaces.RV_CHANNELS`, `labels.fine_channels(..., rv=True)`):

  | ch | name | encoding |
  |---|---|---|
  | .. | `faces_source` | 0 none / 1 rectoverso / 2 ct — the source of this voxel's face channels |
  | .. | `rv_class` | upstream rectoverso class copied through (0 bg / 1 recto / 2 verso / 3 contact) |
  | .. | `hzvt_class` | upstream fibre class copied through (0 bg / 1 hz / 2 vt / 3 exclude) |

  `labels.summary.json` gains `fine.faces.rectoverso`: the source fractions, the per-source
  thickness percentiles (the check that the CT builder is measuring single sheets, not welded
  pairs) and, over the voxels **both** sources supervise, the median / p90 `|delta sdf_in|` and
  `|delta sdf_out|` — the calibration number between the CT geometry and the upstream meshes.
- **CT fallback retuning.**  With a non-CT source the CT builder is only the fallback, so
  `labels.MERGE_CT_DEFAULTS` (body_threshold 80, close_radius 6, side_reach 3, min_thickness 10)
  and `ignore_band [0.35, 0.65]` become the defaults; anything set explicitly in the config wins.
  Measured 2026-09-06: the production settings weld neighbouring sheets (one 26-component per
  brick, median labelled thickness 48 vox against a visible 15-25).
- **Ambiguity ignore radius.**  The `_local_faces` ambiguity now ignores only voxels whose nearest
  face voxel is ambiguous *and* within `side_reach` of it (it used to be `clip`, which produced
  20-voxel teardrops responsible for 99.96 % of all ignores on the crumpled crop).
- **Measured (crumpled crop, 128x1024x1024 at (34176, 13056, 19968), `configs/paris4_faces_rv_crop.json`).**
  `faces_valid` 0/1/2 = 0.00005 / **0.979** / 0.021 (the CT-only store on the same crop:
  0.583 / 0.417); source none/rectoverso/ct = 0.038 / 0.699 / 0.263; thickness median 16 vox
  (rectoverso) against 24 (CT); `|delta sdf_in|` median 9.0 vox, p90 26 (band: 9.0 / 23),
  `|delta sdf_out|` median 8.0, p90 25.  638 s on 16 CPU cores.

- **Brick caveat.**  The body EDT is computed inside the haloed brick, so a sheet thicker than
  `2 * (halo - 2)` is cut by the halo and its `thickness` (and hence the thickness gate) can
  over-read near brick edges; `run_labels` logs a warning when `halo < max_thickness/2 + 2`.
  `sdf_in` / `sdf_out` themselves are halo-exact for `halo >= clip + 2` as before.

### Native 2.4 um winding field (`extra.labels.winding_fine`, `tsm.winding_fine`, 2026-09-07)
`extra.labels.winding_fine` (`true` or an options object) appends eight more channels
**after** every other block (`labels.fine_channels(..., winding_fine=True)`), so a store built
without them is still a prefix.  It needs `faces.enabled` (it is derived from `sdf_in` /
`sdf_out`) and the coarse store (the prior).  Encodings are **exactly the coarse store's**, one
difference: `wf_density` is already in wraps per *fine* voxel (the coarse `density` divided by
the pitch factor is what the training target is anyway).

| ch | name | encoding |
|---|---|---|
| .. | `wf_sin`, `wf_cos` | `sin/cos(2*pi*w)` of the refined phase, `encode_signed_u8` (byte 128 = 0 where `wf_valid = 0`) |
| .. | `wf_density` | `round(1000 * 1/P)`, `P` = local sheet spacing in **fine** voxels |
| .. | `wf_nx`, `wf_ny`, `wf_nz` | unit `grad w`, oriented outward, `encode_signed_u8` |
| .. | `wf_conf` | `round(255 * geo)`, `geo` = the coherence `|sum g| / sum |g|` of the two SDF gradient votes (1 where only one SDF is locally well behaved, `cos(angle/2)` where both are, so it falls off exactly where the sheet geometry is inconsistent); times the prior agreement `1 - |dw|/snap_max` when the gate is on |
| .. | `wf_valid` | **0 / 1 only** (0 = do not supervise) |

**Geometry.**  Both SDFs are positive on the outward side of their own face, so along the
outward normal one period is `in face (d=0) -- sheet body -- out face (d=t) -- air gap --
next in face (d=P)`, and each SDF is a *sawtooth* of period `P` in `[-P/2, P/2]`
(`sdf_in = d` below the half period, `d - P` above; `sdf_out = d - t` likewise).  The winding
increases by exactly 1 per period and is linear in `d` -- which is what makes
`|grad w| = 1/P = 1/(thickness + gap)` -- so its fractional part is simply

    f = (sdf_in / P) mod 1

with the integer turn on the **in** face.  The `mod` picks the branch, so no sheet/gap
classification is needed and `f` is continuous and monotone along the normal.  `P` comes from
the two SDFs alone: `D = sdf_in - sdf_out` equals `+t` on part of every period and `-g` on the
rest, so each is measured somewhere in every period and filled to the box by an EDT feature
transform (`winding_fine.sheet_period`); `P = t + g`.  A box that never shows one of the two
(smaller than half a period, or no gap at all) is entirely invalid.  The normal is the gradient
of whichever SDF is locally well behaved (`|grad| in [0.5, 1.5]`, which excludes the sawtooth
seams), oriented by the coarse normal, else the radial direction.

**The integer turn, and why the coarse prior does not gate anything (2026-09-07).**  The store
holds the phase **modulo 1**, so the spec's snap `w_fine = round(w_coarse - f) + f` never
reaches the store: `f` is fully determined by the two faces.  The prior is therefore **not**
consulted by default (`snap_max: null`).  `wf_valid = 1` requires only `faces_valid == 1`, an
unsaturated `sdf_in` (`|sdf_in| < clip - 0.5`; beyond the clip the phase is unknowable), `P`
inside `[min_period, max_period]` and a determined normal.  The coarse field is used *only* to
orient the normal outward where the radial direction is ambiguous.

That is a measurement, not a preference: on the crumpled crop the coarse period is 6x the
faces' (below), i.e. the 9.6 um teacher aliases the sheets it is supposed to count, so gating
the correct fine phase on it would discard good supervision and keep bad.  Setting `snap_max`
to a number turns the spec's gate back on as an **opt-in diagnostic** --
`resid = wrap(phase_coarse - f)` in `[-0.5, 0.5]` is `|w_fine - w_coarse|`, and
`|resid| >= snap_max` (plus coarse `valid != 1`) makes the voxel invalid.  `resid` is
accumulated into `labels.summary.json` (`fine.winding_fine.abs_resid_turns`) either way, so
the aliasing stays measurable.  A prior off by more than half a turn aliases to the
neighbouring sheet undetectably -- but since only the phase mod 1 is stored, that changes
nothing in the store.

**Phase origin.**  `f = 0` sits on the in face while the coarse lasagna phase has its own zero,
so the two conventions differ by a roughly constant offset (`phase_offset`, in turns, re-anchors
`f` onto the coarse convention if the two are ever mixed in one run; default 0).

**Measured (crumpled crop, 128x1024x1024 at (34176, 13056, 19968), fine store
`slab_faces_rv`, coarse `slab_faces_tta`, `dev/winding_fine_crop.py`, 244 s on 16 cores).**
`wf_valid` = **0.758** of the crop (identical gated or not: with `snap_max = 0.5` the gate
rejected nothing, which is what made it worth removing).  Against the coarse prior:
median `|dw|` **0.147 turns**, p90 0.393, fraction `|dw| > 0.25` = **0.285**, systematic
phase-convention offset only -0.056 turns (so recentring barely helps: 0.143).  The `|dw|`
histogram is peaked at 0 (0.096 of the mass in the first 0.025-turn bin against 0.022 in the
last), i.e. weakly correlated, not independent.

The disagreement is the **coarse field aliasing**, not the refinement.  Sheet spacing from
the faces: p10/50/90 = **10 / 28 / 40** fine voxels (67 um at the median).  From the coarse
winding teacher's own `density`: **134 / 167 / 209** (400 um).  Per voxel the ratio
coarse/fine is p10/50/90 = **4.1 / 6.0 / 13.2** and **99.97 %** of the crop has it above 2.
At 9.6 um the lasagna teacher cannot resolve a 28-voxel period, so its phase there counts a
~400 um envelope while the upstream recto/verso faces resolve ~67 um laminae.  Independently
measured on the same crop: 9.6 % of voxels lie within 1 voxel of an in face, `D > 0` has
median 9.4 and `D < 0` median 11.2 (so `P ~ 21`), and only 1.5 % of the SDFs are clip
saturated -- the geometry is well resolved, the prior is not.

`dev/winding_fine_crop.py` writes the six-panel montage (CT | coarse phase cos | refined
phase cos | refined valid | `|dw|` heat | coarse/fine period ratio) and prints all of the
above, including the aliased fraction.

### Human label corrections (`faces_weight`, `faces_source = 3`, 2026-09-07)

A two-script human-in-the-loop loop over the two-face labels; neither script is part of
`tsm labels` and neither changes anything when it is not run.

**`dev/annot_export.py <config> --out DIR [--n 12] [--size 256] [--select ...] [--pred P]`**
scores every `--size` cube of the region on a stride-`--stride` (128) grid, keeps the top `--n`
**non-overlapping** ones and writes each as a self-contained *packet* directory of raw uint8
volumes for the renderer (https://github.com/SuperOptimizer/render3d): `.u8` = flat uint8,
`index = (z*ny + y)*nx + x`, dims from the manifest, never from a header.

| file | contents |
|---|---|
| `ct.u8` | CT at the config volume level |
| `faces_in.u8` / `faces_out.u8` | 1 on the label zero set of each face (`sdf_* ` byte 128) |
| `ignore.u8` | `faces_valid == 2` |
| `source.u8` | `faces_source` (0 none / 1 rectoverso / 2 ct / 3 human) |
| `rv_class.u8` | upstream `rv_class` |
| `pred_in.u8` / `pred_out.u8` | with `--pred`: the student's zero crossings |
| `meta.json` | origin, size, voxel_um, store path + channels, per-layer sha256, and the sha256 of the label bricks the packet was cut from |
| `correction.u8` | **written by the annotator**: 0 untouched / 1 in / 2 out / 3 ignore / 4 "wrong here, erase" |

`packet.json` at the root lists every packet (path, origin, size, dims, layers) plus the
palette the renderer paints with (faces_in red, faces_out blue, ignore dim, the two label
channels and the five correction classes).  `--tif` adds a `.tif` copy of each layer
(`tifffile`, in the `viz` extra).  No WebKnossos variant is written: it needs a wkw/OME-Zarr
dataset plus a `datasource-properties.json` and the `webknossos` package, which is neither
trivial nor a dependency here.

**Selection.**  The brief's ideal criterion is `mean |delta sdf_in| + |delta sdf_out|` between
the rectoverso- and CT-derived faces, but the store holds only the *merged* result -- one SDF
pair per voxel -- so that number cannot be recovered without re-running both builders over the
region.  `--select disagreement` therefore implements the documented **proxy**

    score = seam_frac + 0.25 * min(cv_thickness, 1)

* `seam_frac` = the fraction of the crop that is CT-sourced (`faces_source == 2`) **and**
  within `--near` (8) voxels of a rectoverso-sourced voxel: exactly the hand-over surface,
  where the two builders disagree by construction;
* `cv_thickness` = std/mean of `thickness` over the crop's `thickness > 0` voxels, clipped to
  1 -- a locally wild thickness is the signature of a welded sheet pair or a delaminated
  layer, the two things the builders differ about.
A crop with less than `--min-valid` (0.10) of `faces_valid == 1` scores -1.
`--select uncertain --pred <pred.zarr>` instead scores the fraction of label face voxels whose
distance to the student's zero crossing of the same face exceeds `--pred-tol` (3) voxels;
`--select random` is the seeded control.  Scoring is one streaming pass: per stride block the
counts and thickness moments above are accumulated (the seam EDT runs per tile with a `--near`
halo, so the block statistics do not depend on the tiling) and a crop's score is the sum over
its blocks.

**`dev/annot_import.py <config> --packets DIR [--weight 5.0]`** folds every `correction.u8`
(or `correction.tif`) back into `fine.zarr` **in place**:

- class 1 / 2 become the in / out face zero sets, 3 sets `faces_valid = 2`, 4 erases whatever
  face sat there;
- `sdf_in` / `sdf_out` are recomputed from the corrected face sets over the crop **plus a
  `clip + 4` halo** read from the store (so the SDF is continuous across the crop boundary;
  only the core is written) with exactly the rectoverso machinery -- `rvfaces._signed_face`,
  sign `sign((p - q) . n_out(q))` at the nearest face voxel `q`, `n_out` the outward radial
  direction refined by the local band PCA (`rvfaces.local_band_normals`) -- but the recomputed
  field is written back **only inside the region of influence** (below);
- `faces_source` becomes **3** (`rvfaces.SOURCE_HUMAN`) on every painted voxel -- a code no
  builder produces, so corrected voxels stay identifiable for ever after;
- the store gains a uint8 **`faces_weight`** channel (appended last, out of place with the
  brick-wise copy-and-rename machinery of `dev/add_fiber_channels.py`): 1 everywhere,
  `--weight` on the painted voxels.  `0` in the channel means "never written" and reads as 1.

**Region of influence -- the import is incremental, and it has to be (fixed 2026-09-07).**
Only voxels within `clip + 1` of a *painted* voxel take the recomputed value; every other stored
byte is copied through untouched.  That is not an optimisation.  A full-crop recompute is **not**
a faithful round trip of what the merge builder wrote, so until 2026-09-07 every import silently
perturbed the whole crop: measured on the real Paris 4 store, 736 painted voxels (480 in-face
brush, 256 out-face line) moved ~10 M of the 16.7 M SDF voxels of a 256³ crop.  Three
independent reasons, none of them recoverable from the store:

1. **The PCA input band is not in the store.**  `rvfaces.build_rv_face_labels` feeds
   `local_band_normals` the *unthinned* upstream band (`rv_class > 0`, 3-7 voxels thick, both
   faces at once) and evaluates it at the *thinned* face voxels.  Re-deriving the band from the
   stored zero sets changes the covariance in every window and so the normal (~5 % of face
   voxels on a tilted synthetic sheet stack), which flips the **sign** of whole slabs --
   `|Δ| = 228` bytes, i.e. `-clip` against `+clip`.  The importer now feeds it the stored
   `rv_class > 0` united with the corrected face voxels, which reproduces the builder exactly on
   rectoverso-sourced geometry; on CT- or human-sourced voxels there is no band to feed it.
2. **The store is a merge of two builders.**  Where `faces_source == 2` the SDF came from
   `labels.build_face_labels`, whose zero set is the CT body boundary split by a local march --
   a different surface, unreachable by the rectoverso machinery.  The union of the two zero sets
   across the hand-over seam is a surface neither builder ever saw.
3. **The store was written brick-wise.**  Every brick saw its own `clip + 4` halo, so beyond
   `clip` the saturated `±clip` bytes carry a brick-dependent sign a packet-shaped box does not
   reproduce.

Outside the region of influence that is harmless: every added or erased face voxel is more than
`clip` away, so the true distance is saturated at `±clip` before and after and only the
(already brick-dependent) sign of a saturated voxel could differ.  Inside it the recompute uses
the corrected face set over the **whole haloed box** -- the untouched stored faces together with
the painted ones -- so the field is continuous at the boundary of the region, where every changed
face voxel is `>= clip + 1` away and therefore invisible to both fields.  The dry-run summary
reports `influence_voxels`, `sdf_in_max_delta` / `sdf_out_max_delta` and
**`sdf_max_delta_outside`, which is 0 by construction**; a non-zero value is logged as an error
and the run returns 1, so a regression is visible on real data.
`tests/test_annot.py::test_import_of_an_empty_correction_is_a_byte_identical_no_op` builds a
store through the real `build_rv_face_labels` + `build_face_labels` + `merge_face_labels` path
(tilted sheets, all three `faces_source` codes present) and asserts that an all-zero correction
leaves `sdf_in`, `sdf_out`, `faces_valid` and `faces_source` byte-identical -- a full-crop
recompute moves 11 % of those bytes on that fixture, with `|Δ|` up to 254.
`test_import_of_a_brush_stroke_only_moves_the_region_of_influence` asserts the bound for a
24-voxel stroke.

Each packet carries the sha256 of the label bricks it was cut from and the import refuses
(without `--allow-stale`) if the store moved under it.  The store is opened through
`BrickWriter`, so the reopen validation applies; the writes go to `writer.array` because a
packet crop is not a brick and must not enter `done.json`.  `--dry-run` reports and writes
nothing.

**Training.**  A fine store carrying `faces_weight` gives `CropDataset` (faces mode only) an
extra target key `surface_weight`, a plain scalar field that moves with the grid like
`ink_prob` through both augmentation paths and is mask-aware pooled for deep supervision.
`train.faces_loss` / `train.surface_loss` take it as an optional per-voxel multiplier and fold
it into every term's mask -- both the masked means and the soft Dice use the mask purely
multiplicatively, so a weight of exactly 1 (and a store without the channel at all) reproduces
the previous loss **bit for bit**.  `tests/test_annot.py` asserts that identity.

## `<out_dir>/labels/coarse.zarr` (9.6 um, level 2; region = fine region / 4)
| ch | name | encoding |
|---|---|---|
| 0 | phase_sin | sin(2*pi*w): u8 = round(127.5 + 127.5*v) |
| 1 | phase_cos | cos(2*pi*w): same |
| 2 | density | |grad w| in wraps per level-2 voxel: u8 = round(v*1000), clipped (same as lasagna grad_mag) |
| 3 | nx | unit sheet normal, oriented OUTWARD from the axis (n . r_hat > 0): u8 = round(127.5 + 127.5*v) |
| 4 | ny | same |
| 5 | nz | same |
| 6 | conf | 0..255 fusion confidence (255 = teachers agree / clean phase) |
| 7 | valid | 0 no data (air), 1 supervise, 2 ignore (teacher disagreement) |

Derivation from the lasagna teacher (channels cos, grad_mag, dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x at level 2):
- normal: decode with tsm.labels.decode_lasagna_normal, flip sign so n . r_hat > 0 (r_hat = radial unit vector from the axis at that z).
- phase: cos(2*pi*w) = 2*cos_ch - 1 (lasagna cos is 0.5+0.5cos). w increases outward, so d(cos)/dr = -2*pi*sin*|grad w| => sin = -sign(d cos / d r_out) * sqrt(1-cos^2), with the derivative taken along n (outward) with a 1-voxel central difference of the smoothed cos channel; where |d cos/dr| is tiny (near extrema) take the sign from the neighbours along n.
- density: grad_mag / 1000 (already wraps/voxel).
- conf: local cos contrast (max-min of cos in a 5^3 window, scaled to 0..255) times CT mask.
- valid: 1 where CT (level 2) smoothed > 30 within 2 vox, else 0; 2 where the decoded normal magnitude < 0.5 or the lasagna dir pairs are inconsistent.
- The winding_model_9um teacher is disabled (WINDING_ENABLED=False); no fusion in v1.

## Student decode conventions
sdf = (u8-128)*CLIP/127; sin/cos/n = (u8-127.5)/127.5; density = u8/1000; ink = u8/255.

## Implementation notes (`tsm labels`, 2026-09-02)
Choices made where the spec above left room; the store encodings are unchanged.
- Fine `sdf` sign uses the outward normal **at the nearest medial voxel m** (not at p): `sign((p - m) . n_out(m))`, `n_out(m)` = trilinearly upsampled coarse normal (level-2 `nx,ny,nz`, zeroed where coarse `valid != 1`, renormalised; used where its norm >= 0.5) else the radial direction at m. Voxels exactly on the medial surface store 128. Voxels farther than CLIP from the medial surface are +-CLIP with the sign of the nearest in-brick medial voxel (halo = CLIP + 4; beyond the halo the choice of medial voxel is brick-dependent, both signs are "which side of the nearest sheet").
- `sdf` u8 is 0 wherever `sdf_valid == 0` (no data); everywhere else it is in 1..255.
- A brick with no medial voxel at all (no recto sheet within CLIP+4) gets `sdf = +CLIP` and `sdf_valid = 2`.
- CT mask (both stores): gaussian-smoothed CT (sigma 2 vox at level 0, 1 vox at level 2) > `ct_mask_threshold` (30), grown by 2 voxels. Inside the ROI this is ~100 % (masked volume: air between sheets reads 30-45, only the outside of the scroll is 0).
- `ignore_band` applies to the recto probability at the voxel: `0.2 <= p <= 0.8` and > 2 vox from the medial surface -> `sdf_valid = 2`.
- Recto mask components < `min_component` (100 vox, 26-connectivity) are dropped per haloed brick (components cut by the brick halo are counted within the haloed brick only).
- Coarse phase: `sin = -sign(grad(smooth(cos, sigma=1)) . n) * sqrt(1 - cos^2)` with `cos` the raw (unsmoothed) channel, so `sin^2 + cos^2 == 1` exactly before u8 rounding. Where `|grad . n| < 2e-3` the sign is propagated iteratively from the 26-neighbourhood (in-plane neighbours share the phase; the two neighbours along n straddle an extremum and cancel) and defaults to +1 if still undecided.
- Coarse `valid = 2` uses `decode_lasagna_normal(..., return_extra=True)`: candidate agreement (|weighted sum of the three per-plane candidates| / sum of magnitudes) < 0.5, or re-encoding RMS error of the decoded normal against the six dir channels > 0.2. Voxels where all 8 lasagna channels are 0 (teacher wrote nothing) are `valid = 0`.
- Coarse `conf = round(255 * (max - min of cos in 5^3) / 2)`, 0 where `valid == 0`.
- Coarse `density` is the lasagna `grad_mag` u8 copied through (already round(v*1000)).
- Level-2 voxel (i) sits at level-0 coordinate 4 i + 1.5; the radial direction and the trilinear upsampling (`ndi.zoom(grid_mode=True)`) both use this alignment.
- Bricks: fine core 128x256x256 (halo 24), coarse core 64x256x256 (halo 4), all edge-replicated at the region faces; resume via `done.json` (`BrickWriter.has_brick`), `--force` recomputes.

### Parallel fine bricks (`extra.labels.workers`, 2026-09-06)
The fine stage is one heavy, embarrassingly parallel computation per brick (the merged
rectoverso+CT build measured ~70 s/brick, so 1152 bricks ~ 22 h single-process).
`extra.labels.workers` (default `1`) fans those bricks out over a process pool:

- Each worker opens its own readers once (`labels._fine_worker_init`: recto/ink/fiber teachers,
  the coarse store, the CT reader, the axis, the rectoverso store) and then answers brick jobs
  with `labels._fine_brick`, which is pure with respect to the process -- it returns the channel
  arrays *and* its per-brick stats deltas.
- The **parent** does every `BrickWriter.write`, every `done.json` update and all stats
  accumulation (`labels.merge_stats`), consuming results strictly **in brick order**.  That is
  what makes the store, `done.json` and `labels.summary.json` identical whatever `workers` is
  (float sums are order-dependent, so completion order must not leak in); the single-process path
  is literally the same functions called in-process.  `tests/test_labels_workers.py` asserts the
  byte-identity for `workers=1` vs `workers=3` over fork/forkserver/spawn.
- At most `workers + 1` bricks are outstanding, so the pool cannot pile up finished results if a
  worker runs ahead of the writer.
- Resume is unchanged: bricks already in `done.json` are never dispatched.
- **Memory.**  `budget.ram_bytes` is a *per process* limit (configs set 8 GiB on a 182 GB box), so
  it is the wrong ceiling for a pool.  The per-brick estimate (`_fine_budget_rows`) is still
  checked against it; the pool is checked against `MemAvailable - budget.min_avail_mb` measured at
  start-up, or against `extra.labels.workers_ram_bytes` when that is set.  Too many workers is a
  `BudgetError` before any I/O (it is checked on `--dry-run` too).  Measured single-brick RSS for
  the merged rv+CT build is ~4.6 GB, so `workers=8` wants ~37 GB free.
- `extra.labels.mp_start` (default `"fork"`) picks the start method; `"forkserver"` / `"spawn"`
  cost a few seconds of start-up each but cannot inherit a lock held by a zarr background thread.

## Dataset conventions (tsm.data / tsm.train, milestone 7)
Choices made where the tables above are silent; the tables are unchanged.
- **No-data sdf.** `sdf` u8 = 0 decodes to -20.16 (outside +-CLIP). `CropDataset` zeroes the decoded sdf wherever `sdf_valid == 0`; every loss is masked by `valid == 1` anyway.
- **Valid logit target.** The surface head's valid logit is trained with BCE against `(sdf_valid == 1)` on all voxels with `sdf_valid != 2`, i.e. air / no-data (0) counts as a *negative* ("no trustworthy SDF here"); `2` voxels are excluded. Same rule is not applied to ink (ink_valid is 0/1 and only gates the loss).
- **One pitch.** The student trains at 2.4 um only (user decision 2026-09-02): every crop is a fine crop; `surface`/`ink` targets come from `fine.zarr`, `winding` targets from the matching box of `coarse.zarr` (global origin / factor, size / factor, +1 coarse voxel margin) upsampled on the fly by `data.upsample_coarse_targets`. The factor is `round(coarse.voxel_um / fine.voxel_um)` from the store attrs (4 today; 2 for a future 4.8 um store). Coarse voxel `i` covers fine voxels `[i*f, (i+1)*f)` (centre-aligned, `F.interpolate(align_corners=False)`); sin/cos, normal and density are mask-aware trilinear (only coarse `valid == 1` voxels contribute), sin/cos and normal are then renormalized, density is divided by the factor (wraps per fine voxel); `conf` and `valid` are nearest. Every item carries all target keys with fixed shapes; supervision is purely by the valid masks (no coarse store => winding masks all zero). The scale input channel is kept (constant 0).
- **Crop origins.** `build_origins` scans the fine store's `sdf_valid` at stride 64 and keeps origins (local store coordinates) with >= 5 % `valid == 1`; on an axis thinner than the patch the window is centred on the store (negative origin; the overhang reads real CT and no-data labels, which the masks ignore). Cached as `<store>/../origins_<store>_p<patch>_s<stride>.npy`. CT is read at `origin_zyx + local` at the config volume level (must have the fine store's pitch).
- **Winding source (`extra.train.winding_source`, 2026-09-07).**  `"coarse"` (default,
  unchanged: the coarse store upsampled on the fly), `"fine"` (the store's `wf_*` block only --
  every voxel with `wf_valid != 1` becomes `winding_valid = 0`, i.e. unsupervised) or `"merge"`
  (the `wf_*` values, conf and validity where `wf_valid == 1`, the upsampled coarse targets
  everywhere else; `data.fine_winding_targets` / `data.merge_winding_targets`).  A fine store
  without the `wf_*` channels warns once and behaves exactly as `"coarse"`, so old stores and
  old configs are unaffected.  The default stays `"coarse"` only so that existing configs do
  not change behaviour; where the coarse field aliases (measured above) `"merge"` is the
  better target, and `configs/paris4_faces_rvfw.json` uses it.
- **Surface mode (`extra.train.surface_mode`, 2026-09-02).**  `"medial"` (default) is unchanged:
  `surface_sdf` is 1 channel from `sdf`, `surface_valid` from `sdf_valid`, and the surface head is
  `[sdf, valid logit]`.  `"faces"` needs a store with the four face channels: `surface_sdf` becomes
  2 channels `[sdf_in, sdf_out]`, `surface_valid` comes from `faces_valid`, crop origins are scanned
  on `faces_valid`, and the surface head becomes `[sdf_in, sdf_out, valid logit]`
  (`student.heads_for`).  Losses (`train.faces_loss`): the gaussian-weighted L1 and the band Dice
  of the medial mode applied to each face separately (`surface/sdf_in_l1`, `surface/sdf_out_l1`,
  `surface/band_dice_in`, `surface/band_dice_out`) plus one `valid_bce` on the shared valid logit,
  with the same masking (`valid == 1` for the SDF terms, `valid != 2` for the BCE).  Every other
  key, the deep-supervision pooling and all augmentation rules are unchanged -- `surface_sdf`
  simply carries 2 channels through the identical per-channel rule.  `evaluate` reports the medial
  metrics per face (`surface/in_*`, `surface/out_*`) plus `surface/thickness_mae`
  (`sdf_in - sdf_out` between the faces).
- **Body mode (`extra.train.surface_mode = "body"`, 2026-09-12).**  The orientation-free target:
  the same store as `"faces"` (origins and `surface_valid` from `faces_valid`, `faces_weight` still
  applies), but `surface_sdf` is the single channel `data.body_sdf(sdf_in, sdf_out) =
  min(sdf_in, -sdf_out)` -- positive inside the papyrus, `0` on both faces *and* on a labelled
  contact plane (touching sheets stay separate `> 0` components), negative outside.  The head is the
  medial one, `[sdf_body, valid logit]`, so the export / TRT layout is unchanged.  Loss
  `train.body_loss` = the medial terms (`surface/sdf_l1`, `surface/band_dice`, `surface/valid_bce`),
  bit-identical to `surface_loss` when no `surface_aux` weight is set; with `surface_aux` it adds the
  zero-set terms named for the body (`surface/shell_body`, `crest_body`, `far_body`) and the
  single-target form of `gap` / `cldice`.  Augmentation is the unchanged scalar-SDF path (isotropic
  scale × `det(S)^(1/3)`, saturation → `valid 2`); `min()` and resampling commute exactly for the
  cube rotations / flips and to sub-voxel accuracy otherwise.  The direction fibre targets use
  `fiber.sheet_normal(prefer_fallback=True)` here (the gradient of a body SDF is degenerate on the
  medial ridge).  `evaluate` uses the single-face keys and adds `surface/body_dice`,
  `surface/body_iou`, `surface/body_vol_ratio` and `surface/body_n_components_ratio` (26-connected,
  ≥ 32 voxels, per-crop median); there is no `thickness_mae`.  Prediction store:
  `tsm.infer.pred_channels("body")` = `sdf_body, valid, ink, sin, cos, density, nx, ny, nz, conf,
  spare, surface_body1` (fibre channels spliced after `spare` as usual), with no thickness pass.
  Configs: `configs/body30k{,_bgf,_bgf_aux}.json`.
- **Auxiliary surface terms (`extra.train.surface_aux`, faces mode, 2026-09-07).**  Three optional
  terms on the *zero set* rather than on the SDF value, all with weight `0` by default -- with the
  defaults the surface loss is byte-identical to the one above and nothing is even computed
  (`train.surface_aux_terms`, logged as `surface/shell_in`, `surface/crest_in`, `surface/far_in`
  and the `_out` twins, already multiplied by their weight; deep-supervised like the main SDF
  loss).  Motivation (measured on the student, 2026-09): the predicted in-face has excellent
  recall of the human bands (band -> pred median 2 vox) but poor precision (45 % of predicted
  in-face voxels are > 5 vox from any band) and fragments in crumpled regions (821 components
  vs 57).
  - `shell` (`shell_radius` 2, `shell_margin` 3.0) -- "de-blob" separation.  Dilate the label zero
    set (`|sdf_tgt| <= 0.5`, `valid == 1`) by `shell_radius`, subtract the zero set itself, keep
    the shell voxels the *label* already puts `>= shell_margin` away, and penalise
    `relu(shell_margin - |sdf_pred|)` there.  The gate is on the label so a perfect prediction
    scores exactly 0.  Note the geometry: the dilation is cubic (one `max_pool3d`), so with
    `shell_radius < shell_margin` only its corners (out to `shell_radius*sqrt(3)`) pass the gate
    and the term is a thin sliver plus whatever label discontinuities contribute -- set
    `shell_radius >= shell_margin` for the intended behaviour (`train.surface_aux_opts` warns).
  - `crest` (`crest_tol` 1.0) -- existence recall: `relu(|sdf_pred| - crest_tol)` averaged **per
    sample** over the label face voxels (`train.per_sample_mean`), so a crop with a sparse face is
    not drowned by the dense crops in the batch.
  - `far` (`far_margin` 6.0, default weight 0) -- the direct precision term: over voxels the label
    puts `>= far_margin` from the face, `relu(far_margin/2 - |sdf_pred|)`.  It pulls *against*
    `crest`, so keep it small (0.1 against 0.5 in `configs/paris4_faces_rvfw_aux.json`).

  Asking for a non-zero weight in `surface_mode = "medial"` is an error, not a silent no-op.
  `configs/ablate_surface_aux.json` ablates `shell_crest` / `shell_crest_far` / `core400`.
- **Umbilicus core mask (`extra.labels.core_radius_vox` / `extra.train.core_radius_vox`, both
  default `0` = off, 2026-09-07).**  Voxels within that **in-plane** radius (level-0 voxels) of the
  umbilicus -- the axis interpolated at each z, `labels.core_mask` -- are dropped from supervision:
  `faces_valid` (`surface_valid`) becomes `2` (ignore) and the winding validity `0` (no data).
  The CT, the SDF values and every other channel are left alone, so the core is still *context*,
  just not a target.  Reason: near the core the sheets wrap so tightly that the in/out convention
  degenerates -- the measured `recto_is_in` falls to ~0.47 there, i.e. a coin flip, so the labels
  are noise with a plausible shape, which is worse for the surface head than no labels at all.
  - **Build time** (`extra.labels.core_radius_vox`): applied in `labels._fine_brick` (`faces_valid`,
    and `wf_valid` when `winding_fine` is on) and in the coarse loop (`valid`).  Because the fine
    stage *reads* the coarse normal field, clearing the coarse `valid` also makes the fine build
    fall back to the radial normal inside (and one coarse voxel around) the disc, which changes the
    `sdf` sign there; through the component-level thickness / side gates that can flip the odd
    voxel elsewhere in the same brick.  Both are inside or beside a region that is now `ignore`.
    `labels.summary.json` reports `fine.faces.core_radius_vox` / `core_masked_voxels`.
  - **Train time** (`extra.train.core_radius_vox`): `CropDataset` applies the same rule to
    `surface_valid` / `winding_valid` per crop, so an existing store can be used without a rebuild.
    It loads the axis even when `input_radial` is off.
  - Geometry of the Paris 4 slab (start `(34176, 10496, 15872)`, size `(256, 6144, 6144)`): the
    axis sits at `(y, x) ~ (13600, 18940)` over those 256 z, well inside the box, so the disc is
    fully contained and the masked fraction is exactly `pi r^2 / 6144^2` -- **1.33 %** at
    `r = 400` and **5.33 %** at `r = 800`.  `configs/paris4_faces_rvfw_aux.json` uses 400 at train
    time.
- **Augmentation and orientation.** Flips of z/y/x and rot90 in the (y, x) plane only. All scalar channels (sdf, phase sin/cos, density, conf, ink, valid) move with the grid unchanged; the normal `(nx, ny, nz)` is transformed as a vector (components permuted with the axes and negated on a flipped axis; the rot90 is composed from flip + transpose exactly as `np.rot90(a, k, axes=(y, x))`). "Outward" is thus carried by the normal channels; the sdf sign and the phase direction are treated as data-defined scalars and are not flipped. `tests/test_data.py` checks gradient-field equivariance for all 32 flip/rot90 combinations.
- **Input radial channels (`extra.train.input_radial`, default `true`, 2026-09-03).** The student input is `[ct z-score, scale const, r_z, r_y, r_x]` (5 channels; `student.input_channels` / `in_channels`, `TSMNet(in_ch=5)`). `r` is the outward unit radial direction of the umbilicus axis at that voxel (`labels.radial_field` at level 0, (z, y, x) order, so `r_z == 0` for a z-parallel axis segment), computed per crop from the crop's *global* origin. Rationale: the sdf sign, the winding phase sign and the normal orientation are all defined as "away from the scroll axis", which a 128^3 crop has no way to know -- without this input the heads can only collapse to sign-free predictions. The axis JSON is `extra.train.axis_path` (default `extra.labels.axis_path`, else `labels.DEFAULT_AXIS`); it is stored in the checkpoint config (`input_radial`, `in_ch`, `axis_path`) so `tsm infer` rebuilds the same input. Near the axis the direction is ill-defined; it is left as is (coarse `valid` is already 2 there). Augmentation treats the three channels **exactly** like the winding normal target: resampled trilinearly, transformed by the sampling matrix's `L^-T` (rotations/flips: the rotation part), renormalised to unit length; v1 uses the same flip/transpose component rule. At inference `sliding.predict_box` hands `infer.StudentNet` (which sets `needs_box_origin = True`) the absolute `(z, y, x)` origin of every window as `box_origin_zyx`, and the wrapper builds the field itself; its own TTA moves the radial channels as a vector (`tta_forward(..., vec=(2, 5))`). A checkpoint without `input_radial` in its config loads as the old 2-channel student.
- **Fibre head (optional, 2026-09-06).** `extra.train.heads.fiber = true` adds a 2-channel head
  `fiber = [vertical logit, horizontal/angular logit]` in **both** surface modes
  (`student.heads_for(surface_mode, fiber)`).  Loss = BCE + soft Dice per channel against the *soft*
  teacher probabilities from `fine.zarr`, masked by `fiber_valid == 1`, weighted by
  `extra.train.loss_weights.fiber` (default 1.0) and included in deep supervision exactly like `ink`
  (`train.fiber_loss`, `train.downsample_targets`).  Each channel is an independent sigmoid rather than
  a 2-way softmax: the teacher's 4-way softmax also carries background and ink, so `vt + hz <= 1`.
  The two classes are scalar fields (they are defined relative to the sheet, not to the volume axes),
  so they need no augmentation rule beyond "scalar" and follow the `ink` rules verbatim.
  Holdout metrics: `fiber/vt_auprc`, `fiber/hz_auprc` (logit vs teacher probability > 0.5 on
  `fiber_valid == 1`, same grouped-threshold AP as the other heads).

- **Fibre modes (`extra.train.fiber_mode`, `tsm.fiber`, 2026-09-07).**  "vertical" and
  "horizontal" are defined **relative to the scroll axis** (vertical = along z, horizontal =
  circumferential), so the pair is *not* a scalar field: the augmentation v2 cube rotations and
  arbitrary 3D rotations move the volume z axis and therefore move the class boundary.  Treating
  the two channels as invariant scalars (the behaviour up to 2026-09-07) fed the head
  inconsistent targets.  Two settings:
  - `"class"` (default, backward compatible): the 2-channel head is unchanged, but a spatial
    transform whose linear part sends the volume z axis more than **45 degrees** away from z
    (`tsm.fiber.axis_swap_needed`; the classes are *axial*, so the angle ignores a sign flip and
    a z flip is not a swap) **swaps the vt/hz target channels**, and with them the human
    `hzvt_class` codes.  Exact for the 24 cube rotations (z -> +-z / +-y / +-x), nearest-class for
    continuous rotations.  `extra.train.fiber_swap_fix: false` opts out and reproduces the bug
    (the ablation baseline only).
  - `"direction"`: the target becomes an **axial direction** per voxel, derived on the fly in the
    dataset from the existing channels -- **no label rebuild**.  With `n` the sheet normal
    (`grad sdf_in` where `|grad|` is in [0.5, 1.5], else the winding normal target) and
    `a = (1, 0, 0)` in (z, y, x): `t_v = normalise(a - (a.n) n)`, `t_h = normalise(n x a)`,
    `d = normalise(p_vt t_v + p_hz t_h)`, `s = max(p_vt, p_hz)`, validity =
    `fiber_valid == 1 & n not parallel to a`.  Where the human `hzvt_class` band exists it
    overrides the teacher (1 -> `t_h`, 2 -> `t_v`, `s = 1`, per-voxel loss weight
    `extra.train.fiber_band_weight`, default 5; 3 -> ignored).  `d` is transformed as a
    **vector** by every augmentation (like the winding normal; the sign is irrelevant).
    Head: 4 channels `[dz, dy, dx, s]`; loss (`train.fiber_dir_loss`): the axial cosine
    `1 - (d^.d)^2` weighted by `s`, the validity and the per-voxel weight, plus a BCE on `s`;
    deep supervision like ink (the direction is pooled strength-weighted and renormalised).
    Holdout metrics: `fiber/angle_deg` (+ median), `fiber/vt_auprc` / `fiber/hz_auprc` of the
    **derived** classes against the teacher, and -- teacher-independent -- `fiber/band_angle_deg`
    and `fiber/band_class_acc` on the human bands.  Inference writes `fiber_dz, fiber_dy,
    fiber_dx, fiber_strength` plus the derived `fiber_vt` / `fiber_hz`
    (`s * |d^.t_v|` vs `s * |d^.t_h|`, with the basis from the *predicted* `grad sdf_in`;
    `infer.write_fiber_class_channels`), so every downstream consumer keeps working.
    `dev/eval_region.py` gains `fiber.upstream` (the human-band class accuracy and, for a
    direction store, the axial angular error).
  `configs/paris4_faces_rvfw_dir.json` is the direction-mode run; `configs/ablate_fiber.json`
  compares the buggy baseline (`fiber_swap_fix: false`), `class_fixed` and `direction`.
- **Scroll-axis input channels (`extra.train.input_axis`, default false, 2026-09-07).**  Three
  more input channels `a_z, a_y, a_x` holding the scroll-axis unit direction in volume
  coordinates: constant `(1, 0, 0)` in the unaugmented frame and transformed as a **vector** by
  every spatial augmentation / TTA transform, exactly like the radial channels
  (`student.input_vec_slices`, `data.Augment(input_vec=...)`, `infer.tta_forward(vec=[...])`).
  The two-face labels survive an all-axes rotation because `input_radial` tells the net which way
  is out; the fibre classes have no such anchor without this channel.  `in_ch` becomes 8 (5 + 3)
  and is recorded in the checkpoint config and in the TensorRT engine cache key; a checkpoint
  without `input_axis` loads unchanged.

  **Backward compatibility:** the head is optional at both ends — `student.load_student_state` accepts a
  checkpoint whose `head.*` keys differ from the model's, printing which heads were left at their random
  initialisation or dropped, and raises on any mismatch outside the heads.  The EMA shadow is a flat
  parameter list, so it is *refused* (with a warning, falling back to the raw weights) when the head set
  changed rather than being zipped out of alignment.

- **Ink class imbalance (2026-09-03).** `loss_weights.ink` defaults to **2.0** (was 1.0) and the ink BCE takes a `pos_weight` from `extra.train.ink_pos_weight`: `"auto"` (default) uses this batch's soft target mass, `pos_weight = sum(1 - p) / sum(p)` over `ink_valid == 1` voxels, clamped to `[1, 50]` (`train.ink_pos_weight`, logged as `ink/pos_weight`); a number is used as given, `null`/`0` disables it. The soft-Dice term is unchanged, so the ink loss is BCE(pos_weight) + soft Dice.
- **CT normalization.** Intensity jitter (gamma [0.7, 1.4], brightness +-0.1 / contrast [0.8, 1.2], gaussian noise sigma <= 0.03, p = 0.5 each) is applied to the u8/255 crop; the per-crop z-score (`student.normalize_ct`, std floor 1e-3, air voxels included) is then applied on the device (`train.model_input`). Channel 1 is the constant `log2(voxel_um / 2.4)`; channels 2..4 are the radial field (above).
- **Winding normal target.** The decoded target normal is re-normalized to unit length before the cosine loss; the density Huber uses `10 * density` (wraps/voxel) with delta 1; phase / density / normal terms are weighted by `conf/255`, the conf logit is trained with BCE against `conf/255 > 0.5`. Channel 8 ("spare") receives no loss.
- **Soft Dice** (ink and the surface-band Dice on `exp(-|sdf|/2)`) uses the squared denominator `2<a,b> / (<a,a> + <b,b>)` so that identical soft maps give zero loss.
- **Deep supervision targets** at half resolution: continuous channels are mask-aware average pooled (only `valid == 1` voxels contribute), masks are nearest-subsampled (`[::2, ::2, ::2]`).

### Several label stores (`extra.train.stores`, 2026-09-07)
Train from several regions / scrolls at once.  `extra.train.stores` is a list of
**self-contained** store entries; when it is present it *replaces* `fine_store` /
`coarse_store` (giving both is an error, and every other `extra.train` key --- patch,
stride, `min_valid_frac`, `surface_mode`, `winding_source`, `heads`, `holdout_origins`,
augmentation --- still applies to every store):

```json
"stores": [
  {"name": "paris4",
   "fine_store":   "/home/forrest/tsm-output/slab_faces_rv/labels/fine.zarr",
   "coarse_store": "/home/forrest/tsm-output/slab_faces_tta/labels/coarse.zarr",
   "axis":         "/home/forrest/.cache/tsm-production/axes/PHercParis4/umbilicus-full-resolution.json",
   "voxel_um": 2.4},
  {"name": "s4",
   "fine_store":   "/home/forrest/tsm-output/s4/slab_faces_rv/labels/fine.zarr",
   "coarse_store": "/home/forrest/tsm-output/s4/slab_faces_rv/labels/coarse.zarr",
   "axis":         "/home/forrest/.cache/tsm-production/axes/PHerc1667/umbilicus-full-resolution.json",
   "voxel_um": 2.399,
   "volume": {"url": "https://.../PHerc1667/volumes/20251217075048-2.399um-...zarr", "voxel_um": 2.399}}
],
"holdout_stores": []
```

| key | meaning |
| --- | --- |
| `name` | unique label; used in the logs and in the per-store metric keys (default `store<i>`) |
| `fine_store` | required; surface / ink / fibre / `wf_*` targets |
| `coarse_store` | optional (null / missing = winding masks all zero for that store) |
| `axis` | umbilicus JSON path (or an inline `[[z,y,x], ...]` list) for the radial input channels of **this** store; default `extra.train.axis_path` |
| `voxel_um` | fine pitch; validated against the store attrs and against the store's volume |
| `weight` | sampling weight; default = that store's number of training origins (i.e. uniform over crops) |
| `volume` | optional CT volume object (same schema as top-level `volume`); default: the top-level one.  Cross-scroll training needs it --- the CT of another scroll lives at another URL |
| `region` | optional; only used to decide whether the local CT cache covers the store.  Default: the fine store's own extent (`origin_zyx`, `shape`) |

**Dataset.** One `CropDataset` per store, wrapped in `data.MultiStoreDataset`: each sample picks a
store with probability ∝ weight (deterministic in the seed, `store_index(i)`) and then delegates to
that store's dataset, which draws the crop.  The items are byte-identical to what the single store
would produce, so augmentation, losses and the model are unchanged; the single-store
`fine_store`/`coarse_store` path still builds a plain `CropDataset` and is unaffected.
Every store must expose the **same channel set** (identical fine channel lists, all-or-none coarse
stores with identical channels) and the same patch / surface mode / winding source / fibre and
input settings --- a mixed batch has to be one supervision contract, so a mismatch is an error
rather than a silently unsupervised head.

**Origins and holdout.** Origins are cached and split per store (`holdout_origins` applies to each
store separately) and carried as `(N, 4)` `[store_index, z, y, x]` in `ds.origins` /
`ds.holdout_origins` (`train/holdout.npy` gains that first column).  `holdout_stores: ["s4"]` holds
out a *whole* store: it trains on nothing (weight 0) and every one of its origins is evaluated ---
cross-scroll generalisation.  Holding out every store is an error.  The pooled holdout list is
shuffled with a seeded permutation so `eval_max_crops` stays spread over the stores.

**Evaluation.** `train.evaluate_holdout` reports the pooled metrics under exactly the usual keys
plus, per store, `<store>/<metric>` --- i.e. `holdout/<store>/<metric>` in `summary.json` /
`train/holdout_metrics.json`.  `dev/ablate.py` and the eval configs are unchanged (the ablation
runner leaves `stores` alone instead of pinning `fine_store`/`coarse_store`).

Shipped examples: `configs/multi_s1_s4.json` (Paris 4 slab + Scroll 4, both trained) and
`configs/multi_holdout_s4.json` (same, with Scroll 4 held out for cross-scroll evaluation).

### Orientation ambiguity (added 2026-09-02)
Coarse: where |n · r_hat| < `orient_min` (0.3) the outward orientation is ill-defined
(flattened windings near the umbilicus) -> coarse `valid` = 2. Fine: wherever a coarse
store exists but its upsampled normal is missing at the nearest medial voxel (i.e. the
coarse field was invalid/ambiguous there), the SDF sign would rest on the radial
fallback -> `sdf_valid` = 2.

## Student prediction store and exports (`tsm infer` / `tsm export`, milestone 8)

### `<out_dir>/student/pred.zarr` (2.4 um, region = config region)
Same container as the label stores (zarr v3 (C,Z,Y,X) uint8, chunks (1,128,128,128), attrs channels / voxel_um / origin_zyx / scale). The sliding engine (`tsm.sliding.run_sliding`) blends windows with gaussian weights and quantises `round(p*255)`, so `tsm.infer.StudentNet` maps every activated head into [0,1] (`to_unit`) such that the quantised byte *is* the label-store encoding; blending is therefore linear in sdf / sin / cos / normal / density and on probabilities for the logits ((sin,cos) and the normal may lose unit length in the blend -> consumers renormalise).
| ch | name | encoding |
|---|---|---|
| 0 | sdf | raw sdf head, clipped: u8 = round(128 + sdf*127/CLIP) in 1..255; 0 = no data (empty / uncovered tile) |
| 1 | valid | sigmoid(valid logit)*255 |
| 2 | ink | sigmoid(ink logit)*255 |
| 3,4 | sin, cos | (sin,cos) renormalised to the unit circle: 127.5 + 127.5 v |
| 5 | density | relu(density)*1000 (wraps per 2.4 um voxel), clipped 255 (training used the raw head, relu only keeps it >= 0) |
| 6..8 | nx, ny, nz | renormalised unit normal: 127.5 + 127.5 v |
| 9 | conf | sigmoid(conf logit)*255 |
| 10 | spare | sigmoid(spare)*255 (unsupervised; kept so the 11-channel head tensor is complete) |
| 11 | surface1 | 255 on the 1-voxel surface = data voxels with sdf <= 0 (byte 128 = exactly 0 counts) that have a 6-neighbour with sdf > 0, restricted to valid > 0.5 (`extract_surface`); 0 elsewhere |

**Fibre head** (a checkpoint trained with `extra.train.heads.fiber = true`; `tsm.infer.pred_channels(mode, fiber)`,
detected from the checkpoint config): two more head channels `fiber_vt, fiber_hz` = `sigmoid(logit)*255`,
inserted directly **after `spare`** and before the derived `surface1` / thickness channels, so the head
channels stay contiguous.  A checkpoint without the head writes no such channels and every other channel
index is unchanged.

**Two-face prediction store** (a checkpoint trained with `surface_mode = "faces"`; `tsm.infer.pred_channels("faces")`,
detected from the checkpoint config or forced with `extra.infer.surface_mode`): 12 head channels + 3 derived, i.e.
`sdf_in, sdf_out, valid, ink, sin, cos, density, nx, ny, nz, conf, spare, surface_in1, surface_out1, thickness`.
`sdf_in` / `sdf_out` use the `sdf` encoding; `surface_in1` / `surface_out1` are `extract_surface` on the matching
SDF channel (identical rule to `surface1`); `thickness` = `round(clip(sdf_in - sdf_out, 0, 255))` voxels, 0 where
either face has no data.  `pred.summary.json` carries `surface_mode` and per-face `surface1` stats plus a
`thickness` percentile block; previews are `preview_{sdf_in,sdf_out,faces,thickness,valid,ink,cos,normal}.png`
(`preview_faces.png` = the middle z slice with the in-face zero set in red and the out-face zero set in blue).
The lasagna export takes `pred_dt` from `extra.export.lasagna.pred_dt_channel`, which defaults to `sdf_in` for a
two-face store (and `sdf` otherwise); the manifest's `preprocess_params.pred_dt_channel` records the choice.

Inference: EMA weights from `<out_dir>/train/latest.pt` (or `extra.infer.checkpoint`), eval, bf16 autocast, channels_last, per-window z-score (`student.normalize_ct`, as trained), scale channel `log2(voxel_um/2.4)`. `pred.summary.json` carries the sliding summary, encodings, clip, `volume_shape_zyx` (used as the lasagna base shape), the checkpoint info and `surface1` stats (`labels.surface_stats` on a central sub-box: thickness, 6-degree histogram, 6- vs 26-components; the negative-side boundary of a tilted sheet is a staircase, i.e. 26-connected, not 6-connected). Previews: `preview_{sdf,valid,ink,cos,normal}.png` + side-by-side of the middle z slice (surface1 in red on the sdf panel). Resume: tiles whose 11 head channels are in `done.json` are skipped; `surface1` is recomputed.

Speed knobs (2026-09-03, docs/student_v2_plan.md A1/A5/A6):
- `extra.infer.backend`: `"torch"` (default, bf16 autocast) or `"trt"` — a TensorRT fp16 engine for the student (`tsm.trt.TRTStudent`, CUDA only). The engine takes the full `[B, in_ch, p, p, p]` input (in_ch = 5 with the radial channels, which are still computed torch-side by `StudentNet.radial` and concatenated before the call) and returns the raw 11- (or 12-, two-face) channel head tensor, split back into `{surface, ink, winding}`; activation / `to_unit` / TTA / blending are unchanged. fp16 only (TensorRT 11 has no bf16 tactic for the decoder's 3D ConvTranspose). Engines are cached per GPU like the teachers' under `~/.cache/tsm-models/trt/`, keyed `student_i<in>o<out>_w<widths>_bs<body_stride>_p<patch>_b<batch>_fp16_<gpu>_trt<ver>.plan`; `extra.infer.trt_precision` / `trt_workspace_gb` tune the build.
- `extra.infer.halo: "rf"`: receptive-field tiling instead of 50 % overlap — `halo = TSMNet.receptive_field_radius() + 2`, `step = patch - 2*halo`, and the blending weight becomes **uniform on the window core** (1 inside `[halo, patch-halo)`, 0 in the halo: `sliding.uniform_core_weight`, `WindowSpec.weight`), so window cores tile the volume and each core voxel is written by exactly one window (an unweighted mean where the last-window alignment makes cores overlap). Requires `2*halo < patch`; it is only *cheaper* than 50 % overlap once `patch > 4*halo`. Measured radii: 122 voxels at `body_stride=1`, 248 at `body_stride=2` — i.e. patch >= 256 to run at all and >= 512 to win. `extra.infer.weight` forces `"gaussian"` / `"uniform"` independently (e.g. an empirically chosen smaller halo).
- `extra.train.body_stride: 2` (+ `fullres_width`, default 32): stride-2 stem, the whole encoder–decoder at 4.8 µm, one full-resolution residual block on `[input, upsampled decoder output]` before the 1×1×1 heads. 20.0 M params (vs 19.97 M) and 216 GMAC per 128³ window vs 706 (3.26×). `extra.train.norm: "batch"` swaps GroupNorm for BatchNorm3d (foldable at export, but geometric TTA hurts BN models — ablation only).

### Un-aliased winding metric (`dev/eval_region.py`, `winding_fine` section, 2026-09-07)
When the fine label store carries the `wf_*` block the report gains a `winding_fine` section:
the student's phase / normal / density scored at the **fine** pitch (no 4x pooling) against
those channels, restricted to `wf_valid == 1`, alongside the existing coarse-referenced
`winding` section.  `wf_density` is already in wraps per fine voxel, like the student head, so
the density MAE is directly comparable.  It is the only winding number in the report that is
not aliased wherever the sheets are finer than the 9.6 um teacher can resolve.

### Teacher-independent face metric (`dev/eval_region.py --rectoverso`, 2026-09-06)
`--rectoverso <rectoverso.zarr>` adds an `upstream_faces` block: the student's `surface_in1` /
`surface_out1` zero sets scored against the **thinned** upstream recto / verso bands
(`tsm.rvfaces.thin_band`), restricted to voxels within 8 voxels of any band, with a Dice that
counts a hit within **2** voxels (the band registration tolerance) plus EDT surface distances
both ways.  It is the only metric in that report that does not involve the teachers the student
was distilled from.  First measurement (eval region, `slab_faces_tta2` student): in-face
Dice@2 = 0.403, out-face 0.339; symmetric median distance 4.1 / 4.5 voxels.

### Lasagna export (`<out_dir>/export/lasagna/<name>.lasagna.json`, `tsm.infer.export_lasagna`)
Layout of villa `preprocess_cos_omezarr.run_preprocess_3d` as if the lasagna model had run on the level-`level_shift` (default 2 = 9.6 um) volume (`input_sd = 2**level_shift`):
- `<name>_{cos,grad_mag,nx,ny,pred_dt}.ome.zarr`: zarr **v2** OME groups (`multiscales` 0.4, `/` chunk separator, chunks 32^3, uint8, `lasagna_pyramid_downsample: mean_pool2x`), 3-D level arrays spanning the whole base volume (`base_shape_zyx`, ceil-halved per level) and filled only over the prediction region; levels `first..n_levels-1` (n_levels >= other_level + 2), coarser levels 2x mean-pooled (normals: decode, average, hemisphere, re-encode).
- `cos` and `pred_dt` at OME level `level_shift + log2(cos_scaledown)` (default 3 = 19.2 um); `grad_mag`, `nx`, `ny` at `level_shift + log2(scaledown)` (default 4 = 38.4 um). Level index = floor(base index / 2**level), like villa's `_ds_index`.
- Bytes: `cos` = round(255*(0.5 + 0.5 cos)); `grad_mag` = round(1000 * wraps per input voxel) (fine density * 2**level_shift) with `grad_mag_factor = 1/2**level_shift` so villa's `load_3d` decodes `grad_mag/(1000/grad_mag_factor)` = wraps per base voxel; `nx`, `ny` = round(v*127 + 128) after flipping the normal to nz >= 0 (villa recovers nz = sqrt(1 - nx^2 - ny^2)); `pred_dt` (`encode_pred_dt`, distances in 2.4 um voxels): inside the sheet band |sdf| <= sheet_half_vox (2) -> 127 + clip(round(half - |sdf| + 1), 1, 48) in [128,175], outside -> 128 - clip(round(|sdf| - half), 1, 48) in [80,127], then mean-pooled to the cos level like villa's INTER_AREA; no data (< min_cover of the fine voxels valid) -> 0 (cos, grad_mag, pred_dt) / 128 (nx, ny).
- Pooling is mask-aware over `sdf != 0 & valid > 0.5` fine voxels; sin/cos and the normal are renormalised after averaging.
- Manifest keys as `LasagnaVolume.save` writes them: `version` 2, `source_to_base` 1.0, `grad_mag_encode_scale` 1000, `grad_mag_factor`, `groups` {cos, grad_mag, nx, ny, pred_dt: {zarr: "<name>_<ch>.ome.zarr/<level>", scaledown, channels}}, `crops` [[x,y,z,w,h,d]] (base voxels), `base_shape_zyx`, `umbilicus_json` (copy of the axis file, `<name>_umbilicus.json`, when an axis is configured); plus a `preprocess_params` block (`source: tsm`, scaledown, cos_scaledown, grad_mag_encode_scale, channels, crop_xyzwhd, encodings) that villa's loader ignores. `tests/test_infer.py::test_villa_load_3d_reads_export` (slow) loads the export with villa's `fit_data.load_3d`.

### Spiral export (`<out_dir>/export/spiral/`, `tsm.infer.export_spiral`)
- `winding.zarr`: label-store container at level `level_shift` (9.6 um), channels `sin, cos, density, nx, ny, nz, conf, valid` with the coarse-store encodings (density = round(1000 * wraps per level-2 voxel); `valid` = 1 where >= min_cover of the fine voxels were valid, all other channels 0 elsewhere); origin_zyx in level-2 voxels (level-2 voxel i sits at level-0 coordinate 4i + 1.5).
- `winding.json`: encodings, origin / shape / pitch, axis path, ray parameters.
- `rays.npz` (with an axis): outward in-plane radial rays from the umbilicus every `ray_spacing` level-2 z, every `ray_angle_deg`, samples every `ray_step` voxels up to `ray_length` (default: the far corner). `starts_zyx` / `centers_zyx` (ray midpoint, as `neural_winding_losses.sample_straight_ray_cache` expects) / `directions_zyx` [rays,3] unit, `spacing`, `points_zyx` [rays,samples,3], `phase` [rays,samples] (unwrapped wraps, first valid sample = 0, NaN where invalid), `phase_angle` (raw atan2(sin,cos)), `density`, `conf`, `valid`; trilinear samples of the decoded field, all coordinates in level-2 voxels.

## Augmentation v2 (`tsm.data.Augment`, `extra.train.augment`, 2026-09-02)
Applied on the device per collated batch, after the DataLoader and before the per-crop z-score
(`train.model_input`); written in torch, every transform truly 3D, each independently switchable
via `AugmentConfig` (`extra.train.augment` = `true` (= preset `"strong"`), a preset name
`"strong" | "strong_scan" | "light" | "none"`, a dict `{"preset": ..., "<transform>": {"p": ..., ...}}` (unknown keys
rejected), `"v1"` for the old CPU flips/rot90 + jitter path, or `false`).  The number of samples
each transform touched per optimizer step is logged as `aug/<name>` in `log.jsonl`.

**Spatial transforms** (flip per axis, rot90 (in-plane by default, `all_axes` for the 24 cube
rotations), small rotation about a random 3D axis (`in_plane` option; default ±15°), isotropic
(default) or anisotropic scaling 0.85–1.2, elastic deformation from a coarse random grid) are
composed into ONE sampling field per sample, `x_in = L^-1 x_out + e(x_out)` with
`L = Q F R S` (rot90 · flips · rotation · scaling) in crop-centred `(z, y, x)` voxel coordinates,
and applied with a single `grid_sample`.  Per-channel rules:

| channel | rule |
|---|---|
| ct | trilinear; samples outside the crop padded by `oob_fill` (reflection default, border, zeros) |
| sdf | mask-aware trilinear (only `sdf_valid == 1` neighbours contribute), × `det(S)^(1/3)` (distances scale with the zoom; exact for isotropic scaling, geometric-mean approximation for anisotropic), clamped to ±CLIP.  In `surface_mode = "faces"` this is a 2-channel field (`sdf_in`, `sdf_out`) and **both channels follow the identical rule**: a flip / rot90 / rotation never exchanges the two faces.  "In" and "out" are defined by the physical outward direction, which is carried by the geometry itself and moves with it under the transform -- unlike a vector channel there is nothing to negate, and unlike a left/right label there is no mirror ambiguity to fix: after a flip the sheet's in face is still the same set of voxels, now at flipped coordinates.  (The ±clip saturation rule below is applied per channel and a sample is marked ignore if *either* face saturated and was shrunk.) |
| sdf_valid / ink_valid / winding_valid / fiber_valid | nearest; 0 (no data) where the sample point falls outside the crop; sdf_valid additionally 2 where a label saturated at ±CLIP was shrunk below the clip by a zoom-out (its true distance is unknown) |
| ink, conf, fiber_vt, fiber_hz | mask-aware trilinear |
| sin, cos | mask-aware trilinear, then renormalised to the unit circle (the phase is a scalar invariant of the point) |
| density | mask-aware trilinear × `|L^-T n|` with `n` the interpolated target normal (= 1/s for isotropic scaling s: wraps per voxel shrink when voxels grow); fallback `det(S)^(-1/3)` where the normal is missing (norm < 0.5) |
| normal (nx, ny, nz) | mask-aware trilinear as a vector, then `normalise(L^-T n)`: the exact rotation part of the sampling matrix — a flip negates the flipped component, rot90 / arbitrary rotations rotate the vector, anisotropic scaling tilts it as a covector (normals transform with `L^-T`) |

Everything invalid after resampling (masks 0 / 2) has its continuous targets zeroed.  The
elastic field is small (default 2 vox std on a 4³ control grid) and its local Jacobian is
ignored for the sdf / density / normal rules.  Orientation conventions are unchanged from v1:
the sdf sign and the phase direction are data-defined scalars; "outward" is carried by the
normal channels.  `tests/test_augment.py` checks all of the above against analytic planar-sheet
fields (flips, in-plane and all-axes rot90, arbitrary rotations, iso/aniso scaling, combinations).

**Intensity transforms** (CT only, per sample, in this order): low-resolution simulation
(3D area downsample by a random factor 1–2, trilinear back — a coarser scan), gaussian blur
(separable 3D; with `axis_only_p` along one random axis only), unsharp-mask sharpening, gamma,
brightness/contrast, multiplicative gaussian noise, additive gaussian noise, ring/streak
artefacts (off by default; plane waves along a random 3D direction + sinusoidal shells about a
random 3D line), 3D cutout (boxes filled with a random constant).  Targets are never touched
by intensity transforms; the CT is *not* clamped afterwards (the z-score follows).

### Scan-domain intensity family (2026-09-07, preset `strong_scan`)
Six further CT-only transforms model what actually differs *between scans* rather than generic
photometric jitter.  All default to `p = 0`, so the `strong` preset is bit-identical to before
(`_fires` short-circuits on `p == 0` and draws nothing, so the generator stream is unchanged —
`tests/test_augment.py::test_strong_preset_is_unchanged_by_the_scan_family` pins a digest of
four strong batches computed with the pre-2026-09-07 code).  The preset `strong_scan` is
exactly `strong` plus this family; the ranges come from the PHercParis4 vs PHercParis3
comparison at 2.4 µm (`dev/scan_stats.py`, `~/tsm-output/scan_stats/report.md` — 24 random
256³ level-0 blocks per volume, same beamline/energy/voxel size/recon pipeline):

| measured (Paris 4 → Paris 3) | transform | what it does | ranges |
|---|---|---|---|
| 8-bit export window `[-0.04, 0.22]` vs `[-0.03, 0.19]` f32 (~±12 % of the window, from the volume metadata) | `window` (p 0.5) | affine re-map of the export window: `x' = clip((x − lo)/(hi − lo), 0, 1)` | `lo ∈ [−12, +12]`, `hi ∈ [220, 300]` grey levels, one draw per crop |
| papyrus std 20.1 vs 27.1 grey levels at nearly the same modes (air 42.5/41.5, papyrus 112.5/114.5) ⇒ contrast ratio 3.49 vs 2.70 | `class_contrast` (p 0.4) | rescales the spread of the **papyrus** mode only: `x' = m_p + (x − m_p)·s` above the air/papyrus midpoint, smoothstep-blended over ±10 grey levels, identity below (air untouched) | `s ∈ [0.7, 1.4]` (= 20.1/27.1 … 27.1/20.1); modes from 2-means (10 Lloyd iterations) on a 4096-voxel strided subsample of the crop, initialised at the measured 42 / 112 (`estimate: false` uses the guesses) |
| ACF 1/e length ratio z 0.80, y 0.88, x 0.99 (13.3/10.9/10.8 vs 10.6/9.6/10.7 vox) | `aniso_blur` (p 0.3) | separable gaussian with the z sigma drawn independently of the in-plane sigma (one draw shared by y and x); runs before the noise transforms | `σ_z ∈ [0, 1.5]`, `σ_yx ∈ [0, 1.0]` vox |
| PSD log-log slope (1/64…1/4) −4.26 vs −4.39, noise ratio (laplacian/local std) 0.67 vs 0.60 | `spectral_noise` (p 0.3) | coloured noise: white noise filtered by `|k|^(β/2)` in Fourier space (DC removed), added on top of the existing white `noise` | `σ ∈ [2, 8]` grey levels, `β ∈ [−1, 1]` (β<0 red, 0 white, >0 blue) |
| helical multi-sub-scan recon (nabu GHBP, 33–36 sub-scans) — ring/detector artefacts are scanner-specific | `ring` (p 0.2) | multiplicative concentric gaussian rings about a centre 4–20 crop widths outside the crop in the y-x plane, so they cross the crop as gently curved stripes, constant along z | amplitude ∈ [0.02, 0.08], 1–3 rings, ring width (σ) 2–8 vox |
| | `stripe` (p 0.05) | a detector line: 1–2 constant y or x planes of multiplicative gain, constant along z | amplitude ∈ [0.01, 0.05], width 1–3 vox |

Pipeline order inside `apply_intensity`: `lowres → blur → aniso_blur → sharpen → class_contrast
→ gamma/contrast/mult_noise/noise → spectral_noise → ring → stripe → window → artefact →
cutout` (the export window is last because that is the last thing the exporter does).  Unlike
the older intensity transforms, each of these clamps its own output to `[0, 1]` (= `[0, 255]`
in 8-bit units): a window or a gain is a physical re-export, not an unbounded jitter.  Targets
and the non-CT input channels are untouched, exactly as for the rest of the intensity family.
`configs/ablate_augment.json` sweeps `base` (= `strong`) / `none` / `geo_only` (every intensity
transform at p=0, geometry kept) / `strong_scan` / `strong_seed1` (`base` with
`extra.train.seed` 1, i.e. the seed-to-seed noise floor of the comparison) over 6000 steps.

**Holdout and evaluation.** `extra.train.holdout_origins` (`{"z_frac": f}`, `{"z_range": [lo, hi]}`
or an explicit list of local origins; `data.split_holdout`) removes crops from training (training
crops never overlap a holdout crop); the EMA model is scored on them at the end of training
(`train.evaluate` -> `train/holdout_metrics.json`): surface sdf MAE in the ±clip band,
zero-crossing Dice vs the label zero crossing, medial-surface distances (pred→label, label→pred,
symmetric: median / p90 / fraction > 3 vox, via EDT), valid AUPRC; ink AUPRC vs the soft target
(and, with the fibre head, `fiber/vt_auprc` / `fiber/hz_auprc` the same way)
> 0.5; winding circular phase error (deg), density MAE, normal angle error (deg), conf AUROC.
`tsm ablate` / `dev/ablate.py` (`tsm.ablate`) trains named `extra.train` overrides
(`{"no_rot": {"augment.rotate.p": 0}}`) for a fixed step budget and seed and tabulates these
metrics in `<out_dir>/ablation/ablation.md` + `.json`.

**Student TTA** (`extra.infer.tta` = `"none" | "flip8" | "flip8_rot4"`, `infer.StudentNet`):
each pass flips the input (all 8 axis-flip subsets; `flip8_rot4` adds the y↔x transpose, i.e.
the 16 distinct elements generated by flips and in-plane rot90s) and maps the raw heads back:
sdf, valid, ink, sin, cos, density, conf, spare and the fibre logits move with the grid
(scalars); the normal is a
vector (transpose swaps nx↔ny, a flip negates the matching component).  Passes are averaged in
the pre-activation space, then activated once.  The sliding engine's own flip-average is not
used for the student (it would average an un-negated normal).

### Orientation defaults (2026-09-02)
`strong` preset: rot90 over all 24 cube rotations (was in-plane), small random-axis
rotations ±30° p=0.5, and with p=0.2 a uniform SO(3) rotation (random unit quaternion).
Reason: the recto teacher was trained with single-plane rot90 and no 3D mirroring and
lasagna with z-only rot90 (measured orientation bias); the student must not inherit it.

## DINO token cache (`tsm dino`, `tsm.dino`, 2026-09-03)

`tsm dino <config>` runs the reimplemented dinovol_2 ViT (`tsm.dino.DinoVolViT`, 216M params,
patch 8, embed 864, depth 24, mixed RoPE, 4 register tokens) over the config region at level 0
in `extra.dino.window`³ (default 128³ = 16³ tokens; 256³ OOMs on 24 GB) with `overlap_tokens`
(default 2) tokens of overlap, and writes under `<out_dir>/dino/`:

| path | dtype | shape | notes |
| --- | --- | --- | --- |
| `tokens.zarr` | float16 | `(pca_dim, D/8, H/8, W/8)`, chunks `(C, 16, 64, 64)` | L2-normalised PCA-reduced patch tokens |
| `ink_likeness.zarr` | uint8 | `(1, D/8, H/8, W/8)`, chunks `(1, 16, 64, 64)` | `u8 = round((cos + 1) * 127.5)`, cosine to the expert reference embedding |
| `pca.npz` | float32 | `mean (864,)`, `components (pca_dim, 864)`, `explained`, `n_samples` | fitted on `pca_sample_windows` evenly spaced windows |

Attrs on both arrays: `origin_zyx` (level-0 voxel origin of the cached region), `scale` = 8
(voxels per token: `token = (voxel - origin_zyx) // 8`), `voxel_um`, `window`, `overlap_tokens`,
`l2_normalised`, `model`.

**Decisions (and why).**
* **Stored at the /8 token pitch, not upsampled ×8.** Both consumers work at /8 (the student's
  `enc3` stage for `feat_distill`, the ink agreement signal in `tsm labels`); ×8 nearest
  upsampling would cost 512× the bytes for zero information.  `scale` in the attrs is the
  contract.
* **Stitching.** Per window the patch tokens are L2-normalised, then blended with upstream's
  separable linear ramp weights (`_window_weight_grid`), then the stitched grid is L2-normalised
  again — identical to dinovol's `compute_patch_embedding_grid`.  Because the blend is a convex
  combination and the PCA is affine, the reduction is applied per window and the result is the
  same as reducing after the blend.
* **Ink-likeness is computed on the full 864-dim tokens** before the PCA reduction (the
  reference embedding lives in the 864-dim space).
* **Input normalisation** is dinovol's default `robust` scheme (the released config carries no
  `normalization_scheme` and no nnU-Net intensity properties): clip to the window's [p1, p99],
  then `(x - median) / (1.4826 * MAD)`.  Applied **per window**, matching upstream's per-view
  normalisation in the SSL dataset.  This means overlapping windows do *not* see identical
  statistics; the ramp blending absorbs the seam.
* **Memory.** The region is tiled into `block_windows`³-window blocks; only one block's 864-dim
  fp32 accumulator plus one window's activations are live at a time.

### `feat_distill` with `"dino"` (`extra.train.feat_distill.teachers`)

Adding `"dino"` to `feat_distill.teachers` adds a third term that runs **no network** in the
train loop: the tokens for the crop are read from `feat_distill.dino_cache` (default
`<out_dir>/dino/tokens.zarr`) at `origin_zyx / 8`, size `patch / 8`, the student's
`enc<dino_stage>` map (default stage 3, /8, 256 ch) is projected to the cache's channel count by
one 1×1×1 conv + GroupNorm, and the loss is `1 - cos` per voxel, averaged — the same shape of
term as the U-Net teachers, and the projector is trained and checkpointed with them.

**Augmentation rule.** The cached tokens are a stack of scalar fields (one per PCA channel), so
they resample with the crop like `sdf`.  But the ViT is not rotation-equivariant: a rotated crop
does not produce rotated tokens, so a resampled target would be wrong.  The term is therefore
applied **only when the sample's spatial transform is a signed permutation** (flips / rot90,
where the resampling is an exact `permute` + `flip` of the token grid, checked against
`Augment.apply_spatial` in the tests); samples with a small rotation, scaling or an elastic
field are dropped from this term for that step.  The kept fraction is logged as
`feat_dino_frac` (with `augment: strong`, rotate p=0.5 and scale/elastic on, expect roughly
a quarter of the samples to qualify).  `Augment.last_params` exposes the per-sample transforms
to the trainer for this.

## Teacher encoder feature cache (`tsm feats`, `tsm.feats`, 2026-09-03)

`tsm feats <config>` runs the frozen teacher **encoders** (no decoder) over the config region
at level 0 and writes, under `<out_dir>/feats/`:

| path | dtype | shape | notes |
| --- | --- | --- | --- |
| `<teacher>_s<stage>.zarr` | float16 | `(C, D/2^k, H/2^k, W/2^k)`, chunks `(C, 8, 32, 32)` | `C` = `pca_dim` (64) or the raw stage width |
| `pca_<teacher>_s<stage>.npz` | float32 | `mean (Craw,)`, `components (pca_dim, Craw)`, `explained`, `n_samples` | fitted on `pca_sample_windows` evenly spaced window cores |
| `progress.json` | — | — | completed `(teacher, block)` keys; the stage resumes from it |

Attrs: `origin_zyx` (level-0 origin of the cached region), `stride` = `scale` = `2**stage`
(`cell = (voxel - origin_zyx) // stride`), `channels`, `raw_channels`, `pca`, `teacher`,
`stage`, `voxel_um`, `window`, `border`, `normalizer`, `dtype`.

`extra.feats` (defaults): `teachers ["recto","ink"]`, `stages [3,4]`, `window 256`, `border 32`,
`pca_dim 64`, `pca_sample_windows 16`, `pca_vectors_per_window 4096`, `dtype "float16"`,
`block_windows 4`, `backend "torch"` (bf16 autocast on cuda), `models_dir`, `device`,
`limit_blocks`.  The stage index **is** the downsampling power for the published ResEnc U-Nets
(`encoder_strides` = `[1,2,4,8,16,…]`, `encoder_channels` = `[32,64,128,256,320,320,320]`), so
stage 3 = 256 ch at /8 and stage 4 = 320 ch at /16; both are verified against the loaded encoder
before anything is written.  On the 256×6144² slab: 1024 windows per teacher, 2.25 GiB per /8
stage, 0.28 GiB per /16 stage, 5.06 GiB total for recto+ink at `pca_dim 64`
(`tsm feats configs/paris4.json --dry-run` prints the table).

**Decisions (and why).**
* **Cores, not blended overlaps.** At /8 the encoder's receptive field is hundreds of voxels, so
  every feature cell near a window face is contaminated by that window's zero padding; blending
  contaminated values with a ramp (what `sliding.predict_box` does for *decoder* outputs) mixes
  the error instead of removing it.  Each 256³ window therefore contributes only its central
  192³ core (`border` = 32 level-0 voxels = 4 cells at /8, 2 at /16), windows step by 192, and
  the cores partition the region exactly once — one value per cell, no weights.  At the region's
  outer faces there is no neighbouring window to prefer, so the border is kept there
  (`feats.core_intervals` returns the exact partition, including the clamped last window).
* **Measured** (`tests/test_feats.py::test_window_cores_agree_with_a_single_pass`): with the norm
  layer removed, each window's core matches a single whole-region forward to ~2 % of the feature
  scale while the discarded ring is >10× worse.
* **The residual seam is InstanceNorm3d, not the halo.** The published encoders normalise over
  the whole window, so a window's features carry a per-channel affine offset that no border can
  remove (`test_instance_norm_is_the_residual_seam_not_the_halo`).  The consequence is that a
  cached feature is only defined relative to a window size: with `window: 256` the cache holds
  exactly what the teacher produces at *inference* (`tsm teacher` runs 256³ windows), whereas the
  live `feat_distill` path normalises over the 128³ training crop.  Set `window` to the training
  patch if you want the cached and live targets to coincide instead.
* **Per-window teacher normalisation** (`teachers.Normalizer`: `zscore_instance` for recto,
  `percentile_minmax` for ink) — the same `norm_scope="window"` rule inference uses.
* **PCA is a plain linear sketch** (`(x - mean) @ Vᵀ`, no per-voxel L2), fitted per teacher and
  stage on window cores; the distillation cosine is then taken in the sketch space.
  `pca_dim: null` stores the raw channels and makes the cached term numerically identical to the
  live one (measured < 1e-3 loss difference on the identity augmentation).
* **One teacher at a time**, blocks of `block_windows³` windows: the CT is read once per block
  (169 MiB for the slab at `block_windows: 4`) and each block is recorded in `progress.json`
  before the next starts, so an interrupted run resumes (`--force` restarts and rewrites).

### `feat_distill.source` — cached vs live teachers

`extra.train.feat_distill.source` is `"live"` (default, the encoder runs in the train loop) or
`"cached"` (the target is read from `feat_distill.feats_dir`, default `<out_dir>/feats`).
`feat_distill.sources` overrides it per teacher, so mixed configurations are allowed, e.g.
`{"teachers": ["recto","ink","dino"], "source": "cached", "sources": {"ink": "live"}}`.
The cached term uses the same 1×1×1 `FeatureProjector` and the same `1 - cos` loss as the live
one, and obeys the **same signed-permutation-only rule as the DINO source**: a conv encoder is
translation- but not rotation-equivariant, so a sample whose spatial transform contains a small
rotation, an anisotropic scale or an elastic field is dropped from the term (kept fraction logged
as `feat_<teacher>_frac`).  Intensity augmentation is likewise not matched by a clean-volume
target — that is the F.2 caveat, measured by `configs/ablate_feats.json`.
