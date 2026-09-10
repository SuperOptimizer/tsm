# `winding_model_9um` teacher — status (milestone 6, best-effort)

**Kill switch: `tsm.winding.WINDING_ENABLED = False`.**  The reimplementation in
`src/tsm/winding.py` reproduces the checkpoint's state-dict keys/shapes exactly
(178 tensors, 9,135,724 parameters, strict `load_state_dict`) and runs, but its
output could not be made self-consistent with the expected semantics on real
data within the time box.  Per the plan, the student's winding head is
supervised from lasagna `cos`/`grad_mag` + axis geometry instead.  Everything
below is kept so the work can be resumed if villa publishes the source.

## What is known (hard evidence)

Sources: tensor inventory + `architecture_config` in
`~/.cache/tsm-production/models/winding_model_9um/*/*.tsm-model/model.json`;
the raw checkpoint `~/.cache/tsm-models/winding_model_9um.pth` (keys `model`,
`optimizer`, `lr_scheduler`, `config`, `step=50000`, `model_ema`); villa's
consumer `scripts/spiral/neural_winding_losses.py`; the recovered Codex
reverse-engineering (reference only).

| Fact | Evidence |
|---|---|
| Encoder = 3 stages `conv3 → norm → act → conv3 → norm → act`, widths 32/64/96, input 2 ch | `stages.{s}.layers.{0,1,3,4}` shapes; slots 2/5 parameter-free |
| Trunk = 1×1 conv 96→192 + norm, 2 residual blocks (`conv1,norm1,conv2,norm2`), 6 axial blocks (`ray_norm`, `ray_attention{qkv,proj,relative_bias[6,257]}`, `transverse_norm`, `transverse_attention{..., relative_bias[6,31,31]}`, `mlp_norm`, `mlp 192→768→192`) | inventory |
| Decoder = `conv3(256→96)`,norm,act,`conv3`,norm,act; head = `1×1(128→32)`,norm,act,`1×1(32→4)` | inventory |
| Transverse plane is 128 → 16×16 tokens (3 stride-2 stages) | bias `31 = 2·16−1`; **AdamW `exp_avg_sq` is non-zero for all 31×31 bins** |
| Ray axis is *not* downsampled (384 tokens) | ray bias `257 = 2·128+1`; all bins have non-zero `exp_avg_sq`; the clamp bins (±128) have ~2000× larger second moments than interior bins, which needs many pairs at offset ≥128 (L=384 fits, L=192 does not) |
| Ray axis is the **last** spatial axis of the 3×3×3 kernels | per-axis kernel variation of `stages.0.layers.0`, `stem_blocks.0.conv1`, `decoder.0` is identical along axes 0,1 and distinct along axis 2 |
| Concat order `[trunk ‖ skip1]` and `[decoder ‖ skip0]` | input-channel weight norms of `decoder.0` jump at index 192 (3.4→5.6) and of `full_resolution_head.0` at 96 (0.39→0.17) |
| `qkv` is `[q; k; v]` blocks of 192 (not head-interleaved) | bias magnitudes per 192-block `[0.066, 0.071, 0.028]` |
| The 4 head channels are 4 copies of *one* quantity (2×2 transverse sub-pixel shuffle) | biases `-2.8391, -2.8402, -2.8389, -2.8401`, near-identical weight rows, `use_crossing_head=false`, `use_variance_head=false`, `full_resolution_head=true` |
| Output = winding-rate logit; `density = softplus(logit)` windings/voxel; `phase = cumsum` | `phase_initial_increment=0.04 = softplus(−3.2)`; trained bias `softplus(−2.84)=0.057`/voxel ≈ 1/17.6 voxels ≈ 170 µm sheet pitch at 9.6 µm; villa integrates `density_windings_per_wv` along polylines vs integer winding counts; losses `lambda_phase/density/span_delta/position_hinge/position_snap` |
| Input channel 1 = validity mask | first conv uses ch 1 with kernel sums 3× larger than ch 0 (a constant-offset channel); reference |
| Training data: 2.4 µm zarrs at `volume_scale=2` (9.6 µm), `ray_length=384`, `spacing=1.0`, `transverse_size=128`, `tile_size=64`, `sampling=trilinear` | `config` |

Unverifiable from shapes (guesses, all made switchable in `WindingArch`):
norm type (`layer` = per-position channel LayerNorm; also `groupN`,
`instance`), activation (`gelu`/`silu`/`relu`/`leaky`), residual-block order
(`residual_preact`), attention order (`transverse_first`), relative-bias sign
(`bias_sign`), upsample mode, per-stage ray strides (`ray_strides`), internal
layout (`ray_first`), input normalisation (`prepare_input` modes).

## What was tried

Harness: 4 real 128×128×384 tiles from PHercParis4 level 2 around
`z=8544` (axis at x≈4740, y≈3406) plus synthetic tiles of perfectly periodic
sheets (period P ∈ {8,12,16,24,32}, and a P=10→20 step).  Metric: windings the
model accumulates per sheet period (`mean density × P`; ideal 1.0 for every
P), and on real tiles `windings_per_ct_period` (mean density × dominant CT
period per 96-sample window; ideal 1.0).

| Variant | synthetic windings/period for P=8,12,16,24,32 | real: median / frac within ±0.3 |
|---|---|---|
| **default** (ray-last, strides (2,2,1)×3, LayerNorm, GELU, z-score, ch1=1) | 0.65 0.65 0.65 0.83 1.93 | 1.28 / 0.45 |
| transverse attention first | 0.53 0.70 0.88 1.40 1.97 | 1.01 / 0.55 |
| bias sign −1 | 0.54 0.76 0.59 0.91 1.86 | 1.62 / 0.19 |
| pre-activation residual blocks | 0.52 0.75 1.00 1.50 2.00 | 1.45 / 0.26 |
| ray-first layout | 0.53 0.76 1.05 1.89 2.53 | 2.18 / 0.13 |
| GroupNorm(1) / (8) / (16) / (32) | 0.75 0.54 0.44 0.62 0.84 … | 0.48–1.54 / 0.13–0.47 |
| SiLU / ReLU / LeakyReLU | 0.68 0.69 0.76 0.90 1.00 / … | 1.41 / 0.34 (ReLU/Leaky: output ~constant) |
| ray strides (1,2,1) (1,1,2) (2,1,1) (2,2,2) … | e.g. (2,1,1): 0.65 0.99 1.04 1.10 1.22 | 1.8–2.7 / ≤0.15 |
| input: `x/255`, `(x/255−0.5)/0.25`, z-score×{0.5,2}, ±1 offset, swapped channels, ch1 ∈ {0, 0.5, −1, ramp, r/1000}, reversed ray | best `(x/255−0.5)/0.25`: 0.70 1.07 1.26 1.22 1.29 | 1.10–1.86 / 0.15–0.51 |

No variant makes the winding rate track the sheet spacing (a correct model
must give ≈1.0 at every P and a 2× density step across the P=10→20 boundary;
the best variants change by ≤1.5× while the spacing changes 4×).  The default
configuration does produce sheet-following band structure in the density map
and a mean rate (0.057–0.075 windings/voxel) equal to the reciprocal of the
measured median CT sheet spacing (13–17 voxels), i.e. the *prior* is right and
the network reacts to the input, but the reaction is far weaker than the
training objective implies.  Most likely one of the unverifiable conventions
(norm type, block internals, attention details) is wrong in a way that keeps
the network in a near-prior regime; distinguishing them would need either the
source or paired input/output samples from villa.

## Real-ROI validation (`validate()`)

`uv run python -m tsm.winding --n-rays 8` (S3 level 2, z=8544, rays from
r=150 to r=2000 level-2 voxels, tiles of 384 stitched with a trapezoid
blend).  Gate: finite; phase non-decreasing on ≥80 % of rays;
`windings_per_ct_period` within ±0.3 of 1 on ≥60 % of periodic windows and
median within ±0.2.  Diagnostics: `/home/forrest/tsm-output/v1/winding/`
(`winding_ray{k}.png`: CT / density / phase mod 1 along each ray with CT
sheet peaks as green bars; `winding_tile{k}.png`: CT slices with density
overlay; `winding_validation.json`).  Result of the final 8-ray run:
**FAILED** — finite: yes; monotone: 8/8 rays; `windings_per_ct_period`
median 0.95 but only 42 % of windows within ±0.3 (per ray 19–69 %; gate
≥60 %); phase step per CT sheet peak median 0.76, 38 % within ±0.3; density
peaks within 2 voxels of a CT peak 38 %; mean density 0.054 windings/voxel vs
median CT sheet spacing 13 voxels (1/13 = 0.077).  The prior is right, the
per-window rate is not.

## What is usable now

* `WindingNet` / `infer_winding_arch` / `build_winding_net` /
  `load_winding_net`: exact key parity, strict load, forward on
  `[B, 2, 128, 128, 384]` in 0.45 s (bf16, RTX 5080, 2.5 GB).
* `Axis` (render3d JSON, per-z interpolation, level scaling), `ray_frame`,
  `sample_ray_tile` (trilinear `grid_sample`, validity mask), `ArrayReader`,
  `WindingTeacher.predict_tile/predict_ray`, `validate`, `ray_metrics`,
  `windings_per_ct_period` — all tested in `tests/test_winding.py`
  (synthetic cylinder volume, parity vs inventory, tiny forward, slow real
  load).

## Next steps if resumed

1. Obtain villa's `winding_model` source (the training repo is not in
   `volume-cartographer`; `neural_winding_losses.py` says "contains no H2
   inference code") or one saved `(input tile, output)` pair; a single pair
   pins every convention above.
2. Failing that, brute-force the remaining discrete conventions jointly
   (norm × act × block order × attention order × shuffle order ≈ 200 combos,
   ~1 s each on the cached tiles in the scratch harness) against the
   synthetic-period test; the correct one should give ≈1.0 for all P.
