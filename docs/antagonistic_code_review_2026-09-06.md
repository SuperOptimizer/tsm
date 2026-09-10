# TSM antagonistic code review

Date: 2026-09-06. Scope: the complete local TSM source tree, tests, configuration examples, documentation, and development utilities. This is a review document only; no implementation or test changes were made.

## Assessment

The repository has substantial implementation and test coverage, but several silent correctness failures remain in cache reuse, spatial transformations, and evaluation. The highest priorities are preventing stale or incomplete artifacts from being treated as valid, repairing the radial TTA transform, correcting feature stitching and training/validation separation, and removing whole-volume allocations from face statistics. These should precede expensive training runs or decisions based on small differences in evaluation scores.

The current test suite is green: **452 default tests passed in 147.14 seconds; all 3 additional slow tests passed in 8.47 seconds**. Passing tests do not resolve the findings below: several adversarial probes exercise missing cases, and two existing tests reproduce the implementation's faulty assumptions in their expected results.

This review does not establish that existing trained checkpoints are unusable. Each finding identifies its trigger; many supplied configurations avoid particular edge cases. No production datasets, cached models, or output directories were modified to investigate them.

## Method, evidence, and coverage

Four agents participated. The primary reviewer examined configuration, CLI, volume I/O, cache/resume behavior, resource guards, packaging, and integration. Three independent reviewers examined training/data/model/equivariance, inference/teachers/DINO/TensorRT/features, and labels/winding/evaluation utilities. Reviewers sought concrete counterexamples, checked existing tests, and exchanged findings for challenge. The primary reviewer reproduced storage failures, inspected reported code paths, consolidated duplicates, and rejected or narrowed unsupported claims.

Evidence labels used below:

- **Reproduced:** a small executable probe demonstrated the issue, using CPU/in-memory data or temporary stores. Temporary diagnostic fixtures are not code changes.
- **Source-confirmed:** the relevant control flow establishes the failure, but a complete affected production workflow was not executed.
- **Risk / profiling target:** a plausible limitation or expensive operation whose production impact needs measurement; not presented as a measured regression.

Priorities: **P1** = fix before relying on the affected large run or result; **P2** = substantive correctness/robustness defect with a narrower trigger; **P3** = diagnostic accuracy, clarity, or lower-priority operational improvement. These are review priorities, not claims that every issue affects every configuration.

| Area | Coverage |
|---|---|
| Package and entry points | All 19 `src/tsm/*.py` files; CLI dispatch and packaging metadata |
| Training and architecture | Losses, sampling, splits, augmentation, checkpoint/EMA behavior, feature supervision, ablations, evaluation |
| Teachers and inference | Architecture loading, normalizers, activations, sliding windows, TTA, DINO attention/RoPE/PCA, feature caching, export, TensorRT cache/stream logic |
| Geometry and labels | Medial/face construction, masks, SDF encoding, normal/radial conventions, winding diagnostics |
| Development utilities | All 14 `dev/*.py` files inspected by the reviewers; deeper checks on evaluation, orientation/TTA tools, student benchmarks, and synthetic overfit driver |
| Tests | All 455 collected tests executed across default and slow selections; related tests inspected against each major finding |
| Configuration/docs | All 23 JSON configurations passed `load_config`; README and three existing design/status/schema documents inspected |

Additional checks: `uv pip check` reported all 102 installed packages compatible. CUDA was available with one device; the environment reported Torch 2.14.0, NumPy through the installed dependency set, Zarr 3.3.0, and Python 3.12.14. The slow tests covered real recto loading/forward, real winding loading/forward, and external export-reader compatibility. No new long training, real-volume throughput benchmark, TensorRT engine comparison, or upstream parity campaign was run. The default suite emitted one multiprocessing-fork warning and three ONNX symbolic-axis warnings.

The working tree was already largely untracked: only `.gitignore`, `LICENSE`, and `README.md` were tracked. Findings refer to local file contents and line numbers, not a committed revision. The source/test/dev Python inventory totals approximately 24,061 lines. No claim of exhaustive proof of correctness is implied by reading this inventory.

## Priority findings

### R01 — P1: TensorRT cache can silently run a different checkpoint

**Source-confirmed.** `src/tsm/trt.py:646`, `:737`, `:741`; teacher equivalent `:581`.

Student ONNX/engine identity includes channel counts, widths, body stride, patch, batch, precision, and platform details, but not weights or the complete architecture configuration. Two trained checkpoints with the same listed dimensions map to the same cache. After exporting checkpoint A, requesting B reuses A's engine. Deleting only the engine plan is insufficient because the ONNX export is also reused. Normalization mode and full-resolution width can also escape the identity. Teacher weight changes and INT8 calibration changes have analogous provenance omissions.

Inference `--force` refreshes predictions; it does not invalidate this shared engine cache. Immutable teacher releases reduce the teacher exposure, but do not protect a student that is continually retrained.

**Recommendation:** bind both ONNX and engine artifacts to a model-state and complete architecture/export fingerprint; include calibration provenance for quantized exports. Test two same-shape models with deliberately different outputs, including a missing-plan/existing-ONNX case. Existing synthetic export validity tests do not establish checkpoint identity.

### R02 — P1: Unfinished CT caches are accepted as complete input

**Reproduced.** `src/tsm/volume.py:527`, especially `:545` through `:555`; selection in `src/tsm/cli.py:166`.

`open_cached_reader` checks that an array opens and that its declared footprint covers the requested region. It does not verify completion. Cache arrays are created at their final full shape before all bricks are written, so an interrupted build is indistinguishable here from a complete cache. Unwritten chunks read as zero. Training or inference then receives fake air with a logged cache HIT.

Probe: create a 128³ cache with just its first 64³ brick populated with value 7 and a matching one-brick progress sidecar. `open_cached_reader` accepts it; reading voxel `(100,100,100)` returns 0 although the source returns 7.

**Recommendation:** publish a validated completion manifest after the required footprint is committed, or validate requested coverage and use a source-backed reader for missing pieces. Missing physical chunks alone cannot distinguish legitimate all-fill chunks from unfinished work; completion requires explicit provenance.

### R03 — P1: Partial chunk existence incorrectly marks unwritten bricks complete

**Reproduced.** `src/tsm/cache.py:153`, `:237`.

The resume fallback treats the existence of every backing chunk file as proof that a brick was written. This is false when multiple configured bricks share a 128³ storage chunk. The code accepts a 64³ brick with only a warning. Writing one 64³ sub-brick creates the shared chunk file, with zero fill in its unwritten portions. On restart, the fallback marks every other sub-brick in that chunk complete, even when `done.json` lists only the first brick.

Probe using the R02 fixture: `run_cache` printed `8/8 bricks already cached`, wrote zero bricks, and left the far source-7 voxel at 0. The fallback is applied even when a progress sidecar exists, not solely to genuinely old caches without one. `tests/test_cache.py` uses 64³ bricks but completes the first build before exercising resume, missing this interruption case.

**Recommendation:** completion must track written extents. Restrict the physical-chunk fallback to cases where complete chunk coverage is provable, or remove it for shared/partial chunk writes. Test interruption after exactly one sub-chunk brick and a changed brick size.

### R04 — P1: Existing output/cache arrays are reused with incompatible metadata

**Reproduced.** `src/tsm/volume.py:330`; `src/tsm/cache.py:223`; `src/tsm/volume.py:423`.

`BrickWriter` reopens an existing array without checking shape, origin, channel names/order, dtype, pitch, or scale against its new arguments. Its Python attributes then describe the new request while the underlying array and attributes describe the previous one. Its completion record also describes only origin/channel completion, with no brick extent or generating-run identity.

Probe: reopen a one-channel 8³ array as a differently named channel, 16³ shape, and origin `(100,0,0)`. No exception is raised; the object advertises the new geometry while the persisted store remains `(1,8,8,8)` at origin zero. Old completion keys remain accepted. Reusing an output directory after changing the model, region, channels, or tile size can therefore skip stale data or combine incompatible results.

The CT builder checks only origin when reusing a same-origin cache. A second probe grew a cache from `(128,128,128)` to `(256,128,128)`: the builder reported writing the new brick and updated `size_zyx` to the larger shape, but the actual array remained 128³. Zarr silently discarded the entirely out-of-bounds assignment in this environment.

**Recommendation:** validate structural metadata and generating-run identity before resume, including brick extents. Choose explicit rebuild/resize/new-artifact behavior for changed geometry and semantics. Validate stored shape independently of descriptive attributes. This is distinct from R01's engine cache and R02's incomplete cache publication.

### R05 — P1: Radial-input TTA applies vector signs in the wrong coordinate frame

**Reproduced.** `src/tsm/infer.py:291`; incorrect oracle in `tests/test_radial.py:178`.

The forward spatial transform flips axes and then transposes Y/X. Vector components instead undergo the transpose before receiving signs indexed in the original frame. These operations do not commute.

For a radial vector in `(z,y,x)` order equal to `(1,2,3)`, flip Y followed by transpose Y/X must produce `(1,3,-2)`. `tta_forward(..., axes=(1,), transpose=True)` produces `(1,-3,2)`. Four of the sixteen `flip8_rot4` transforms are affected: a transpose with exactly one of Y/X flipped, with or without a Z flip. Plain `flip8` is unaffected. The output inverse ordering does not have this same error.

The radial test reconstructs the same faulty sign/permutation order, so passing it is not independent evidence.

**Recommendation:** apply component signs before the forward permutation, or permute signs together with components. Test against a separately constructed signed transformation matrix and a model that exposes the transformed radial input.

### R06 — P1: DINO block stitching overwrites overlap instead of combining it

**Reproduced with a small deterministic model/loop oracle.** `src/tsm/dino.py:877` through `:895`.

Blocks partition window starts, but output spans overlap. Each block normalizes its own accumulator and writes the entire span; subsequent blocks overwrite preceding contributions in shared regions. The final token features and ink-likeness values therefore depend on `block_windows`, a memory/processing knob, rather than incorporating all overlapping windows.

CPU probe: 32³ input, 16-voxel windows, one-token overlap, one window per block, and context-dependent deterministic tokens. Maximum difference from one shared accumulator was `0.033947758` before PCA. Attention and per-window normalization naturally make real features context-dependent, so identical overlapping predictions cannot be assumed.

Tests establish external shared-accumulator behavior and end-to-end shape/finiteness, but not equivalence across block partitions.

**Recommendation:** assign disjoint output cores and gather every contributing window for each core, or persist numerator/denominator contributions and normalize after accumulation. Test several block sizes and traversal orders against a full-accumulator reference.

### R07 — P1: Face statistics expand a histogram into a whole-volume allocation

**Source-confirmed; arithmetic checked without allocating the large array.** `src/tsm/labels.py:1988`, `:1997`; `src/tsm/infer.py:544`.

Both paths use `np.repeat(np.arange(256), histogram)`, producing an int64 entry for every counted voxel, including background, before selecting positive thicknesses. This reverses the memory savings of brickwise processing at the end of a run. Label quantiles also repeatedly construct positive masks/subsets.

A fresh `(256,6144,6144)` face-label run, matching the supplied Paris4 face configurations, creates a **72 GiB** repeat array alone; these configurations set an **8 GiB** RAM budget. A 1024³ inference result needs 8 GiB for the repeat before masks/subsets. Even an all-zero histogram population expands the background counts. On resumed labeling, the histogram covers processed bricks rather than necessarily the entire region; that narrows the allocation but introduces the separate reporting issue noted later.

**Recommendation:** derive exact positive-value quantiles from the 256-bin cumulative histogram, preserving percentile interpolation semantics. Test counts representing billions of voxels while keeping actual memory constant. The memory watchdog may terminate the process; it does not make the allocation safe.

### R08 — P1: Interior holdout ranges leak training voxels into validation

**Reproduced.** `src/tsm/data.py:272`; faulty assertion in `tests/test_augment.py:324`.

The range split selects held-out crop origins in `[lo,hi)` but permits training origins at `hi`, even if the last held-out crop extends beyond `hi`. With patch 32, stride 16, and `z_range=[32,64]`, held-out origin 48 covers Z `[48,80)` and training origin 64 covers `[64,96)`: a 16-voxel overlap.

This violates the no-overlap premise of held-out evaluation/ablation. The test explicitly permits the faulty predicate. Top-fraction splits avoid this particular upper-bound problem because the held range reaches the final origins; explicit-list exclusion is a different path.

**Recommendation:** exclude training boxes overlapping the actual held-out crop extents on both sides. Verify pairwise box nonintersection for interior ranges on all three axes, including anisotropic origin arrangements.

### R09 — P1: Legacy spatial augmentation misaligns cached feature supervision

**Source-confirmed by two reviewers.** `src/tsm/train.py:925`, `:933`, `:969`; `src/tsm/data.py:692`; `src/tsm/feats.py:263`; `src/tsm/dino.py:715`.

With `augment="v1"`, the CPU dataset flips/rotates the sample but does not return the spatial transform. The training-loop augmentation object remains `None`, and feature distillation receives `spatial=None`. Cached teacher/DINO features are sampled in the original frame while the student sees the transformed frame.

This applies when cached features and legacy spatial augmentation are combined. Live teachers see the same transformed CT; v2 supplies transform metadata and avoids this specific mismatch.

**Recommendation:** propagate the legacy signed permutation to feature sampling or reject the unsupported combination. Test a coordinate-coded feature cache against a forced nonidentity legacy augmentation. Testing feature sampling and augmentation separately does not cover their integration.

### R10 — P1: Feature-cache progress can skip a newly requested feature stage

**Source-confirmed.** `src/tsm/feats.py:402`, `:443`, `:480`; existing-array handling at `:327`.

Completion is keyed by teacher/block index rather than teacher/stage/run identity. After completing a stage-3 cache, add stage 4 using the same output directory. The code creates the stage-4 array, then sees the teacher's blocks as done and skips them, leaving zero feature targets while completing normally. Removing a feature store while keeping progress creates a similar failure.

Geometry, block size, teacher state, normalization, and PCA changes also need validation. Merely recording some settings in progress metadata is insufficient if they are never compared on resume. Existing resume tests repeat an unchanged configuration.

**Recommendation:** validate a full run manifest and track completion per required stage/output. Test adding a stage, deleting one output, and changing model or PCA identity after a completed cache.

### R11 — P1: A declared installation is missing the required SciPy dependency

**Source/installed-metadata confirmed.** `pyproject.toml:9`; imports in `src/tsm/labels.py:33`, `src/tsm/equivariance.py:36`, and `src/tsm/train.py:1045`.

SciPy is imported unconditionally by core label/equivariance code, and training imports label code. It is absent from the project's dependencies and lockfile. The installed project's metadata confirms that omission; `edt` depends on NumPy, not SciPy. The current enriched environment has enough packages for the suite, so `uv pip check` passes, but it cannot detect imports that the project failed to declare. A clean install from the declared dependency closure cannot run these core paths.

**Recommendation:** declare SciPy as a runtime dependency and verify an isolated install of the built package. Keep optional visualization/export packages in explicit extras with actionable errors. The dev test `test_is_transient_classification` also imports optional `botocore` unconditionally, so a clean dev-only installation needs either the S3 extra for that test or an optional-dependency-aware test.

## Additional correctness findings

### R12 — P2: BrickWriter's bounds check compares tuples lexicographically

**Reproduced.** `src/tsm/volume.py:390`.

`(z_end,y_end,x_end) > shape` compares the first differing coordinate, not each axis. A small Z extent can hide a Y or X overflow. Probe: an 8³ writer accepted a `(1,2,1)` array at `(0,7,0)`, persisted only one of its two ones, and marked the brick complete. The existing bounds test exceeds Z, which the lexicographic check catches.

**Recommendation:** reject any per-axis overflow and validate the channel index. Test independent Y/X overflow, wholly out-of-range writes, and confirmation that rejected writes never update progress.

### R13 — P2: Uniform window weights allow internal holes

**Reproduced.** `src/tsm/sliding.py:133`, `:178`, `:579`.

Uniform weights are zero in the halo. Validation checks `2*halo < patch` but not that `step <= patch-2*halo`. A valid `WindowSpec(patch=8,step=8,halo=1,weight="uniform")` leaves uncovered output voxels. An everywhere-one model on a nonempty 16³ box produced **1,016 zeros in a 2,744-voxel cropped core**; the remaining bytes were 255.

Automatically derived RF-halo settings select a compatible step; explicit supported settings can violate coverage. **Recommendation:** validate effective core spacing and assert nonzero accumulated weight over every output core. Use a constant-output oracle.

### R14 — P2: Crop-origin cache ignores its selection criteria

**Cache-key behavior reproduced; integration source-confirmed.** `src/tsm/data.py:210`.

The default filename includes store basename, patch, and stride, but omits `valid_channel`, `min_frac`, and label-store generation. Changing the valid-fraction threshold or switching medial to face validity reuses the previous origins. Rebuilding labels in place also preserves a stale selection. Tests avoid one collision by assigning a different explicit cache name instead of checking invalidation.

**Recommendation:** include semantic selection settings and store identity in the cache manifest. Test that a changed threshold, validity channel, or label generation actually changes the sampled origins. A lower-level `force`/custom-name escape hatch does not make default training behavior correct.

### R15 — P2: Average precision depends on voxel order when scores tie

**Reproduced.** `src/tsm/train.py:1036`; `src/tsm/equivariance.py:415`.

Both functions sort scores stably and score each positive individually instead of treating equal scores as one threshold. Four identical scores produce AP `1.0` for labels `[1,1,0,0]` and `0.4166667` for `[0,0,1,1]`; threshold-based AP is `0.5` in both cases. Quantized probability stores and low-precision outputs make ties routine. Evaluation results and model rankings can therefore depend on traversal order.

**Recommendation:** consolidate AP into one implementation with grouped thresholds. Test tied-score permutations, perfect/reversed rankings, and documented empty/single-class behavior. Unique-score tests miss this defect.

### R16 — P2: Completely missed surfaces vanish from distance summaries

**Reproduced.** `src/tsm/train.py:1098`.

If either surface is empty, `_surface_distances` returns two empty arrays. With a nonempty target and no prediction, target-to-prediction misses should contribute a failure measure, but they disappear when distances are concatenated across crops. A model finding one crop perfectly and missing others can retain perfect-looking distance percentiles. Dice and counts still penalize misses, so the defect is specific to the distance summaries.

**Recommendation:** use a documented infinite/censored distance or explicit missing-surface counts and gate them in acceptance. The equivariance implementation already treats missing counterparts differently. Test a mixed evaluation containing one correct crop and one total miss.

### R17 — P2: BatchNorm state updates twice under activation checkpointing

**Reproduced.** `src/tsm/student.py:248`.

Checkpoint recomputation re-executes BatchNorm in training mode, updating its running statistics a second time. A tiny batch-normalized network with `act_ckpt=1` produced `num_batches_tracked=2` in checkpointed stem/encoder layers after one forward/backward; `act_ckpt=0` produced 1. A memory setting therefore changes evaluation-state behavior and confounds BatchNorm architecture comparisons. BatchNorm is a supported/shipped ablation option; GroupNorm has no such running buffers.

**Recommendation:** prevent recomputation from changing BN buffers, or explicitly disallow/calibrate this combination. Test parameter gradients and running state against a noncheckpointed reference. EMA's separate handling of buffers is discussed as a calibration consideration, not asserted as an additional proven defect.

### R18 — P2: Nonaligned fine crops lose the coarse-grid remainder

**Reproduced with a coarse ramp.** `src/tsm/data.py:525`, `:611`.

The coarse lookup floors the global fine origin by the scale factor, then crops interpolation at a fixed margin. The discarded remainder is not restored. With factor 4, fine origins X=8 and X=9 produced identical winding targets although the CT crop moved one voxel. This shifts targets by up to factor minus one fine voxels for custom origins, nonaligned region offsets, or some centered thin-slab origins.

Standard aligned configurations avoid it. **Recommendation:** include the remainder in interpolation/cropping and read sufficient surrounding data, or reject unsupported origins explicitly. Test coordinate ramps across all four residues of each axis.

### R19 — P2: Validity evaluation removes every ordinary negative label

**Reproduced.** `dev/eval_region.py:265`.

The selector promises to retain validity classes 0/1 but also requires encoded SDF to be nonzero. Label construction uses SDF byte zero for validity-zero/no-data voxels. The extra condition therefore eliminates the negative class; a probe with 640 validity-zero and 960 validity-one voxels retained only 960 positives, and validity AP was undefined even for an all-zero validity predictor.

Some outside-volume voxels may legitimately need exclusion, but SDF availability cannot also define the negative class for this metric. **Recommendation:** define an independent evaluation-domain mask, then retain its positive and negative validity labels. Test a mixed-class label store through `voxel_pass`, not only the AP helper.

### R20 — P2: Equal per-brick quotas bias purported voxel-level metrics

**Reproduced for the development reservoir; source-confirmed for training.** `dev/eval_region.py:157`; `src/tsm/train.py:1079`.

These samplers assign equal quotas to bricks/batches, not equal inclusion probability to eligible voxels. A brick with 1,000 negative eligible voxels and another with 10 positive voxels can yield a 50% positive sample under a small equal quota although prevalence is 0.99%. AP and AUROC then measure a reweighted population. The training quota is additionally divided by number of crops while `add` receives a batch, so evaluation batch size changes the effective sample count.

Equal block weighting could be a deliberate metric, but it is not the claimed reservoir/pooled voxel population and requires an explicit definition. **Recommendation:** use a real stream reservoir or sampling weights/inclusion probabilities, and test invariance to brick/batch partitioning using unequal occupancy and nonperfect scores.

### R21 — P2: Clipped face SDFs can place the evaluation medial surface incorrectly

**Reproduced.** `dev/eval_region.py:375`, `:388`.

The zero crossing of half the sum of two face SDFs identifies the midpoint only when their useful distances have not saturated. For parallel faces at X=10 and X=70 with clip 20, the true midpoint is 40; clipped opposite SDFs create a zero plateau and the helper returns X=50. It then measures a reconstruction artifact as a student medial error.

**Recommendation:** exclude ambiguous doubly saturated regions and report coverage, or reconstruct geometry from face locations rather than assuming the clipped sum remains a distance field. Test thicknesses below, at, and above twice the clip. Thin sheets within the represented distance range avoid this specific ambiguity.

### R22 — P2: Orientation/TTA diagnostic caches can mix unrelated experiments

**Source-confirmed.** `dev/orientation_bias.py:400`, `:425`; `dev/tta_side.py:413`; related student benchmark cache at `dev/bench_student.py:100`, `:244`.

Prediction reuse is based on transform names/files, without validating all source volume, region/offset, model checkpoint, pitch, and transform settings. Reusing an output location for a different experiment can load old predictions while assembling a report for the new box; partial caches can mix models. Reading `run_info.json` does not validate it against the new request. Student benchmark CT/reference caches have related missing provenance.

**Recommendation:** persist and compare an experiment manifest before reuse, including exact transform matrices. Existing recompute flags or fresh output directories are workarounds, not automatic protection. Verify rejection/recompute after changing one input property.

### R23 — P2: Arbitrary-rotation side metrics ignore their valid-support mask

**Source-confirmed.** `dev/tta_side.py:411`, `:460`.

The code constructs a rotation-valid mask, then passes no mask to equivariance metrics and does not apply it to side metrics. Interpolation/padding outside the inverse-rotated support can be scored as model orientation bias. Exact signed permutations do not have this issue. A sufficiently restricted scoring window may avoid invalid support for a specific run, but the code does not enforce that condition.

**Recommendation:** crop/transform the support mask with the scoring window and use it consistently in side, surface, and image metrics. Test a rotation-consistent predictor whose only differences occur in unsupported corners.

### R24 — P2: Flat CT can pass a winding periodicity check

**Reproduced; winding feature currently disabled in the main pipeline.** `src/tsm/winding.py:844`.

For constant CT, both the FFT maximum and median are zero. The rejection test `max < 3*median` is false, so the first frequency bin is accepted as a dominant period. A 384-sample flat profile with constant density `1/32` yielded seven apparently ideal winding-per-period values of 1.0. This diagnostic can falsely support a constant predictor.

**Recommendation:** require nontrivial signal energy/variance and a meaningful peak before computing a period. Test flat zero/nonzero CT, low-amplitude noise, and synthetic periodic sheets. The disabled status limits present training impact; retain the test before reconsidering winding enablement.

## Performance, reporting, and robustness observations

These observations are retained separately so unmeasured optimization ideas do not acquire the confidence of the reproduced correctness failures above.

| Priority / status | Location | Observation and next step |
|---|---|---|
| P2, source-confirmed scaling | `infer.py:898` | Pyramid export loads a whole level region into NumPy and forms multiple floating normal arrays. The export brick setting does not bound this stage. Use output-tile downsampling with the required input halo; measure peak RSS for representative levels. |
| P2, source-confirmed complexity | `labels.py:1472` | Face component statistics repeatedly scan the complete label volume with `(lab == c).sum()` for each component, plus repeated component masks. Work approaches O(components × voxels) for fragmented inputs. Use `bincount` for counts and grouped component statistics; preserve exact median/percentile meaning. |
| P2/P3, profiling target | `train.py:370`, `:389`, `:392` | Python `float` conversions materialize GPU loss terms on every microbatch even when logging is infrequent. They force host/device synchronization. Keep detached summaries on device and materialize at logging boundaries; measure actual step time and ensure diagnostics do not change graph retention. |
| P3, source-confirmed accounting | `train.py:985`; `ablate.py:182` | Throughput numerator does not consistently match steps since the previous log. Step 1 to step 10 covers nine updates but can use ten; a final partial interval can use one for several updates. Ablation averages these values. Track actual elapsed update counts and state clearly whether checkpoint/evaluation I/O is included. |
| P3, reproduced accounting | `train.py:687`; `student.py:503` | Training's duplicate MAC estimator charges transposed convolution by output voxels; the student helper correctly uses input voxels. Tiny-network totals were 6,757,888 versus 6,499,840. Consolidate the estimator and test stride-2 transposed convolutions independently. |
| P2/P3, memory risk | `train.py:1202` | Evaluation retains all phase/normal errors for quantiles; probability `sample_cap` does not bound these lists. At 64 fully valid 128³ crops, the two float arrays alone are about 1 GiB before concatenation overhead. Use bounded quantiles or document and estimate this separate budget. |
| P3, profiling target | `data.py:1254` | Full random noise volumes are generated on CPU and moved to the device in GPU augmentation. A batch of two 256³ float32 volumes is 128 MiB per noise family. Evaluate device RNG while preserving reproducibility requirements. |
| P3, profiling target | `sliding.py:720`; `limits.py:192` | Clearing the CUDA allocator cache after every tile discards reusable allocations. Measure with and without this action under realistic pressure; retained-cache memory and usable memory are different quantities. No speedup is claimed without profiling. |
| P3, profiling target | `teachers.py:711`; `feats.py:188` | Percentile normalization materializes a scalar on the host; feature extraction normalizes NumPy input on CPU; TTA can prepare the same input repeatedly. Profile normalization, transfer, inference, and blending separately before tuning. |
| P3, profiling target | `dino.py:553` | PCA computes a full economy SVD while retaining a small subspace, using a default sample matrix around 98,304×864. Consider randomized/truncated methods only with reconstruction and downstream-quality checks. |
| P2/P3, source-confirmed integration | `ablate.py:90` | Rebased variant output directories pin label sources but do not automatically preserve default cached-feature/DINO source locations. A valid base configuration using implicit feature paths can fail after rebasing. Shipped feature ablations explicitly specify those paths and avoid this case. Preserve source inputs separately from variant outputs; also handle explicit null label paths. |
| P3, source-confirmed reporting | `ablate.py:37` | Markdown columns use medial-surface metric keys, so face-mode results can display `-` for surface quality even though JSON contains per-face metrics. Select columns by surface mode and include thickness/coverage. |
| P3, source-confirmed reporting | `dev/bench_student.py:258` | Comparison logic hardcodes medial channel positions; face mode can interpret outer SDF as validity. A broad “probability” difference also mixes phase/density/normal channels. Resolve channel semantics from metadata and report comparable quantities. |
| P2, operational risk | `dev/overfit_synth.py:37`, `:51` | `--fresh` recursively removes arbitrary `--root`, and `root/train` is removed even without `--fresh`. The review did not run this utility. Make reset explicit, validate ownership of a generated fixture directory, and disclose what will be removed. |
| P3, source-confirmed resource accounting | `cache.py:201` | Free-space checks require the full planned cache size even for a fully completed or nearly completed resume. A completed cache can be refused for insufficient free space despite needing no data writes. Estimate remaining allocation after validating the existing layout. |
| P3, source-confirmed validation | `cache.py:185` | Missing `cache_root` becomes `abspath("")`, so `run_cache` bypasses the intended missing-root validation in `plan_levels` and targets the working directory. Validate the raw setting before normalization. |
| P2, provenance risk | `volume.py:411`, `:545` | CT cache identity uses the URL basename and does not validate `source_url` on a hit. Different sources sharing a basename can collide. Preserve explicit primary/alternate equivalence while distinguishing unrelated volumes. |
| P2, boundary risk | `volume.py:498`, `:545`; `cli.py:166` | Cache selection checks the nominal region, not every later halo/context read. Non-strict local reads outside the cached footprint return zero even inside the real volume. `cache_margin` can mitigate this, but there is no check against the selected augmentation/inference halo. Require actual requested coverage or fall back per read. Existing tests deliberately verify zero padding, so this is an integration-contract issue. |
| P3, source-confirmed reporting | `labels.py:1972` | On partially resumed labeling, statistics accumulate newly processed bricks, then some summaries divide by whole-region volume or describe the whole store. The final summary-preservation code handles fully completed sections, but not partial sections. Persist/reduce per-brick statistics or rescan the completed output; label partial-run statistics explicitly. |
| P3, static edge case | `labels.py:648`, `:685` | If all initially selected audit candidates are removed by the maximum-distance filter, `_pct` can return only `n`, but logging unconditionally indexes `median`. Handle the post-filter empty case before computing/logging offsets. A separate initial-empty tuple-unpack allegation was retracted during review. |
| P3, source-confirmed scaling | `winding.py:644`, `:970` | Getting the central ray or an affine bounding box constructs all transverse sample points. T=128/R=1851 creates about 347 MiB of float32 XYZ coordinates, plus float64 construction temporaries. Compute the central ray directly and bounding extrema from corners. This module is currently disabled in the main pipeline. |
| P2/P3, memory risk | `dev/eval_region.py`, `faces_pass` | Whole-region surface masks, connected components/EDTs, multiple face masks, and retained phase/normal error arrays mean the complete evaluator is not bounded by its brick size. Estimate combined live memory, process local comparisons by tiles where valid, and explicitly bound global analyses. |
| P3, avoidable allocation | `dev/tta_side.py:64`; `dev/tta_side_faces.py:101` | Side analysis loads face-label volumes before narrowing to a model window and can upsample whole coarse cubes before slicing. The face helper also eagerly allocates `np.zeros(BOX_SIZE)` as a `dict.get` default even when the key exists. Read only required intersections and allocate fallback arrays lazily. |
| P3, source-confirmed planning discrepancy | `dino.py:830`, `:879` | For region axes smaller than the DINO window, planning clamps the span to the region while execution reads/allocates a full window. Zarr clips oversized writes in the tested version; this is not a demonstrated assignment failure. Make context semantics explicit and estimate the actual execution span. |

## Clarity and design assessment

There are useful foundations: explicit Z/Y/X origin names, small normalization/encoding helpers, mask-aware label interpolation, atomic checkpoint/progress replacement, strict teacher state loading, synthetic geometry tests, and property tests over signed permutations. The pure helper functions make small independent counterexamples practical. The existing suite is valuable and should be extended around the identified invariants.

The main clarity problem is that important invariants live in comments or implicit conventions rather than shared contracts. Storage completion needs an agreed definition of source identity, region, channel schema, processing settings, required outputs, and committed extents. An array opening successfully, a chunk existing, and a whole result being complete are three different facts. The current code sometimes treats them as interchangeable.

Geometry needs equally explicit contracts. Output normals use XYZ component order while radial inputs use ZYX. Transform direction and component order should be visible in types/helper names, and tests should use independently derived matrices. The R05 failure shows why hand-repeating a transformation in a test is weak protection.

Metric implementations have already diverged: AP is duplicated with the same tie bug, distance helpers disagree on missing surfaces, and MAC counters disagree on transposed convolutions. Consolidate these computations and specify their population, tie, empty-set, and unit conventions. Report coverage/missingness alongside scalar quality; otherwise attractive values can hide excluded failures.

Configuration validation is uneven. The top-level parser rejects unknown keys and validates coordinate triples, while many stage options rely on coercion with `int`/`bool`, late indexing, or untyped dictionaries. `Budget.from_dict` does not consistently reject non-object/type-invalid values; positive-number checks can admit nonfinite values such as NaN where only `<=0` is used. Validate stage combinations and finite numeric ranges before opening models or output stores. Avoid presenting all failures as network unavailability when an underlying budget/configuration exception is the cause.

The module-level promise that every large NumPy allocation goes through memory guards is stronger than the implementation. R07 and pyramid export directly contradict it; `VolumeReader.read` also separately guards allocations without proving a total live-memory bound, and local-cache wrapping adds copies. The watchdog measures the parent process RSS and system available memory, not a precise combined parent/worker budget. Document which bounds are enforced preallocation, which are estimates, and which are reactive process termination.

The README is only two lines, while runtime knowledge lives in a large proposal/status document and personal paths in scripts/configs. Add a concise supported-workflow guide: installation/extras, stage inputs/outputs, encoding conventions, resume/force semantics, external checkpoint requirements, implemented versus proposed features, and which metrics are acceptance criteria. Mark one-off visualization utilities as such. `gallery.py` loads entire volumes and uses a hardcoded source; `peek_montage.py` hardcodes a configuration, scale factor, and SDF clip, making them unsuitable as general correctness evidence without checking metadata.

Two specific diagnostic descriptions need qualification. `winding.py:501` calls its normalization validated on real data despite the recorded failed semantic validation; say what was tested without implying the teacher passed its semantic gate. Face component summaries count local haloed component observations across bricks, not unique global sheets. Likewise, tests for a tiny winding forward, strict key matching, positivity, or cumulative monotonicity establish those properties but do not establish physical winding semantics. The candid disabled status in `docs/winding_status.md` is appropriate.

`labels.py` combines audit statistics, field conversions, geometry, output persistence, and visualization in roughly 2,000 lines. A future split along those responsibilities would make review easier, particularly the distinction between a recto-teacher ridge and a CT-sheet medial surface. Existing fine-label brick/whole-array comparisons are useful, but do not establish invariance of arbitrary halo-truncated face components or thickness gates. Compare face labels across brick sizes on connected bodies crossing boundaries before generalizing those tests.

The CLI appends several stages after its `if __name__ == "__main__"` block. Installed `tsm` and `python -m tsm` import the complete module and work; direct `python -m tsm.cli` reaches `main` before the later registrations. Move registration before execution if that direct entry point is intended to work. This is a clarity/entry-point inconsistency rather than failure of the advertised installed command.

## Modeling limitations and claims not established by this review

- **RF halo is not an exact-equivalence proof for the actual student.** `student.py:291` acknowledges global GroupNorm statistics. Per-window CT normalization also changes with window extent. A convolutional receptive-field radius cannot remove those global dependencies. Local toy-convolution tests with box normalization prove their restricted case. Treat RF tiling as a speed/quality tradeoff and compare the real model across window sizes, seams, and whole-crop references.
- **Cached-feature warping is a consistency objective.** A non-equivariant teacher's features on transformed CT are generally not exactly its cached features spatially transformed. That may be a useful training target; comments saying it is exact should distinguish coordinate consistency from teacher functional equivalence. R09 is stronger: it fails even to transform the cached coordinates.
- **Anisotropic/elastic SDF augmentation is approximate.** The code documents determinant-based distance scaling and limits on elastic vector/Jacobian handling. These are known modeling approximations, not newly demonstrated implementation bugs. Measure geometry/quality if strengthening these augmentations.
- **EMA and BatchNorm need calibration policy.** EMA averages parameters while running buffers follow a different policy. This alone was not classified as a definite bug; latest-buffer handling can be intentional. R17's duplicate recomputation update is independently reproduced.
- **TensorRT teacher redimensioning remains a validation gap.** The student exporter has dynamic-shape handling to avoid baked GroupNorm reshape sizes; teacher export lacks the analogous path. No actual failing teacher TRT build was reproduced, so this is not listed as a confirmed defect.
- **No speculative TensorRT buffer race is reported.** Review of the double-buffer stream wait ordering did not demonstrate an ownership error.
- **An initial empty-audit tuple-unpack allegation was rejected.** The source returns the required three arrays in that branch. Only the separate post-filter empty-summary risk remains above.
- **A small-DINO-region assignment-failure allegation was rejected.** A direct Zarr in-memory check showed oversized assignment is clipped rather than rejected. The surviving concern is the planning/context discrepancy in the table, not an asserted exception.

## Small reproduction examples

These examples can run from the existing environment without editing code or accessing production data. They supplement the storage and geometry probe descriptions above; they are not replacement tests or proposed fixes.

```python
import numpy as np
import torch
from tsm.data import split_holdout
from tsm.infer import tta_forward
from tsm.train import average_precision, _surface_distances

# R05: flip Y then transpose Y/X must transform (1,2,3) to (1,3,-2).
x = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1, 1)
print(tta_forward(x, (1,), True, vec=(0, 3)).flatten().tolist())
# Current: [1.0, -3.0, 2.0]

# R08: the returned held/training cubes overlap on [64,80).
origins = np.array([[z, 0, 0] for z in range(0, 128, 16)], np.int32)
train, held = split_holdout(origins, 32, {"z_range": [32, 64]})
print(train[:, 0].tolist(), held[:, 0].tolist())

# R15: identical scores must not gain information from label traversal order.
print(average_precision(np.ones(4), np.array([1, 1, 0, 0])))
print(average_precision(np.ones(4), np.array([0, 0, 1, 1])))
# Current: 1.0 and 0.41666666666666663; grouped-threshold AP: 0.5 for both.

# R16: a missing predicted surface contributes no distances at all.
pred = np.zeros((4, 4, 4), bool)
target = pred.copy()
target[:, :, 2] = True
print([a.size for a in _surface_distances(pred, target)])
# Current: [0, 0], despite 16 target surface voxels being missed.
```

## Recommended order and acceptance checks

1. **Protect artifact integrity:** R01–R04 and R10–R11. Establish compatible manifests and completion semantics across CT, outputs, ONNX/engines, and features. Test interrupted builds, changed sources/checkpoints/settings, added outputs, missing files, and clean installation.
2. **Repair spatial correctness:** R05–R06, R08–R09, R12–R14, R17–R18. Use independent signed-matrix, coordinate-ramp, constant-output, and pairwise-box-overlap oracles. Test composed options, not only individual helpers.
3. **Repair evaluation before selecting models:** R15–R16 and R19–R24. Include tied scores, total misses, unequal occupancy, clipped distances, stale experiment caches, unsupported rotation corners, and flat CT. Re-evaluate affected comparisons after these changes; this review does not retroactively calculate corrected scores.
4. **Restore bounded memory:** R07, pyramid export, evaluation quantiles, and component statistics. Use large logical sizes with small actual fixtures; separately benchmark representative real volumes to record peak parent/worker RSS and VRAM.
5. **Measure speed after correctness stabilizes:** correct throughput/MAC reporting first; then profile preprocessing, data transfer, model execution, blending, writing, and logging. Report cold/warm artifact state, checkpoint identity, patch/halo/step, precision, TTA count, batch, occupied-voxel fraction, and hardware. No quantified optimization gain is claimed by this review.
6. **Consolidate and document contracts:** shared metrics, provenance validation, coordinate conversion, and supported configuration combinations. Add concise operational documentation that matches implemented behavior.

No fixes were applied. This document is the deliverable for the requested multi-agent antagonistic review.

## Verification and resolution (2026-09-06, same day)

Every finding was independently re-derived from the source with a fresh reproduction probe (synthetic data, CPU) before any fix. Verdicts: R01, R02, R04–R07, R09–R24 confirmed as written; R03 confirmed but only reachable with a brick size below the 128-voxel storage chunk (no config does this); R08 confirmed for the interior `_range` split only — the `_frac` splits used by every config keep a full patch gap. Nothing was refuted.

Impact on results reported before this date: all evals ran with `backend=torch`, so R01 never selected a wrong engine; every CT cache on the production machine was checked (chunk count equals the planned grid, no all-zero 64³ samples), so R02 never fed zeros; the metric findings (R15 tie bias ≈1e-3, R16, R19 validity AP was null rather than wrong, R20 sampler weighting, R21 medial displacement in doubly-saturated sheets) shift numbers by amounts below the differences any conclusion rested on.

Fixes landed for all 24 findings (see the per-finding tests: `tests/test_trt_identity.py`, `tests/test_volume.py`, `tests/test_cache.py`, `tests/test_feats.py`, `tests/test_radial.py`, `tests/test_dino.py`, `tests/test_labels_audit.py`, `tests/test_augment.py`, `tests/test_student.py`, `tests/test_student_speed.py`, `tests/test_data.py`, `tests/test_metrics.py`, `tests/test_metrics_dev.py`, `tests/test_eval_region.py`, `tests/test_train.py`, `tests/test_winding.py`). Full suite after the fixes: 523 passed, 1 skipped (sklearn cross-check), 3 deselected (slow). Compatibility: legacy flat `done.json` sidecars are accepted and upgraded in place when the reopened store passes the new geometry validation, and CT caches whose `done.json` covers the planned brick grid get a derived `complete.json`; both were exercised on the production machine by resuming an in-progress label build (resumed at brick 304 of 1152) and by a cache HIT on the slab cache. Stale shape-keyed student engines were moved to `~/.cache/tsm-models/trt_stale/`. The performance/reporting table and the clarity section remain open items.
