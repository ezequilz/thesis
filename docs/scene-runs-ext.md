# ArtiFixer scene reconstruction

The extended path lets the explorer select a bundle of recorded views and jointly
fit an existing splat to ArtiFixer-generated targets. Geometry, opacity and color
remain trainable. There is no image-score acceptance gate or new geometry bound.

## Nested artifact inspection

The outer agent uses `tasks/artifact_hunt_3.py` and its original tool schemas
unchanged. Every `report_artifact` starts the inner loop regardless of
`regenerate` or the baseline repair-trigger setting.

The inner agent can only `move`, `move_toward`, and `rotate`, with normal
navigation limits and collision handling. It moves around the same stationary
object for ten turns, keeping it framed while creating parallax. Views after
each movement are recorded automatically. There are no capture, report, map,
or waypoint actions in this phase, and no map images are sent to the agent.

After movement, the agent receives a numbered contact sheet and only the
`select_repair_views` tool. Each tile retains the full exploration-frame
resolution: ten 640×480 views produce a 1920×2112 overview, sent with high image
detail. The agent selects exactly five distinct tile numbers. The first becomes
the reconstruction anchor and the other four become references; neither the
original discovery view nor the last camera view is automatically included.
Tile numbers are resolved to recorded camera poses by the harness. Invalid
selections are retried up to three times, then collection is cancelled.
The original task and tools return after selection or cancellation.

Tune `extended.local_candidate_count` from 5 to 10 (default 10). Reconstruction
always uses five selected views; `local_view_count` is retained only for saved
configuration compatibility. `local_max_turns` bounds movement-phase attempts
(with enough turns allowed for the candidate count). Old `local_step_fraction`
and `local_rotation_degrees` settings remain ignored.
`extended.repair_limit=1` ends a research run after one successful repair; zero
retains deadline-driven exploration.

The harness supplies accepted step IDs to the worker, which resolves them against
completed GPU render requests. All selected views are rendered from the same
scene version before editing. The current anchor is edited using up to four raw
neighboring views; subsequent edits receive that shared repaired anchor, the raw
anchor, and up to two other views. This respects the relay's five-image request
limit. Each request explicitly identifies its target camera and shared physical
object. Changed output aspect ratios fail instead of assigning false calibration.
These remain synthetic references, not captured photographs or guaranteed
multiview-consistent geometry. The lower-level bundle API still supports explicit
`view_steps` and scene scope for controlled experiments.

The GPT-edited observations are the intended corrected views. A cloned splat is
jointly fitted to these images to initialize reconstruction, but that approximate
fit no longer supplies RGB or opacity to ArtiFixer. Each camera gets a separate
translated trajectory seeded directly by its own calibrated GPT edit.

The `gpt-starter-kv-v1` adapter encodes the edit as the first causal VAE latent,
keeps it clean, and performs a timestep-zero transformer pass to populate the
temporal KV cache **before generating subsequent frames**. Later frames start
from noise, with zero rendered RGB/opacity, calibrated camera conditioning and
all edited references. Opacity one at the starter means that image is observed;
it is not measured scene opacity. The starter is not copied across camera poses.
Each trajectory resets caches. The short sequence retains its full history;
generated blocks refresh their cache from clean output. Exporting the exact first
RGB image removes VAE roundtrip loss, rather than being the mechanism that seeds
generation. Every trajectory requires an edited reference at its starting pose.

This is a custom single-GPU inference adaptation of the pinned upstream KV
pipeline, not an upstream image-to-video flag or a guarantee of geometric
consistency. It removes scene-appearance conditioning deliberately. Sparse or
inconsistent references can therefore hallucinate or drift. The scene still
provides camera-trajectory depth and reconstruction initialization. Saved
`inputs/` and opacity arrays remain diagnostic renders, **not generation inputs**;
`inference.json` records the actual conditioning mode and starter frame indices.

Final joint fitting uses generated nearby views while preserving the original
GPT edits as direct targets at each known reference camera, including the loop's
identical closing camera. At those poses the edit replaces the generated target,
so conflicting images are not fitted at the same camera. Original conditioning
renders/opacity are retained as `inputs-original/` and `opacity-original.npy`.
Before/after validation still compares against the untouched incoming scene.
This is synthetic supervision, not a guarantee that the invented details are
physically correct or that the fixed-topology splat can reconstruct them.

## Native reconstruction resolution

Exploration remains inexpensive (normally 640×480). Propagation and fitting use
the **largest exact-aspect multiple-of-16 resolution no larger than the returned
edited anchor**, rather than the exploration resolution or requested edit size.
For a 1448×1086 returned image this is 1408×1056. Camera pose and FOV stay fixed;
pixel intrinsics are recomputed. There is no crop, stretch or implicit upsampling
of the current edited anchor. Previously edited references may be resampled to
match the current bundle. Aspect-ratio changes are rejected because the original
camera would no longer be calibrated to that image.

`extended.max_repair_pixels=0` (default) selects this native-aligned resolution.
An explicit positive pixel budget allows a smaller exact-aspect resolution when
GPU memory requires it. There is no silent fallback to exploration resolution.
High resolution and many selected views increase GPU memory and runtime.

Defaults: 25 frames per selected view, trajectory radius 4% of central median
depth, four inference steps, and 1,000 fitting updates per selected view. Thus
two selected views receive 2,000 initialization updates followed by 50 targets
and 2,000 refinement updates; adding coverage
does not dilute the per-view fitting budget. Loss remains
`0.8 L1 + 0.2 (1 - SSIM)`. This continuation budget is not an upstream paper
hyperparameter.

## Diagnostics and transaction safety

Each request saves calibrated cameras, reference images, RGB/actual opacity
conditions, generated targets, native resolution and selected step IDs.
`extended/validation/before-NNN.png` and `after-NNN.png` use identical cameras:
even indices are selected anchors; odd indices are held-out translated views at
twice the fitting trajectory radius. These images are for visual review, not an
automated correctness score. All validation cameras are in `bundle.json`.

Metrics distinguish exploration, returned-reference and actual fitting
resolution, plus the number of generation segments and reference images.
Target L1/PSNR measure agreement with synthetic targets, not scene correctness.
A complete candidate is published atomically; failure or cancellation leaves
the incumbent untouched.

## Investigation of run_20260922_212041

The saved first door-repair targets already contain the bright glass haze seen
in the user's comparison. The optimizer closely reproduced them: mean target L1
fell from 0.13125 to 0.00872. This is evidence that more fitting iterations alone
would not fix the unwanted generated appearance. An isolated CUDA ablation using
the unedited anchor as reference also produced the haze, so the synthetic edit
alone does not explain it.

The prior implementation discarded most of the edited image's detail by reducing
1448×1086 to 640×480. It also fitted one tiny local loop at a time, with no joint
supervision across other selected scene views. This revision addresses those two
limitations. It does not assert that higher resolution or multiple synthetic
references guarantee correct geometry or reflections.

## Relationship to the original method

The [paper](https://arxiv.org/html/2603.00492v2) uses generated views as
pseudo-supervision for 3D reconstruction. Its reference conditioning can use
captured observations. Our interface currently supplies synthetic edited renders.
Four inference steps are intentional for the distilled causal model.

This remains an asset-only adaptation. The
[official reconstruction entry point](https://github.com/nv-tlabs/ArtiFixer/blob/main/data_processing/run_artifixer3d.py)
uses a fresh 3DGRUT reconstruction from prepared scene data. Here we continue an
existing fixed-topology scene with gsplat and RGB/DC appearance. No densification,
pruning or higher-order spherical harmonics is implemented in this extended path.
Consequently, missing surfaces and view-dependent glass/reflections remain
important limitations. Whole-scene coverage requires the agent to select enough
observed viewpoints; one reference and a short loop do not reproduce the paper's
reconstruction protocol.

## Validation

Run `python -m pytest tests/test_scene_runs_ext.py tests/test_scene_run_gpu_worker.py
tests/test_scene_run_runner.py tests/test_scene_run_studio.py`. CPU tests cover
resolution/frustum preservation, view-ID validation, reference calibration,
joint fitting and transaction safety. CUDA validation must additionally exercise
ArtiFixer generation and inspect matched native before/after views. Synthetic
fitting loss alone is not evidence of improved reconstruction.

CUDA smoke test on the existing LRZ A100 allocation completed two selected views
and two saved edited references at 1408×1056, with 9 frames per view and 500 fit
updates per view (18 targets, 1,000 joint updates). Total pipeline time was
150.3 seconds. All geometry/opacity/color parameter groups trained. Results and
matched renders are in `outputs/artifixer-audit-20260923/native-multiview/`.
The door haze persists in the generated targets and fitted result: this verifies
the high-resolution joint pipeline, **not** a solved visual repair. The candidate
is separate from the original run and was not published as its repaired scene.

## Local railing experiment

`scene_runs_ext.local_region` prepares an isolated calibrated crop experiment;
it does not fit or publish a scene. `crop_transforms` shifts the principal point
by the crop origin while retaining focal length and camera poses. The experiment
uses a 512×512 crop at native pixel density from repair-00005 of
`run_20260922_222300`, with five translated reference frames (0, 6, 11, 14, 18).
Selection currently maximizes camera translation coverage within the existing
short trajectory; it does not yet establish visibility or wide angle coverage.

`GptImageEditBackend.edit_with_references` accepts a target followed by explicit
reference images. The local experiment supplies the same clean railing anchor
and three other damaged views to every edit. This supplies shared context, not a
guarantee of multiview consistency. The relay must support multiple images; an
unsupported request fails instead of silently dropping the references.

The intended controlled comparison is single versus five edited references,
then a diagnostic with both corrupted RGB and opacity removed. Keep seed, crop
and cameras fixed, and inspect non-reference views before any scene fitting.
Five image edits completed after user approval. They visually recover separate
balusters and openings, but this does not yet establish correct 3D geometry.
The first ArtiFixer crop test exposed upstream rejection of a negative principal
point. The bridge now computes full-frame rays and crops them, transforming K
matrices without changing the original viewing rays. GPU verification completed successfully for all three crop experiments.
With actual damaged RGB/opacity, both one and five references still produce a
smeared, nearly solid railing. Removing both RGB and opacity changes the output
toward the reference appearance, but produces unstable blurry structure rather
than a usable repair. This rules out reference count alone as a sufficient fix
for this crop; neither result was fitted into or published as the repaired scene.
Prepared inputs, edited references, and experiment scripts are under
`outputs/artifixer-local-railing/`. A subsequent user-run live test exposed overly
small inner-loop movements, motivating the object-coverage revision below.

## Object-coverage revision validation

Local regression checks cover unchanged outer v3 tools, normal inner movement,
large rotations, waypoint rejection, and migration away from old micro-step
settings. Pipeline tests verify that edited images initialize the cloned splat
and remain direct targets even when generated frames disagree. Starter-inference
CPU tests verify clean-context ordering, camera/opacity alignment, dependence of
future output on the starter, and cache isolation between trajectories. These
tensor tests require PyTorch; lightweight bundle tests do not. The starter
adaptation has not yet been visually validated in a live GPU run. Final fitting
still uses the existing target sampling and fixed topology; those separate
limitations remain.
