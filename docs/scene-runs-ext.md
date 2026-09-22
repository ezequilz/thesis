# ArtiFixer scene reconstruction

The extended path lets the explorer select a bundle of recorded views and jointly
fit an existing splat to ArtiFixer-generated targets. Geometry, opacity and color
remain trainable. There is no image-score acceptance gate or new geometry bound.

## Agent-selected views

`report_artifact` accepts `repair_scope` (`local` or `scene`) and `view_steps`
(up to eight earlier observed step IDs). The current view is always included.
Explore with `regenerate=no` before requesting a repair:

- For a local defect, select overlapping views with useful parallax.
- For scene reconstruction, select diverse views covering the observed scene.
  Scene scope requires at least one earlier view; it does not imply unobserved
  parts of the asset have been reconstructed.

The worker resolves IDs against completed GPU render requests. The agent cannot
supply invented camera poses or filesystem paths. Selected views are rendered
from the current scene. Previous edited images at exactly those recorded cameras
are reused as additional synthetic references when available. A selected view
without an edit contributes a rendered trajectory, not a clean reference image.

Each selected view gets a calibrated translated trajectory. ArtiFixer generates
these as separate sequences with shared references, avoiding false camera motion
across cuts between viewpoints. All generated frames are then fitted **jointly**
with a shared optimizer. The edited references condition generation only; they
are not extra, potentially conflicting supervision at the same target cameras.

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
two selected views receive 50 targets and 2,000 joint updates; adding coverage
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
