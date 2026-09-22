# ArtiFixer scene reconstruction

The extended path lets the explorer select a bundle of recorded views and jointly
fit an existing splat to ArtiFixer-generated targets. Geometry, opacity and color
remain trainable. There is no image-score acceptance gate or new geometry bound.

## Nested artifact inspection

The outer agent retains the normal artifact-hunting task. A triggered
`report_artifact` starts a separate local collection loop for the same object.
The report is the first accepted view. The inner agent then uses `move`, `rotate`,
`view_depth`, `capture_repair_view`, and `cancel_local_repair`; waypoint jumps and
`move_toward` are unavailable and rejected at runtime. Maps are withheld inside
this loop. Captures require translation from every previously accepted pose.
The agent judges overlap and continued object visibility; this is not a geometric
visibility guarantee. Collection cancels after its turn budget or excess travel.
The original task and tools return after collection or cancellation.

Defaults are five views, 30 inner turns, movement capped at 2.5% of initial median
visible depth (also capped by the normal movement limit), and five degrees per
rotation. Tune `extended.local_view_count` (5–9), `local_max_turns`,
`local_step_fraction`, and `local_rotation_degrees` independently of exploration.
`extended.repair_limit=1` completes a research run after one successful repair;
zero retains deadline-driven exploration.

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
`outputs/artifixer-local-railing/`. The live nested-loop launch awaits explicit
approval for restarting the idle manager and uploading full-frame views.
