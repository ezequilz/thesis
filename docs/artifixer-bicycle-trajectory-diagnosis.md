# Bicycle trajectory and inference diagnosis

Investigated `run_20261005_142842` on 2026-10-06. Frame numbers below are zero-based
generated-target indices unless explicitly described as full-path indices.

The zigzag is real and is explained by the chosen camera waypoints and the
authors' azimuth ordering. It is not caused by an incorrect implementation of
their interpolation in our adapter. The later output degradation is also real,
but the available evidence does not isolate camera motion, autoregressive
history, input reconstruction ambiguity, and model capacity from each other.

## What we actually compute

`splatfix/benchmark.py` passes the three published photographic references
(`_DSC8681.JPG`, `_DSC8707.JPG`, `_DSC8790.JPG`) and all 25 published test poses to
`splatfix/author_trajectory.py`. This benchmark does not use the VLM view finder,
GPT edits, or the older `scene_runs_ext` trajectory code.

The adapter converts OpenGL C2W to OpenCV C2W by right-multiplying by
`diag(1,-1,-1,1)`, calls the pinned authors' helper, then converts back. The
translation column is unchanged by this conversion.

The authors' `Renderer.interpolate_orbit_poses`:

1. Estimates a focus point from the target cameras' viewing axes.
2. Combines reference and target camera positions and obtains a principal orbit
   plane with PCA.
3. Sorts all positions by their projected `atan2` angle in that plane, rotating
   the order to begin at the first reference.
4. Interpolates translation linearly and rotation with SLERP between consecutive
   sorted poses. There is no spline across junctions and no height penalty.
5. For each segment, uses `ceil(sqrt(||delta_t||² + theta²) / d)` steps, where
   `theta` is relative rotation in radians and our `d = 0.1 / metric_scale`.

For this run, `metric_scale = 0.7089871485`, so `d = 0.1410462802`. This mixed
translation/rotation distance is the authors' convention, not a pure metric
translation bound. `loop=False` matches their orbit-rendering entry point.

There are 28 input nodes and 346 interpolated full-path poses. The adapter removes
the three exact reference poses from generated supervision, leaving 343 targets.
The official preparation appends those references as contexts at catalogue
indices 343, 344, and 345. They are available as image cross-attention conditions
throughout inference; their catalogue location does not delay their availability.
They are not literal photographic keyframes inside the generated video.

I fetched ArtiFixer revision `a392c4dfe17459ef9952407accdb9fcdcdddba98`, resolved its
3DGRUT submodule to `62e1038b74b2edc01440fd4ddf5f080109b6faba`, and executed the
original CPU interpolation helper against the saved inputs. All 343 output poses
match the saved trajectory, with maximum absolute matrix difference
`5.551115123125783e-16`.

## Why smooth interpolation still zigzags

The selected photographs include different capture elevations. Sorting solely
by azimuth interleaves those elevations: a high camera can sit angularly between
two low cameras. Linear interpolation faithfully climbs and descends between
them. It guarantees continuity of position, not continuity of velocity across
waypoints, constant height, visibility overlap, or a useful inference curriculum.
Adding more interpolated frames would retain the same geometric zigzag and
lengthen the autoregressive sequence.

Using world Z (the full source set's mean camera-up direction aligns with +Z),
the full 346-pose path measures:

- 21 height reversals.
- Height range 2.151 scene units; accumulated vertical travel 30.693 units.
- Total translation 45.563 scene units.
- Accumulated orientation change 635.53 degrees. This includes pitch and roll;
  it does not mean the path circles the bicycle 1.77 times.
- Maximum full-path step 0.1369 scene units and 2.655 degrees of rotation.

The small per-frame motion and large accumulated vertical motion are compatible:
the issue is the route, not a single abrupt camera jump. Removing context poses
can join two interpolation steps at reference locations; target-only diagnostics
are now recorded separately from the full path.

The pinned authors' repository also contains `render_ellipse_orbit.py`. It fits
an ellipse using camera-position percentiles, looks toward a focus point, uses
mean training-camera Z for automatic height, and defaults to **81 frames and
zero height variation**. That is a different path family from the test-camera
orbit. The paper's training-camera sampling algorithm is yet another operation:
it selects sparse reconstruction inputs, rather than prescribing a novel-view
inference route. Do not confuse these three procedures.

The exact website Bicycle camera path is not established by the available
artifacts. The existence of the flat ellipse helper is not proof that it produced
the website video. The release also explicitly supports arbitrary calibrated
trajectory JSON files.

## What the image evidence establishes

At the 25 published photographic test cameras, first-pass ArtiFixer averages:

- Targets 0–99: 17.00 dB PSNR and 0.500 LPIPS over seven cameras.
- Targets 100–199: 14.36 dB and 0.571 LPIPS over six cameras.
- Targets 200–342: 14.93 dB and 0.556 LPIPS over twelve cameras.

The early-to-middle decline is substantial, but the final group partially
recovers. These are different viewpoints, so the groups confound time with
visibility and reconstruction difficulty. Initial-render PSNR also decreases
across these groups: 13.57, 13.04, and 12.86 dB. The inference cannot be diagnosed
from frame number alone.

Matched images show malformed/duplicated bicycle geometry already in the initial
render. At target 22, the diffusion output looks cleaner but still invents
structure. At target 321 it removes much of a wheel. Early perceptual plausibility
therefore does not establish reliable 3D supervision. The prior saved opacity
audit also reports high opacity over corrupted geometry: opacity measures
accumulated coverage, not correctness, so opacity mixing can retain a misleading
rendered prior.

The final distilled splat is not proven worse than the initial splat. On the
saved 25-view evaluation, mean PSNR improves from 13.10 to 16.82 dB; LPIPS improves
from 0.607 to 0.545. This does not rule out serious local wheel defects or prove
that including all late generated views was beneficial. That requires a
controlled reconstruction comparison.

Existing controls in this workspace add useful evidence:

- Restarting inference at target 280 improved mean PSNR at the six shared late
  test poses from 14.536 to 15.198 dB. Four improved and two worsened; target 321
  improved by 1.13 dB and recovered much of the missing wheel. The experiment is
  sensitive to history and/or sampling, but is a single unpaired-noise trial.
- The completed 1088×720 control scored 15.427 versus 15.758 dB for the historical
  output under the same 720p comparison resize. It does not support reducing
  resolution as the immediate fix. These scores are not directly comparable to
  the native-resolution scores above.

The source caption and scale preparation used all 194 source photographs, as
already disclosed in the benchmark manifest. Only three photographs are image
references during inference. This remains an execution test, not a verified
reproduction of the paper's complete evaluation protocol.

## Cache behavior and the apparent 100-frame threshold

The pinned inference uses four denoising steps, seven latent frames per block,
21 latent frames of self-attention cache including a seven-frame sink, and three
photographic references in a separate cross-attention cache. With temporal VAE
factor four, the first block covers 25 RGB frames and later blocks cover 28.
Rolling eviction begins with the fourth block, at target index 81. The retained
sink is the first generated block, not seven photographs or seven RGB frames.

This is close enough to the observed deterioration to justify a controlled
experiment, but is expected release behavior, not evidence of a bug. Reference
cross-attention is not evicted with temporal self-attention. The pinned revision
already fixes the positional-argument bug that previously allowed the progress
flag to disable reference conditioning.

The paper trains on 81-frame samples and explicitly claims arbitrary-length
generation with a rolling cache. Thus 81 is neither an inference hard limit nor
an instruction to reset every 81 images. Resetting changes the VAE boundary
context, sink content, first block, and random sampling as well as history.
An offset divisible by 28 aligns subsequent block boundaries but does not make
two runs otherwise temporally equivalent.

## Recommended next experiments

First separate **generation coverage** from **evaluation camera locations**.
Keep the 25 photographic test cameras fixed for scoring the final 3D model, but
do not require every one of them to be a waypoint in one generation video. Use
one coherent, modest-elevation orbit near supported geometry, then a second
elevation band if needed. Inspect its actual RGB, opacity, framing, clearance,
and overlap before spending GPU time on diffusion. A fitted ellipse can cross
unobserved or obstructed regions; mathematical smoothness alone is insufficient.

Choose frame density using angular motion and image overlap, rather than simply
asking for fewer images. The authors' 81-frame ellipse preview has zero height
reversals but maximum orientation step 6.36 degrees, versus 2.66 degrees on the
current full path. It is not automatically the better sampling density. A
fairer geometry control initially retains comparable motion bounds; a separate
trial tests sequence length. Keep the exact original three photographic
references for the published baseline. For our non-benchmark workflow, select
references with overlapping content and complementary parallax, not merely
maximum pose separation.

Next compare the same cameras under full history and several shorter,
overlapping sequences, anchored near reliable observations. Compare local
object crops, LPIPS/PSNR at shared photographed views, and geometry consistency
on overlap. Repeat random trials. A global seed alone does not pair noise when
sequence shapes differ; truly paired tests must map noise to the same targets
and document boundary changes. A rotated-start or reversed-order control is
also useful: defects that follow absolute pose implicate coverage; defects that
follow sequence position implicate history. Neither test alone fully separates
the effects.

Before final distillation, retain trusted photographs and filter/weight generated
supervision using validated multiview consistency and coverage. Do not promote
sharp-looking hallucinations or treat high opacity as confidence in geometry.
Do not indiscriminately discard everything after frame 100: that removes view
coverage and the late group is not uniformly bad. Compare full-supervision and
quality-screened reconstructions at the same held-out camera set. Depth-based
checks from the damaged initial splat need caution and should be combined with
image correspondences and explicit inspection of the bicycle's thin structures.

Only after these controls should we attribute remaining errors to 1.3B capacity,
try 14B with the same inputs, or alter cache size/sink/recaching behavior. Changing
cache semantics away from the trained release recipe is not a high-confidence
fix. Shorter sequences can introduce cross-sequence seams unless overlap is
checked before fitting.

## Changes made and verification

Added `splatfix/trajectory_diagnostics.py` and integrated it into future authored
orbit provenance. It records total/max translation and rotation, height range,
vertical travel, and reversal indices for both full and target-only paths.
Distances are explicitly scene units. It handles ambiguous estimated up and
does not change poses or model behavior.

Updated the history-control manifest to document the 25/28 RGB block schedule
and restart confounders. Existing completed results are preserved.

All 18 focused tests passed, including existing CPU integration tests executed
against the pinned original orbit helper and trajectory reader. Tests cover
coordinate transformations, scale, plateaus, singleton trajectories, and invalid
poses. GPU inference and visual improvement have not been validated for a new
path in this investigation.

Generated review artifacts under
`outputs/benchmarks/bicycle/trajectory-review-20261006/`: `audit.json`,
`trajectory-and-quality.png`, `matched-views.jpg`, and
`authors-flat-ellipse-81-preview.json`. The last file is a calibrated, target-only
pose preview from the pinned ellipse helper, not an accepted production path.

Primary code sources:
[orbit interpolation](https://github.com/nv-tlabs/3DGRUT-ArtiFixer/blob/62e1038b74b2edc01440fd4ddf5f080109b6faba/threedgrut/render.py),
[ellipse generator](https://github.com/nv-tlabs/3DGRUT-ArtiFixer/blob/62e1038b74b2edc01440fd4ddf5f080109b6faba/render_ellipse_orbit.py),
[KV pipeline](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/kv_cache_pipeline.py),
and the supplied ArtiFixer paper, especially Sections 4.1–4.2 and supplement E/G.
