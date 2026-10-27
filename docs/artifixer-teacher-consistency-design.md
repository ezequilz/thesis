# Teacher consistency: evidence and proposed experiments

Status: design, 27 October 2026. No inference policy was changed and no GPU
experiment was launched for this investigation. Quality descriptions below are
visual observations of saved frames, not calibrated quality scores. These runs
do not isolate a single cause.

## What the comparisons establish

The main working hypothesis is dependence on the complete clip's conditioning:
which repaired view starts it, temporal encoding, the subsequent camera path,
and the damaged views supplied alongside the target. High opacity and damaged
RGB occur in both successful and unsuccessful examples, so neither identifies
failure on its own. Reducing opacity globally is not the proposed remedy.

### Same camera, different result

Camera 27 is visually much better in `run_20261027_012036` than in
`run_20261027_060217`: the former has recognizable flames and reflections; the
latter has duplicated plant-like structure and deformed fireplace contents.
The camera matrices match exactly, all six repaired reference PNGs are
byte-identical, and the caption checksum and camera scale match. The original
RGB images are not byte-identical, but their mean absolute channel difference is
only 0.756 on the 0–255 scale (maximum 15). Camera 29 has the same pattern,
with mean difference 0.750 and maximum 19. The supplied `00029.png` exactly
matches the saved `012036` prediction.

This is much stronger evidence of context sensitivity than comparing cameras
27 and 228. It is still not a controlled experiment: context, input order,
reference preservation and random-number consumption changed together.

### A preserved first reference helps explain one success, but is insufficient

In `012036`, Clip 2 uses repaired camera 32 followed by targets 31 down to 17.
The first latent is restored after every one of the 50 UniPC steps; the manifest
records successful preservation. Camera 29 is the third exported target, but
the fourth actual input after the starter. The older gallery label `input 3/15`
omits that starter because its manifest lacks the newer input-order array.

Clip 3 uses the identical repaired camera 32 and the same preservation policy,
but targets 33 through 65. Its first target, camera 33, is already structurally
poor, and the failure persists at inspected cameras 37, 49 and 65. Therefore:

- Locking the first latent alone cannot be the complete solution.
- Distance from the starting reference alone cannot explain failure.
- Shortening every clip indiscriminately is not yet a validated solution.

The complete Clip 2 spans about 16.44 degrees of rotation and 0.482 scene units
of translation. Clip 3 spans about 56.53 degrees and 0.770 scene units. However,
Clip 4 spans 54.82 degrees and is substantially cleaner in inspected later
frames, so large angular extent alone also fails as a classifier.

### A strong discontinuity at an artificial ownership boundary

In `012036`, camera 65 is the last target of Clip 3, generated from camera 32.
Camera 66 is the last target of reverse Clip 4, generated from camera 98.
Camera 65 is severely distorted; camera 66 has much more coherent furniture,
plant and fire. The two poses are only 1.713 degrees and 0.02335 scene units
apart. The mid-leg split changes the generation context abruptly; the physical
camera trajectory does not jump there.

This is not a same-camera direction ablation. Nevertheless, it motivates
overlapping predictions on identical cameras instead of assigning each half
exclusively to one endpoint with no comparison at the join.

### Separate initial blur from persistent structural failure

In `012036`, inspected cameras 1 and 2 are broadly blurred, while camera 5 and
later views contain more detail. Reverse Clip 4 also starts softly at camera 97
and improves by camera 93. Clip 2 becomes detailed very early. These observations
suggest an initial temporal transition problem in some clips, but do not prove
whether the VAE, the preservation adapter, or denoising is responsible.

This differs from the persistent duplicated/stippled structures in Clip 3.
Extrapolated warm-up cannot be assumed to solve both problems.

### Why the later full-leg policy did not solve it

`060217` Clip 1 has 79 actual inputs: 40 leading extrapolated views, repaired
camera 0, cameras 1–31, repaired camera 32, and six trailing extrapolated views.
It has no preserved latent. The first extrapolated view is visibly extreme and
distorted. Clip 5 has 48 inputs, starts at repaired camera 211, has no extension,
and produces the good camera 228 example without preserving its first latent.
Thus preservation is not necessary for every successful clip either.

All six references are available via reference attention in both cases. A
reference appearing late inline does not mean the model had no clean reference
conditioning beforehand. The teacher denoises all frames jointly; it does not
first generate the prefix and subsequently reach camera 27.

## Mechanism to investigate

The teacher transformer has bidirectional temporal attention, but the complete
system is not invariant to reversing a clip. Wan's temporal VAE is causal and
temporally compressed. The first frame has special handling; subsequent frames
are grouped in fours, with temporal receptive fields extending beyond simple
non-overlapping groups. ArtiFixer also groups camera conditioning in fours.
Changing input order, start pose, reference placement or clip extent changes
these representations even when one requested output camera is unchanged.

The plausible failure is that some clips settle on a temporally persistent
interpretation of the damaged source instead of a clean interpretation supported
by the references. Full attention permits distant damaged context to affect
even the first target. This is an inference, not a measured attention attribution.
Successful reverse Clip 2 directly contradicts a general inability to handle
reverse camera motion.

The `033950` experiment used target opacity ×0.5 and reference opacity 0.8,
with an unlocked reference and eight-view extensions. The inspected saved
outputs have flattened, simplified details. The user also observed weaker
reflections. This is evidence against that recipe, not an isolated estimate of
the target-opacity multiplier's effect, because reference opacity changed too.
Keep both opacity settings at 1.0 in the next experiments.

## First diagnostic experiments

Use inference-only probes and a frozen copy of one input bundle. Keep model,
caption, six reference PNGs, poses, intrinsics, opacity, resolution, sampler,
step count and source RGB fixed except for the stated intervention. Save all
input IDs, padding, starting latents, RNG state, and decoded anchors. Avoid
using the existing `--segment` switch as if it were a clip selector: it can
change the planner and the comparison itself.

1. **Reproduce success and failure.** Reproduce `32,31,...,17` and
   `32,33,...,65` with the historical preservation adapter. Reset sampling RNG
   independently for each probe. A single global seed set before model loading
   does not give every differently packed clip matching noise.
2. **Test distant context.** Compare the failing `32,33,...,65` with
   `32,33,...,48`, scoring only common target cameras 33–48. Keep the same seed
   frame and corresponding initialization noise where alignment permits. If
   early targets improve, distant clip content/extent is implicated. If they
   remain poor, shortening alone is insufficient. This test includes the
   changed temporal endpoint; it does not isolate attention from VAE effects.
3. **Test preservation.** For the successful clip and the short failing clip,
   compare preservation on/off with identical tensors and RNG state. This tests
   whether the experimental clamp helps or causes an initial transition.
4. **Test the join.** Generate overlapping cameras around 65/66 from both
   repaired endpoints, following the actual intervening cameras. Compare the
   same camera IDs. This changes the reference-led context deliberately; it is
   not a pure direction-only experiment.

Before interpreting generative quality, record VAE-only round trips of the
clean starter and source sequence, decoded preserved prefixes, and a few
intermediate denoising outputs. If the artifact is present before denoising,
debug encoding/decoding first. A numerically unchanged first latent does not
guarantee unchanged decoded pixels or a healthy transition to latent two.

For a promising change, repeat a small fixed set of seeds and inspect continuous
video, not only selected stills. Use separate reported judgments for source
artifact removal, detail/reflections, structural agreement, camera following
and temporal stability. Sharpness alone is not an acceptance metric.

## Proposed production design, conditional on those tests

### Generate locally from trusted evidence

Start at an actual repaired camera with full-opacity rendered targets. Keep
all original repaired references available initially; changing reference
selection/weighting is a separate experiment. Do not synthesize distant views
merely to fill the frame budget. Use bounded windows chosen by visibility,
parallax and rotation as well as frame count. Initial experimental sizes of
16–32 new views and 4–8 shared views are candidates, not universal thresholds.

For startup context, test accepted frames at their exact poses. At the first
anchor, a stationary hold of the exact repaired image at the exact same pose
is a separate diagnostic option. It must never be copied into different target
cameras, and it must not be assumed to cure VAE transitions. Do not clamp a
whole interior latent merely because one of the RGB frames it encodes is clean.

### Give adjacent windows a shared result to agree on

Generate overlapping target cameras. For continuations, use a short prefix of
accepted output frames at their original poses and keep original repaired
references present. Test a prefix-conditioned teacher adapter against simply
adding these frames as reference images. The legacy
`generated_context_references` flag is disabled for the teacher; this requires
a teacher-specific implementation, not switching on the causal cache adapter.

If clamping an accepted prefix proves useful, encode that prefix in the new
window's temporal layout and clamp only fully supported prefix latents. Do not
copy arbitrary latent slices between clips with different temporal alignment
or conditioning history. Preserve source provenance and generation depth for
accepted frames; they remain generated evidence, not ground truth.

### Retry the difficult region, not the whole run

At a doubtful interval, generate a candidate from the other repaired endpoint
with overlap. Select a coherent sequence using quality and transition costs;
do not choose each frame independently or average mismatched furniture/flames.
If neither candidate passes, shorten the interval or introduce an additional
trusted repaired keyframe there, checked against the existing references.
Do not propagate a weak generated frame indefinitely as the next trusted seed.

The initial pass need not run twice everywhere. Opposite-end generation is a
targeted retry, with a finite retry budget and an explicit unresolved status.
If the desired camera path is intrinsically unsupported, adding evidence is
preferable to silently changing that path or accepting a hallucination.

### Validate before committing or fitting

Use shared camera predictions to detect structural disagreement, stable-surface
feature tracks/reprojection to check camera following, and temporal checks to
detect popping and duplication. Require both plausibility and consistency;
two candidates can agree on the same error. Calibrate acceptance against human
review of known good/bad clips before making it fully automatic.

Keep the evaluation aware of view-dependent appearance. Changing flames and
reflections can be desirable video behavior. Do not impose pixel equality on
them across viewpoints or suppress highlights to minimize temporal residuals.
For static 3D reconstruction, treat these regions separately from stable
geometry constraints. Original source depth is damaged too, so reprojection
agreement is a diagnostic with visibility checks, not unquestionable truth.

Only accepted wanted frames enter reconstruction. Context-only and extrapolated
frames do not become supervision merely because inference produced a PNG.
Save rejected candidates and reasons. Report the fraction of requested cameras
that passed, rather than presenting a selected successful subset as full coverage.

## Local evidence and implementation locations

- `outputs/scene-runs/run_20261027_012036/gpu/results/edited_6c272c8f4cd2/`:
  requests, trajectory, manifest, 96 completed predictions, reference/input PNGs.
  The run was stopped; its manifest is incomplete. Do not infer ungenerated legs.
- `outputs/scene-runs/run_20261027_060217/gpu/results/edited_d7f64d8fadf8/`:
  completed initial inference manifest and 311 predictions, including warm-up.
- `outputs/scene-runs/run_20261027_033950/gpu/results/edited_58cbc83fab47/`:
  saved request and 24 locally available predictions; no local teacher manifest.
- `src/splat_explorer/splatfix/artifixer/teacher_seeding.py`: preservation adapter.
- `src/splat_explorer/splatfix/artifixer/teacher_inference.py`: inputs, sampling,
  export ownership and single global RNG initialization.
- `src/splat_explorer/splatfix/artifixer/teacher_clips.py` and
  `teacher_short_extensions.py`: independent clip construction and extensions.
- `src/splat_explorer/web/teacher_frame_order.py`: old-manifest display fallback.

## Primary sources

- [Pinned teacher sampler](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/pipeline.py).
- [Pinned camera grouping](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/data/utils.py#L519-L551).
- [Wan temporal VAE](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py).
- [ArtiFixer paper](https://arxiv.org/html/2603.00492v2).
