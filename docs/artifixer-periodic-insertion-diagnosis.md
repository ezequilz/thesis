# Periodic insertion investigation, 30 September 2026

## Evidence from the reported run

Both supplied PNGs match the saved `targets/00100.png` and `targets/00101.png`
byte-for-byte in
`outputs/scene-runs/run_20260929_211629/requests/repair-00014/extended/`.
This is the v3 periodic schedule, 1.3B model, rendered source conditioning,
last-denoising generated cache, four inference steps, and seed 42.

The requested camera motion from frame 100 to 101 is 0.7594 degrees and
0.03321 scene units. This confirms a small requested change, but neither verifies
the generated image's camera nor establishes physical distance in metres.

The pristine exported frame 100 is copied from `refresh/00100/fixed.png` after
generation. The actual decoder result is `refresh/00100/vae-roundtrip.png`.
It already has visibly softer brickwork, mouldings, handles, and floor details.
Its mean absolute RGB error against the fixed image is 0.02096 on [0,1], versus
0.005824 for the initial starter's recorded roundtrip (about 3.6 times higher).
These compare different views and are descriptive, not a controlled quality test.
Before insertion, frame 100's error against the edit was 0.09012: the insertion
does improve the boundary considerably, but does not preserve it exactly.

## Why initial and later insertions were different

The initial image is the first causal VAE frame, occupying its own latent.
It is cached at timestep zero with opacity one, its exact camera, fresh temporal
memory, and frame offset zero. The first latent also occupies the persistent
attention sink.

V3 instead generated through frame 100, decoded the whole prefix, replaced just
RGB frame 100, re-encoded that video, and copied only its last latent back into
the original latent prefix. That latent represents the four-frame group 97–100
in a causal video context. Re-encoding the prefix can also alter earlier latent
values through VAE reconstruction loss; the splice retained the original earlier
latents instead. Consequently this is neither an independent first-image
encoding nor a complete re-encoded video reconstruction.

The corrected block was recached, but it kept the original rendered opacity,
the group's averaged target camera, and older temporal memory. The persistent
sink stayed at frame zero. The following latent group represents frames
101–104, and denoising still uses encoded degraded scene renders as its source.
This is learned generation with compressed temporal conditioning, not a direct
translation of the pristine exported frame 100.

All prepared repairs were also independently encoded in neighbor attention from
generation start. That reference path was correctly populated; it does not make
the temporal splice equivalent to a clean starter. Inspection of the pinned
upstream cache update supports overwriting the current block; no missing
corrected-cache write was identified. The old tests' dictionary cache proved
that values were passed along, not that the real VAE retained the image.

The code establishes these mismatches. It does not establish their individual
contributions to frame 101's degradation. Render conditioning, reference
conflicts, learned denoising, and VAE compression remain possible contributors.

## Implemented change

Periodic mode now uses `gpt-periodic-exact-starter-v4`: overlapping sequences
0–20, 20–40, …, 100–120. Each repair becomes a true first frame through the
existing Exact starter path. Each sequence gets fresh transformer caches,
independent VAE encoding/decoding, starter opacity one, and camera conditioning
recomputed relative to its own first camera. The new repair becomes the sink.
No standalone-image latent is inserted into a later video group.

The overlap is saved once under the original global frame index, with the new
starter taking precedence. All prepared neighbor references remain available.
Generated continuation keeps the selected source/cache policies. Closure still
exports the initial image at its verified matching pose, without another GPT
request. This exported closure is not a generation-consistency measurement.

Each restart retains `before.png`, its actual `vae-roundtrip.png`, and roundtrip
MAE in `inference.json`; metadata also records all restart indices and the
number of inference sequences. The old mixed-prefix callback is removed.

CPU regressions cover both generated-cache policies, clean starter propagation,
source mixing, multiple periodic restarts and trajectory offsets, cache reset,
camera recomputation, all-frame coverage, full reference availability, loop
closure, and preservation of diagnostics before exact export. Targeted suite:
81 passed, 2 GPU-dependent loading tests skipped.

This is a correction to make repeated insertion use the established initial
insertion semantics, not a GPU-validated image-quality result. Independent
sequences may introduce seams and lose useful long-range temporal context.
Compare v3 and v4 using the same prepared edits and cameras, inspecting actual
VAE roundtrips and frames immediately after each restart before fitting. If
continuations remain degraded, compare source conditioning separately against
the earlier Exact starter run; the prior reported success is not a controlled
comparison unless its source/cache/model settings also match.

Upstream source inspected at pinned revision
[`a392c4d`](https://github.com/nv-tlabs/ArtiFixer/tree/a392c4dfe17459ef9952407accdb9fcdcdddba98):
`model_training/pipeline/pipeline_base.py`, `kv_cache_pipeline.py`,
`model_training/net/transformer.py`, `model_training/data/utils.py`, and
`model_eval/run_inference.py`. No pretrained weights or prior run outputs were
modified. GPU generation was not run during this investigation.
