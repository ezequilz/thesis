# ArtiFixer 14B setup memory

The September 29 setup failure was a Slurm host-memory OOM during pipeline
construction, after the checkpoint download. The HF authentication and symlink
deprecation warnings were not the cause. The allocation had 64 GiB host RAM and
an A100 with 80 GiB VRAM; the downloaded checkpoint occupied about 63 GiB.

`scene_runs_ext/artifixer_bridge.py` now shares one loader between preflight and
inference. It constructs the base transformer with empty parameters using
Accelerate, materializes those parameters directly on the GPU in BF16, then
lets upstream add its conditioning layers. Real buffers, including positional
frequencies, retain their initialized values. The pinned upstream loader copies
the memory-mapped checkpoint with strict state-dict validation. A missing or
incompatible checkpoint still fails setup.

This avoids the large random CPU model allocation before checkpoint loading.
No checkpoint redownload, quantization, or smaller model is required. The full
14B preflight succeeded on allocation 5815398; a sample during checkpoint
loading showed 33,157 MiB GPU memory and less than 1 GiB process RSS. These are
loading observations, not peak inference requirements: frame resolution,
temporal caches, references, and other GPU processes still affect inference.

Use the normal GPU setup flow to produce readiness markers. The bridge prints
construction and checkpoint-copy stages, followed by `ARTIFIXER_READY` only
after loading succeeds. Model downloads remain cached on DSS.

## Step 4 teacher experiment (October 26)

The September 29 incident is independently visible in Slurm accounting:
`5815398.33` ended `OUT_OF_MEMORY`, with MaxRSS `59389960K` (about
56.6 GiB). This is host-memory accounting, not a CUDA VRAM error. The allocation
had 64 GiB of host RAM. Step 4's separate `official_worker.inference` had still
been using upstream random CPU construction rather than the preflight fix.
Large Step 4 models now use the same strict, empty-parameter BF16 loader as
preflight, factored into `scene_runs_ext/model_loading.py`.

There are two different 14B releases. `artifixer-14b.pt` is the distilled causal
model. `artifixer-s1-14b.pt` is the stage-1 bidirectional teacher. Both use
`Wan-AI/Wan2.1-T2V-14B-Diffusers`; the checkpoint must also match the sampling
pipeline. Merely substituting teacher weights into the causal sampler is not a
valid upgrade. The Step 4 teacher option explicitly selects `bidirectional`,
UniPC sampling with 50 steps and the upstream default CFG scale of 5. The
one-step probe is a memory test, not a quality result.

The 14B variant actually contains about 16.91 billion parameters after adding
ArtiFixer conditioning: about 31.5 GiB for BF16 transformer weights alone. Its
FP32 checkpoint occupies about 63 GiB on disk. Storage capacity, CPU RAM and
GPU VRAM are separate limits; DSS does not extend GPU-addressable memory.

For the **distilled** model, the upstream temporal KV allocation is another
large term: at 1280 × 720, 40 layers × K/V × 21 latent frames × 3600 spatial
tokens × 5120 channels × 2 bytes is about 57.7 GiB. This excludes weights,
reference/text cross-attention caches, activations, VAE and allocator overhead.
It is allocated even for short ordinary sequences. Existing connected bridge
calls already shrink their cache to seven latent frames; ordinary outer legs
retain 21. Input staggering alone cannot fix that allocation. The new teacher
path does not allocate this causal KV cache, but full-clip activations and
bidirectional attention still grow with resolution and clip length.

### Using the teacher

In `/splatfix`, Step 4 → **Use 14B teacher inference (default)** is enabled
for new jobs. Defaults are up to 24 generated/context views plus one repaired
reference per independent clip (25 input frames),
50 diffusion steps and native run resolution. The optional `480p` inference-only
limit changes a 1280 × 720 input to 832 × 464 (both dimensions VAE-aligned).
When this limit is selected, RGB and opacity are resized
together; all per-frame intrinsics are scaled and poses/distortion retained.
Predictions are bilinearly restored to the original camera dimensions for
reconstruction. This does not create native-resolution detail.

The teacher reads one clip's RGB/opacity from disk at a time. Depth is not a
teacher inference input here; it is used elsewhere in preparation/fitting.
All selected reference images remain conditioning (subject to the existing
endpoint reference policy), and the VAE encodes references individually.
Predictions exclude trusted anchors and preserve global frame IDs.

New jobs use `teacher_clip_layout: full-leg`. The teacher combines the legacy
inward half-segments into captured legs before taking 24-view windows, preserving
the picked camera order between anchors. The closest permitted repaired endpoint
seeds each window. A short final window borrows distinct views from the same leg
as context; those overlapping views are masked out of exports so every target
has exactly one reconstruction image. A leg with fewer than 24 available views
stays shorter. No new cameras are extrapolated, and windows do not cross a saved
anchor or missing-capture gap. For example, a 31-target leg now exports 24 then 7
new images; both clips see 24 target/context views plus their repaired starter.

The UI counts views excluding the starter; the CLI/API `teacher_max_frames`
retains its total 1+4*n budget convention (25 for 24 views). With seeding disabled,
the extra slot is reserved for temporal padding. Historical requests without a
layout keep `segment` behavior and their original cap. The already running
`run_20261027_012036` retains its saved 37-frame, segment-based recipe in both
inference phases; changing defaults does not rewrite an active experiment.


Reducing the number of selected references can also reduce conditioning memory,
at the cost of scene coverage. Encoding references one at a time retains all
references while reducing the VAE's temporary peak; it does not remove their
latent attention cost. The teacher uses upstream's unrestricted bidirectional
attention (`frames_per_block=None`). The input clip limit does not alter the
distilled checkpoint's trained causal block/window geometry.

Independent teacher clips have full bidirectional attention within each clip;
they do not share temporal state. Clip boundaries can introduce seams. The
connected-bridge, generated-context and GPT-handoff adapters are disabled.
The teacher starting-image default seeds each clip with its leg’s repaired
reference at the exact reference camera, with opacity 1. The first encoded
latent is restored after every UniPC step, leaving other latents to denoise
around it. This is an experimental conditioning adapter, not a persistent KV
cache or the unmodified teacher recipe. The 37-frame cap includes the starter,
which is never exported as a generated target. Its original RGB remains direct
reconstruction supervision. The UI also offers seed-only and reference-only
comparisons. Effective settings, reference checksums and preservation checks are saved. No
automatic smaller-model or lower-quality fallback occurs in normal Step 4.
Turn **Use 14B teacher inference** off to restore the legacy distilled model
selector and generation controls, including 1.3B. New API jobs default to
`teacher_inference: true`; explicit distilled model selections still opt out.
Historical jobs without this setting retain their saved model and sampling policy.
The CLI accepts `--no-teacher-inference --inference-model 1.3b` for the legacy path.
The full reconstruction workflow still follows inference automatically.

CLI equivalents:

```bash
splatfix repair CHECKPOINT --output OUTPUT \
  --inference-model 14b-teacher --teacher-resolution 480p \
  --teacher-max-frames 25 --teacher-clip-layout full-leg --teacher-steps 50
```

Use `--teacher-resolution native` for the original run resolution. To test only
inference against an existing prepared Step 4 request, inside its GPU environment:

```bash
python -m splat_explorer.splatfix.artifixer.teacher_inference \
  --request /path/to/request.json --output /new/probe-directory \
  --segment 0 --max-clips 1 --max-frames 17 --steps 50 --resolution 480p
```

The probe creates no reconstruction and makes no image-repair API calls.
`inference-teacher.json` records the checkpoint, sampling policy, actual sizes,
frame IDs and completion. `inference-memory.jsonl` records loading and each
clip's allocated/reserved CUDA peaks, device free memory and process peak host
RSS. The corresponding plus-pass files use the `plus-` prefix. Host RSS is not
Slurm cgroup accounting or total node RAM. Monitor Slurm separately for host
OOMs, which can kill a process without a Python traceback.

The teacher checkpoint must already be cached under the configured model
checkpoint directory. The October 26 test replaced the cached distilled 14B
checkpoint at the user's request, retaining 1.3B. The download initially failed
with `Disk quota exceeded`; the aggregate DSS free-space report did not reflect
the shared project fileset quota. `mmlsquota` subsequently confirmed that the
user has no individual block limit, while `pn25pi-dss-0000` has a 10,485,760,000 KiB
fileset limit. Before staging the full replay, about 16 GiB remained after
allowing for the reported in-doubt allocations. The teacher's published SHA256 is
`8ce0d5333803e42574d229ed702038833f4c9a44feb9dcf354ad035e93172940`.

### Measured single-A100 results (October 26–27)

The teacher checksum matched the published value. On allocation `5875153`,
node `lrz-dgx-a100-004`, the isolated inference test completed on one
A100-SXM4-80GB, inside a 52 GiB Slurm memory step within the 64 GiB allocation.
It used all six repaired references, seed 42, the existing caption/cameras and
source frames 23–39 (17 targets). No quantization or transformer CPU offload
was used. The sampler was the pinned upstream bidirectional pipeline, BF16,
native attention backend, UniPC and CFG 5.

- 832 × 464, one-step memory test: **36.08 GiB peak allocated**, 37.45 GiB
  peak reserved; 20.75 seconds including input preparation and PNG export.
- 1280 × 720, one-step memory test: **41.91 GiB peak allocated**, 46.73 GiB
  peak reserved; 33.81 seconds.
- 1280 × 720, full **50-step** inference: **41.91 GiB peak allocated**,
  45.08 GiB peak reserved; **709.72 seconds** (11 minutes 50 seconds).
  Denoising itself took about 690 seconds, averaging 13.8 seconds per step.

- The longest saved segment, 36 frames (padded to 37 for the VAE), also passed
  at 1280 × 720 with all six references through the production
  `official_worker.py --phase infer` entry point. This **one-step memory test**
  peaked at **42.32 GiB allocated / 45.87 GiB reserved**, taking 68.85 seconds.
  Its results are in `outputs/artifixer-teacher-probe/results/smoke-native-full-segment-v2/`.
  A 37-frame cap therefore preserves all existing segments of this saved
  trajectory without extra subdivision. The 50-step test above used 17 frames;
  the full 36-frame segment has not had a 50-step quality evaluation.

Model loading took approximately 48–80 seconds per fresh process, separately
from those clip times. The loaded pipeline allocated 31.86 GiB on the GPU.
Peak process host RSS was about 52 GiB, including reclaimable memory-mapped
checkpoint pages; this must not be mistaken for extra GPU memory. PyTorch
allocator peaks also exclude driver/library allocations outside that allocator.

All 17 full-inference predictions decoded at 1280 × 720, had the expected
global frame IDs and nonconstant pixels. The results and telemetry are under
`outputs/artifixer-teacher-probe/results/quality-native/`; frame `00031.png`
is available in `inference/splatfix/frames/batch_0000/pred/`. These tests prove
hardware feasibility for the stated inputs, not a reconstruction-quality win.
The inspected teacher sample is softer and retains more artifacts than the
same-index output from the earlier 1.3B run. That prior run used different
seeding/context behavior, so it is not a controlled model-quality comparison.
No teacher reconstruction or completed full-trajectory quality evaluation was
run in these probes.

On October 27, a full inference replay of `run_20261026_212233` was launched
through the same production worker on Slurm step `5875153.31`. Its isolated
bundle is `outputs/artifixer-teacher-replay/run_20261026_212233/` locally and
under the remote workspace. It preserves all 202 saved camera positions, six
references, the caption and seed, and generates the remaining 196 frames across
the original ten segments at native 1280 × 720, with a 37-frame cap and 50 steps.
All 426 staged input files were checksum-verified before inference. This replay
does not start reconstruction. `status.json` reports live segment/denoising
progress; `results/validation.json` is written only after all expected frame
IDs, image dimensions and nonconstant predictions have passed validation.
During the first segment, `nvidia-smi` measured 42,361 MiB used of 81,920 MiB.
The model-load process peak RSS reached 62.76 GiB in this run, higher than the
earlier probes; mapped checkpoint pages are reclaimable, so these RSS readings
are not a guarantee of a smaller host-memory requirement. The existing 64 GiB
host allocation passed model loading.
The first 11-frame segment subsequently completed all 50 steps in 558.79
seconds, peaking at 41.83 GiB allocated / 45.12 GiB reserved. All eleven PNGs
were downloaded and validated at 1280 × 720, and segment two started without
reloading the model. This is an intermediate success, not confirmation that
the full 196-frame replay has finished; consult its live status and final
validation artifact.

Validation also included 528 passing local tests (41 skipped without local
PyTorch), 17 passing tests in the remote PyTorch environment, and a targeted
161-test recheck after setting native resolution and a 37-frame cap as the teacher defaults.
Step 4's controls were checked in the live browser; the dashboard and queue
manager were restarted with the updated code. The later default change enables the teacher for new Step 4 jobs; the separate
published-dataset benchmark keeps its 1.3B default.

### When two A100s would help

The measured teacher clips do not need a second GPU. Two GPUs are useful if larger
native-resolution bidirectional clips exceed the measured memory budget or
runtime is unacceptable. Upstream supports teacher context parallelism with
`torchrun --nproc_per_node=2` and `--context_parallel_size 2`: use two allocated
GPUs **on the same DGX node**, with NCCL communicating over its interconnect.
Each rank still has a model replica; context parallelism is not 160 GiB of
pooled memory and does not halve model-weight memory. Each rank reads the same
selected item from shared storage; temporal token work is divided and attention
collectives synchronize ranks. Disk storage is not the inter-GPU data channel.

Our Step 4 worker currently has a single-rank contract and strips torchrun
environment variables. A two-GPU extension needs one Slurm task launching two
ranks, upstream context meshes/padding, synchronized input/RNG, rank-zero output
ownership and distributed cancellation. Starting two existing workers would
replicate work and cause output races. Also, upstream's default causal
7-frame-block/21-frame-window settings have a context-parallel divisibility
constraint; two GPUs are not a drop-in fix for that distilled configuration.
Host RAM must be budgeted for both ranks as well; adding another GPU does not
itself fix the original CPU-memory construction failure.

Sources: [NVIDIA checkpoint/model card](https://huggingface.co/nvidia/ArtiFixer),
[teacher checkpoint and checksum](https://huggingface.co/nvidia/ArtiFixer/blob/main/artifixer-s1-14b.pt),
[pinned inference entry point](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_eval/run_inference.py),
[pinned KV-cache allocation](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/kv_cache_pipeline.py),
[pinned teacher sampler](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/pipeline.py).

### Reviewing teacher results

Normal Step 4 teacher jobs publish completed clips through the existing inference
preview and download mirror. The results detail page shows all generated teacher
frames, originals, repaired references and the inference/plus teacher manifests.
The mirror refreshes progressive previews until all clips are complete.

Isolated bundles under `outputs/artifixer-teacher-replay/run_YYYYMMDD_HHMMSS/`
are shown as **ArtiFixer inference** cards. Their controller status and completed
clip manifests drive review; partially written clips stay out of the gallery.
These cards are separate from the original repair run and do not enter the job
queue. They expose review artifacts only; queue controls and reconstruction
actions are unavailable for these externally launched inference-only tests.

### Progress updates

The studio and run review show separate inference, reconstruction and subsequent
3D+ inference progress. Teacher inference uses completed clips and the current
denoising step; reconstruction reports actual optimizer iterations and remains
below 100% until export finishes. Saved-frame counts appear separately.

A visible studio or results page sends a local activity heartbeat. The queue
manager shares one remote observation per run across all tabs, at most once every
60 seconds while visible and every five minutes in the background. The last
attempt time persists across manager restarts. Browser refreshes read the local
snapshot; they do not initiate SSH queries. Explicit cancellation and required
capture/control operations remain responsive independently of progress polling.

### Endpoint blur experiments (October 27)

These are optional, off by default, and have not yet been validated for GPU image
quality. The active 37-frame run keeps its saved settings. New normal runs use
24 target/context views plus a preserved reference, with whole-leg packing.

**Endpoint extension** accepts 4 or 8 requested views. Preparation extends the
endpoint's translation tangent, using the adjacent authored camera step and
fixed endpoint orientation/intrinsics. Every full movement segment is checked
against the existing 3-sigma Gaussian collision index with camera clearance.
The first colliding view and all farther views are excluded. Available context
is rounded down to complete groups of four for inference; a collision leaving
fewer than four safe views means no warm-up at that endpoint.

The inference order is farthest safe extension → nearer extension → repaired
endpoint → wanted views. There is no jump from a leading reference into the
extension. Extension mode explicitly unlocks the repaired reference: it is
inserted at its actual camera with opacity 1, but its latent is allowed to
denoise. Interior windows that do not reach a reference use reference attention
without inserting a distant endpoint into the video. All input views count
against the 25-frame total budget. Used warm-up views are exported once as generated supervision and included in
reconstruction at their calibrated cameras. Unused rendered context and collision-
rejected views stay excluded. Source captures label extension context in review. All original wanted cameras and repaired-reference
supervision are retained.

**Rendered opacity strength** is a multiplier in [0,1], default 1. It affects
only teacher RGB/opacity conditioning, in both inference passes. It changes
neither stored depth nor camera geometry, and the repaired reference is restored
to opacity 1 after scaling. Lower values increase the noise component in the
initial latent and reduce supplied opacity conditioning. They are not a precise
spatial mask for invalid geometry, and can increase hallucination or drift.

The upstream encoder groups temporal frames in fours after its initial frame,
and initializes latents using max-pooled opacity. Warm-up lengths divisible by
four place the interior reference at the end of a temporal group, preventing its
full opacity from being pooled into the next group's wanted frames. This is a
conditioning hypothesis, not proof of improved sharpness.

Suggested controlled comparison: same source, camera IDs, resolution and random
seed; 50 steps at opacity 1 (control), 50 steps at opacity .75 with extensions
off, then 50 steps at opacity 1 with 8-frame extensions and an unlocked reference.
Only then compare 75 steps at otherwise identical settings. Compare the same
wanted cameras near each endpoint, not different clip/gallery positions.
More denoising steps do not by themselves remove strong blurry-render conditioning.
