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

The October 26–27 experiment enabled **Use 14B teacher inference** for new
jobs. It is now opt-in; new jobs default to the 14B distilled student. Teacher defaults are forward windows between consecutive selected repaired
views, up to 81 total input frames including references and warm-ups,
50 diffusion steps and native run resolution. A full-size memory preflight on
October 27 completed one 49-input clip at 1280 × 720 with 46.94 GiB peak reserved
and 42.57 GiB peak allocated on the A100 80GB. It used one denoising step plus
VAE decode, eight warm-up views, unchanged rendered opacity and reference opacity
1.0. This validates that probe’s memory footprint, not full-run image quality or
completion; the queued production comparison uses 150 steps.
A subsequent 81-input probe (74 real inputs plus seven masked repetitions of the
last frame) also completed denoising and decoding: 52.54 GiB peak reserved and
43.64 GiB peak allocated. Padding was confined to the memory test. The then-current whole-leg
recipe kept that captured route in five clips with 40, 74, 72, 41 and 47 actual
inputs. Those measurements describe the earlier layout, not the new forward plan.
The official default is documented in [NVIDIA’s inference parser](https://github.com/nv-tlabs/ArtiFixer/blob/main/model_eval/run_inference.py#L806-L811).

The optional `480p` inference-only
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

New jobs use `teacher_clip_layout: forward`, `initial_image_seeding: true`,
`image_cache_insertion: false`, `teacher_endpoint_extension: 8`, and
`teacher_warmup_supervision: false`. This follows the connected picked path in
increasing camera order, with each consecutive pair of repaired references
normally bounding a clip. All six selected references remain available to
reference attention. There are no reversed half-legs or distant starter jumps.

The input order is leading warm-up → repaired reference → interpolated views →
repaired reference → trailing extension. Each end gets up to eight safe extension
views, reduced in groups of four for collision clearance and the total budget.
For example, 73 captured frames including both references leave room for four
warm-ups per end. If the captured interval itself needs the full budget, it takes
priority over extensions. Warm-ups provide context only by default; they do not
become generated exports or reconstruction supervision in forward mode.

Intervals longer than the cap use overlapping forward windows. Overlap is up to
eight inputs, with output ownership divided between neighboring windows so every
wanted camera is exported once. Interior windows do not insert a distant repaired
camera; all repairs remain available through reference attention. These windows
have independent temporal attention and do not carry generated state. Their
seams and image quality still require a GPU comparison.

At each included reference camera, the repaired RGB replaces rendered RGB before
encoding the complete input sequence with the temporal VAE. This seeds the
reference at its actual time position without inserting an independently encoded
single-image latent into a compressed temporal group. No latent is restored after
a scheduler step. The causal VAE makes temporal order significant, but does not
prohibit reversed video; forward order removes an unnecessary source of context
variation. The teacher transformer still uses bidirectional attention.

The dashboard, CLI and API all count **total input frames**, including references,
warm-ups and padding (`teacher_max_frames`, 1+4*n, maximum 81). Select the legacy
`full-leg` or `segment` layout to reproduce older policies. Latent preservation is
an explicit legacy fallback with extensions disabled; forward mode rejects it.
Historical requests retain their saved layout, cap, seeding and preservation
settings, including requests that predate the layout field.

Reducing the number of selected references can also reduce conditioning memory,
at the cost of scene coverage. Encoding references one at a time retains all
references while reducing the VAE's temporary peak; it does not remove their
latent attention cost. The teacher uses upstream's unrestricted bidirectional
attention (`frames_per_block=None`). The input clip limit does not alter the
distilled checkpoint's trained causal block/window geometry.

Independent teacher clips have full bidirectional attention within each clip;
they do not share temporal state. Clip boundaries can introduce seams. The
connected-bridge, generated-context and GPT-handoff adapters are disabled.
Reference cameras are never exported as generated targets; their original repaired
RGBs remain direct reconstruction supervision. Effective settings, exact inline
reference positions, reference checksums and input paths are recorded. The legacy
preservation adapter restores the first latent after every UniPC step only when
explicitly enabled; it is not the default or an interior-frame preservation method.
No automatic smaller-model or lower-quality fallback occurs in normal Step 4.
Turn **Use 14B teacher inference** off to restore the legacy distilled model
selector and generation controls, including 1.3B. New API jobs now default to
`teacher_inference: false, inference_model: 14b`; teacher experiments opt in.
Historical jobs without this setting retain their saved model and sampling policy.
The CLI accepts `--no-teacher-inference --inference-model 1.3b` for the legacy path.
Reconstruction still follows inference automatically. New jobs finish after exporting
the repaired PLY. The optional **Run final ArtiFixer3D+ image pass** is off by
default (`artifixer_plus_pass: false`); `--artifixer-plus-pass` enables the
additional renders and denoising for testing. The + pass produces images only
and does not improve or replace the reconstructed PLY.

CLI equivalents:

```bash
splatfix repair CHECKPOINT --output OUTPUT \
  --inference-model 14b-teacher --teacher-resolution 480p \
  --teacher-max-frames 81 --teacher-clip-layout forward --teacher-steps 50
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

The forward default now uses fixed endpoint extensions with seed-only references.
The fill-short-clip and opacity experiments remain optional, off by default, and
have not been validated as a quality improvement. Existing runs keep their saved
settings. The legacy whole-leg extension behavior below remains available.

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
inserted at its actual camera with the selected reference opacity (default 1), but its latent is allowed to
denoise. Interior windows that do not reach a reference use reference attention
without inserting a distant endpoint into the video. All input views count
against the selected total budget (81 frames by default). Legacy whole-leg mode exports used warm-up views once as generated supervision
at their calibrated cameras; forward mode excludes them by default. Unused rendered context and collision-
rejected views stay excluded. Source captures label extension context in review. All original wanted cameras and repaired-reference
supervision are retained.

**Extend both ends of short clips** (`teacher_extend_short_clips`, optional and
off by default, legacy whole-leg mode only) replaces the fixed warm-up count for short complete legs. It
checks both outward endpoint translation tangents, including the two sides of
interior references. Each side stops at its first continuous Gaussian collision.
The budget includes both reference cameras, wanted views, and safe extensions.
Remaining space is shared between the two sides, or reassigned to the safe side.
Leading extensions use groups of four to align the first reference with the VAE;
any remaining temporal padding is not exported. Paths stay extension → reference
→ captured leg → reference → extension. Long legs retain bounded windows.
Both generated extension sets enter fitting once at their exact cameras. The
plan cache includes the mode and frame budget so a fixed warm-up plan cannot be
reused accidentally. This experiment may improve endpoint context, but image
sharpness and reconstruction quality still require review.

**Use GPT-repaired RGB at exact reference poses**
(`teacher_reference_rgb_replacement`, on by default to retain existing seeded
behavior) explicitly controls replacement of rendered RGB by the calibrated
GPT repair. It applies to both inline endpoints of extended short clips, and
only those exact camera IDs. Both use the separate reference-opacity setting;
interpolated RGB stays unchanged. These reference frames are masked from generated
exports, and their original repairs supervise reconstruction. Turning replacement
off requires reference-latent preservation to be off; the dashboard switches to
seed-once mode. Two-sided extension always leaves reference latents unlocked.

Teacher review orders generated images by their recorded position in the model
input, with clip number, input position and warm-up labels. Global camera IDs
remain stable for reconstruction; sorting them numerically would put appended
extension cameras after the ordinary targets and falsely show a teleport. The
worker validates the extension/reference/target join before loading the model
and records each planned path in `inference-teacher.json`. Ordering validation
does not guarantee sharp output or remove the teacher's temporal VAE artifacts.

**Interpolated-view opacity multiplier** is a multiplier in [0,1], default 1.
It affects the rendered opacity supplied to the teacher in both inference passes,
including interpolated and warm-up views. For example, 0.5 maps opacity 0.8 to
0.4 before inference. RGB pixels, stored depth and camera geometry are unchanged.
The repaired reference uses its separate
**Repaired-reference opacity** value after scaling (default 1, absolute rather
than multiplied). Values below 1 require an unlocked reference: Seed once or
endpoint extension. Preserving a fixed reference latent requires opacity 1.
Lower values increase the noise component in the
initial latent and reduce supplied opacity conditioning. They are not a precise
spatial mask for invalid geometry, and can increase hallucination or drift.

The upstream encoder groups temporal frames in fours after its initial frame,
and initializes latents using max-pooled opacity. Warm-up lengths divisible by
four place the interior reference at the end of a temporal group, preventing its
full opacity from being pooled into the next group's wanted frames. This is a
conditioning hypothesis, not proof of improved sharpness.

Compare the same source, camera IDs, resolution and random seed. The current
opacity experiment uses 50 steps, interpolated opacity ×0.5, reference opacity
0.8 and the same 8-frame extensions with an unlocked reference. The prior run
is the opacity-1 control; this changes both opacity controls, so it cannot
isolate their individual effect.
Only then compare 75 steps at otherwise identical settings. Compare the same
wanted cameras near each endpoint, not different clip/gallery positions.
More denoising steps do not by themselves remove strong blurry-render conditioning.


## 14B distilled student in Step 4 (October 27)

New jobs now select `artifixer-14b.pt`, not `artifixer-s1-14b.pt`. The student
checkpoint was downloaded again and verified against NVIDIA's SHA256
`c1a6d31fb849211d4c682a28b40980549cd8f807ee309e7bc0141a336ffcd16b`
(67,644,337,412 bytes). Both the teacher and small student remain cached.
Old job staging was cleared at the user's explicit request; incomplete remote
archives were permitted to be discarded for this cleanup. Existing local files
were retained. Qwen caption weights were already absent; the remaining small
Qwen directory contained tokenizer/configuration metadata.

Both students use the same `official_worker.inference` path, authors' KV sampler,
four steps, seven-frame latent blocks, camera/ray construction, source opacity,
reference ordering and exports. Both expose identical generation controls in
Step 4. Teacher controls remain opt-in and do not route a student through the
bidirectional clip planner. `--inference-model 14b` selects the large student;
`--inference-model 1.3b` selects the small student. Historical requests without a
selection continue to derive their original model from their checkpoint.

The large student uses the shared strict BF16 empty-parameter loader. Its
`student_memory.py` adapter changes cache residency only. It reserves activation
headroom and a single staged layer on GPU, keeps as many complete layer caches
resident as fit, and places the remainder in host RAM. Each offloaded block
materializes its temporal/text/reference cache on CUDA, executes the original
block and attention processors, then copies cache changes back synchronously.
This preserves the full requested context, attention sinks, rolling eviction,
reference count, and dtype. Connected bridges retain their existing seven-frame
cache policy. References are VAE-encoded one at a time. There is no quantization
or automatic smaller-model/resolution/context fallback.

Cache planning considers free device memory (capped at 80 GiB), estimated temporal,
reference and text caches, and Linux/Slurm host memory limits. It prefers 12 GiB
of GPU headroom plus one staged cache layer and 4 GiB of host headroom. For
reference-heavy calls it can use 10/3 or 8/2 GiB GPU/host headroom, retaining the
same tensors and full context. Immutable bridge-preservation K/V is budgeted
separately when that optional adapter is selected. Estimates
are preflight checks, not guarantees for arbitrary inputs. Telemetry is written
to `inference-memory.jsonl` (or `plus-memory.jsonl`). CPU/GPU transfers add latency.
The adapter is single-GPU, inference-only, and tied to the pinned authors' block
cache interface; it does not patch the clean upstream checkout.

### Native long-window validation

Slurm step `5875153.56` completed successfully on A100-SXM4-80GB (exit 0).
The production worker generated **92 frames at 1280 × 720**, retaining all six
repaired references, the saved caption, camera scale and poses, seed 42 and four
sampling steps per seven-latent-frame block. There were 93 real sequence inputs
including the starter, padded by upstream to 109 RGB inputs / 28 latent frames.
The fourth autoregressive block exercised eviction in the full 21-frame cache.
No smaller resolution, shorter context, lower precision or teacher sampler was used.

Peak allocated CUDA memory was **69.10 GiB**, peak reserved **73.16 GiB**;
a live `nvidia-smi` sample showed 75,435 MiB total device usage. Slurm reported
MaxRSS **50,243,676 KiB** (47.92 GiB) within the 62 GiB step / 64 GiB allocation.
The measured worker duration was **603.65 seconds**, including model load,
conditioning, sampling, decoding and PNG export (Slurm elapsed 10m10s).
The planner put 24 of 40 layers' caches in host RAM (estimated 44.73 GiB) and
kept the other 16 on CUDA. Complete cache size including references/text was
estimated at 74.55 GiB, before weights and activations.

All expected global frame IDs and output sizes were verified. Results and
telemetry are in `outputs/artifixer-student-14b/success.json`,
`inference-memory.jsonl`, and `previews/`. This validates the stated inference
workload, not a completed reconstruction or a multiview-quality improvement.
The initial conservative host check rejected clean checkpoint pages counted as
active file cache; the corrected check treats clean unmapped pages as
reclaimable while excluding mapped/shared/dirty pages and anonymous tensors.

### Connected generation validation

A second native-resolution probe used the existing connected planner on the
first real 23-camera interval, including both inward outer legs, RGB-seeded
middle bridging and generated-context references. It completed all **21 target
frames** in **396.70 seconds**. Reference counts grew from 2 to 8 to 12 as the
existing context policy reused generated images. Peak allocated memory was
**71.36 GiB** and peak reserved **75.50 GiB**. Original references and retained
outer images passed the existing checksum assertions; every target camera was
exported exactly once. See `outputs/artifixer-student-14b/connected/success.json`.

### Twelve references with the full temporal cache

A third probe kept all six repaired references plus six actual generated
context images and the full 21-frame temporal cache. It generated **21 frames
at 1280 × 720** in **249.61 seconds**, using the same four-step causal sampler.
Peak allocated CUDA memory was **73.60 GiB**, peak reserved **76.38 GiB**.
Slurm step `5875153.59` completed with exit 0 and MaxRSS **61,716,592 KiB**
(58.86 GiB), within the 62 GiB step limit. The planner used 25 CPU cache layers,
15 GPU cache layers and the 8/2 GiB activation/host headroom option. Full caches
were estimated at 91.03 GiB, with 56.89 GiB on the host. See
`outputs/artifixer-student-14b/references-12/success.json`.

All three probes' output images, manifests and telemetry were saved locally.
These measured workloads fit the current A100 allocation; larger resolutions,
more references or other workloads sharing memory still require preflight checks.
