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

In `/splatfix`, Step 4 → **ArtiFixer inference model** → **14B bidirectional
teacher · experimental**. Defaults are 17 real frames per independent clip,
50 diffusion steps and the `480p` inference-only limit. A 1280 × 720 input
becomes 832 × 464 (both dimensions VAE-aligned). RGB and opacity are resized
together; all per-frame intrinsics are scaled and poses/distortion retained.
Predictions are bilinearly restored to the original camera dimensions for
reconstruction. This does not create native-resolution detail.

The teacher reads one clip's RGB/opacity from disk at a time. Depth is not a
teacher inference input here; it is used elsewhere in preparation/fitting.
All selected reference images remain conditioning (subject to the existing
endpoint reference policy), and the VAE encodes references individually.
Predictions exclude trusted anchors and preserve global frame IDs.

Reducing the number of selected references can also reduce conditioning memory,
at the cost of scene coverage. Encoding references one at a time retains all
references while reducing the VAE's temporary peak; it does not remove their
latent attention cost. The teacher uses upstream's unrestricted bidirectional
attention (`frames_per_block=None`). The input clip limit does not alter the
distilled checkpoint's trained causal block/window geometry.

Independent teacher clips have full bidirectional attention within each clip;
they do not share temporal state. Clip boundaries can introduce seams. The
causal starter/cache, connected-bridge, generated-context and GPT-handoff
adapters are explicitly disabled and the effective settings are saved. No
automatic smaller-model or lower-quality fallback occurs in normal Step 4.
The existing 1.3B model remains the default until a comparison is chosen.

CLI equivalents:

```bash
splatfix repair CHECKPOINT --output OUTPUT \
  --inference-model 14b-teacher --teacher-resolution 480p \
  --teacher-max-frames 17 --teacher-steps 50
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
this account quota. The teacher's published SHA256 is
`8ce0d5333803e42574d229ed702038833f4c9a44feb9dcf354ad035e93172940`.

### When two A100s would help

First measure a bounded teacher clip on one GPU. Two GPUs are useful if larger
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

Sources: [NVIDIA checkpoint/model card](https://huggingface.co/nvidia/ArtiFixer),
[teacher checkpoint and checksum](https://huggingface.co/nvidia/ArtiFixer/blob/main/artifixer-s1-14b.pt),
[pinned inference entry point](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_eval/run_inference.py),
[pinned KV-cache allocation](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/kv_cache_pipeline.py),
[pinned teacher sampler](https://github.com/nv-tlabs/ArtiFixer/blob/a392c4dfe17459ef9952407accdb9fcdcdddba98/model_training/pipeline/pipeline.py).
