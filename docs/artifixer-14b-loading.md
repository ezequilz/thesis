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
