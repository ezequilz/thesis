# WorldExplorer: independent generative starter

Open `/splatfix/worldexplorer` (the **Generate scene** link on `/splatfix`).
Restart the dashboard after updating the code. WorldExplorer does not consume a
checkpoint or enter the existing reconstruction queue. Its code lives in
`src/splat_explorer/splatfix/worldexplorer/`.

The pinned authors' implementation is
<https://github.com/mschneider456/worldexplorer/tree/e661fcc71fb6cf087bcd9cc676345e30e11aa6b2>.
This is an integration of that pipeline, not a reimplementation of its models.

## First setup

Connect eduVPN and `scripts/lrz/ssh-session.sh`, then select your existing running
GPU reservation on `/repair/gpu`. OracleGS's micromamba bootstrap and the existing
PyTorch squashfs must already exist. Click **Prepare WorldExplorer runtime**, or:

```sh
.venv/bin/python worldexplorer.py setup
```

Preparation itself runs detached on the reserved GPU; refresh its job to inspect
progress. Only WorldExplorer's control files are uploaded. It uses its own named
Pyxis container, Python environment, extension cache and CUDA toolkit when the
container's CUDA is not 12.8. It never installs packages into OracleGS environments.
The installer requires the existing SEVA weight to be locally available and
checks gated FLUX access, CUDA availability and model imports before publishing
the readiness marker. Full FLUX/inpainting weights download on first generation.
The saved Hugging Face token is read on LRZ, never copied to this project.

The account's `~/.cache/huggingface` is resolved through the passwd database,
exactly as OracleGS does, instead of using the redirected container HOME.
SEVA v1.1 (`modelv1.1.safetensors`) is reused from OracleGS's cache instead of
WorldExplorer's default v1.0. Setup validates checkpoint keys and tensor shapes
against WorldExplorer's model; generation subprocesses use the v1.1 checkpoint.
Other compatible cached model blobs are also shared. WorldExplorer's modified
SEVA **code** stays separate: substituting OracleGS's SEVA code would lose the
WorldExplorer global-scene-memory behavior. Metric Hypersim depth and video-depth
Small weights differ from OracleGS's monocular depth model and must be fetched;
they are cached once and symlinked into WorldExplorer's source checkout.

## Generate

```sh
.venv/bin/python worldexplorer.py generate "bioluminescent, gravity-defying, telepathic cosmic jellyfish hive"
```

With the project's venv activated, the requested `python worldexplorer.py generate
"…"` spelling works as well. An installed package also exposes `worldexplorer`.
Optional flags: `--mode automatic|fast|manual`, `--translation-scaling 3.0`, and
`--num-images-for-vggt 40`. The original prompt is stored verbatim and passed as a
literal CLI argument. A bounded ASCII folder slug prevents shell/path parsing
issues in upstream's internally assembled commands.

The VGGT budget is for generated views **in addition to** the eight scaffold
views, as defined by the authors. The integration accepts 32–100 so every one
of the 32 trajectories can contribute. Sampling retains the authors' algorithm:
divide the budget by the trajectory count, sort numeric frame names, then take
every `frame_count // per_trajectory_budget` frame starting with the first.
Integer rounding makes this an approximate budget, not a strict image cap.
The default 40 selects one generated frame per trajectory plus eight inputs.
Before copying images, the adapter checks the selected camera-center spread.
If the stride samples have effectively coincident centers, it replaces each
trajectory's first sample with its farthest translated view from the scaffold
origin, retaining the same image count. This avoids fitting VGGT's scale to
only the panorama origin. Rotation-only trajectories with no usable baseline
fail before VGGT runs; non-finite camera centers are also rejected.

The runtime adapter fixes two upstream path assumptions: input images are
identified by their immediate `input` directory, and trajectories are grouped
by their enclosing trajectory directory rather than the shared `samples-rgb`
name. This prevents `input_scaffold` in a scene name from excluding every frame.
Empty generated sets and insufficient budgets raise explicit errors; short
trajectories use stride one. Repeat selection replaces the derived selection
folder so old frames cannot enter VGGT. These fixes do not change generation,
VGGT inference/alignment, training, or export.

For the one-shot quick-start command, the CLI explicitly selects
**automatic** mode, so generation proceeds through panorama creation, 32 SEVA
trajectory videos, VGGT alignment, splatfacto-big training and PLY export without
human curation. Fast mode changes panorama selection only. The dashboard defaults to the authors' **manual** mode. Custom four-direction
prompts are not exposed by this starter.

## Manual candidate selection

Select **Manual · choose candidates** in the dashboard, or run:

```sh
.venv/bin/python worldexplorer.py generate "your scene description" --mode manual
```

The remote scaffold stage creates four original cardinal views and 88 inpainting
variants for each of the four intermediate views, following the authors' seeds
and guidance scales. It exits with `awaiting_selection`, releasing the job step;
it does not hold a Python process waiting for browser input. Your reservation's
sleep-hold remains allocated normally. Keep the page visible or reopen the dashboard after startup; the candidate gallery synchronizes automatically.
The UI shows both neighboring originals for each gap, selectable candidate
thumbnails, filenames carrying guidance/seed settings, and full-size image links.
Pick one candidate each for `001`, `003`, `005`, `007`. Browser choices persist
locally across reloads.

**Expand selected panorama on LRZ** verifies candidate hashes and creates a new
expansion run with the four untouched originals and your four choices renamed
`000.png` through `007.png`. The original candidates and scaffold are preserved.
A `selection.json` records exact choices and hashes. Select a fresh reservation
on Compute first if the original reservation is no longer suitable. The expansion
run uses the current allocation, not the old scaffold job's allocation.

CLI: after downloading the scaffold, write a JSON object mapping the four indices
to their relative paths listed in `candidates.json`, then run
`python worldexplorer.py expand RUN_ID --selection choices.json`.

Submission has no minimum remaining-time requirement on the selected reservation. Allocate
more for a first generation's model downloads. WorldExplorer does not extend the
reservation or allocate a second GPU. Use the GPU exclusively: a per-allocation
lock prevents two WorldExplorer jobs from overlapping, but existing reconstruction
methods have their own queue and must not run concurrently with it.

## Sleep, reconnect, download

Wait until submission returns a job ID and **Refresh remote** confirms startup.
`nohup` runs the `srun --overlap` launcher on the login node, with stdin closed and
logs on DSS. The GPU process does not depend on the laptop, dashboard, VPN or SSH
ControlMaster remaining connected. Slurm allocation expiry and cluster failures
still terminate it; recovery is explicitly submitted as a new run.

### Continue from completed trajectories

After the archive download is verified, failed/interrupted generation runs offer
**Continue from saved trajectories**. Completed runs offer **Retrain from saved
trajectories**. CLI:

```sh
.venv/bin/python worldexplorer.py resume we_20261020_025819_5d494b3a09a5
```

Recovery checks the archive manifest and SHA-256 of the merged camera transforms
and every referenced image. All 32 generated trajectories and eight scaffold
views must exist. Incomplete trajectory stages are rejected before submission.
Select a running GPU allocation on Compute; recovery uses that current allocation.

A new run receives only the merged transforms and referenced images. Absolute
image paths are relocated to the new run. The failed run and local archive stay
unchanged. `source_run` and `recovery.json` record provenance. An active recovery,
including an uncertain submission, is returned instead of launching a duplicate.

The adapter resumes at image selection, then runs the authors' existing VGGT
alignment, Gaussian training and export code. It extracts the suffix of the
pinned upstream `TrajectoryTransformer.run` at the checked
`select_images_for_vggt` boundary; an unknown boundary fails explicitly. It does
not regenerate panorama/trajectory images. Training starts fresh from the
VGGT initialization; this is not optimizer-checkpoint continuation. Changing
the generation parameters or resuming partially generated trajectories is not
supported by this recovery path.

After reconnecting VPN/SSH and restarting the local dashboard, open the WorldExplorer page. Finished runs synchronize automatically; **Refresh
remote** remains available. CLI equivalents:

```sh
.venv/bin/python worldexplorer.py list
.venv/bin/python worldexplorer.py status RUN_ID
.venv/bin/python worldexplorer.py download RUN_ID
```

Jobs persist in `outputs/worldexplorer/RUN_ID/job.json`. Each records its original
allocation and DSS workspace, so switching the GPU dashboard to a new reservation
does not redirect old-job queries. A lost SSH reply during submission is shown as
`launch_unknown`; refresh that job before considering a new submission. Network
errors preserve the last known remote state. Refresh reads DSS files and launcher
liveness and a GPU-worker heartbeat, without scheduler polling.

DSS results live under `WORKSPACE/worldexplorer-runs/RUN_ID/`. Downloads use
checksum rsync with partial-transfer support (the compatible bundled Samba binary
on macOS) into
`outputs/worldexplorer/RUN_ID/results/`, then verify generated artifact SHA-256
hashes. All panoramas, videos, intermediate geometry, training checkpoints,
configs, native/rotated PLY exports and logs remain available. Shared model/source
symlinks are excluded. The local manager checks once after startup and on every new submission. While
the WorldExplorer page is visible, it requests at most one remote check per minute;
hidden pages do not poll. A single-flight guard prevents overlapping checks.
Local UI progress refreshes do not make SSH or scheduler calls.

On discovering a finished run (including a manual scaffold or a failed setup),
the manager waits for confirmed launcher exit, builds a complete archive manifest,
downloads the files, verifies every regular file and internal link, then removes
only that run's DSS directory. A second remote guard rejects changed files before
deleting. Shared runtime/source/model links are unlinked, never traversed. Their
targets, Python/CUDA environments, model caches and credentials remain reusable.
The manifest and cleanup confirmation persist locally. Lost cleanup replies are
recovered on the next check. Failed transfers, hash mismatches, uncertain exits,
and offline laptops retain the remote run. Interrupted jobs with no confirmed
launcher exit require investigation before automatic cleanup is safe.

The dashboard provides local PLY download links and an artifact browser after
synchronization; no separate transfer button is required. CLI `download` can
explicitly trigger the same verified synchronization and cleanup. Manual scaffold
selection continues from the verified local candidates after remote cleanup.
Use `--output PATH` before the subcommand if your dashboard uses an output directory
other than the default `outputs` (pass its `worldexplorer` child).

## Integration checks and limitations

The adapter fails on nonzero upstream video/training/export subprocess exits.
It adds a `splatfacto` alias if Nerfstudio writes `splatfacto-big`, matching the
upstream export lookup, and refuses completion without a nonempty exported PLY.
The pinned authors' source and model weights remain unchanged. Each run records
its request, source revision and installed package list.

CPU tests cover literal prompt handling, mode validation, detached launch syntax,
restart/disconnect recovery, launcher failures, path confinement, missing exports,
shared-cache environment and verified downloads. Full scene quality and the
6–7-hour end-to-end GPU pipeline require a real generation run. This starter
exports results independently; importing a generated PLY into the existing
SplatFix reconstruction workflow remains a separate user action.

### SD2 inpainting source recovery

The authors' `stabilityai/stable-diffusion-2-inpainting` repository returns 404.
We use `sd2-community/stable-diffusion-2-inpainting` at
`5f74973cbb64c8568780732c17f43eb269d63a0d`. Its 13 required Diffusers files match
the Git blob IDs and LFS SHA-256 values of the separate original-repository copy
`alwold/stable-diffusion-2-inpainting` at
`e8255bd00e6a3d8dd489d2a9f03955eb08262d6e`. This cross-check supports the mirror's
provenance; the unavailable original repository cannot be queried directly.

`inpainting-lock.json` pins SHA-256 for every loaded file. Downloads stay in the
shared remote Hugging Face cache, and identical existing model blobs (including
the SD2 VAE) are reused. The runtime redirects only this exact upstream model
lookup to the verified local snapshot, using safetensors and the authors' FP16
loading setting. Setup also loads this pipeline and runs a two-step GPU inpaint
smoke check; this is not an end-to-end scene quality test.
