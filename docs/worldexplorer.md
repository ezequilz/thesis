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
SEVA and compatible cached model blobs are shared. WorldExplorer's modified
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
Manual scaffold submission requires two hours left; expansion requires eight.

Full-pipeline/expansion submission requires eight hours remaining on the selected reservation. Allocate
more for a first generation's model downloads. WorldExplorer does not extend the
reservation or allocate a second GPU. Use the GPU exclusively: a per-allocation
lock prevents two WorldExplorer jobs from overlapping, but existing reconstruction
methods have their own queue and must not run concurrently with it.

## Sleep, reconnect, download

Wait until submission returns a job ID and **Refresh remote** confirms startup.
`nohup` runs the `srun --overlap` launcher on the login node, with stdin closed and
logs on DSS. The GPU process does not depend on the laptop, dashboard, VPN or SSH
ControlMaster remaining connected. Slurm allocation expiry and cluster failures
still terminate it; there is no automatic resumption of an interrupted pipeline.

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
