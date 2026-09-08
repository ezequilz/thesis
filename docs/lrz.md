# LRZ GPU: reserve, connect, CUDA repair

Handover for this repo’s GSFix3D photometric lift on LRZ AI Systems.
Password is never stored. Do not put NGC keys in git.

Official cluster docs: <https://doku.lrz.de/display/PUBLIC/LRZ+AI+Systems>  
Status: <https://status.lrz.de/affected/ai-systems>  
Enroot/NGC: <https://doku.lrz.de/4-2-enroot-images-from-nvidia-ngc-2747831798.html>

## What you are allocating

A 6-hour **hold job** (`sleep`) that keeps one GPU. Repair work is extra
`srun --overlap` steps on that job. Enroot/Pyxis exist **only on compute
nodes**, not on the login node.

Default account used here: `go73kaf2` @ `login.ai.lrz.de`  
DSS workspace: `/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer`

Copy config once on your laptop:

```bash
cp configs/lrz.example.yaml configs/lrz.local.yaml
# set user / workspace / job_id (job_id is rewritten by allocate.sh)
```

`configs/lrz.local.yaml` is gitignored.

## Every session (laptop)

1. Connect **eduVPN**.
2. From the **project repo**, open a multiplexed SSH session (type the LRZ
   password once):

```bash
scripts/lrz/ssh-session.sh
```

That is the login that already worked, plus ControlMaster:

```bash
/usr/bin/ssh -4 -F /dev/null \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  -o NumberOfPasswordPrompts=1 \
  go73kaf2@login.ai.lrz.de
```

Do **not** use `ssh -fN` (Cursor’s terminal then rejects a correct password).
Do **not** offer SSH keys first (`PubkeyAuthentication=no`). If the login
node answers `Exceeded MaxStartups`, wait a minute and retry.

Socket: `~/.ssh/cm-lrz` (8h). Close with:

```bash
/usr/bin/ssh -o ControlPath=$HOME/.ssh/cm-lrz -O exit go73kaf2@login.ai.lrz.de
```

3. If you do not already have a running GPU job, allocate **once** (no
   `squeue` loop):

```bash
scripts/lrz/allocate.sh
```

That submits:

```bash
sbatch --job-name=gs-debug \
  --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
  --time=06:00:00 --output=gs-debug-%j.log \
  --wrap='sleep 21600'
```

Both A100 partitions are listed on purpose. Submitting only
`lrz-hgx-a100-80x4` can sit in `PD (Priority)` while a DGX A100 is free.
If that happens, one `scontrol` (not a loop):

```bash
scontrol update JobId=<JOBID> Partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
```

`allocate.sh` writes `job_id` into `configs/lrz.local.yaml`. Confirm with
one `squeue --me` (already printed by `ssh-session.sh`). Job state must be
`R` before anything else. Never poll `squeue` in a loop — LRZ treats that
as a DoS on the controller.

4. Open a GPU shell (needs `--overlap` because the batch step is `sleep`):

```bash
scripts/lrz/gpu-shell.sh
```

Equivalent:

```bash
/usr/bin/ssh -4 -t -F /dev/null \
  -o ControlMaster=no -o ControlPath="$HOME/.ssh/cm-lrz" \
  go73kaf2@login.ai.lrz.de \
  'srun --jobid=<JOBID> --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --pty bash'
```

Without `--overlap` you get `Job … step creation temporarily disabled,
retrying (Requested nodes are busy)`.

## Once per machine: NGC credentials (compute-node home)

Enroot on LRZ talks to NGC with a credentials file. Anonymous import of
`nvcr.io/nvidia/pytorch` fails.

1. Create an NVIDIA account: <https://ngc.nvidia.com/signin>
2. Setup → Get API Key
3. On the **GPU node** (or any node that sees the same home):

```bash
mkdir -p ~/enroot
cat > ~/enroot/.credentials <<'EOF'
machine nvcr.io login $oauthtoken password PASTE_NGC_API_KEY_HERE
machine authn.nvidia.com login $oauthtoken password PASTE_NGC_API_KEY_HERE

EOF
chmod 600 ~/enroot/.credentials
```

Leave the blank line at the end. `$oauthtoken` is the literal login name
NGC expects, not a shell variable.

Home for this account is `/dss/dsshome1/0F/go73kaf2`.

## Once per DSS workspace: import PyTorch squashfs

Enroot is only on the GPU node. From the shell opened by `gpu-shell.sh`:

```bash
cd /dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer
mkdir -p containers inputs outputs logs code
```

**URI that failed** (slash is parsed as Docker Hub, hence
`registry-1.docker.io/.../nvcr.io/nvidia/pytorch` → 401):

```bash
# WRONG
enroot import -o containers/pytorch.sqsh docker://nvcr.io/nvidia/pytorch:24.10-py3
```

**URI that hits NGC** (`#` is Enroot’s registry separator):

```bash
enroot import -o containers/pytorch.sqsh 'docker://nvcr.io#nvidia/pytorch:24.10-py3'
```

This is large and slow; it only needs to succeed once. Reuse
`containers/pytorch.sqsh` across later allocations.

If you have no NGC key yet, public Docker Hub (CUDA devel, includes `nvcc`
so `gsplat` can compile):

```bash
enroot import -o containers/pytorch.sqsh docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Driver on the A100 nodes has been CUDA 13.0; a CUDA 12.x image is fine.

## Once per allocation: named Pyxis container

Still on the GPU node, after the `.sqsh` exists. `--overlap` again:

```bash
srun --jobid=<JOBID> --overlap --gres=gpu:1 \
  --container-image=/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer/containers/pytorch.sqsh \
  --container-name=splat-repair \
  --container-mounts=/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer:/workspace \
  --pty bash
```

Inside the container (first time this job; compiles `gsplat`, slow):

```bash
python -c 'import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)'
# code/ is rsynced by the first dashboard repair; PYTHONPATH is enough:
#   PYTHONPATH=/workspace/code/src
```

The dashboard worker is:

```text
PYTHONPATH=/workspace/code/src python -m splat_explorer.repair_lrz --job-dir /workspace/inputs/<id>
```

## Dashboard (after squashfs exists)

Keep eduVPN up and `~/.ssh/cm-lrz` alive.

1. Local stack: `./scripts/start.sh` → <http://localhost:8090/repair>
2. Backend **gsplat CUDA (GSFix3D)** — paper photometric lift, no color
   stamp. **gsplat CUDA (baseline)** is the frozen pre-paper lift for A/B.
3. Repair this view. First run rsyncs `src/` to DSS, then `srun --overlap`
   on job `job_id`.

If the socket is gone: `scripts/lrz/ssh-session.sh` again. If the job is
not `R`, allocate a new one and update `job_id`.

## Scripts

| Script | What it does |
| --- | --- |
| `scripts/lrz/ssh-session.sh` | Password once → `~/.ssh/cm-lrz`; one `squeue --me` |
| `scripts/lrz/allocate.sh` | One `sbatch`; writes `job_id` into `lrz.local.yaml` |
| `scripts/lrz/gpu-shell.sh` | `srun --overlap --pty bash` on the allocated node |
| `scripts/lrz/bootstrap.sh` | Prints/checks workspace + the import commands above |
| `scripts/lrz/run-repair.sh` | Manual rsync + remote worker (dashboard usually does this) |

## Gotchas we already hit

- Password-only SSH. Keys first → `Permission denied` after three prompts.
- `ssh -fN` in Cursor → same, even with the right password.
- Hold job is `sleep` → extra steps need `--overlap`.
- Enroot `docker://nvcr.io/nvidia/...` goes to Docker Hub. Use `nvcr.io#…`.
- NGC still 401s without `~/enroot/.credentials`.
- Do not automate `squeue`/`sinfo` in a loop.
- Jobs without `--gres=gpu:N` stay `PD` with `QOSMinGRES`.
- Login-node sessions older than 30 days are killed.
- Do not `sudo`.
