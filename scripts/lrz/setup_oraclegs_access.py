#!/usr/bin/env python3
"""Run locally: securely authenticate LRZ and cache SEVA's official v1.1 assets."""
import argparse
import getpass
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

# Code travels in argv; the secret travels only over SSH stdin. Never add a
# token argument, shell interpolation, environment dump, or traceback here.
REMOTE = r'''
import json, os, sys
from huggingface_hub import login, hf_hub_download
os.umask(0o077)
copy_only = '--copy-only' in sys.argv[1:]
token = None if copy_only else sys.stdin.readline().strip()
if not copy_only and not token:
    print("No token received.", file=sys.stderr)
    sys.exit(2)
try:
    if not copy_only:
        login(token=token, add_to_git_credential=False)
    paths = [hf_hub_download(repo_id="stabilityai/stable-virtual-camera",
                            filename=name, token=token, local_files_only=copy_only)
             for name in ("config.yaml", "modelv1.1.safetensors")]
except Exception as exc:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    print("SEVA setup failed: " + type(exc).__name__ +
          (" (HTTP " + str(status) + ")" if status else "") +
          ". Check model access, token permissions, network and disk space.", file=sys.stderr)
    sys.exit(1)
finally:
    token = None
print("ORACLEGS_ASSETS=" + json.dumps(paths))
'''


def remote_command(cfg, *, copy_only=False):
    from splat_explorer import repair_lrz as lrz
    inner = shlex.join([
        'env', '-u', 'PYTHONPATH', '-u', 'PYTHONHOME', 'PYTHONNOUSERSITE=1',
        'LD_LIBRARY_PATH=/workspace/oraclegs-model-env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
        '/workspace/oraclegs-model-env/bin/python', '-c', REMOTE,
        *(['--copy-only'] if copy_only else []),
    ])
    return lrz.ssh_argv(cfg, multiplex=True) + [
        lrz.container_srun_prefix(cfg) + 'bash -lc ' + shlex.quote(inner)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-local', type=Path, metavar='DIRECTORY',
                        help='Also copy the two downloaded assets to this local directory (several GB).')
    parser.add_argument('--copy-only', action='store_true',
                        help='Copy already cached worker files without prompting for a token or downloading again.')
    parser.add_argument('--check-only', action='store_true',
                        help='Verify cached worker files without a token prompt or any local model copy.')
    args = parser.parse_args()
    if args.check_only and (args.copy_only or args.download_local):
        parser.error('--check-only cannot be combined with copying options')
    if args.copy_only and not args.download_local:
        parser.error('--copy-only requires --download-local DIRECTORY')
    from splat_explorer import repair_lrz as lrz
    cfg = lrz.load_lrz_config()
    if not lrz.lrz_session_alive(cfg):
        parser.error('Connect to LRZ in Splatfix first; this script reuses its SSH session.')
    token = None
    if args.copy_only or args.check_only:
        print('Locating cached SEVA files on LRZ; no token required.', flush=True)
        command = remote_command(cfg, copy_only=True)
    else:
        print('First accept access at https://huggingface.co/stabilityai/stable-virtual-camera')
        print('Use a Hugging Face read token authorized for that gated model.')
        print("The token will be saved in the worker account's Hugging Face credential store.")
        token = getpass.getpass('Hugging Face token (hidden): ').strip()
        if not token or any(c.isspace() for c in token):
            parser.error('A nonempty token without whitespace is required.')
        print('Authenticating and downloading SEVA v1.1 on LRZ; this may take several minutes.', flush=True)
        command = remote_command(cfg)
    try:
        result = subprocess.run(command, input=(token+'\n') if token else '', text=True,
                                stdout=subprocess.PIPE, check=False)
    finally:
        token = None
    if result.returncode:
        raise SystemExit('Worker setup failed; no reconstruction was launched.')
    lines = [s for s in result.stdout.splitlines() if s.startswith('ORACLEGS_ASSETS=')]
    if len(lines) != 1:
        raise SystemExit('Worker did not confirm both model files. Please retry.')
    paths = json.loads(lines[0].split('=', 1)[1])
    if args.download_local:
        destination = args.download_local.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for source in paths:
            # Hub cache files are symlinks; copy the bytes, not dangling links.
            try:
                subprocess.run(['rsync', '-aL', '--partial', '-e', lrz.rsync_ssh_cmd(cfg),
                                f"{cfg['user']}@{cfg['host']}:{shlex.quote(source)}",
                                str(destination)+'/'], check=True)
            except subprocess.CalledProcessError:
                raise SystemExit('Worker files remain cached. Local copy failed; retry with '
                                 '--copy-only --download-local ' + shlex.quote(str(destination))) from None
        print('Local files saved to:', destination)
    print('Both official SEVA files are cached on LRZ. Ready to retry OracleGS.')


if __name__ == '__main__':
    main()
