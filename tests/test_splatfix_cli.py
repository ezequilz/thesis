"""Exercise the public offline selection command with real rendering and disk IO."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from splat_explorer.scene import GaussianScene, save_ply
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.cli import add_parser, repair_splat


def test_public_select_command_without_api(tmp_path):
    random = np.random.default_rng(8)
    count = 60
    scene = GaussianScene(
        means=random.uniform(-2, 2, (count, 3)).astype(np.float32),
        scales=np.full((count, 3), .15, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1)),
        opacities=np.full(count, .9, np.float32),
        colors=random.uniform(0, 1, (count, 3)).astype(np.float32),
    )
    source = tmp_path / "scene.ply"
    save_ply(scene, source)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "scene": {"path": str(source)},
        "camera": {"start_position": [0, 0, 0], "up_axis": "+y"},
        "renderer": {"width": 64, "height": 48},
        "agent": {"vlm_backend": "scripted"},
    }))
    output = tmp_path / "runs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([
        sys.executable, "-m", "splat_explorer.splatfix.cli", "--config", str(config),
        "select", "--output", str(output), "--views", "8", "--select-only",
    ], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = Checkpoint.load(next(output.iterdir()))
    assert checkpoint.complete and len(checkpoint.views) == 8
    assert len(list(checkpoint.root.rglob("*.png"))) == 8
    assert all("repaired_rgb" not in view for view in checkpoint.views)
    assert checkpoint.manifest["metadata"]["scene_load"]["min_opacity"] == .05
    assert (checkpoint.root / "actions.jsonl").is_file()


def test_cli_defaults_and_invalid_view_count():
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(["splatfix", "select"])
    assert args.views == 6
    with pytest.raises(SystemExit):
        parser.parse_args(["splatfix", "select", "--views", "0"])


def test_repair_command_passes_only_saved_run_and_runtime(monkeypatch, tmp_path):
    from splat_explorer.config import Config
    from splat_explorer.splatfix import repair

    received = {}
    def run(checkpoint, output, **kwargs):
        received.update(checkpoint=checkpoint, output=output, **kwargs)
        return {"splat_path": str(output / "model.ply")}
    monkeypatch.setattr(repair, "run_repair", run)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(["splatfix", "repair", str(tmp_path / "saved"),
                             "--output", str(tmp_path / "out"), "--mode", "baseline",
                             "--artifixer-repo", "/gpu/official"])
    repair_splat(Config({"splatfix": {"runtime": {"python": "/gpu/python"}}}), args)
    assert received["mode"] == "baseline"
    assert received["runtime"] == {"repo": "/gpu/official", "python": "/gpu/python"}
    assert received["checkpoint"] == tmp_path / "saved"
