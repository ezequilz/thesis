"""Startup install planning: reinstall only when a required version changed."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "start_deps", ROOT / "scripts" / "start_deps.py"
)
start_deps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start_deps)


class Dist:
    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name):
        if name == "direct_url.json" and self._direct_url is not None:
            import json
            return json.dumps(self._direct_url)
        return None


def _lookup(packages):
    def lookup(name):
        return packages.get(name.lower()) or packages.get(name)
    return lookup


def test_release_already_satisfying_the_specifier_is_kept():
    reason = start_deps.requirement_reason("viser>=0.2.7", Dist("1.1.0"))
    assert reason is None


def test_release_below_the_specifier_is_reinstalled():
    reason = start_deps.requirement_reason("viser>=0.2.7", Dist("0.2.0"))
    assert "0.2.0" in reason
    assert "viser" in reason


def test_missing_package_is_reinstalled():
    reason = start_deps.requirement_reason("numpy>=1.26", None)
    assert "not installed" in reason


def test_matching_git_commit_is_kept():
    commit = "9e53709a94e9c56fb66db2708d7d728832176713"
    url = f"git+https://github.com/RobotFlow-Labs/gsplat-mlx.git@{commit}"
    dist = Dist("0.1.0", {
        "url": "https://github.com/RobotFlow-Labs/gsplat-mlx.git",
        "vcs_info": {"vcs": "git", "commit_id": commit},
    })
    assert start_deps.requirement_reason(f"gsplat-mlx @ {url}", dist) is None


def test_moved_git_commit_is_reinstalled():
    url = (
        "git+https://github.com/RobotFlow-Labs/gsplat-mlx.git@"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    dist = Dist("0.1.0", {
        "url": "https://github.com/RobotFlow-Labs/gsplat-mlx.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "9e53709a94e9c56fb66db2708d7d728832176713",
        },
    })
    reason = start_deps.requirement_reason(f"gsplat-mlx @ {url}", dist)
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in reason


def test_git_install_is_replaced_when_the_pin_becomes_a_release():
    dist = Dist("1.4.0", {
        "url": "https://github.com/example/gsplat.git",
        "vcs_info": {"vcs": "git", "commit_id": "abc"},
    })
    reason = start_deps.requirement_reason("gsplat>=1.4", dist)
    assert "release" in reason


def test_false_environment_marker_is_ignored():
    reason = start_deps.requirement_reason(
        'mlx>=0.31.0; sys_platform == "never"', Dist("0.0.1")
    )
    assert reason is None


def test_editable_project_at_this_checkout_matches(tmp_path):
    project = {"name": "splat-explorer", "version": "0.1.0", "dependencies": []}
    dist = Dist("0.1.0", {
        "url": tmp_path.resolve().as_uri(),
        "dir_info": {"editable": True},
    })
    assert start_deps.project_reason(project, tmp_path, dist) is None


def test_non_editable_project_is_reinstalled(tmp_path):
    project = {"name": "splat-explorer", "version": "0.1.0"}
    reason = start_deps.project_reason(project, tmp_path, Dist("0.1.0"))
    assert "editable" in reason


def test_plan_installs_only_the_stale_requirement(tmp_path):
    project = {
        "name": "splat-explorer",
        "version": "0.1.0",
        "dependencies": ["numpy>=1.26", "pillow>=10.0"],
        "optional-dependencies": {"viewer": ["viser>=0.2.7"]},
    }
    packages = {
        "splat-explorer": Dist("0.1.0", {
            "url": tmp_path.resolve().as_uri(),
            "dir_info": {"editable": True},
        }),
        "numpy": Dist("2.0.0"),
        "pillow": Dist("10.1.0"),
        "viser": Dist("0.1.0"),
    }
    items = start_deps.stale_items(
        project, ["viewer"], tmp_path, _lookup(packages)
    )
    assert start_deps.pip_install_args(["viewer"], items, force=False) == [
        "viser>=0.2.7"
    ]


def test_force_reinstalls_the_selected_extras_together():
    args = start_deps.pip_install_args(
        ["viewer", "vlm", "apple"], [], force=True
    )
    assert args == ["-e", ".[viewer,vlm,apple]"]


def test_repo_pyproject_pins_are_readable():
    project = start_deps.load_project(ROOT / "pyproject.toml")
    texts = [
        text for text, _extra in start_deps.selected_requirements(
            project, ["viewer", "vlm", "apple"]
        )
    ]
    assert "viser>=0.2.7" in texts
    assert any(text.startswith("gsplat-mlx @ git+") for text in texts)


def test_fingerprint_tracks_library_bytes_and_ignores_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.py").write_text("v1")
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "lib.pyc").write_bytes(b"stale")
    (tmp_path / "lrz.local.yaml").write_text("job: 1")
    first = start_deps.fingerprint(
        ["src", "lrz.local.yaml"], {"lrz.local.yaml"}, tmp_path
    )
    (cache / "lib.pyc").write_bytes(b"other")
    (tmp_path / "lrz.local.yaml").write_text("job: 2")
    assert start_deps.fingerprint(
        ["src", "lrz.local.yaml"], {"lrz.local.yaml"}, tmp_path
    ) == first
    (src / "lib.py").write_text("v2")
    assert start_deps.fingerprint(
        ["src", "lrz.local.yaml"], {"lrz.local.yaml"}, tmp_path
    ) != first


def test_install_returns_10_after_pip_runs(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "splat-explorer"\nversion = "0.1.0"\n'
        'dependencies = ["numpy>=1.26"]\n'
        '[project.optional-dependencies]\nviewer = ["viser>=0.2.7"]\n'
    )
    monkeypatch.setattr(
        start_deps,
        "stale_items",
        lambda *args, **kwargs: [(["viser>=0.2.7"], "viser 0.1.0 does not satisfy >=0.2.7")],
    )
    monkeypatch.setattr(start_deps, "_pip_cache_dir", lambda: "/cache/pip")
    calls = []
    monkeypatch.setattr(
        start_deps.subprocess,
        "check_call",
        lambda cmd, cwd: calls.append((cmd, cwd)),
    )
    assert start_deps.cmd_install(["viewer"], False, tmp_path) == 10
    cmd, cwd = calls[0]
    assert cwd == tmp_path
    assert cmd[-1] == "viser>=0.2.7"
    assert "only-if-needed" in cmd
    assert "--no-cache-dir" not in cmd


def test_install_skips_pip_when_nothing_is_stale(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "splat-explorer"\nversion = "0.1.0"\n'
        'dependencies = ["numpy>=1.26"]\n'
    )
    monkeypatch.setattr(
        start_deps, "stale_items", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(start_deps, "_pip_cache_dir", lambda: "/cache/pip")
    def fail(*args, **kwargs):
        raise AssertionError("pip should not run")
    monkeypatch.setattr(start_deps.subprocess, "check_call", fail)
    assert start_deps.cmd_install([], False, tmp_path) == 0
