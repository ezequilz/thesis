#!/usr/bin/env python3
"""Version checks and reload fingerprints for scripts/start.sh.

``install`` compares the selected pyproject extras with what is already
installed and runs pip only for requirements whose version (or git
revision) no longer matches. Unchanged wheels stay in pip's download cache.

``fingerprint`` hashes library and config files so the startup script can
reload processes after those files change and leave them running otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - startup venv is 3.11+
    tomllib = None

from importlib.metadata import PackageNotFoundError, distribution

_HEX_REV = re.compile(r"[0-9a-fA-F]{7,40}")
_SKIP_PARTS = {"__pycache__", ".git"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


def load_project(path: Path):
    if tomllib is None:
        raise SystemExit("python >= 3.11 is required to read pyproject.toml")
    data = tomllib.loads(path.read_text())
    project = data.get("project") or {}
    return project


def selected_requirements(project, extras):
    """Core dependencies plus each requested extra, as written in pyproject."""
    opt = project.get("optional-dependencies") or {}
    unknown = [name for name in extras if name not in opt]
    if unknown:
        raise SystemExit("unknown extra(s): " + ", ".join(unknown))
    items = [(text, False) for text in project.get("dependencies") or []]
    for extra in extras:
        items.extend((text, True) for text in opt[extra])
    return items


def pinned_revision(url: str):
    base = url.split("#", 1)[0]
    if "://" in base:
        tail = base.split("://", 1)[1]
    else:
        tail = base
    if "@" not in tail:
        return None
    rev = tail.rsplit("@", 1)[1]
    if not rev or "/" in rev:
        return None
    return rev


def repo_identity(url: str) -> str:
    value = url.split("#", 1)[0]
    if value.startswith("git+"):
        value = value[4:]
    if "://" in value:
        head, tail = value.split("://", 1)
        if "@" in tail:
            tail = tail.rsplit("@", 1)[0]
        value = head + "://" + tail
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/").lower()


def _direct_url(dist):
    raw = dist.read_text("direct_url.json")
    if not raw:
        return {}
    return json.loads(raw)


def _file_url_path(url: str):
    if not url or not url.startswith("file:"):
        return None
    return Path(unquote(urlparse(url).path)).resolve()


def requirement_reason(original: str, dist):
    """Why this requirement must be reinstalled, or None when it already matches."""
    from packaging.requirements import Requirement

    req = Requirement(original)
    if req.marker is not None and not req.marker.evaluate():
        return None
    if dist is None:
        return f"{req.name} is not installed ({original})"
    if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
        return f"{req.name} {dist.version} does not satisfy {req.specifier}"
    meta = _direct_url(dist)
    vcs = meta.get("vcs_info") or {}
    if req.url:
        rev = pinned_revision(req.url)
        if rev is not None:
            commit = (vcs.get("commit_id") or "").lower()
            requested = vcs.get("requested_revision")
            if _HEX_REV.fullmatch(rev):
                if not commit.startswith(rev.lower()):
                    shown = vcs.get("commit_id") or "unknown"
                    return f"{req.name} commit {shown} != {rev}"
            elif requested != rev:
                shown = requested or "unknown"
                return f"{req.name} revision {shown} != {rev}"
        elif repo_identity(req.url) != repo_identity(meta.get("url") or ""):
            return f"{req.name} is installed from a different URL"
        return None
    if vcs:
        return f"{req.name} is installed from git; pyproject now asks for a release"
    return None


def project_reason(project, project_dir: Path, dist):
    name = project.get("name") or "splat-explorer"
    if dist is None:
        return f"{name} is not installed"
    meta = _direct_url(dist)
    editable = bool((meta.get("dir_info") or {}).get("editable"))
    if not editable:
        return f"{name} is not an editable install"
    installed_at = _file_url_path(meta.get("url") or "")
    if installed_at is not None and installed_at != project_dir.resolve():
        return f"{name} editable install points at {installed_at}"
    from packaging.version import Version

    wanted = str(project.get("version") or "")
    if wanted and Version(dist.version) != Version(wanted):
        return f"{name} {dist.version} != {wanted}"
    return None


def lookup_distribution(name: str):
    try:
        return distribution(name)
    except PackageNotFoundError:
        return None


def stale_items(project, extras, project_dir: Path, lookup=lookup_distribution):
    """Pip arguments and human reasons for everything that must be installed."""
    items = []
    project_name = project.get("name") or "splat-explorer"
    reason = project_reason(project, project_dir, lookup(project_name))
    if reason:
        items.append((["-e", "."], reason))
    seen = set()
    for original, _is_extra in selected_requirements(project, extras):
        if original in seen:
            continue
        seen.add(original)
        from packaging.requirements import Requirement

        req = Requirement(original)
        if req.marker is not None and not req.marker.evaluate():
            continue
        reason = requirement_reason(original, lookup(req.name))
        if reason:
            items.append(([original], reason))
    return items


def pip_install_args(extras, items, force: bool):
    if force:
        extra = ",".join(extras)
        spec = f".[{extra}]" if extra else "."
        return ["-e", spec]
    args = []
    for item_args, _reason in items:
        args.extend(item_args)
    return args


def fingerprint(paths, exclude_names, root: Path):
    digest = hashlib.sha256()
    files = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise SystemExit(f"missing path: {path}")
        if path.is_file():
            if path.name not in exclude_names:
                files.append(path)
            continue
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            if _SKIP_PARTS.intersection(file.parts):
                continue
            if file.suffix in _SKIP_SUFFIXES:
                continue
            if file.name in exclude_names:
                continue
            files.append(file)
    for file in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        rel = file.relative_to(root).as_posix().encode()
        digest.update(rel)
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _pip_cache_dir() -> str:
    env = os.environ.get("PIP_CACHE_DIR")
    if env:
        return env
    result = subprocess.run(
        [sys.executable, "-m", "pip", "cache", "dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def cmd_install(extras, force: bool, root: Path) -> int:
    os.environ.pop("PIP_NO_CACHE_DIR", None)
    project = load_project(root / "pyproject.toml")
    try:
        items = stale_items(project, extras, root)
    except ImportError:
        spec = "." if not extras else f".[{','.join(extras)}]"
        items = [(["-e", spec], "packaging is not installed yet")]
    cache = _pip_cache_dir()
    if not items and not force:
        labels = ", ".join(extras) if extras else "core"
        print(f"    Host libraries already match pyproject.toml ({labels})")
        print(f"    Package cache: {cache}")
        return 0
    args = pip_install_args(extras, items, force)
    if force and not items:
        print("    Reinstalling host libraries (--force)")
    else:
        print("    Installing updated libraries:")
        for _args, reason in items:
            print(f"      {reason}")
    print(f"    Package cache: {cache}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        *args,
    ]
    subprocess.check_call(command, cwd=root)
    return 10


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install")
    install.add_argument("--extras", default="", help="comma-separated extras")
    install.add_argument("--force", action="store_true")
    install.add_argument("--root", default=".")

    finger = sub.add_parser("fingerprint")
    finger.add_argument("paths", nargs="+")
    finger.add_argument("--exclude", action="append", default=[])
    finger.add_argument("--root", default=".")

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "fingerprint":
        print(fingerprint(args.paths, set(args.exclude), root))
        return 0
    extras = [part for part in args.extras.split(",") if part]
    return cmd_install(extras, args.force, root)


if __name__ == "__main__":
    sys.exit(main())
