"""Configuration loading.

Configs are plain YAML files (see configs/default.yaml). We keep them as nested
dicts wrapped in a small attribute-access helper rather than a rigid schema,
since the project is in the scaffolding stage and sections will evolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access and recursive wrapping: cfg.renderer.width."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"No config key {name!r}") from exc
        return Config(value) if isinstance(value, dict) else value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_default_config() -> Path:
    """Locate configs/default.yaml: cwd first (Docker WORKDIR, repo root),
    then relative to the source tree (editable installs from elsewhere)."""
    candidates = [
        Path.cwd() / "configs" / "default.yaml",
        Path(__file__).resolve().parents[2] / "configs" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "configs/default.yaml not found. Run from the repo root or mount ./configs."
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load configs/default.yaml, optionally deep-merged with an override file."""
    default_path = _find_default_config()
    with open(default_path) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != default_path:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return Config(cfg)
