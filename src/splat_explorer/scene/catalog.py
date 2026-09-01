"""Scene catalog: named splat assets the dashboard can switch between.

The dropdown is driven by `scene.catalog` in configs/default.yaml, plus any
extra .sog / streamed-SOG folders found next to the configured path. A live
pointer at outputs/live/scene.json lets the viser viewer (a separate process)
hot-swap the WebGL splat when the dashboard changes scene.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LIVE_SCENE_PATH = Path("outputs/live/scene.json")
# Snapshot of configs/default.yaml navigation, restored on every scene switch
# before applying that entry's catalog overlay. Starter Scene has no overlay.
_NAV_DEFAULTS_KEY = "_navigation_defaults"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "scene"


def portable_scene_path(path: Path | str) -> str:
    """Repo-relative path both the host dashboard and Docker visor can open.

    The visor runs in Linux Docker with ``./outputs`` and ``./3dgs_rooms``
    mounted at ``/app``. Absolute Mac paths from a host dashboard are
    invisible inside that container, so live-scene pointers must stay
    relative (``outputs/episodes/.../scene_repaired.ply``).
    """
    raw = Path(path)
    search = [raw]
    try:
        search.append(raw.expanduser().resolve())
    except (OSError, RuntimeError):
        pass
    cwd = Path.cwd().resolve()
    for candidate in search:
        try:
            return str(candidate.relative_to(cwd))
        except ValueError:
            pass
        parts = candidate.parts
        for marker in ("outputs", "3dgs_rooms"):
            if marker in parts:
                index = parts.index(marker)
                return str(Path(*parts[index:]))
    return str(raw)


def openable_scene_path(path: Path | str) -> Path:
    """Resolve a live-scene path, rewriting host-absolute prefixes for Docker."""
    raw = Path(path)
    if raw.exists():
        return raw
    rewritten = Path(portable_scene_path(raw))
    if rewritten.exists():
        return rewritten
    return raw


@dataclass(frozen=True)
class SceneSpec:
    id: str
    label: str
    path: Path
    up_axis: str = "+y"
    lod_level: int = 0
    # Optional per-scene overlay on cfg.navigation (spawn height, bird's-eye
    # framing, …). Empty for indoor scenes (Starter, Venetian) so those keep
    # the original defaults.
    nav_overrides: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        payload = {
            "id": self.id,
            "label": self.label,
            "path": str(self.path),
            "up_axis": self.up_axis,
            "lod_level": self.lod_level,
        }
        if self.nav_overrides:
            payload["navigation"] = dict(self.nav_overrides)
        return payload


def _label_from_path(path: Path) -> str:
    name = path.name
    if path.suffix.lower() in {".sog", ".ply"}:
        name = path.stem
    return name.replace("_", " ").strip() or path.name


def _spec_from_mapping(raw: dict, default_up: str, default_lod: int) -> SceneSpec:
    path = Path(raw["path"])
    raw_nav = raw.get("navigation")
    return SceneSpec(
        id=str(raw.get("id") or slugify(path.stem if path.suffix else path.name)),
        label=str(raw.get("label") or _label_from_path(path)),
        path=path,
        up_axis=str(raw.get("up_axis") or default_up),
        lod_level=int(raw.get("lod_level", default_lod)),
        nav_overrides=dict(raw_nav) if isinstance(raw_nav, dict) else {},
    )


def is_scene_path(path: Path) -> bool:
    """True if `path` is a splat asset this loader can open."""
    if path.is_file():
        return path.suffix.lower() in {".sog", ".ply"}
    if path.is_dir():
        return (path / "lod-meta.json").is_file() or (path / "meta.json").is_file()
    return False


def discover_scenes(root: Path, default_up: str = "+y", default_lod: int = 0) -> list[SceneSpec]:
    """Pick up .sog files and unbundled / streamed-SOG folders under `root`."""
    if not root.is_dir():
        return []
    found: list[SceneSpec] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith("."):
            continue
        if child.suffix.lower() == ".ply":
            # Uncompressed debug exports — only listed when put in scene.catalog.
            continue
        if not is_scene_path(child):
            continue
        found.append(SceneSpec(
            id=slugify(child.stem if child.suffix else child.name),
            label=_label_from_path(child),
            path=child,
            up_axis=default_up,
            lod_level=default_lod,
        ))
    return found


def list_scenes(cfg) -> list[SceneSpec]:
    """Catalog entries from config, then undiscovered files in the rooms folder.

    Missing files are dropped so the dropdown never offers a scene that cannot
    be opened. Explicit catalog order is preserved; extras append alphabetically.
    """
    scene_cfg = cfg["scene"] if isinstance(cfg, dict) else cfg
    camera_cfg = cfg["camera"] if isinstance(cfg, dict) else cfg
    if hasattr(scene_cfg, "get"):
        raw_catalog = scene_cfg.get("catalog") or []
        default_lod = int(scene_cfg.get("lod_level", 0) or 0)
        configured_path = Path(scene_cfg.get("path") or "")
        catalog_dir = scene_cfg.get("catalog_dir")
    else:
        raw_catalog, default_lod, configured_path, catalog_dir = [], 0, Path(""), None
    default_up = "+y"
    if hasattr(camera_cfg, "get"):
        default_up = str(camera_cfg.get("up_axis") or "+y")

    entries: list[SceneSpec] = []
    seen: set[str] = set()
    seen_paths: set[Path] = set()

    def add(spec: SceneSpec) -> None:
        if spec.id in seen:
            return
        resolved = spec.path
        try:
            resolved = spec.path.resolve()
        except OSError:
            pass
        if resolved in seen_paths:
            return
        if not is_scene_path(spec.path):
            logger.warning("Skipping catalog scene %s: not found at %s", spec.id, spec.path)
            return
        entries.append(spec)
        seen.add(spec.id)
        seen_paths.add(resolved)

    for raw in raw_catalog:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        add(_spec_from_mapping(raw, default_up=default_up, default_lod=default_lod))

    if configured_path and is_scene_path(configured_path) and not any(
        e.path.resolve() == configured_path.resolve() for e in entries
        if e.path.exists()
    ):
        add(SceneSpec(
            id=slugify(configured_path.stem if configured_path.suffix else configured_path.name),
            label=_label_from_path(configured_path),
            path=configured_path,
            up_axis=default_up,
            lod_level=default_lod,
        ))

    scan_root = Path(catalog_dir) if catalog_dir else (
        configured_path.parent if configured_path.name else Path("3dgs_rooms")
    )
    for extra in discover_scenes(scan_root, default_up="+y", default_lod=default_lod):
        add(extra)

    return entries


def spec_by_id(cfg, scene_id: str) -> SceneSpec | None:
    for spec in list_scenes(cfg):
        if spec.id == scene_id:
            return spec
    return None


def catalog_id_from_live(live: dict | None) -> str | None:
    """Catalog room id from a live pointer, ignoring repair-original/repaired previews."""
    if not live:
        return None
    for key in ("catalog_id", "id"):
        value = str(live.get(key) or "").strip()
        if value and not value.startswith("repair-"):
            return value
    return None


def live_display_spec(cfg, live: dict) -> SceneSpec:
    """Splat the visor should decode: episode PLY preview if present, else catalog."""
    catalog = None
    cid = catalog_id_from_live(live)
    if cid:
        catalog = spec_by_id(cfg, cid)
    path = live.get("path")
    if path:
        opened = openable_scene_path(path)
        if opened.exists():
            up = live.get("up_axis") or (catalog.up_axis if catalog is not None else "+y")
            lod = int(live.get("lod_level") or (catalog.lod_level if catalog is not None else 0) or 0)
            return SceneSpec(
                id=str(live.get("id") or (catalog.id if catalog is not None else "live")),
                label=str(live.get("label") or (catalog.label if catalog is not None else opened.name)),
                path=opened,
                up_axis=str(up),
                lod_level=lod,
            )
    if catalog is not None:
        return catalog
    return current_spec(cfg)


def live_scene_reload_action(req: dict, state: dict) -> str:
    """Decide whether the visor should decode a live pointer.

    Returns ``load``, ``ack-same``, or ``skip``. A dashboard restart resets
    generation to 0; a newer ``updated_at`` on a different path still loads
    so we do not stay stuck on Starter Scene.
    """
    if not req or not req.get("path"):
        return "skip"
    gen = int(req.get("generation") or 0)
    path = str(req["path"])
    prev_path = str(state.get("path") or "")
    updated_at = float(req.get("updated_at") or 0.0)
    last_updated = float(state.get("updated_at") or 0.0)
    path_changed = path != prev_path
    if path_changed and updated_at + 1e-9 >= last_updated:
        return "load"
    if gen < int(state.get("generation") or 0):
        return "skip"
    same = (not path_changed) and state.get("status") == "ready"
    mtime = float(req.get("mtime") or 0.0)
    newer_file = mtime > float(state.get("mtime") or 0.0) + 1e-6
    force = bool(req.get("reload")) and (
        gen > int(state.get("generation") or 0) or newer_file
    )
    if same and not force:
        return "ack-same"
    return "load"


def spec_for_path(cfg, path: str | Path) -> SceneSpec | None:
    target = Path(path)
    try:
        target_res = target.resolve()
    except OSError:
        target_res = target
    for spec in list_scenes(cfg):
        try:
            if spec.path.resolve() == target_res:
                return spec
        except OSError:
            if spec.path == target:
                return spec
    return None


def current_spec(cfg) -> SceneSpec:
    """The catalog entry matching cfg.scene.path, or a synthetic one."""
    scene_cfg = cfg["scene"]
    path = Path(scene_cfg["path"])
    matched = spec_for_path(cfg, path)
    if matched is not None:
        return matched
    camera = cfg["camera"]
    return SceneSpec(
        id=slugify(path.stem if path.suffix else path.name),
        label=_label_from_path(path),
        path=path,
        up_axis=str(camera.get("up_axis") or "+y"),
        lod_level=int(scene_cfg.get("lod_level", 0) or 0),
    )


def apply_spec(cfg, spec: SceneSpec) -> None:
    """Point the live config at this catalog entry (path + up axis + LOD).

    Navigation is restored to the YAML defaults, then this entry's optional
    `navigation:` overlay is applied. Switching back to Starter Scene therefore
    gets the original indoor spawn / bird's-eye settings, not leftover outdoor
    overrides.
    """
    cfg["scene"]["path"] = str(spec.path)
    cfg["scene"]["lod_level"] = int(spec.lod_level)
    cfg["camera"]["up_axis"] = spec.up_axis
    nav = cfg.setdefault("navigation", {})
    if _NAV_DEFAULTS_KEY not in cfg:
        cfg[_NAV_DEFAULTS_KEY] = dict(nav)
    nav.clear()
    nav.update(cfg[_NAV_DEFAULTS_KEY])
    nav.update(spec.nav_overrides)


def publish_live_scene(
    spec: SceneSpec,
    generation: int,
    reload: bool = False,
    catalog_id: str | None = None,
) -> None:
    """Atomic write of the pointer the viser viewer polls.

    `reload=True` forces the viewer to re-decode even when the path is
    unchanged (used after overwriting scene_repaired.ply in place).
    `catalog_id` is the dropdown room (venetian-balcony, …). Preview ids
    like ``repair-repaired`` must not be used as the catalog id — a visor
    restart would otherwise fall back to Starter Scene.
    """
    LIVE_SCENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {**spec.to_json(), "generation": int(generation), "updated_at": time.time()}
    payload["path"] = portable_scene_path(spec.path)
    ply = Path(payload["path"])
    try:
        payload["mtime"] = ply.stat().st_mtime
    except OSError:
        payload["mtime"] = 0.0
    cid = str(catalog_id or "").strip()
    if not cid and not str(spec.id).startswith("repair-"):
        cid = spec.id
    if cid and not cid.startswith("repair-"):
        payload["catalog_id"] = cid
    if reload:
        payload["reload"] = True
    tmp = LIVE_SCENE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, LIVE_SCENE_PATH)


def read_live_scene() -> dict | None:
    try:
        return json.loads(LIVE_SCENE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
