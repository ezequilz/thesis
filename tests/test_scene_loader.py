"""Scene catalog + SOG / streamed-SOG loading."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.config import Config
from splat_explorer.scene import load_scene, load_sog, load_sog_lod
from splat_explorer.scene.catalog import (
    apply_spec,
    discover_scenes,
    list_scenes,
    slugify,
    spec_by_id,
)
from splat_explorer.scene.sog_loader import _lod_file_indices
from splat_explorer.scene.types import GaussianScene


def _codebook(n: int = 256) -> list[float]:
    return [0.0] * n


def _write_webp(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, format="WEBP", lossless=True)


def write_sog_dir(root: Path, count: int = 4, origin: float = 0.0) -> Path:
    """Minimal unbundled SOG v2 directory with `count` gaussians around `origin`."""
    root.mkdir(parents=True, exist_ok=True)
    side = max(1, int(np.ceil(np.sqrt(count))))
    rgb = np.full((side, side, 3), 128, dtype=np.uint8)
    rgba = np.full((side, side, 4), 128, dtype=np.uint8)
    rgba[..., 3] = 252  # omit w → identity-ish quaternion
    sh0 = np.full((side, side, 4), 128, dtype=np.uint8)
    sh0[..., 3] = 200  # opacity
    _write_webp(root / "means_l.webp", rgb)
    _write_webp(root / "means_u.webp", rgb)
    _write_webp(root / "scales.webp", rgb)
    _write_webp(root / "quats.webp", rgba)
    _write_webp(root / "sh0.webp", sh0)
    meta = {
        "version": 2,
        "count": count,
        "means": {
            "mins": [origin - 1.0, origin - 1.0, origin - 1.0],
            "maxs": [origin + 1.0, origin + 1.0, origin + 1.0],
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {"codebook": _codebook(), "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": _codebook(), "files": ["sh0.webp"]},
    }
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def write_sog_zip(path: Path, count: int = 3) -> Path:
    tmp = path.parent / (path.stem + "_dir")
    write_sog_dir(tmp, count=count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for child in tmp.iterdir():
            zf.write(child, child.name)
    return path


def test_slugify():
    assert slugify("Pond shelter") == "pond-shelter"
    assert slugify("Calico Basin at sunrise") == "calico-basin-at-sunrise"
    assert slugify("ArchInteriors_for_UE2_Atlux") == "archinteriors-for-ue2-atlux"


def test_load_sog_directory_and_zip(tmp_path: Path):
    folder = write_sog_dir(tmp_path / "room", count=4)
    from_dir = load_sog(folder)
    assert from_dir.num_gaussians == 4
    assert from_dir.means.shape == (4, 3)
    assert from_dir.opacities.shape == (4,)

    via_meta = load_scene(folder / "meta.json")
    assert via_meta.num_gaussians == 4

    zipped = write_sog_zip(tmp_path / "room.sog", count=3)
    from_zip = load_scene(zipped)
    assert from_zip.num_gaussians == 3


def test_load_streamed_sog_one_lod(tmp_path: Path):
    root = tmp_path / "landscape"
    write_sog_dir(root / "0_0", count=4, origin=0.0)
    write_sog_dir(root / "0_1", count=5, origin=3.0)
    write_sog_dir(root / "1_0", count=2, origin=0.0)  # coarser LOD, must not mix in
    write_sog_dir(root / "env", count=1, origin=10.0)
    lod_meta = {
        "version": 1,
        "count": 11,
        "counts": [9, 2],
        "lodLevels": 2,
        "environment": "env/meta.json",
        "filenames": ["0_0/meta.json", "0_1/meta.json", "1_0/meta.json"],
        "tree": {
            "bound": {"min": [-2, -2, -2], "max": [12, 12, 12]},
            "children": [
                {
                    "bound": {"min": [-2, -2, -2], "max": [2, 2, 2]},
                    "lods": {
                        "0": {"file": 0, "offset": 0, "count": 4},
                        "1": {"file": 2, "offset": 0, "count": 2},
                    },
                },
                {
                    "bound": {"min": [2, 2, 2], "max": [6, 6, 6]},
                    "lods": {"0": {"file": 1, "offset": 0, "count": 5}},
                },
            ],
        },
    }
    (root / "lod-meta.json").write_text(json.dumps(lod_meta))

    assert _lod_file_indices(lod_meta["tree"], 0) == [0, 1]
    assert _lod_file_indices(lod_meta["tree"], 1) == [2]

    finest = load_sog_lod(root, lod_level=0)
    # LOD 0 chunks (4+5) plus environment (1)
    assert finest.num_gaussians == 10

    coarse = load_scene(root, lod_level=1)
    assert coarse.num_gaussians == 3  # 2 + env

    no_env = load_sog_lod(root, lod_level=0, include_environment=False)
    assert no_env.num_gaussians == 9


def test_concatenate_and_opacity_filter():
    a = GaussianScene(
        means=np.zeros((2, 3), np.float32),
        scales=np.ones((2, 3), np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (2, 1)),
        opacities=np.array([0.9, 0.01], np.float32),
        colors=np.ones((2, 3), np.float32),
    )
    b = GaussianScene(
        means=np.ones((1, 3), np.float32),
        scales=np.ones((1, 3), np.float32),
        quats=np.array([[1, 0, 0, 0]], np.float32),
        opacities=np.array([0.5], np.float32),
        colors=np.ones((1, 3), np.float32),
    )
    stacked = GaussianScene.concatenate([a, b])
    assert stacked.num_gaussians == 3
    kept = stacked.filtered_by_opacity(0.1)
    assert kept.num_gaussians == 2


def test_list_scenes_catalog_and_discovery(tmp_path: Path):
    rooms = tmp_path / "3dgs_rooms"
    write_sog_zip(rooms / "ArchInteriors_for_UE2_Atlux.sog", count=2)
    write_sog_zip(rooms / "Pond shelter.sog", count=2)
    write_sog_dir(rooms / "Calico Basin at sunrise" / "0_0", count=2)
    (rooms / "Calico Basin at sunrise" / "lod-meta.json").write_text(json.dumps({
        "version": 1, "count": 2, "counts": [2], "lodLevels": 1,
        "filenames": ["0_0/meta.json"],
        "tree": {"bound": {"min": [0, 0, 0], "max": [1, 1, 1]},
                 "lods": {"0": {"file": 0, "offset": 0, "count": 2}}},
    }))
    write_sog_zip(rooms / "Extra hall.sog", count=1)

    cfg = Config({
        "scene": {
            "path": str(rooms / "ArchInteriors_for_UE2_Atlux.sog"),
            "lod_level": 0,
            "catalog": [
                {"id": "arch-interiors", "label": "Arch Interiors",
                 "path": str(rooms / "ArchInteriors_for_UE2_Atlux.sog"),
                 "up_axis": "-y"},
                {"id": "pond-shelter", "label": "Pond shelter",
                 "path": str(rooms / "Pond shelter.sog"), "up_axis": "+y"},
                {"id": "calico-basin", "label": "Calico Basin at sunrise",
                 "path": str(rooms / "Calico Basin at sunrise"), "up_axis": "+y"},
                {"id": "missing", "label": "Gone", "path": str(rooms / "nope.sog")},
            ],
        },
        "camera": {"up_axis": "-y"},
    })
    scenes = list_scenes(cfg)
    ids = [s.id for s in scenes]
    assert ids[0] == "arch-interiors"
    assert "pond-shelter" in ids
    assert "calico-basin" in ids
    assert "missing" not in ids
    assert "extra-hall" in ids
    pond = spec_by_id(cfg, "pond-shelter")
    assert pond is not None and pond.up_axis == "+y"
    apply_spec(cfg, pond)
    assert cfg["scene"]["path"] == str(pond.path)
    assert cfg["camera"]["up_axis"] == "+y"

    discovered = discover_scenes(rooms)
    assert {s.id for s in discovered} >= {"pond-shelter", "extra-hall", "calico-basin-at-sunrise"}
