"""CNAS 测试集加载与样本收集."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cnas_test.runner.config import CLASS_NAMES, PROJECT_ROOT, TILE_DATASET_ROOT, TILES_DIRS


def load_test_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_test_stems(path: Path) -> list[str]:
    manifest = load_test_manifest(path)
    return [item["stem"] for item in manifest["images"]]


def collect_tile_paths(stems: list[str]) -> list[Path]:
    all_tiles: list[Path] = []
    for tiles_dir in TILES_DIRS:
        all_tiles.extend(sorted(tiles_dir.glob("*.jpg")))
    selected = [tile for tile in all_tiles if tile.stem.rsplit("_", 2)[0] in stems]
    if not selected:
        dirs = ", ".join(str(path) for path in TILES_DIRS)
        raise FileNotFoundError(f"在 {dirs} 中未找到匹配样本，请确认测试集清单和样本目录一致。")
    return selected


def label_path_for_tile(tile_path: Path) -> Path:
    relative = tile_path.relative_to(TILE_DATASET_ROOT / "images")
    return TILE_DATASET_ROOT / "labels" / relative.with_suffix(".txt")


def summarize_tiles(tile_paths: list[Path]) -> dict:
    class_counts: Counter[str] = Counter()
    n_boxes = 0
    backgrounds = 0
    missing_labels: list[str] = []

    for tile_path in tile_paths:
        label_path = label_path_for_tile(tile_path)
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue
        lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not lines:
            backgrounds += 1
        n_boxes += len(lines)
        for line in lines:
            cls_idx = int(line.split()[0])
            class_counts[CLASS_NAMES.get(cls_idx, str(cls_idx))] += 1

    return {
        "images": len({tile.stem.rsplit("_", 2)[0] for tile in tile_paths}),
        "tiles": len(tile_paths),
        "boxes": n_boxes,
        "background_tiles": backgrounds,
        "missing_labels": missing_labels,
        "class_counts": {name: class_counts.get(name, 0) for name in CLASS_NAMES.values()},
    }


def build_val_dataset_yaml(tile_paths: list[Path], save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    list_file = save_dir / "cnas_val_list.txt"
    list_file.write_text("\n".join(str(path) for path in tile_paths) + "\n", encoding="utf-8")

    yaml_path = save_dir / "cnas_val.yaml"
    yaml_path.write_text(
        f"path: {PROJECT_ROOT}\n"
        f"train: {list_file}\n"
        f"val:   {list_file}\n"
        "nc: 3\n"
        "names: ['scratch', 'spot', 'critical']\n",
        encoding="utf-8",
    )
    return yaml_path
