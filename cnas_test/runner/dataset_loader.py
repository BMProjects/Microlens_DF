"""CNAS 测试集加载与切片收集."""

from __future__ import annotations

import json
from pathlib import Path

from cnas_test.runner.config import PROJECT_ROOT, TILES_DIR


def load_test_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_test_stems(path: Path) -> list[str]:
    manifest = load_test_manifest(path)
    return [item["stem"] for item in manifest["images"]]


def collect_tile_paths(stems: list[str]) -> list[Path]:
    all_tiles = sorted(TILES_DIR.glob("*.jpg"))
    selected = [tile for tile in all_tiles if tile.stem.rsplit("_", 2)[0] in stems]
    if not selected:
        raise FileNotFoundError(
            f"在 {TILES_DIR} 中未找到匹配切片，请确认测试集清单和切片目录一致。"
        )
    return selected


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
