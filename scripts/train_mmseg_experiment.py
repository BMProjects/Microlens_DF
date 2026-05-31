#!/usr/bin/env python3
"""第二批分割模型实验入口：生成标准化 manifest 与命令清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE3_ROOT = PROJECT_ROOT / "output/experiments/phase3_segmentation"
CONFIG_ROOT = PROJECT_ROOT / "configs/segmentation/mmseg"


EXPERIMENTS = {
    "segformer_b2_msd": {
        "model_family": "SegFormer-B2",
        "stage": "msd_pretrain",
        "dataset_root": PHASE3_ROOT / "msd_prepared",
        "work_dir": PHASE3_ROOT / "batch2_segformer_b2_msd",
        "config_template": CONFIG_ROOT / "segformer_b2_msd_template.py",
        "notes": "轻量 Transformer 基线；验证多尺度编码与分辨率鲁棒性。",
    },
    "segformer_b2_private": {
        "model_family": "SegFormer-B2",
        "stage": "private_weak_finetune",
        "dataset_root": PHASE3_ROOT / "private_weak_masks",
        "work_dir": PHASE3_ROOT / "batch2_segformer_b2_private",
        "config_template": CONFIG_ROOT / "segformer_b2_private_template.py",
        "notes": "在私有弱标签上微调，重点看 scratch IoU 与长度误差。",
    },
    "hrnet_ocr_w18_msd": {
        "model_family": "HRNet-OCR-W18",
        "stage": "msd_pretrain",
        "dataset_root": PHASE3_ROOT / "msd_prepared",
        "work_dir": PHASE3_ROOT / "batch2_hrnet_ocr_w18_msd",
        "config_template": CONFIG_ROOT / "hrnet_ocr_w18_msd_template.py",
        "notes": "高分辨率保持对照，重点看细线连通性与交叉结构。",
    },
    "hrnet_ocr_w18_private": {
        "model_family": "HRNet-OCR-W18",
        "stage": "private_weak_finetune",
        "dataset_root": PHASE3_ROOT / "private_weak_masks",
        "work_dir": PHASE3_ROOT / "batch2_hrnet_ocr_w18_private",
        "config_template": CONFIG_ROOT / "hrnet_ocr_w18_private_template.py",
        "notes": "重点关注细长 scratch 与交叉结构保真。",
    },
    "hrnetv2_w18_msd": {
        "model_family": "HRNetV2-W18-Seg",
        "stage": "msd_pretrain",
        "dataset_root": PHASE3_ROOT / "msd_prepared",
        "work_dir": PHASE3_ROOT / "batch2_hrnetv2_w18_msd",
        "config_template": CONFIG_ROOT / "hrnetv2_w18_msd_template.py",
        "notes": "作为不带 OCR 的高分辨率对照。",
    },
    "hrnetv2_w18_private": {
        "model_family": "HRNetV2-W18-Seg",
        "stage": "private_weak_finetune",
        "dataset_root": PHASE3_ROOT / "private_weak_masks",
        "work_dir": PHASE3_ROOT / "batch2_hrnetv2_w18_private",
        "config_template": CONFIG_ROOT / "hrnetv2_w18_private_template.py",
        "notes": "对照 OCR 增益，重点看复杂区域误分割。",
    },
}


def build_shell_commands(name: str, spec: dict) -> list[str]:
    cfg_path = spec["config_template"]
    commands = [
        "uv pip install 'mmcv-lite>=2.1,<2.2.0' 'mmengine>=0.10,<1.0' 'mmsegmentation>=1.2,<1.3'",
        f"uv run python scripts/train_mmseg_experiment.py --experiment {name} --write-manifest",
        "# 正式训练命令:",
    ]
    if spec["stage"] == "private_weak_finetune":
        parent_name = spec["work_dir"].name.replace("_private", "_msd")
        parent_dir = PHASE3_ROOT / parent_name
        commands.append(
            f"uv run python scripts/train_mmseg_config.py {cfg_path} "
            f"--work-dir {spec['work_dir']} "
            f"--cfg-options load_from={parent_dir / 'latest.pth'}"
        )
    else:
        commands.append(
            f"uv run python scripts/train_mmseg_config.py {cfg_path} --work-dir {spec['work_dir']}"
        )
    return commands


def _balanced_split(stems: list[str], val_ratio: float = 0.15) -> tuple[list[str], list[str]]:
    if not stems:
        return [], []
    stems = sorted(stems)
    val_count = max(1, int(round(len(stems) * val_ratio)))
    if len(stems) <= 3:
        val_count = 1
    val = stems[:val_count]
    train = stems[val_count:] or stems[:-1]
    if not train:
        train = stems
        val = stems[:1]
    return train, val


def ensure_private_split_files(dataset_root: Path) -> dict[str, str]:
    split_dir = dataset_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    image_stems = sorted(p.stem for p in (dataset_root / "images").glob("*.png"))
    manifest_path = PHASE3_ROOT / "data_stratification" / "complexity_manifest.json"
    levels: dict[str, list[str]] = {"L1": [], "L2": [], "L3": []}
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        levels = {k: [s for s in v if s in image_stems] for k, v in data.get("levels", levels).items()}
    else:
        levels["L3"] = image_stems

    assigned = set(levels["L1"]) | set(levels["L2"]) | set(levels["L3"])
    remaining = [stem for stem in image_stems if stem not in assigned]
    levels["L3"].extend(remaining)

    train_all: list[str] = []
    val_all: list[str] = []
    for level in ("L1", "L2", "L3"):
        train, val = _balanced_split(levels[level])
        train_all.extend(train)
        val_all.extend(val)

    train_unique = sorted(set(train_all) - set(val_all))
    val_unique = sorted(set(val_all))
    if not train_unique:
        train_unique = sorted(set(image_stems) - set(val_unique))
    if not train_unique:
        train_unique = image_stems

    train_path = split_dir / "segmentation_train.txt"
    val_path = split_dir / "segmentation_val.txt"
    train_path.write_text("\n".join(train_unique) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_unique) + "\n", encoding="utf-8")

    split_info = {
        "n_total": len(image_stems),
        "n_train": len(train_unique),
        "n_val": len(val_unique),
        "levels_used": {k: len(v) for k, v in levels.items()},
    }
    (split_dir / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "train_split": str(train_path),
        "val_split": str(val_path),
        "split_info": str(split_dir / "split_info.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="第二批分割实验入口（生成 manifest 与命令）")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    spec = EXPERIMENTS[args.experiment]
    private_split_info = None
    if spec["stage"] == "private_weak_finetune":
        private_split_info = ensure_private_split_files(spec["dataset_root"])
    manifest = {
        "experiment": args.experiment,
        "model_family": spec["model_family"],
        "stage": spec["stage"],
        "dataset_root": str(spec["dataset_root"]),
        "work_dir": str(spec["work_dir"]),
        "config_template": str(spec["config_template"]),
        "pixel_size_mm": 0.0068,
        "recommended_patch_mm": 3.5,
        "recommended_overlap_ratio": 0.25,
        "notes": spec["notes"],
        "commands": build_shell_commands(args.experiment, spec),
        "private_split_info": private_split_info,
    }

    if args.write_manifest:
        spec["work_dir"].mkdir(parents=True, exist_ok=True)
        (spec["work_dir"] / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   第二批分割实验入口                                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  实验:     {args.experiment}")
    print(f"  模型:     {spec['model_family']}")
    print(f"  阶段:     {spec['stage']}")
    print(f"  数据:     {spec['dataset_root']}")
    print(f"  输出:     {spec['work_dir']}")
    print(f"  配置模板: {spec['config_template']}")
    print(f"  说明:     {spec['notes']}")
    if args.write_manifest:
        print(f"  Manifest: {spec['work_dir'] / 'manifest.json'}")
    if private_split_info is not None:
        print(f"  Train split: {private_split_info['train_split']}")
        print(f"  Val split:   {private_split_info['val_split']}")
    print()
    print("建议命令:")
    for cmd in manifest["commands"]:
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
