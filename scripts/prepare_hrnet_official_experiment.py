#!/usr/bin/env python3
"""为 HRNet 官方语义分割仓生成可执行实验包."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE3_ROOT = PROJECT_ROOT / "output/experiments/phase3_segmentation"
DEFAULT_REPO_DIR = PROJECT_ROOT / "third_party/HRNet-Semantic-Segmentation"
OFFICIAL_REPO = "https://github.com/HRNet/HRNet-Semantic-Segmentation"


EXPERIMENTS = {
    "msd": {
        "dataset_root": PHASE3_ROOT / "msd_prepared",
        "output_dir": PHASE3_ROOT / "batch2_hrnet_ocr_official_msd",
        "dataset_name": "darkfield_msd",
        "notes": "MSD 结构对照阶段；用于补齐高分辨率 OCR 路线的源域结果。",
        "train_batch_per_gpu": 4,
        "test_batch_per_gpu": 2,
        "end_epoch": 60,
        "base_lr": 0.01,
        "base_size": 512,
        "image_size": [512, 512],
        "private_style": False,
    },
    "private": {
        "dataset_root": PHASE3_ROOT / "private_weak_masks",
        "output_dir": PHASE3_ROOT / "batch2_hrnet_ocr_official_private",
        "dataset_name": "darkfield_private",
        "notes": "私有弱标签微调阶段；用于与 FPN / SegFormer-B2 做 batch2 对照。",
        "train_batch_per_gpu": 2,
        "test_batch_per_gpu": 1,
        "end_epoch": 40,
        "base_lr": 0.005,
        "base_size": 512,
        "image_size": [512, 512],
        "private_style": True,
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_split_stems(stage: str, dataset_root: Path) -> tuple[list[str], list[str]]:
    if stage == "msd":
        train_names = sorted(p.stem for p in (dataset_root / "images" / "train").glob("*.png"))
        val_names = sorted(p.stem for p in (dataset_root / "images" / "val").glob("*.png"))
        return train_names, val_names

    split_dir = dataset_root / "splits"
    train_file = split_dir / "segmentation_train.txt"
    val_file = split_dir / "segmentation_val.txt"
    train_names = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_names = [line.strip() for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return train_names, val_names


def list_line(stage: str, stem: str) -> str:
    if stage == "msd":
        return f"images/train/{stem}.png masks/train/{stem}.png"
    return f"images/{stem}.png masks/{stem}.png"


def val_list_line(stage: str, stem: str) -> str:
    if stage == "msd":
        return f"images/val/{stem}.png masks/val/{stem}.png"
    return f"images/{stem}.png masks/{stem}.png"


def build_dataset_lists(stage: str, dataset_root: Path, package_dir: Path) -> tuple[Path, Path, dict[str, int]]:
    train_names, val_names = read_split_stems(stage, dataset_root)
    list_dir = package_dir / "data/list"
    ensure_dir(list_dir)
    train_lst = list_dir / "train.lst"
    val_lst = list_dir / "val.lst"
    train_lst.write_text("\n".join(list_line(stage, stem) for stem in train_names) + "\n", encoding="utf-8")
    val_lst.write_text("\n".join(val_list_line(stage, stem) for stem in val_names) + "\n", encoding="utf-8")
    return train_lst, val_lst, {"n_train": len(train_names), "n_val": len(val_names)}


def build_dataset_patch(dataset_name: str) -> str:
    return textwrap.dedent(
        f"""\
        import os
        import torch

        from .base_dataset import BaseDataset


        class DarkField(BaseDataset):
            def __init__(self,
                         root,
                         list_path,
                         num_samples=None,
                         num_classes=4,
                         multi_scale=True,
                         flip=True,
                         ignore_label=255,
                         base_size=512,
                         crop_size=(512, 512),
                         downsample_rate=1,
                         scale_factor=16,
                         mean=[0.5],
                         std=[0.5]):
                super(DarkField, self).__init__(
                    ignore_label,
                    base_size,
                    crop_size,
                    downsample_rate,
                    scale_factor,
                    mean,
                    std)

                self.root = root
                self.list_path = list_path
                self.num_classes = num_classes
                self.multi_scale = multi_scale
                self.flip = flip
                self.img_list = [line.strip().split() for line in open(os.path.join(root, list_path), 'r')]
                self.files = self.read_files()
                if num_samples:
                    self.files = self.files[:num_samples]

                # background / scratch / spot / damage
                self.class_weights = torch.FloatTensor([0.1, 2.0, 1.0, 1.0])

            def read_files(self):
                files = []
                for item in self.img_list:
                    image_path, label_path = item
                    name = os.path.splitext(os.path.basename(image_path))[0]
                    files.append({{
                        "img": os.path.join(self.root, image_path),
                        "label": os.path.join(self.root, label_path),
                        "name": name,
                    }})
                return files
        """
    )


def build_init_patch() -> str:
    return textwrap.dedent(
        """\
        from .darkfield import DarkField

        # 在数据集工厂 / 名称映射中注册:
        # 'darkfield': DarkField
        """
    )


def build_yaml(stage: str, dataset_root: Path, output_dir: Path, spec: dict) -> str:
    train_list_rel = "data/list/train.lst"
    val_list_rel = "data/list/val.lst"
    image_w, image_h = spec["image_size"]
    return textwrap.dedent(
        f"""\
        CUDNN:
          BENCHMARK: true
          DETERMINISTIC: false
          ENABLED: true

        GPUS: (0,)
        OUTPUT_DIR: '{output_dir}'
        LOG_DIR: '{output_dir}/logs'
        WORKERS: 8
        PRINT_FREQ: 20

        DATASET:
          DATASET: darkfield
          ROOT: '{dataset_root}'
          TRAIN_SET: '{train_list_rel}'
          TEST_SET: '{val_list_rel}'
          NUM_CLASSES: 4

        MODEL:
          NAME: seg_hrnet_ocr
          PRETRAINED: ''
          ALIGN_CORNERS: true
          NUM_OUTPUTS: 2
          OCR:
            MID_CHANNELS: 512
            KEY_CHANNELS: 256
            DROPOUT: 0.05
            SCALE: 1
          EXTRA:
            FINAL_CONV_KERNEL: 1
            STAGE1:
              NUM_MODULES: 1
              NUM_BRANCHES: 1
              BLOCK: BOTTLENECK
              NUM_BLOCKS: [4]
              NUM_CHANNELS: [64]
              FUSE_METHOD: SUM
            STAGE2:
              NUM_MODULES: 1
              NUM_BRANCHES: 2
              BLOCK: BASIC
              NUM_BLOCKS: [4, 4]
              NUM_CHANNELS: [18, 36]
              FUSE_METHOD: SUM
            STAGE3:
              NUM_MODULES: 4
              NUM_BRANCHES: 3
              BLOCK: BASIC
              NUM_BLOCKS: [4, 4, 4]
              NUM_CHANNELS: [18, 36, 72]
              FUSE_METHOD: SUM
            STAGE4:
              NUM_MODULES: 3
              NUM_BRANCHES: 4
              BLOCK: BASIC
              NUM_BLOCKS: [4, 4, 4, 4]
              NUM_CHANNELS: [18, 36, 72, 144]
              FUSE_METHOD: SUM

        LOSS:
          USE_OHEM: false
          OHEMTHRES: 0.9
          OHEMKEEP: 131072
          BALANCE_WEIGHTS: [0.4, 1.0]

        TRAIN:
          IMAGE_SIZE: [{image_w}, {image_h}]
          BASE_SIZE: {spec["base_size"]}
          DOWNSAMPLERATE: 1
          FLIP: true
          MULTI_SCALE: true
          SCALE_FACTOR: 16
          BATCH_SIZE_PER_GPU: {spec["train_batch_per_gpu"]}
          SHUFFLE: true
          BEGIN_EPOCH: 0
          END_EPOCH: {spec["end_epoch"]}
          RESUME: false
          OPTIMIZER: sgd
          LR: {spec["base_lr"]}
          EXTRA_LR: 0.001
          MOMENTUM: 0.9
          WD: 0.0005
          NESTEROV: false
          IGNORE_LABEL: 255

        TEST:
          IMAGE_SIZE: [{image_w}, {image_h}]
          BASE_SIZE: {spec["base_size"]}
          BATCH_SIZE_PER_GPU: {spec["test_batch_per_gpu"]}
          FLIP_TEST: false
          MULTI_SCALE: false
          SCALE_LIST: [1]
          MODEL_FILE: ''
        """
    )


def build_readme(stage: str, dataset_root: Path, output_dir: Path, repo_dir: Path, spec: dict, stats: dict[str, int]) -> str:
    cfg_name = f"seg_hrnet_ocr_w18_{spec['dataset_name']}.yaml"
    return textwrap.dedent(
        f"""\
        # HRNet-OCR 官方实验包

        - 阶段: `{stage}`
        - 官方仓库: `{OFFICIAL_REPO}`
        - 建议本地仓路径: `{repo_dir}`
        - 数据根目录: `{dataset_root}`
        - 结果目录: `{output_dir}`
        - 训练/验证样本: `{stats['n_train']} / {stats['n_val']}`

        ## 已生成内容

        - `data/list/train.lst`
        - `data/list/val.lst`
        - `patch/lib/datasets/darkfield.py`
        - `patch/lib/datasets/__init__.py.append.txt`
        - `patch/experiments/custom_darkfield/{cfg_name}`
        - `run_train.sh`
        - `run_test.sh`

        ## 推荐步骤

        1. `git clone {OFFICIAL_REPO} {repo_dir}`
        2. 在官方仓中安装依赖，建议独立环境
        3. 将本目录下 `patch/` 中的文件复制到官方仓对应位置
        4. 在官方仓里注册 `darkfield` 数据集类
        5. 执行 `bash {output_dir / 'run_train.sh'}`
        6. 训练完成后执行 `bash {output_dir / 'run_test.sh'}`
        7. 将结果用 `scripts/register_hrnet_official_result.py` 回填到本项目

        ## 回填结果示例

        ```bash
        uv run python scripts/register_hrnet_official_result.py \\
          --stage {stage} \\
          --best-epoch 40 \\
          --best-val-miou 0.8123 \\
          --best-loss 0.2741 \\
          --checkpoint /path/to/best.pth
        ```
        """
    )


def build_train_script(repo_dir: Path, output_dir: Path, cfg_name: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        cd "{repo_dir}"
        python tools/train.py --cfg "{output_dir / 'patch/experiments/custom_darkfield' / cfg_name}"
        """
    )


def build_test_script(repo_dir: Path, output_dir: Path, cfg_name: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        cd "{repo_dir}"
        python tools/test.py --cfg "{output_dir / 'patch/experiments/custom_darkfield' / cfg_name}" \\
          TEST.MODEL_FILE "{output_dir}/best.pth"
        """
    )


def build_commands(stage: str, repo_dir: Path, spec: dict) -> list[str]:
    output_dir = spec["output_dir"]
    cfg_name = f"seg_hrnet_ocr_w18_{spec['dataset_name']}.yaml"
    return [
        f"git clone {OFFICIAL_REPO} {repo_dir}",
        f"cd {repo_dir} && uv sync --extra hrnet  # 或按官方 README 安装 requirements.txt",
        f"bash {output_dir / 'run_train.sh'}",
        f"bash {output_dir / 'run_test.sh'}",
        (
            "uv run python scripts/register_hrnet_official_result.py "
            f"--stage {stage} --best-epoch <EPOCH> --best-val-miou <MIOU> "
            f"--best-loss <LOSS> --checkpoint {output_dir / 'best.pth'}"
        ),
        f"# 官方配置文件: {output_dir / 'patch/experiments/custom_darkfield' / cfg_name}",
    ]


def write_package(stage: str, repo_dir: Path, spec: dict) -> dict[str, str | int]:
    output_dir = spec["output_dir"]
    ensure_dir(output_dir)
    package_dir = output_dir / "hrnet_official_package"
    ensure_dir(package_dir)

    train_lst, val_lst, stats = build_dataset_lists(stage, spec["dataset_root"], package_dir)

    patch_dataset_dir = package_dir / "patch/lib/datasets"
    patch_exp_dir = package_dir / "patch/experiments/custom_darkfield"
    ensure_dir(patch_dataset_dir)
    ensure_dir(patch_exp_dir)

    dataset_py = patch_dataset_dir / "darkfield.py"
    init_append = patch_dataset_dir / "__init__.py.append.txt"
    cfg_name = f"seg_hrnet_ocr_w18_{spec['dataset_name']}.yaml"
    cfg_path = patch_exp_dir / cfg_name
    dataset_py.write_text(build_dataset_patch(spec["dataset_name"]), encoding="utf-8")
    init_append.write_text(build_init_patch(), encoding="utf-8")
    cfg_path.write_text(build_yaml(stage, spec["dataset_root"], output_dir, spec), encoding="utf-8")

    readme = output_dir / "README.md"
    readme.write_text(build_readme(stage, spec["dataset_root"], output_dir, repo_dir, spec, stats), encoding="utf-8")

    train_script = output_dir / "run_train.sh"
    test_script = output_dir / "run_test.sh"
    train_script.write_text(build_train_script(repo_dir, output_dir, cfg_name), encoding="utf-8")
    test_script.write_text(build_test_script(repo_dir, output_dir, cfg_name), encoding="utf-8")
    train_script.chmod(0o755)
    test_script.chmod(0o755)

    summary_stub = {
        "framework": "hrnet_official",
        "stage": stage,
        "status": "prepared",
        "model": "HRNet-OCR-W18",
        "best_epoch": None,
        "best_val_miou": None,
        "best_loss": None,
        "checkpoint": None,
        "output_dir": str(output_dir),
        "notes": spec["notes"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_stub, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "package_dir": str(package_dir),
        "train_list": str(train_lst),
        "val_list": str(val_lst),
        "config": str(cfg_path),
        "train_script": str(train_script),
        "test_script": str(test_script),
        "n_train": stats["n_train"],
        "n_val": stats["n_val"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 HRNet 官方实验包")
    parser.add_argument("--stage", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--write-package", action="store_true")
    args = parser.parse_args()

    spec = EXPERIMENTS[args.stage]
    package_info: dict[str, str | int] = {}
    if args.write_package:
        package_info = write_package(args.stage, args.repo_dir, spec)

    manifest = {
        "framework": "hrnet_official",
        "repo_url": OFFICIAL_REPO,
        "repo_dir": str(args.repo_dir),
        "stage": args.stage,
        "dataset_root": str(spec["dataset_root"]),
        "output_dir": str(spec["output_dir"]),
        "pixel_size_mm": 0.0068,
        "notes": spec["notes"],
        "commands": build_commands(args.stage, args.repo_dir, spec),
        "package": package_info,
    }

    if args.write_manifest:
        spec["output_dir"].mkdir(parents=True, exist_ok=True)
        (spec["output_dir"] / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   HRNet-OCR 官方实验入口                                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  阶段:       {args.stage}")
    print(f"  官方仓库:   {OFFICIAL_REPO}")
    print(f"  repo_dir:   {args.repo_dir}")
    print(f"  数据目录:   {spec['dataset_root']}")
    print(f"  输出目录:   {spec['output_dir']}")
    if args.write_package:
        print(f"  实验包:     {spec['output_dir'] / 'hrnet_official_package'}")
    if args.write_manifest:
        print(f"  Manifest:   {spec['output_dir'] / 'manifest.json'}")
    print()
    print("建议命令:")
    for cmd in manifest["commands"]:
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
