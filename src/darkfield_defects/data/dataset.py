"""Patch 训练数据集 — 用于从图像+掩码生成 512×512 训练 patch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch is required: pip install darkfield-defects[ml]")


# 多类掩码编码: 0=background, 1=scratch, 2=spot, 3=damage
NUM_CLASSES = 4
CLASS_NAMES = ["background", "scratch", "spot", "damage"]


class DefectPatchDataset(Dataset):
    """从全图+掩码对生成随机 patch 的 Dataset.

    目录结构:
        images/
            img001.png
        masks/
            img001.png   # 灰度, 像素值 = 类别 ID (0/1/2/3)

    Args:
        image_dir: 图像目录.
        mask_dir: 掩码目录.
        patch_size: 裁切尺寸.
        patches_per_image: 每张全图随机采样 patch 数.
        augment: 是否数据增强.
        candidate_dir: 可选候选图目录 (用作第二通道输入).
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        patch_size: int = 512,
        patches_per_image: int = 8,
        augment: bool = True,
        candidate_dir: str | Path | None = None,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.candidate_dir = Path(candidate_dir) if candidate_dir else None

        # 找匹配的图像-掩码对
        mask_stems = {p.stem for p in self.mask_dir.glob("*.png")}
        self.pairs: list[tuple[Path, Path, Optional[Path]]] = []

        for img_path in sorted(self.image_dir.glob("*.png")):
            if img_path.stem in mask_stems:
                mask_path = self.mask_dir / f"{img_path.stem}.png"
                cand_path = None
                if self.candidate_dir and (self.candidate_dir / f"{img_path.stem}.png").exists():
                    cand_path = self.candidate_dir / f"{img_path.stem}.png"
                self.pairs.append((img_path, mask_path, cand_path))

        if not self.pairs:
            raise ValueError(f"未找到匹配的图像-掩码对: {image_dir} / {mask_dir}")

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_idx = idx // self.patches_per_image
        img_path, mask_path, cand_path = self.pairs[pair_idx]

        # 加载图像
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"加载失败: {img_path}")

        # 可选候选图
        candidate = None
        if cand_path is not None:
            candidate = cv2.imread(str(cand_path), cv2.IMREAD_GRAYSCALE)

        # 随机裁切 patch
        image_patch, mask_patch, cand_patch = self._random_crop(
            image, mask, candidate
        )

        # 数据增强
        if self.augment:
            image_patch, mask_patch, cand_patch = self._augment(
                image_patch, mask_patch, cand_patch
            )

        # 构建输入通道
        channels = [image_patch.astype(np.float32) / 255.0]
        if cand_patch is not None:
            channels.append(cand_patch.astype(np.float32) / 255.0)

        x = np.stack(channels, axis=0)  # (C, H, W)
        y = mask_patch.astype(np.int64)  # (H, W), values 0-3

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
        }

    def _random_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        candidate: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """随机裁切 patch, 优先选择含缺陷区域."""
        h, w = image.shape[:2]
        ps = self.patch_size

        # 50% 概率—偏向含缺陷区域
        if random.random() < 0.5 and np.any(mask > 0):
            defect_coords = np.argwhere(mask > 0)
            center = defect_coords[random.randint(0, len(defect_coords) - 1)]
            r = max(0, min(center[0] - ps // 2, h - ps))
            c = max(0, min(center[1] - ps // 2, w - ps))
        else:
            r = random.randint(0, max(0, h - ps))
            c = random.randint(0, max(0, w - ps))

        img_crop = image[r:r + ps, c:c + ps]
        mask_crop = mask[r:r + ps, c:c + ps]

        # 若裁切太小则 pad
        if img_crop.shape[0] < ps or img_crop.shape[1] < ps:
            img_crop = self._pad_to(img_crop, ps)
            mask_crop = self._pad_to(mask_crop, ps)

        cand_crop = None
        if candidate is not None:
            cand_crop = candidate[r:r + ps, c:c + ps]
            if cand_crop.shape[0] < ps or cand_crop.shape[1] < ps:
                cand_crop = self._pad_to(cand_crop, ps)

        return img_crop, mask_crop, cand_crop

    @staticmethod
    def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
        """Zero-pad 到目标尺寸."""
        h, w = arr.shape[:2]
        result = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
        result[:h, :w] = arr
        return result

    @staticmethod
    def _augment(
        image: np.ndarray,
        mask: np.ndarray,
        candidate: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """简单数据增强: 翻转 + 旋转 + 亮度扰动."""
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            if candidate is not None:
                candidate = np.fliplr(candidate).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            if candidate is not None:
                candidate = np.flipud(candidate).copy()

        # 随机 90° 旋转
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
            if candidate is not None:
                candidate = np.rot90(candidate, k).copy()

        # 亮度扰动 (仅 image)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image, mask, candidate

    @property
    def in_channels(self) -> int:
        """输入通道数."""
        return 2 if self.candidate_dir else 1
