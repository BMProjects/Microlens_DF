"""数据加载模块 — 目录扫描 + 图像读取 + 元数据管理."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from darkfield_defects.exceptions import ImageLoadError
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


class LensSide(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    UNKNOWN = "?"


class ImageType(str, Enum):
    LENS = "lens"
    BACKGROUND = "background"
    CALIBRATION = "calibration"
    REFERENCE = "reference"
    OTHER = "other"


@dataclass
class ImageInfo:
    """图像文件元数据."""
    path: Path
    filename: str
    image_type: ImageType = ImageType.OTHER
    lens_side: LensSide = LensSide.UNKNOWN
    lens_id: Optional[str] = None
    batch: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


def parse_filename(filename: str) -> ImageInfo:
    """从文件名解析镜片元数据.

    支持的命名模式:
      - "123l.bmp" / "123r.bmp" → 编号123, 左/右镜片
      - "37l-ljj.bmp" → 编号37, 左镜片, 标记 ljj
      - "bg-1.bmp" → 背景图
      - "2.5d-1.bmp" → 2.5D 类型图像
    """
    stem = Path(filename).stem.lower()
    info = ImageInfo(path=Path(filename), filename=filename)

    # 背景图
    if stem.startswith("bg") or "背景" in filename:
        info.image_type = ImageType.BACKGROUND
        return info

    # 编号+左右镜片模式: "123l", "37l-ljj"
    match = re.match(r"^(\d+)([lr])(?:-(.+))?$", stem)
    if match:
        info.lens_id = match.group(1)
        info.lens_side = LensSide.LEFT if match.group(2) == "l" else LensSide.RIGHT
        info.image_type = ImageType.LENS
        if match.group(3):
            info.tags.append(match.group(3))
        return info

    # 2.5D 图像
    if stem.startswith("2.5d"):
        info.image_type = ImageType.CALIBRATION
        info.tags.append("2.5d")
        return info

    # 中文文件名分析
    if any(kw in filename for kw in ["在用", "旧眼镜", "换腿", "钢化膜"]):
        info.image_type = ImageType.LENS
    elif "白片" in filename or "树脂" in filename:
        info.image_type = ImageType.REFERENCE
    elif "培养皿" in filename or "分辨率板" in filename:
        info.image_type = ImageType.CALIBRATION
    elif "离焦" in filename or "星优" in filename or "思问" in filename or "艾沐" in filename:
        info.image_type = ImageType.LENS
    elif "单片" in filename:
        info.image_type = ImageType.LENS

    # 粗略左右判断
    if "L" in Path(filename).stem and info.lens_side == LensSide.UNKNOWN:
        info.lens_side = LensSide.LEFT
    if ("R" in Path(filename).stem or "Ｒ" in Path(filename).stem) and info.lens_side == LensSide.UNKNOWN:
        info.lens_side = LensSide.RIGHT

    return info


def scan_directory(
    root: str | Path,
    extensions: tuple[str, ...] = (".bmp", ".tif", ".tiff", ".png"),
    recursive: bool = True,
) -> list[ImageInfo]:
    """扫描目录中的图像文件."""
    root = Path(root)
    if not root.exists():
        raise ImageLoadError(f"目录不存在: {root}")

    images: list[ImageInfo] = []
    walker = root.rglob("*") if recursive else root.iterdir()

    for filepath in walker:
        if filepath.is_file() and filepath.suffix.lower() in extensions:
            info = parse_filename(filepath.name)
            info.path = filepath
            if filepath.parent != root:
                info.batch = filepath.parent.name
            images.append(info)

    images.sort(key=lambda x: x.filename)
    logger.info(f"扫描到 {len(images)} 张图像: {root}")
    return images


def load_image(path: str | Path, grayscale: bool = True) -> np.ndarray:
    """读取图像文件为 numpy 数组."""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"文件不存在: {path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ImageLoadError(f"无法读取图像: {path}")

    logger.debug(f"加载图像: {path.name}, shape={img.shape}, dtype={img.dtype}")
    return img


def load_background(
    path_or_dir: str | Path,
    extensions: tuple[str, ...] = (".bmp", ".png"),
) -> np.ndarray:
    """加载背景图像（单张或多张平均），返回 float64."""
    p = Path(path_or_dir)

    if p.is_file():
        return load_image(p, grayscale=True).astype(np.float64)

    if p.is_dir():
        bg_files = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in extensions and "bg" in f.stem.lower()
        )
        if not bg_files:
            raise ImageLoadError(f"目录中未找到背景图: {p}")

        bg_sum: Optional[np.ndarray] = None
        for bf in bg_files:
            img = load_image(bf, grayscale=True).astype(np.float64)
            bg_sum = img if bg_sum is None else bg_sum + img

        bg = bg_sum / len(bg_files)  # type: ignore[operator]
        logger.info(f"从 {len(bg_files)} 张图平均计算背景")
        return bg

    raise ImageLoadError(f"路径无效: {p}")
