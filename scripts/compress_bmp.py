#!/usr/bin/env python3
"""BMP → PNG 无损压缩脚本.

读取 BMP 图像数据，根据图像属性自动判断通道数和 bit 深度。
对多通道图像仅保留绿色通道（暗场灰度强度代理），并保持位深不变；
随后使用 PNG 无损压缩算法进行压缩，保持文件名不变（仅更换扩展名）。

Usage:
    python scripts/compress_bmp.py /path/to/image/dir
    python scripts/compress_bmp.py /path/to/image/dir --delete-original
    python scripts/compress_bmp.py /path/to/image/dir --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def read_bmp_info(filepath: Path) -> tuple[dict, cv2.typing.MatLike | None]:
    """使用 OpenCV 读取 BMP 图像和属性."""
    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {}, None

    if img.ndim == 2:
        channels = 1
    else:
        channels = int(img.shape[2])

    bits_per_channel = int(img.dtype.itemsize * 8)
    h, w = img.shape[:2]
    bpp = channels * bits_per_channel

    return {
        "width": int(w),
        "height": int(h),
        "bpp": int(bpp),
        "channels": channels,
        "bits_per_channel": bits_per_channel,
    }, img


def compress_bmp_to_png(
    bmp_path: Path,
    output_dir: Path | None = None,
    delete_original: bool = False,
    dry_run: bool = False,
) -> dict:
    """将单个 BMP 文件无损压缩为 PNG.

    Args:
        bmp_path: BMP 文件路径.
        output_dir: 输出目录，默认与原文件同目录.
        delete_original: 是否删除原始 BMP 文件.
        dry_run: 仅报告，不实际操作.

    Returns:
        压缩结果统计字典.
    """
    bmp_path = Path(bmp_path)
    if not bmp_path.exists():
        return {"error": f"文件不存在: {bmp_path}"}

    # OpenCV 读取属性与图像
    info, img = read_bmp_info(bmp_path)
    if img is None:
        return {"error": f"OpenCV 无法读取: {bmp_path}"}

    # 确定输出路径
    if output_dir is None:
        output_dir = bmp_path.parent
    png_path = output_dir / (bmp_path.stem + ".png")

    result = {
        "source": str(bmp_path),
        "target": str(png_path),
        "width": info["width"],
        "height": info["height"],
        "bpp": info["bpp"],
        "channels": info["channels"],
        "bits_per_channel": info["bits_per_channel"],
        "original_size": bmp_path.stat().st_size,
    }

    if dry_run:
        result["action"] = "dry_run"
        return result

    # 暗场图像预处理：
    # - 单通道: 保持不变
    # - 多通道: 仅保留绿色通道（OpenCV 默认通道顺序 BGR/BGRA）
    output_channels = 1
    if img.ndim == 3:
        img = img[:, :, 1]

    bits_per_channel_after = int(img.dtype.itemsize * 8)

    # PNG 压缩级别 9（最高无损压缩率）
    compress_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    success = cv2.imwrite(str(png_path), img, compress_params)

    if not success:
        return {"error": f"PNG 写入失败: {png_path}"}

    compressed_size = png_path.stat().st_size
    ratio = (1.0 - compressed_size / result["original_size"]) * 100

    result["compressed_size"] = compressed_size
    result["compression_ratio"] = ratio
    result["action"] = "compressed"
    result["output_channels"] = output_channels
    result["output_bits_per_channel"] = bits_per_channel_after

    # 删除原始文件
    if delete_original:
        bmp_path.unlink()
        result["action"] = "compressed+deleted"

    return result


def process_directory(
    directory: Path,
    output_dir: Path | None = None,
    delete_original: bool = False,
    dry_run: bool = False,
    recursive: bool = True,
) -> list[dict]:
    """批量处理目录中的 BMP 文件.

    Args:
        directory: 输入目录.
        output_dir: 输出目录（默认原地输出）.
        delete_original: 是否删除原始 BMP.
        dry_run: 仅报告不执行.
        recursive: 是否递归子目录.

    Returns:
        每张图像的压缩结果列表.
    """
    directory = Path(directory)
    if not directory.is_dir():
        print(f"❌ 目录不存在: {directory}", file=sys.stderr)
        return []

    if recursive:
        bmp_files = sorted(directory.rglob("*.bmp"))
    else:
        bmp_files = sorted(directory.glob("*.bmp"))

    # 同时检查大写 .BMP
    if recursive:
        bmp_files += sorted(directory.rglob("*.BMP"))
    else:
        bmp_files += sorted(directory.glob("*.BMP"))

    # 去重
    bmp_files = sorted(set(bmp_files))

    if not bmp_files:
        print(f"⚠️ 目录中未找到 BMP 文件: {directory}")
        return []

    print(f"📁 扫描到 {len(bmp_files)} 个 BMP 文件")
    if dry_run:
        print("🔍 [DRY RUN] 仅分析，不执行压缩\n")
    else:
        print(f"🚀 开始无损压缩 (PNG level 9)...\n")

    results = []
    total_orig = 0
    total_comp = 0

    for i, bmp_path in enumerate(bmp_files, 1):
        # 确定输出目录
        if output_dir is not None:
            # 保持目录结构
            rel = bmp_path.parent.relative_to(directory)
            out = output_dir / rel
            out.mkdir(parents=True, exist_ok=True)
        else:
            out = None

        result = compress_bmp_to_png(bmp_path, out, delete_original, dry_run)
        results.append(result)

        # 打印进度
        fname = bmp_path.name
        if "error" in result:
            print(f"  [{i}/{len(bmp_files)}] ❌ {fname}: {result['error']}")
        elif dry_run:
            orig_mb = result["original_size"] / 1024 / 1024
            output_ch = 1 if result["channels"] > 1 else result["channels"]
            print(
                f"  [{i}/{len(bmp_files)}] 📄 {fname}: "
                f"{result['width']}×{result['height']} "
                f"{result['channels']}ch→{output_ch}ch {result['bits_per_channel']}bit "
                f"({orig_mb:.1f} MB)"
            )
            total_orig += result["original_size"]
        else:
            orig_mb = result["original_size"] / 1024 / 1024
            comp_mb = result["compressed_size"] / 1024 / 1024
            ratio = result["compression_ratio"]
            print(
                f"  [{i}/{len(bmp_files)}] ✅ {fname}: "
                f"{result['channels']}ch→{result.get('output_channels', 1)}ch "
                f"{result['bits_per_channel']}bit→{result.get('output_bits_per_channel', result['bits_per_channel'])}bit, "
                f"{orig_mb:.1f} MB → {comp_mb:.1f} MB "
                f"({ratio:.1f}% 压缩)"
            )
            total_orig += result["original_size"]
            total_comp += result["compressed_size"]

    print()
    if dry_run:
        print(f"📊 总计: {len(bmp_files)} 文件, {total_orig/1024/1024:.1f} MB")
    else:
        if total_orig > 0:
            overall_ratio = (1.0 - total_comp / total_orig) * 100
            print(
                f"📊 总计: {len(bmp_files)} 文件\n"
                f"   原始总大小: {total_orig/1024/1024:.1f} MB\n"
                f"   压缩后总大小: {total_comp/1024/1024:.1f} MB\n"
                f"   整体压缩率: {overall_ratio:.1f}%"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BMP → PNG 无损批量压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="包含 BMP 文件的目录路径")
    parser.add_argument(
        "--output", "-o",
        help="输出目录（默认原地输出到 BMP 同目录）",
        default=None,
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="压缩成功后删除原始 BMP 文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅分析 BMP 文件属性，不执行压缩",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归子目录",
    )

    args = parser.parse_args()
    directory = Path(args.directory)
    output = Path(args.output) if args.output else None

    process_directory(
        directory,
        output_dir=output,
        delete_original=args.delete_original,
        dry_run=args.dry_run,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
