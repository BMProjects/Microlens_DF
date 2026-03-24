#!/usr/bin/env bash
# =============================================================================
# 全量切图 + 缺陷标注数据集构建
# 用法: bash scripts/run_build_dataset.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  暗场镜片缺陷 — 切图标注数据集全量构建"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

IMG_DIR="output/dataset_v2/images"
ROI="output/dataset_v2/roi_mask.png"
OUT_DIR="output/tile_dataset"
CONFIG="configs/detect_tile.yaml"

# 检查输入
if [ ! -d "$IMG_DIR" ]; then
    echo "[ERROR] 预处理图像目录不存在: $IMG_DIR"
    echo "  请先运行: python scripts/run_batch_preprocess_dataset.py"
    exit 1
fi

if [ ! -f "$ROI" ]; then
    echo "[ERROR] ROI 掩膜不存在: $ROI"
    exit 1
fi

echo "  输入目录 : $IMG_DIR"
echo "  输出目录 : $OUT_DIR"
echo "  配置文件 : $CONFIG"
echo "  预计耗时 : ~50 分钟 (247 张图, 每图约 12s)"
echo ""
echo "  [提示] 若要中断: Ctrl+C  重启后加 --resume 可跳过已处理图像"
echo ""

python scripts/build_tile_dataset.py \
    --img-dir "$IMG_DIR" \
    --roi "$ROI" \
    --out-dir "$OUT_DIR" \
    --config "$CONFIG" \
    --all

echo ""
echo "============================================================"
echo "  构建完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  下一步: 打开 $OUT_DIR/overlays/ 人工抽样审核标注质量"
echo "  训练配置: $OUT_DIR/defects.yaml"
echo "============================================================"
