#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/experiments/phase3_segmentation/logs"

mkdir -p "$LOG_DIR"

echo "[1/6] DeepLabV3+ on MSD"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_msd_segmentation.py" \
  --model-name deeplabv3plus \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --max-minutes 60 \
  --save-metric miou \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/deeplabv3plus_r34" \
  2>&1 | tee "$LOG_DIR/deeplabv3plus_r34.log"

echo "[2/6] FPN on MSD"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_msd_segmentation.py" \
  --model-name fpn \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --max-minutes 60 \
  --save-metric miou \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/fpn_r34" \
  2>&1 | tee "$LOG_DIR/fpn_r34.log"

echo "[3/6] Unet++ private weak-label finetune"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/unetplusplus_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 40 \
  --lr 5e-4 \
  --max-minutes 90 \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/private_finetuned_unetplusplus_r34" \
  2>&1 | tee "$LOG_DIR/private_finetuned_unetplusplus_r34.log"

echo "[4/6] LightUNet private weak-label finetune"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/light_unet_baseline/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 40 \
  --lr 5e-4 \
  --max-minutes 90 \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/private_finetuned_light_unet" \
  2>&1 | tee "$LOG_DIR/private_finetuned_light_unet.log"

echo "[5/6] DeepLabV3+ private weak-label finetune"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/deeplabv3plus_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 40 \
  --lr 5e-4 \
  --max-minutes 90 \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/private_finetuned_deeplabv3plus_r34" \
  2>&1 | tee "$LOG_DIR/private_finetuned_deeplabv3plus_r34.log"

echo "[6/6] FPN private weak-label finetune"
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/model_zoo/fpn_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 40 \
  --lr 5e-4 \
  --max-minutes 90 \
  --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/private_finetuned_fpn_r34" \
  2>&1 | tee "$LOG_DIR/private_finetuned_fpn_r34.log"

echo
echo "All scheduled segmentation experiments completed."
echo "Logs:"
echo "  $LOG_DIR"
