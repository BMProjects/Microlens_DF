#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/experiments/phase3_segmentation/batch2_logs"
FROM_STEP=1
TO_STEP=6
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_segmentation_batch2.sh [--from-step N] [--to-step M] [--prepare-only] [--list-steps]

Examples:
  bash scripts/run_segmentation_batch2.sh --list-steps
  bash scripts/run_segmentation_batch2.sh
  bash scripts/run_segmentation_batch2.sh --from-step 1 --to-step 4
  bash scripts/run_segmentation_batch2.sh --from-step 5 --to-step 6 --prepare-only
EOF
}

list_steps() {
  cat <<'EOF'
1. FPN (SMP) on MSD
2. FPN (SMP) on private weak labels
3. SegFormer-B2 (HF Transformers) on MSD
4. SegFormer-B2 (HF Transformers) on private weak labels
5. HRNet-OCR-W18 on MSD
6. HRNet-OCR-W18 on private weak labels
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-step)
      FROM_STEP="${2:?missing value for --from-step}"
      shift 2
      ;;
    --to-step)
      TO_STEP="${2:?missing value for --to-step}"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --list-steps)
      list_steps
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$FROM_STEP" =~ ^[1-6]$ && "$TO_STEP" =~ ^[1-6]$ ]]; then
  echo "--from-step and --to-step must be integers in [1, 6]" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_cmd() {
  local log_file="$1"
  shift
  (
    cd "$ROOT_DIR"
    "$@"
  ) 2>&1 | tee "$log_file"
}

run_step() {
  local step_num="$1"
  local label="$2"
  local log_file="$3"
  shift 3

  if (( step_num < FROM_STEP || step_num > TO_STEP )); then
    echo "[skip ${step_num}/6] ${label}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[run ${step_num}/6] ${label}"
  echo "============================================================"
  echo "log: $log_file"

  if (( PREPARE_ONLY == 1 )); then
    echo "prepare-only mode: skip actual execution."
    return 0
  fi

  run_cmd "$log_file" "$@"
}

echo "[Plan] Batch-2 segmentation experiments"
echo "  Local mainline:"
echo "    - SMP 继续做 CNN 系列 (FPN)"
echo "    - HF Transformers 补 SegFormer-B2"
echo "    - HRNet-OCR-W18 基于官方结构做本地实现"
echo "Run range: step $FROM_STEP -> step $TO_STEP"
echo "Prepare:   $PREPARE_ONLY"

run_step \
  1 \
  "FPN (SMP) on MSD" \
  "$LOG_DIR/step01_fpn_msd.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_msd_segmentation.py" \
    --model-name fpn \
    --encoder-name resnet34 \
    --encoder-weights imagenet \
    --device cuda \
    --batch-size 8 \
    --num-workers 8 \
    --max-minutes 90 \
    --save-metric miou \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_fpn_msd"

run_step \
  2 \
  "FPN (SMP) on private weak labels" \
  "$LOG_DIR/step02_fpn_private.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
    --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_fpn_msd/best.pt" \
    --model-name fpn \
    --encoder-name resnet34 \
    --encoder-weights imagenet \
    --device cuda \
    --batch-size 4 \
    --num-workers 8 \
    --max-minutes 120 \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_fpn_private"

run_step \
  3 \
  "SegFormer-B2 (HF Transformers) on MSD" \
  "$LOG_DIR/step03_segformer_b2_hf_msd.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_msd_segmentation.py" \
    --model-name segformer_b2_hf \
    --device cuda \
    --batch-size 4 \
    --num-workers 8 \
    --max-minutes 120 \
    --save-metric miou \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_segformer_b2_hf_msd"

run_step \
  4 \
  "SegFormer-B2 (HF Transformers) on private weak labels" \
  "$LOG_DIR/step04_segformer_b2_hf_private.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
    --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_segformer_b2_hf_msd/best.pt" \
    --model-name segformer_b2_hf \
    --device cuda \
    --batch-size 2 \
    --num-workers 8 \
    --max-minutes 150 \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_segformer_b2_hf_private"

run_step \
  5 \
  "HRNet-OCR-W18 on MSD" \
  "$LOG_DIR/step05_hrnet_ocr_w18_msd.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_msd_segmentation.py" \
    --model-name hrnet_ocr_w18 \
    --device cuda \
    --batch-size 6 \
    --num-workers 8 \
    --max-minutes 120 \
    --save-metric miou \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_hrnet_ocr_w18_msd"

run_step \
  6 \
  "HRNet-OCR-W18 on private weak labels" \
  "$LOG_DIR/step06_hrnet_ocr_w18_private.log" \
  env UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$ROOT_DIR/scripts/train_private_segmentation.py" \
    --pretrained "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_hrnet_ocr_w18_msd/best.pt" \
    --model-name hrnet_ocr_w18 \
    --device cuda \
    --batch-size 3 \
    --num-workers 8 \
    --max-minutes 150 \
    --output-dir "$ROOT_DIR/output/experiments/phase3_segmentation/batch2_hrnet_ocr_w18_private"

echo
echo "Batch-2 experiment run completed."
