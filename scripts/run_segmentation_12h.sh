#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE3_DIR="$ROOT_DIR/output/experiments/phase3_segmentation"
LOG_DIR="$PHASE3_DIR/logs"
ANALYSIS_DIR="$PHASE3_DIR/analysis/segmentation_review_20260325_12h"
BRIDGE_DIR="$PHASE3_DIR/bridge_review_20260325_12h"

mkdir -p "$LOG_DIR" "$ANALYSIS_DIR" "$BRIDGE_DIR"

FROM_STEP=1
TO_STEP=6

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_segmentation_12h.sh [--from-step N] [--to-step M] [--list-steps]

Examples:
  bash scripts/run_segmentation_12h.sh
  bash scripts/run_segmentation_12h.sh --from-step 5
  bash scripts/run_segmentation_12h.sh --from-step 4 --to-step 6
  bash scripts/run_segmentation_12h.sh --from-step 5 --to-step 5
EOF
}

list_steps() {
  cat <<'EOF'
1. Unet++ private weak-label refinement
2. FPN private weak-label refinement
3. DeepLabV3+ private weak-label refinement
4. Refresh quantitative analysis + 24-image visual review
5. Full-image bridge review with Unet++
6. Full-image bridge review with FPN
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
if (( FROM_STEP > TO_STEP )); then
  echo "--from-step cannot be greater than --to-step" >&2
  exit 1
fi

run_step() {
  local step_num="$1"
  local label="$2"
  local log_path="$3"
  shift 3

  if (( step_num < FROM_STEP || step_num > TO_STEP )); then
    echo "[skip ${step_num}/6] ${label}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[run ${step_num}/6] ${label}"
  echo "============================================================"

  UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$@" 2>&1 | tee "$log_path"
}

echo "[Plan] 12h segmentation window"
echo "  1. Private-domain refinement for Unet++ / FPN / DeepLabV3+"
echo "  2. Refresh 24-image visual + quantitative review"
echo "  3. Full-image bridge review for top candidates"
echo
echo "Run range: step $FROM_STEP -> step $TO_STEP"
echo "Logs: $LOG_DIR"
echo "Analysis: $ANALYSIS_DIR"
echo "Bridge review: $BRIDGE_DIR"

run_step \
  1 \
  "Unet++ private weak-label refinement (target-domain priority)" \
  "$LOG_DIR/private_finetuned_unetplusplus_r34_12h.log" \
  "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$PHASE3_DIR/model_zoo/unetplusplus_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 60 \
  --lr 3e-4 \
  --max-minutes 150 \
  --output-dir "$PHASE3_DIR/private_finetuned_unetplusplus_r34"

run_step \
  2 \
  "FPN private weak-label refinement (balanced geometry + stability)" \
  "$LOG_DIR/private_finetuned_fpn_r34_12h.log" \
  "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$PHASE3_DIR/model_zoo/fpn_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 60 \
  --lr 3e-4 \
  --max-minutes 150 \
  --output-dir "$PHASE3_DIR/private_finetuned_fpn_r34"

run_step \
  3 \
  "DeepLabV3+ private weak-label refinement (check area expansion persists)" \
  "$LOG_DIR/private_finetuned_deeplabv3plus_r34_12h.log" \
  "$ROOT_DIR/scripts/train_private_segmentation.py" \
  --pretrained "$PHASE3_DIR/model_zoo/deeplabv3plus_r34/best.pt" \
  --device cuda \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 60 \
  --lr 3e-4 \
  --max-minutes 150 \
  --output-dir "$PHASE3_DIR/private_finetuned_deeplabv3plus_r34"

run_step \
  4 \
  "Refresh quantitative analysis + 24-image visual review" \
  "$LOG_DIR/analyze_segmentation_results_12h.log" \
  "$ROOT_DIR/scripts/analyze_segmentation_results.py" \
  --device cuda \
  --sample-count 24 \
  --analysis-dir "$ANALYSIS_DIR"

if (( TO_STEP >= 5 && FROM_STEP <= 6 )); then
  mapfile -t REVIEW_STEMS < <(
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python - <<'PY'
import json
from pathlib import Path

path = Path("output/experiments/phase3_segmentation/analysis/segmentation_review_20260325_12h/selected_samples.json")
samples = json.loads(path.read_text(encoding="utf-8"))
for name in samples[:8]:
    print(Path(name).stem)
PY
  )

  if [ "${#REVIEW_STEMS[@]}" -eq 0 ]; then
    echo "未能从 $ANALYSIS_DIR/selected_samples.json 读取桥接核查样本。" >&2
    exit 1
  fi
fi

run_step \
  5 \
  "Full-image bridge review with Unet++ (top-1 candidate)" \
  "$LOG_DIR/bridge_unetplusplus_r34_12h.log" \
  "$ROOT_DIR/scripts/run_bridge_review.py" \
  --stems "${REVIEW_STEMS[@]}" \
  --seg-weights "$PHASE3_DIR/private_finetuned_unetplusplus_r34/best.pt" \
  --out-dir "$BRIDGE_DIR/unetplusplus_r34"

run_step \
  6 \
  "Full-image bridge review with FPN (top-2 candidate)" \
  "$LOG_DIR/bridge_fpn_r34_12h.log" \
  "$ROOT_DIR/scripts/run_bridge_review.py" \
  --stems "${REVIEW_STEMS[@]}" \
  --seg-weights "$PHASE3_DIR/private_finetuned_fpn_r34/best.pt" \
  --out-dir "$BRIDGE_DIR/fpn_r34"

echo
echo "Requested segmentation steps completed."
echo "Check logs under: $LOG_DIR"
echo "Check refreshed analysis under: $ANALYSIS_DIR"
echo "Check full-image bridge review under: $BRIDGE_DIR"
