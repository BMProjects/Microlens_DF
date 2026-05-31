#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE3E_DIR="$ROOT_DIR/output/experiments/phase3e"
LOG_DIR="$PHASE3E_DIR/logs"
SEG_ANALYSIS_DIR="$ROOT_DIR/output/experiments/phase3_segmentation/analysis/segmentation_review_phase3e"
DET_TRAIN_PROJECT="$PHASE3E_DIR/detection_training"
DET_TRAIN_NAME="b2_nwd_only_phase3e"
DET_WEIGHTS="$DET_TRAIN_PROJECT/$DET_TRAIN_NAME/weights/best.pt"
DET_EVAL_DIR="$PHASE3E_DIR/detection_eval/b2_nwd_only"
DET_COMPARE_DIR="$ROOT_DIR/output/experiments/comparison/phase3e_detection_compare"

mkdir -p "$LOG_DIR" "$PHASE3E_DIR" "$SEG_ANALYSIS_DIR" "$DET_TRAIN_PROJECT" "$DET_EVAL_DIR" "$DET_COMPARE_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

FROM_STEP=1
TO_STEP=6
RUN_SUMMARY=0
SUMMARY_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_phase3e.sh [--from-step N] [--to-step M] [--summary] [--summary-only] [--list-steps]

Examples:
  bash scripts/run_phase3e.sh
  bash scripts/run_phase3e.sh --summary
  bash scripts/run_phase3e.sh --from-step 4
  bash scripts/run_phase3e.sh --from-step 4 --to-step 6
  bash scripts/run_phase3e.sh --from-step 1 --to-step 3
  bash scripts/run_phase3e.sh --summary-only
EOF
}

list_steps() {
  cat <<'EOF'
1. S-E01 Unet++ bridge rerun
2. S-E02 FPN bridge rerun
3. S-E03 Refresh 24-image segmentation analysis
4. D-E01 Reproduce B2 NWD-only training
5. D-E02 Run CNAS evaluation for B2 NWD-only
6. D-E03 Compare A0 baseline vs B2 NWD-only
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
    --summary)
      RUN_SUMMARY=1
      shift
      ;;
    --summary-only)
      RUN_SUMMARY=1
      SUMMARY_ONLY=1
      shift
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
if (( SUMMARY_ONLY == 1 )); then
  FROM_STEP=1
  TO_STEP=0
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
  "$@" 2>&1 | tee "$log_path"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

echo "[Plan] Phase 3E P0 automation"
echo "  1. Segmentation bridge rerun (Unet++ / FPN)"
echo "  2. Refresh 24-image segmentation review"
echo "  3. Reproduce and re-evaluate B2 NWD-only"
echo "  4. Optional Phase 3E HTML summary"
echo
if (( SUMMARY_ONLY == 1 )); then
  echo "Run range: summary only"
else
  echo "Run range: step $FROM_STEP -> step $TO_STEP"
fi
echo "Logs: $LOG_DIR"
echo "Segmentation analysis: $SEG_ANALYSIS_DIR"
echo "Detection train root: $DET_TRAIN_PROJECT"
echo "Detection eval: $DET_EVAL_DIR"
echo "Detection compare: $DET_COMPARE_DIR"
echo "Summary: $PHASE3E_DIR/summary/phase3e_summary.html"
echo
echo "Note: step 4 now writes into an isolated Phase 3E detection directory"
echo "      $DET_TRAIN_PROJECT/$DET_TRAIN_NAME"

run_step \
  1 \
  "S-E01 Unet++ bridge rerun" \
  "$LOG_DIR/step1_unetpp_bridge.log" \
  bash "$ROOT_DIR/scripts/run_segmentation_12h.sh" --from-step 5 --to-step 5

run_step \
  2 \
  "S-E02 FPN bridge rerun" \
  "$LOG_DIR/step2_fpn_bridge.log" \
  bash "$ROOT_DIR/scripts/run_segmentation_12h.sh" --from-step 6 --to-step 6

run_step \
  3 \
  "S-E03 Refresh 24-image segmentation analysis" \
  "$LOG_DIR/step3_segmentation_analysis.log" \
  uv run --no-sync python "$ROOT_DIR/scripts/analyze_segmentation_results.py" \
    --device cuda \
    --sample-count 24 \
    --analysis-dir "$SEG_ANALYSIS_DIR"

run_step \
  4 \
  "D-E01 Reproduce B2 NWD-only training" \
  "$LOG_DIR/step4_b2_train.log" \
  uv run --no-sync python "$ROOT_DIR/scripts/train_phase2_sfe_nwd.py" \
    --exp B2 \
    --device cuda \
    --epochs 60 \
    --batch 32 \
    --workers 8 \
    --lr0 0.001 \
    --project "$DET_TRAIN_PROJECT" \
    --run-name "$DET_TRAIN_NAME" \
    --cache disk \
    --no-countdown

require_file "$DET_WEIGHTS"
require_file "$ROOT_DIR/output/training/stage2_cleaned/weights/best.pt"

run_step \
  5 \
  "D-E02 Run CNAS evaluation for B2 NWD-only" \
  "$LOG_DIR/step5_b2_cnas_eval.log" \
  uv run --no-sync python "$ROOT_DIR/scripts/run_cnas_eval.py" \
    --weights "$DET_WEIGHTS" \
    --save-dir "$DET_EVAL_DIR"

run_step \
  6 \
  "D-E03 Compare A0 baseline vs B2 NWD-only" \
  "$LOG_DIR/step6_detection_compare.log" \
  uv run --no-sync python "$ROOT_DIR/scripts/compare_experiments.py" \
    --weights "$ROOT_DIR/output/training/stage2_cleaned/weights/best.pt:A0_baseline" \
    --weights "$DET_WEIGHTS:B2_nwd_only_phase3e" \
    --save-dir "$DET_COMPARE_DIR"

if (( RUN_SUMMARY == 1 )); then
  echo
  echo "============================================================"
  echo "[summary] Phase 3E HTML summary"
  echo "============================================================"
  uv run --no-sync python "$ROOT_DIR/scripts/generate_phase3e_summary.py" \
    --output-dir "$PHASE3E_DIR/summary" 2>&1 | tee "$LOG_DIR/step_summary_phase3e.log"
fi

echo
echo "Requested Phase 3E P0 steps completed."
echo "Check logs under: $LOG_DIR"
echo "Check segmentation analysis under: $SEG_ANALYSIS_DIR"
echo "Check detection eval under: $DET_EVAL_DIR"
echo "Check detection comparison under: $DET_COMPARE_DIR"
if (( RUN_SUMMARY == 1 )); then
  echo "Check HTML summary under: $PHASE3E_DIR/summary"
fi
