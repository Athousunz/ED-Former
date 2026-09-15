#!/usr/bin/env bash
# Paper-style prediction visualization for TimeBridge variants on Solar.
#
# Usage:
#   1) First finish training/testing so that results/<setting>/{pred,true,input}.npy exist.
#   2) Fill MODEL_SPECS below (Name:result_dir), then run:
#        bash scripts/visualize_predictions.sh
#
# Quick discover available result folders:
#   ls -1 results

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-Solar}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
CHANNEL="${CHANNEL:--1}"
OUT_DIR="${OUT_DIR:-./_logs/vis_${DATASET}}"
OUT_NAME="${OUT_NAME:-prediction_comparison}"

# Edit these paths after your experiments finish.
# Format: "DisplayName:./results/<setting_folder>"
MODEL_SPECS=(
  "TimeBridge:./results/REPLACE_ME_BASELINE"
  "TimeBridge+ED-SRA:./results/REPLACE_ME_EDSRA"
  "TimeBridge+EA-RevIN:./results/REPLACE_ME_EAREVIN"
  "TimeBridge+Both:./results/REPLACE_ME_BOTH"
)

echo "Available result folders:"
ls -1 results 2>/dev/null || echo "(no results/ yet)"

python -u experiments/visualize_predictions.py \
  --dataset "$DATASET" \
  --sample_idx "$SAMPLE_IDX" \
  --channel "$CHANNEL" \
  --out_dir "$OUT_DIR" \
  --out_name "$OUT_NAME" \
  --models "${MODEL_SPECS[@]}"
