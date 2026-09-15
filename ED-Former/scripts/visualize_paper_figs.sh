#!/usr/bin/env bash
# Paper-style visualizations (Fig.7 cointegration / Fig.8-9 attention heatmaps)
#
# Examples:
#   bash scripts/visualize_paper_figs.sh
#   MODE=cointegration bash scripts/visualize_paper_figs.sh
#   MODE=attention CHECKPOINT=./checkpoints/<setting>/checkpoint.pth bash scripts/visualize_paper_figs.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-all}"
DATA="${DATA:-Solar}"
ROOT_PATH="${ROOT_PATH:-./dataset/Solar/}"
DATA_PATH="${DATA_PATH:-solar_AL.txt}"
OUT_DIR="${OUT_DIR:-./_logs/vis_paper}"
CHECKPOINT="${CHECKPOINT:-}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
DEVICE="${DEVICE:-auto}"

# Solar defaults aligned with TimeBridge.sh
ENC_IN="${ENC_IN:-137}"
PERIOD="${PERIOD:-48}"
NUM_P="${NUM_P:-12}"
IA_LAYERS="${IA_LAYERS:-1}"
PD_LAYERS="${PD_LAYERS:-1}"
CA_LAYERS="${CA_LAYERS:-1}"
SEQ_LEN="${SEQ_LEN:-720}"
PRED_LEN="${PRED_LEN:-336}"
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-128}"

cmd=(
  python -u experiments/visualize_paper_figs.py
  --mode "$MODE"
  --data "$DATA"
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --out_dir "$OUT_DIR"
  --sample_idx "$SAMPLE_IDX"
  --device "$DEVICE"
  --enc_in "$ENC_IN"
  --period "$PERIOD"
  --num_p "$NUM_P"
  --ia_layers "$IA_LAYERS"
  --pd_layers "$PD_LAYERS"
  --ca_layers "$CA_LAYERS"
  --seq_len "$SEQ_LEN"
  --pred_len "$PRED_LEN"
  --d_model "$D_MODEL"
  --d_ff "$D_FF"
)

if [[ -n "$CHECKPOINT" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
