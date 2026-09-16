#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
DATASET_TAG="${DATASET_TAG:-Solar}"
DATA="${DATA:-Solar}"
ROOT_PATH="${ROOT_PATH:-./dataset/Solar/}"
DATA_PATH="${DATA_PATH:-solar_AL.txt}"
PRED_LEN="${PRED_LEN:-336}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
PERIOD="${PERIOD:-48}"
NUM_P="${NUM_P:-12}"
ENC_IN="${ENC_IN:-137}"
IA_LAYERS="${IA_LAYERS:-1}"
CA_LAYERS="${CA_LAYERS:-1}"
ALPHA="${ALPHA:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"

cmd=(
  python -u experiments/run_edsra_sensitivity.py
  --gpu "$GPU"
  --dataset_tag "$DATASET_TAG"
  --data "$DATA"
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --pred_len "$PRED_LEN"
  --train_epochs "$TRAIN_EPOCHS"
  --patience "$PATIENCE"
  --period "$PERIOD"
  --num_p "$NUM_P"
  --enc_in "$ENC_IN"
  --ia_layers "$IA_LAYERS"
  --ca_layers "$CA_LAYERS"
  --alpha "$ALPHA"
  --learning_rate "$LEARNING_RATE"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
