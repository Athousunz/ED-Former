#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -f "experiments/profile_ablation.py" ]] && [[ -f "TimeBridge-main/experiments/profile_ablation.py" ]]; then
  cd "TimeBridge-main"
fi

# Handle invalid OMP_NUM_THREADS values from some cluster/container presets.
if [[ -n "${OMP_NUM_THREADS:-}" ]] && ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]] ; then
  echo "Warning: invalid OMP_NUM_THREADS='${OMP_NUM_THREADS}', reset to 1"
  export OMP_NUM_THREADS=1
fi

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
ENC_IN="${ENC_IN:-137}"
PERIOD="${PERIOD:-48}"
NUM_P="${NUM_P:-12}"
IA_LAYERS="${IA_LAYERS:-1}"
CA_LAYERS="${CA_LAYERS:-1}"
WARMUP="${WARMUP:-15}"
ITERS="${ITERS:-30}"
REPEATS="${REPEATS:-3}"
CSV_OUT="${CSV_OUT:-}"
MD_OUT="${MD_OUT:-}"
LATEX_OUT="${LATEX_OUT:-}"

cmd=(
  python -u experiments/profile_ablation.py
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --warmup "$WARMUP"
  --iters "$ITERS"
  --repeats "$REPEATS"
  --pred_lens $PRED_LENS
  --enc_in "$ENC_IN"
  --period "$PERIOD"
  --num_p "$NUM_P"
  --ia_layers "$IA_LAYERS"
  --ca_layers "$CA_LAYERS"
)

if [[ -n "$CSV_OUT" ]]; then
  cmd+=(--csv_out "$CSV_OUT")
fi
if [[ -n "$MD_OUT" ]]; then
  cmd+=(--md_out "$MD_OUT")
fi
if [[ -n "$LATEX_OUT" ]]; then
  cmd+=(--latex_out "$LATEX_OUT")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
