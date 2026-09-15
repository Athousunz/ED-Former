#!/usr/bin/env bash
# ==============================================================================
# TimeBridge 全数据集脚本（旧接口，推荐改用 scripts/run_all.sh）
# AB_MODE=D0|D1|D2|D3 与 run_all.sh 的 D0-D3 消融档位一致。
# CMoS 风格 CLI 见: bash scripts/run_all.sh [--ea-revin] [--ed-sra] ...
#
# Traffic 多卡：自动检测 nvidia-smi 可见 GPU 数；仅 1 卡时不启用 DataParallel。
#   TRAFFIC_MULTI_GPU=0  强制单卡
#   TRAFFIC_NGPU=2       最多用 2 卡（0,1）
#   TRAFFIC_BATCH=2      单卡 OOM 时减小 batch
# ==============================================================================
set -euo pipefail

mkdir -p ./logs/LongForecasting/TimeBridge

# ------------------------------------------------------------------------------
# Ablation mode switch (integrated from TimeBridge_EDSRA.sh idea)
#   D0: baseline      (no EA-RevIN, no ED-SRA)
#   D1: EA only       (--use_ea_revin ...)
#   D2: ED only       (--use_ed_sra ...)
#   D3: EA + ED       (both on)
#
# Usage:
#   AB_MODE=D0 bash ./scripts/TimeBridge.sh
#   AB_MODE=D1 bash ./scripts/TimeBridge.sh
#   AB_MODE=D2 bash ./scripts/TimeBridge.sh
#   AB_MODE=D3 bash ./scripts/TimeBridge.sh
# ------------------------------------------------------------------------------
ab_mode="${AB_MODE:-D0}"
ab_flags=()
case "${ab_mode}" in
  D0)
    ab_flags=()
    ;;
  D1)
    ab_flags=(--use_ea_revin --revin_affine --revin_dyn_bound 0.05 --revin_entropy_gate 0.55)
    ;;
  D2)
    ab_flags=(--use_ed_sra --ed_sra_init_gamma_min 0.2 --ed_sra_init_gamma_max 0.95 --ed_sra_N_threshold 300)
    ;;
  D3)
    ab_flags=(
      --use_ea_revin --revin_affine --revin_dyn_bound 0.05 --revin_entropy_gate 0.55
      --use_ed_sra --ed_sra_init_gamma_min 0.2 --ed_sra_init_gamma_max 0.95 --ed_sra_N_threshold 300
    )
    ;;
  *)
    echo "Invalid AB_MODE=${ab_mode}. Expected one of: D0, D1, D2, D3" >&2
    exit 1
    ;;
esac
echo "[$(date '+%F %T')] Ablation mode: ${ab_mode}"

model_name=TimeBridge
seq_len=720
GPU="${GPU:-0}"
root=./dataset

# Detect GPU count for traffic multi-GPU (override: TRAFFIC_NGPU=4 TRAFFIC_MULTI_GPU=0)
gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 1
  fi
}

traffic_gpu_setup() {
  local n="${TRAFFIC_NGPU:-$(gpu_count)}"
  if [[ "$n" -ge 4 ]]; then
    n=4
  elif [[ "$n" -lt 1 ]]; then
    n=1
  fi
  if [[ "$n" -le 1 ]] || [[ "${TRAFFIC_MULTI_GPU:-auto}" == "0" ]]; then
    TRAFFIC_CUDA="${TRAFFIC_GPU:-0}"
    TRAFFIC_MP_ARGS=()
    echo "[traffic] single GPU: CUDA_VISIBLE_DEVICES=${TRAFFIC_CUDA}"
  else
    local ids=()
    local i=0
    while [[ $i -lt $n ]]; do
      ids+=("$i")
      i=$((i + 1))
    done
    TRAFFIC_CUDA=$(IFS=,; echo "${ids[*]}")
    TRAFFIC_MP_ARGS=(--use_multi_gpu --devices "${TRAFFIC_CUDA}")
    echo "[traffic] multi GPU (${n}): CUDA_VISIBLE_DEVICES=${TRAFFIC_CUDA}"
  fi
}

alpha=0.2
data_name=electricity
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/electricity/ \
    --data_path electricity.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --des "Exp_${ab_mode}" \
    --n_heads 32 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 2 \
    --pd_layers 1 \
    --ia_layers 1 \
    --attn_dropout 0.1 \
    --num_p 4 \
    --stable_len 4 \
    --alpha $alpha \
    --batch_size 16 \
    --learning_rate 0.0005 \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.35
data_name=traffic
traffic_gpu_setup
for pred_len in 336 720 192 96; do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${TRAFFIC_CUDA}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$TRAFFIC_CUDA \
  python -u run.py \
    --is_training 1 \
    --root_path $root/traffic/ \
    --data_path traffic.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des "Exp_${ab_mode}" \
    --num_p 8 \
    --n_heads 64 \
    --stable_len 2 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 3 \
    --pd_layers 1 \
    --ia_layers 1 \
    --batch_size "${TRAFFIC_BATCH:-4}" \
    --attn_dropout 0.15 \
    --patience 5 \
    --train_epochs 100 \
    "${TRAFFIC_MP_ARGS[@]}" \
    --alpha $alpha \
    --learning_rate 0.0005 \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

# ---------------------------------------------------------------------------
# PeMS short-term forecasting (paper: I=96, O=12; L1 / MAE metric)
# PEMS03(358) PEMS04(307) PEMS07(883) PEMS08(170)
# ---------------------------------------------------------------------------
GPU=0
pems_seq_len=96
pems_pred_len=12

alpha=0.35
data_name=PEMS03
log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pems_pred_len}_${ab_mode}.logs"
echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pems_pred_len} gpu=${GPU}"
echo "log_file=${log_file}"
CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/PEMS/ \
  --data_path PEMS03.npz \
  --model_id ${data_name}_${pems_seq_len}_${pems_pred_len}_${ab_mode} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $pems_seq_len \
  --label_len 48 \
  --pred_len $pems_pred_len \
  --enc_in 358 \
  --des "Exp_${ab_mode}" \
  --period 12 \
  --num_p 4 \
  --stable_len 2 \
  --ia_layers 1 \
  --pd_layers 1 \
  --ca_layers 2 \
  --n_heads 32 \
  --d_model 512 \
  --d_ff 512 \
  --attn_dropout 0.1 \
  --batch_size 16 \
  --alpha $alpha \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  "${ab_flags[@]}" \
  --itr 1 2>&1 | tee "${log_file}"
echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pems_pred_len}"

alpha=0.35
data_name=PEMS04
log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pems_pred_len}_${ab_mode}.logs"
echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pems_pred_len} gpu=${GPU}"
echo "log_file=${log_file}"
CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/PEMS/ \
  --data_path PEMS04.npz \
  --model_id ${data_name}_${pems_seq_len}_${pems_pred_len}_${ab_mode} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $pems_seq_len \
  --label_len 48 \
  --pred_len $pems_pred_len \
  --enc_in 307 \
  --des "Exp_${ab_mode}" \
  --period 12 \
  --num_p 4 \
  --stable_len 2 \
  --ia_layers 1 \
  --pd_layers 1 \
  --ca_layers 2 \
  --n_heads 32 \
  --d_model 512 \
  --d_ff 512 \
  --attn_dropout 0.1 \
  --batch_size 16 \
  --alpha $alpha \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 10 \
  "${ab_flags[@]}" \
  --itr 1 2>&1 | tee "${log_file}"
echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pems_pred_len}"

alpha=0.35
data_name=PEMS07
log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pems_pred_len}_${ab_mode}.logs"
echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pems_pred_len} gpu=${GPU}"
echo "log_file=${log_file}"
CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/PEMS/ \
  --data_path PEMS07.npz \
  --model_id ${data_name}_${pems_seq_len}_${pems_pred_len}_${ab_mode} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $pems_seq_len \
  --label_len 48 \
  --pred_len $pems_pred_len \
  --enc_in 883 \
  --des "Exp_${ab_mode}" \
  --period 12 \
  --num_p 4 \
  --stable_len 2 \
  --ia_layers 1 \
  --pd_layers 1 \
  --ca_layers 3 \
  --n_heads 64 \
  --d_model 512 \
  --d_ff 512 \
  --attn_dropout 0.15 \
  --batch_size 8 \
  --alpha $alpha \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 10 \
  "${ab_flags[@]}" \
  --itr 1 2>&1 | tee "${log_file}"
echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pems_pred_len}"

alpha=0.35
data_name=PEMS08
log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pems_pred_len}_${ab_mode}.logs"
echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pems_pred_len} gpu=${GPU}"
echo "log_file=${log_file}"
CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/PEMS/ \
  --data_path PEMS08.npz \
  --model_id ${data_name}_${pems_seq_len}_${pems_pred_len}_${ab_mode} \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $pems_seq_len \
  --label_len 48 \
  --pred_len $pems_pred_len \
  --enc_in 170 \
  --des "Exp_${ab_mode}" \
  --period 12 \
  --num_p 4 \
  --stable_len 2 \
  --ia_layers 1 \
  --pd_layers 1 \
  --ca_layers 2 \
  --n_heads 16 \
  --d_model 512 \
  --d_ff 512 \
  --attn_dropout 0.1 \
  --batch_size 32 \
  --alpha $alpha \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  "${ab_flags[@]}" \
  --itr 1 2>&1 | tee "${log_file}"
echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pems_pred_len}"

# ------------------------------------------------------------------------------
# 以下 ETT / Weather / Solar 块默认关闭；取消注释 if false 改为 if true 即可启用
# ------------------------------------------------------------------------------
if false; then

alpha=0.35
data_name=ETTh1
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des "Exp_${ab_mode}" \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 64 \
    --alpha $alpha \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --patience 10 \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.35
data_name=ETTh2
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --period 48 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des "Exp_${ab_mode}" \
    --n_heads 4 \
    --d_model 128 \
    --d_ff 128 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --patience 15 \
    --alpha $alpha \
    --batch_size 16 \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.35
data_name=ETTm1
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des "Exp_${ab_mode}" \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 128 \
    --period 48 \
    --num_p 6 \
    --lradj 'TST' \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --pct_start 0.2 \
    --patience 15 \
    --batch_size 64 \
    --alpha $alpha \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.35
data_name=ETTm2
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des "Exp_${ab_mode}" \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 128 \
    --lradj 'TST' \
    --period 48 \
    --train_epochs 100 \
    --learning_rate 0.0002 \
    --pct_start 0.2 \
    --patience 10 \
    --batch_size 64 \
    --alpha $alpha \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.1
data_name=weather
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des "Exp_${ab_mode}" \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

alpha=0.05
data_name=Solar
for pred_len in 96 192 336 720
do
  log_file="logs/LongForecasting/TimeBridge/${data_name}_${alpha}_${model_name}_${pred_len}_${ab_mode}.logs"
  echo "[$(date '+%F %T')] START data=${data_name} pred_len=${pred_len} gpu=${GPU}"
  echo "log_file=${log_file}"
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/Solar/ \
    --data_path solar_AL.txt \
    --model_id ${data_name}_${seq_len}_${pred_len}_${ab_mode} \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 137 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des "Exp_${ab_mode}" \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 15 \
    "${ab_flags[@]}" \
    --itr 1 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] DONE  data=${data_name} pred_len=${pred_len}"
done

fi
