#!/usr/bin/env bash
# ==============================================================================
# TimeBridge × ED-SRA（实验线 2b：仅接入 ED-SRA，亦可与 EA-RevIN 组合）
# ------------------------------------------------------------------------------
# 用途：
#   1) ablation-A：基线，原 TimeBridge（默认）；
#   2) ablation-B：仅启用 ED-SRA（--use_ed_sra）；
#   3) ablation-C：EA-RevIN + ED-SRA 组合（"非平稳感知的双重自适应"完整故事）；
# 默认运行 C；如需对照请按需注释。
# ==============================================================================

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/EDSRA" ]; then
    mkdir ./logs/EDSRA
fi

model_name=TimeBridge
seq_len=720
GPU=0
root=./dataset

alpha=0.35
data_name=ETTh1
pred_len=96

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --is_training 1 \
  --root_path $root/ETT-small/ \
  --data_path $data_name.csv \
  --model_id ${data_name}_${seq_len}_${pred_len}_EARevIN_EDSRA \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 7 \
  --ca_layers 1 \
  --pd_layers 1 \
  --ia_layers 3 \
  --des 'EARevIN_EDSRA' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 64 \
  --alpha $alpha \
  --learning_rate 0.0002 \
  --train_epochs 1 \
  --patience 10 \
  --use_ea_revin \
  --revin_affine \
  --revin_dyn_bound 0.05 \
  --revin_entropy_gate 0.55 \
  --use_ed_sra \
  --ed_sra_init_gamma_min 0.2 \
  --ed_sra_init_gamma_max 0.95 \
  --ed_sra_N_threshold 300 \
  --itr 1 > logs/EDSRA/${data_name}_${alpha}_${model_name}_${pred_len}_EARevIN_EDSRA_smoke.log 2>&1
