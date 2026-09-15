#!/usr/bin/env bash
# ==============================================================================
# TimeBridge × EA-RevIN（实验线 1：仅接入 EA-RevIN）
# ------------------------------------------------------------------------------
# 用途：
#   1) ablation-A：use_ea_revin=False，行为与原 TimeBridge 完全一致（基线复现）；
#   2) ablation-B：use_ea_revin=True + revin_dyn_bound=0   ，等价 vanilla RevIN（公平对比）；
#   3) ablation-C：use_ea_revin=True + revin_dyn_bound=0.05，启用 EA-RevIN 动态 affine。
# 默认运行 C；如需对照请按上方注释切换 --revin_dyn_bound 与 --use_ea_revin。
# ==============================================================================

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/EARevIN" ]; then
    mkdir ./logs/EARevIN
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
  --model_id ${data_name}_${seq_len}_${pred_len}_EARevIN \
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
  --des 'EARevIN' \
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
  --itr 1 > logs/EARevIN/${data_name}_${alpha}_${model_name}_${pred_len}_EARevIN_smoke.log 2>&1
