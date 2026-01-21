#!/bin/bash
set -e

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

EXP_NAME=sotabv2
EXP_DIR=outputs/$EXP_NAME
CONFIG_FILE=configs/config_sotabv2.yaml

python scripts/quick_start.py \
  --config $CONFIG_FILE \
  --experiment_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --logging_dir $EXP_DIR \
  --trainer_mode acd \
  --use_aum \
  --correction_type glc \
  --anchor_budget 50 \
  --trusted_weight 5.0

aum_cutoff=0.3

python scripts/hitl_pipeline.py --config $CONFIG_FILE --step sample --base_dir $EXP_DIR --aum "$aum_cutoff"
python scripts/hitl_pipeline.py --config $CONFIG_FILE --step correct --base_dir $EXP_DIR --aum "$aum_cutoff"

python scripts/quick_start.py \
  --config $CONFIG_FILE \
  --experiment_name $EXP_NAME \
  --output_dir $EXP_DIR \
  --logging_dir $EXP_DIR \
  --trainer_mode acd \
  --correction_type glc \
  --anchor_budget 50 \
  --trusted_weight 5.0

