#!/bin/bash

# [사용법]
# ./train.sh [CONFIG_PATH] [RESUME_OPTION]

CONFIG_PATH=${1:-"configs/train_config.yaml"}
RESUME_VAL=${2:-""}

# GPU 설정 (필요시 수정)
NUM_GPUS=2
MASTER_PORT=29500

export WANDB_PROJECT="SCA_Comedy_Project"
export WANDB_WATCH="false"

# 명령어 구성
CMD="train.py --config $CONFIG_PATH"


if [ -n "$RESUME_VAL" ]; then
    CMD="$CMD --resume $RESUME_VAL"
fi

echo "========================================================"
echo "🚀 RunPod Environment - SCA Training"
echo "Command: $CMD"
echo "GPUs: $NUM_GPUS"
echo "========================================================"

# DDP 실행
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    $CMD