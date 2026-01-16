#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ACCELERATE_CONFIG="${SCRIPT_DIR}/../configs/accelerate/memory_efficient_2gpus.yaml"
SCA_CONFIG="${SCRIPT_DIR}/../configs/sca/duplex.yaml"
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_PROJECT=${WANDB_PROJECT:-SCA_Duplex}

cd "${SCRIPT_DIR}/.." || exit 1
uv run accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m sca_train.train_duplex \
  --config_file "${SCA_CONFIG}"
