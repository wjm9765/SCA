#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ACCELERATE_CONFIG="${SCRIPT_DIR}/../configs/accelerate/memory_efficient_2gpus.yaml"
SCA_CONFIG="${SCRIPT_DIR}/../configs/sca/default.yaml"
export NCCL_P2P_DISABLE=1

uv run accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m sca_train.train \
  --config_file "${SCA_CONFIG}"
