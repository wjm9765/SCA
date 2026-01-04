#!/usr/bin/env -S uv run
"""Minimal example: Load Qwen3-Omni with trained LoRA adapter."""

import torch
from peft import PeftModel
from sca_train.modeling import Qwen3OmniMoeWithProperForward, Qwen3OmniMoeWithProperForwardConfig


def load_model(
    base_model_id: str,
    adapter_path: str,
    device_map: str = "auto",
    max_memory: dict | None = None,
):
    """Load base model with LoRA adapter and CPU offload."""
    
    # Load config
    config = Qwen3OmniMoeWithProperForwardConfig.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    
    # Load base model
    model = Qwen3OmniMoeWithProperForward.from_pretrained(
        base_model_id,
        config=config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model


if __name__ == "__main__":
    model = load_model(
        base_model_id="huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
        adapter_path="./SCA_finetune/final_consolidated",
        max_memory={0: "20GB", "cpu": "50GB"},
    )
    print("Model loaded successfully!")
