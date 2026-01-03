#!/usr/bin/env python3
"""
Consolidate FSDP sharded checkpoint into a single PEFT-compatible checkpoint.

Usage:
    python scripts/merge_final_ckpt.py \
        --checkpoint-dir ./SCA_finetune \
        --output-dir ./SCA_finetune/final_consolidated
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def find_latest_checkpoint(base_dir: Path) -> Path:
    """Find checkpoint-XXX with highest step number."""
    checkpoints = [
        c for c in base_dir.glob("checkpoint-*")
        if c.is_dir() and c.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {base_dir}")
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def load_fsdp_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Load FSDP sharded checkpoint using dcp_to_torch_save."""
    fsdp_dir = checkpoint_path / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        raise ValueError(f"FSDP shard directory not found: {fsdp_dir}")

    # Convert DCP to torch.save format (temp file)
    temp_path = checkpoint_path / "_temp_consolidated.pt"
    print("  Converting DCP to torch format...")
    dcp_to_torch_save(str(fsdp_dir), str(temp_path))

    # Load the consolidated state dict
    print("  Loading consolidated state dict...")
    state_dict = torch.load(temp_path, map_location="cpu", weights_only=False)

    # Clean up temp file
    temp_path.unlink()

    return state_dict


def extract_trained_weights(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Extract LoRA adapters and modules_to_save from state dict."""
    trained_weights = {}

    for key, tensor in state_dict.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            continue

        # Clean key (remove FSDP/PEFT wrapper prefixes)
        clean_key = key
        for prefix in ["_fsdp_wrapped_module.", "base_model.model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        # Check if this is a trained weight (LoRA or modules_to_save)
        is_lora = "lora_" in key
        is_module_to_save = any(
            m in key for m in ["speaker_projection", "code_predictor"]
        )

        if is_lora or is_module_to_save:
            trained_weights[clean_key] = tensor.to(torch.bfloat16).cpu()

    return trained_weights


def create_adapter_config(base_model_id: str) -> dict:
    """Create PEFT adapter configuration matching training config."""
    # These values match configs/sca/default.yaml
    return {
        "base_model_name_or_path": base_model_id,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        # Thinker/Talker shared LoRA params (from default.yaml)
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_dora": False,
        # Modules to save (trained in full precision)
        "modules_to_save": ["speaker_projection", "talker.code_predictor"],
        # Target modules (matches the regex patterns in default.yaml)
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    }


def save_checkpoint(
    output_dir: Path, weights: Dict[str, torch.Tensor], config: dict
):
    """Save consolidated checkpoint in PEFT format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    save_file(weights, output_dir / "adapter_model.safetensors")

    # Save config
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save README
    readme = f"""# Consolidated PEFT Checkpoint
Load with: `PeftModel.from_pretrained(base_model, "{output_dir}")`
Base model: {config['base_model_name_or_path']}
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate FSDP checkpoint to PEFT format"
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("FSDP Checkpoint Consolidation")
    print("=" * 60)

    # Find checkpoint
    print("\n[1/4] Finding checkpoint...")
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    print(f"  Using: {checkpoint_path.name}")

    # Load FSDP checkpoint
    print("\n[2/4] Loading FSDP checkpoint...")
    loaded = load_fsdp_checkpoint(checkpoint_path)
    # Handle nested state dict (may have 'model' key from FSDP)
    if "model" in loaded:
        state_dict = loaded["model"]
        assert isinstance(state_dict, dict), "Expected 'model' to be a dict"
    else:
        state_dict = loaded
    print(f"  State dict has the following keys: {state_dict.keys()}")
    print(f"  Loaded {len(state_dict)} keys")

    # Extract trained weights
    print("\n[3/4] Extracting trained weights...")
    trained_weights = extract_trained_weights(state_dict)

    lora_count = sum(1 for k in trained_weights if "lora_" in k)
    module_count = len(trained_weights) - lora_count
    print(f"  LoRA weights: {lora_count}")
    print(f"  Module weights: {module_count}")

    if not trained_weights:
        print("ERROR: No trained weights found!")
        sys.exit(1)

    # Save checkpoint
    print("\n[4/4] Saving checkpoint...")
    config = create_adapter_config(args.base_model_id)
    save_checkpoint(output_dir, trained_weights, config)

    print("\n" + "=" * 60)
    print(f"Done! Output: {output_dir}")
    print(f"  Weights: {len(trained_weights)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
