#!/usr/bin/env -S uv run
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
from sca_train.config.loader import load_config
from sca_train.config.lora import SCALoraConfig
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
    state_dict: Dict[str, torch.Tensor],
    modules_to_save: list[str],
) -> Dict[str, torch.Tensor]:
    """Extract LoRA adapters and modules_to_save from state dict.
    
    PEFT checkpoint format (per official docs, adapter names are STRIPPED on save):
    - LoRA weights: base_model.model.<path>.lora_A.weight
    - modules_to_save: base_model.model.<module_path>.<submodule_path>
    
    Note: When PEFT calls save_pretrained(), it strips adapter-specific parts like
    '.default' from LoRA keys and '.modules_to_save.default' from modules_to_save.
    Our FSDP checkpoint already has the correct format - we just need to ensure
    the base_model.model. prefix exists.
    """
    trained_weights = {}

    for key, tensor in state_dict.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            continue

        # Strip FSDP wrapper prefix only
        clean_key = key
        if clean_key.startswith("_fsdp_wrapped_module."):
            clean_key = clean_key[len("_fsdp_wrapped_module."):]

        # Check if this is a trained weight
        is_lora = "lora_" in clean_key
        is_module_to_save = any(m in clean_key for m in modules_to_save)

        if is_lora or is_module_to_save:
            # Ensure 'base_model.model.' prefix exists
            if not clean_key.startswith("base_model.model."):
                clean_key = "base_model.model." + clean_key
            
            # No additional transformation needed - PEFT checkpoint format
            # does NOT include adapter names (.default) in saved files
            trained_weights[clean_key] = tensor.to(torch.bfloat16).cpu()

    return trained_weights


def create_adapter_config(lora_config: SCALoraConfig, base_model_id: str) -> dict:
    """Create PEFT adapter configuration from SCALoraConfig."""
    return {
        "base_model_name_or_path": base_model_id,
        "peft_type": "LORA",
        "task_type": lora_config.task_type,
        "inference_mode": False,
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "bias": lora_config.lora_bias,
        "use_dora": lora_config.use_dora,
        "modules_to_save": lora_config.modules_to_save,
        "target_modules": lora_config.target_modules_regex,
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
    parser.add_argument("--checkpoint-dir", type=str, default="./SCA_finetune")
    parser.add_argument("--output-dir", type=str, default="./SCA_finetune/final_consolidated")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=(Path(__file__).parent / ".." / "configs" / "sca" / "default.yaml").resolve().as_posix(),
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=(Path(__file__).parent / ".." / ".hf_cache").resolve().as_posix(),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    config_file = Path(args.config_file)

    print("=" * 60)
    print("FSDP Checkpoint Consolidation")
    print("=" * 60)

    # Load training config
    print("\n[1/5] Loading training config...")
    if not config_file.exists():
        print(f"  ERROR: Config file not found: {config_file}")
        sys.exit(1)
    training_config = load_config(config_file)
    lora_config = training_config.lora_config
    print(f"  Config loaded from: {config_file}")
    print(f"  LoRA r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"  Target modules regex: {lora_config.target_modules_regex}")
    print(f"  Modules to save: {lora_config.modules_to_save}")

    # Find checkpoint
    print("\n[2/5] Finding checkpoint...")
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    print(f"  Using: {checkpoint_path.name}")

    # Load FSDP checkpoint
    print("\n[3/5] Loading FSDP checkpoint...")
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
    print("\n[4/5] Extracting trained weights...")
    trained_weights = extract_trained_weights(state_dict, lora_config.modules_to_save)

    lora_count = sum(1 for k in trained_weights if "lora_" in k)
    module_count = len(trained_weights) - lora_count
    print(f"  LoRA weights: {lora_count}")
    print(f"  Module weights (with 'base_model.model.' prefix): {module_count}")

    # Show sample keys for verification
    lora_sample = [k for k in trained_weights if "lora_" in k][:2]
    module_sample = [k for k in trained_weights if "lora_" not in k][:2]
    if lora_sample:
        print(f"  Sample LoRA key: {lora_sample[0]}")
    if module_sample:
        print(f"  Sample module key: {module_sample[0]}")

    if not trained_weights:
        print("ERROR: No trained weights found!")
        sys.exit(1)

    # Save checkpoint
    print("\n[5/5] Saving checkpoint...")
    config = create_adapter_config(lora_config, args.base_model_id)
    save_checkpoint(output_dir, trained_weights, config)

    print("\n" + "=" * 60)
    print(f"Done! Output: {output_dir}")
    print(f"  Weights: {len(trained_weights)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
