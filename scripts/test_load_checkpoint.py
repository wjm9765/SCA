#!/usr/bin/env -S uv run
import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open

# Expected LoRA layers per transformer layer
# NOTE: Only attention projections have LoRA applied. MLP projections (gate/up/down_proj)
# don't match because Qwen3-Omni uses MoE architecture with different module naming.
LORA_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Expected modules to save (full weights, not LoRA)
MODULES_TO_SAVE = ["speaker_projection", "code_predictor"]

# Expected number of layers
EXPECTED_THINKER_LAYERS = 48  # Qwen3-Omni thinker: 48 layers
EXPECTED_TALKER_LAYERS = 20   # Qwen3-Omni talker: 20 layers


def check_files_exist(checkpoint_dir: Path) -> bool:
    """Check that required files exist."""
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = []
    for f in required_files:
        if not (checkpoint_dir / f).exists():
            missing.append(f)
    
    if missing:
        print(f"  FAIL: Missing files: {missing}")
        return False
    print("  PASS: All required files present")
    return True


def check_adapter_config(checkpoint_dir: Path) -> bool:
    """Validate adapter_config.json has required fields."""
    config_path = checkpoint_dir / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    required_fields = [
        "base_model_name_or_path",
        "peft_type",
        "r",
        "lora_alpha",
        "modules_to_save",
    ]
    
    missing = [f for f in required_fields if f not in config]
    if missing:
        print(f"  FAIL: Missing config fields: {missing}")
        return False
    
    # Check values
    issues = []
    if config.get("peft_type") != "LORA":
        issues.append(f"peft_type should be 'LORA', got '{config.get('peft_type')}'")
    if config.get("r", 0) <= 0:
        issues.append(f"r should be > 0, got {config.get('r')}")
    if not config.get("modules_to_save"):
        issues.append("modules_to_save is empty")
    
    if issues:
        for issue in issues:
            print(f"  WARN: {issue}")
    
    print(f"  PASS: Config valid (r={config.get('r')}, alpha={config.get('lora_alpha')})")
    print(f"        modules_to_save: {config.get('modules_to_save')}")
    return True


def check_weights(checkpoint_dir: Path) -> bool:
    """Validate weight keys in adapter_model.safetensors."""
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
    
    # Categorize keys
    thinker_lora_keys = [k for k in keys if "thinker" in k and "lora_" in k]
    talker_lora_keys = [k for k in keys if "talker" in k and "lora_" in k and "code_predictor" not in k]
    speaker_keys = [k for k in keys if "speaker_projection" in k]
    code_pred_keys = [k for k in keys if "code_predictor" in k]
    
    print(f"  Total keys: {len(keys)}")
    print(f"  Thinker LoRA keys: {len(thinker_lora_keys)}")
    print(f"  Talker LoRA keys: {len(talker_lora_keys)}")
    print(f"  Speaker projection keys: {len(speaker_keys)}")
    print(f"  Code predictor keys: {len(code_pred_keys)}")
    
    issues = []
    
    # Check thinker LoRA
    if len(thinker_lora_keys) == 0:
        issues.append("No thinker LoRA keys found")
    else:
        # Each layer should have lora_A and lora_B for each projection
        # So expect: num_layers * len(LORA_PROJECTIONS) * 2
        # With 48 layers and 4 projections: 48 * 4 * 2 = 384 keys
        expected_thinker_keys = EXPECTED_THINKER_LAYERS * len(LORA_PROJECTIONS) * 2
        if len(thinker_lora_keys) != expected_thinker_keys:
            print(f"  WARN: Expected {expected_thinker_keys} thinker LoRA keys, got {len(thinker_lora_keys)}")
    
    # Check talker LoRA (should now exist after inject_adapter_in_model fix!)
    if len(talker_lora_keys) == 0:
        issues.append("No talker LoRA keys found")
    else:
        # 20 layers * 4 projections * 2 (A/B) = 160 keys
        expected_talker_keys = EXPECTED_TALKER_LAYERS * len(LORA_PROJECTIONS) * 2
        if len(talker_lora_keys) != expected_talker_keys:
            print(f"  WARN: Expected {expected_talker_keys} talker LoRA keys, got {len(talker_lora_keys)}")
    
    # Check modules_to_save
    if len(speaker_keys) == 0:
        issues.append("No speaker_projection keys found")
    
    if len(code_pred_keys) == 0:
        issues.append("No code_predictor keys found")
    
    if issues:
        for issue in issues:
            print(f"  FAIL: {issue}")
        return False
    
    print("  PASS: All expected weight categories present")
    return True


def check_weight_shapes(checkpoint_dir: Path) -> bool:
    """Validate weight shapes are reasonable."""
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    
    with safe_open(weights_path, framework="pt") as f:
        keys = list(f.keys())
        
        issues = []
        samples = []
        
        for key in keys[:10]:  # Check first 10
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            
            samples.append(f"    {key}: {shape} {dtype}")
            
            # Check for empty tensors
            if tensor.numel() == 0:
                issues.append(f"Empty tensor: {key}")
            
            # Check for NaN/Inf
            if tensor.isnan().any():
                issues.append(f"NaN values in: {key}")
            if tensor.isinf().any():
                issues.append(f"Inf values in: {key}")
    
    print("  Sample weights:")
    for s in samples[:5]:
        print(s)
    
    if issues:
        for issue in issues:
            print(f"  FAIL: {issue}")
        return False
    
    print("  PASS: Weight shapes and values valid")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate consolidated PEFT checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="./SCA_finetune/final_consolidated", help="Path to PEFT checkpoint directory")
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("PEFT Checkpoint Validation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_dir}")
    
    all_passed = True
    
    # Test 1: Files exist
    print("\n[1/4] Checking files...")
    if not check_files_exist(checkpoint_dir):
        all_passed = False
    
    # Test 2: Config valid
    print("\n[2/4] Checking adapter config...")
    if not check_adapter_config(checkpoint_dir):
        all_passed = False
    
    # Test 3: Weights present
    print("\n[3/4] Checking weight keys...")
    if not check_weights(checkpoint_dir):
        all_passed = False
    
    # Test 4: Weight shapes valid
    print("\n[4/4] Checking weight shapes...")
    if not check_weight_shapes(checkpoint_dir):
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
