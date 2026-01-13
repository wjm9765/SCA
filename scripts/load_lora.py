#!/usr/bin/env -S uv run
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from sca_train.modeling import (
    Qwen3OmniMoeWithProperForward,
    Qwen3OmniMoeWithProperForwardConfig,
)


def load_model(
    base_model_id: str,
    adapter_path: str,
    device_map: str = "auto",
    max_memory: dict | None = None,
    use_4bit: bool = True,
):
    # Setup quantization config (same as training)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
            # Skip quantization for these modules (kept in bf16)
            llm_int8_skip_modules=[
                "code_predictor",
                "mimi_model",
                "mimi_feature_extractor",
                "code2wav",
                "speaker_projection",
            ],
        )

    # Load config
    config = Qwen3OmniMoeWithProperForwardConfig.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    config.torch_dtype = torch.bfloat16

    # Load base model with quantization
    model = Qwen3OmniMoeWithProperForward.from_pretrained(
        base_model_id,
        config=config,
        quantization_config=bnb_config,
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
    import argparse

    parser = argparse.ArgumentParser(description="Load model with LoRA adapter")
    parser.add_argument(
        "--base-model",
        type=str,
        default="huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Instruct-abliterated",
        help="Base model ID or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./SCA_finetune/final_consolidated",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use bf16 instead)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.base_model}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"4-bit quantization: {not args.no_4bit}")

    model = load_model(
        base_model_id=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )
    print("Model loaded successfully!")

    # Print model info
    print(f"\nModel type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
