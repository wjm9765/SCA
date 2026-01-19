"""Training script for Full Duplex training.

This script trains Qwen3-Omni on interleaved audio-text data for
full duplex conversational AI.

Usage:
    python -m sca_train.train_duplex --config-file configs/sca/duplex.yaml
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from sca_data.dataset_utils import easy_load
from transformers import BitsAndBytesConfig, Qwen3OmniMoeProcessor, TrainingArguments

from sca_train import logger
from sca_train.config import SCADuplexTrainingConfig
from sca_train.data_collator_duplex import FullDuplexCollator
from sca_train.modeling import Qwen3OmniDuplexConfig, Qwen3OmniDuplexModel
from sca_train.trainer import QwenTrainer
from sca_train.utils import get_local_rank, is_fsdp, prepare_model_for_kbit_training


def load_duplex_config(config_path: Path) -> SCADuplexTrainingConfig:
    """Load duplex training configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return SCADuplexTrainingConfig(**config_dict)


def load_duplex_dataset(config: SCADuplexTrainingConfig):
    """Load the duplex dataset using easy_load.

    Returns HuggingFace Dataset with "dataset_row_obj" key containing DatasetRow.
    """
    logger.info(config, "Loading duplex dataset via easy_load...")
    dataset = easy_load(format="duplex")
    logger.info(config, f"Dataset loaded with {len(dataset)} samples")
    return dataset


def train_duplex(config: SCADuplexTrainingConfig):
    """Main training function for duplex training."""
    local_rank = get_local_rank()

    logger.debug(
        config,
        f"Detected FSDP mode as: {is_fsdp()} at local rank {local_rank}",
        rank0_only=False,
    )
    torch.cuda.set_device(local_rank)
    grad_ckpt = config.gradient_checkpointing

    if is_fsdp():
        device_map = None
        # Keep gradient_checkpointing enabled for FSDP activation checkpointing
        logger.debug(
            config,
            "FSDP detected: gradient_checkpointing will be enabled via prepare_model_for_kbit_training",
        )
    else:
        device_map = {"": local_rank}

    logger.debug(
        config,
        f"Using device map: {device_map} at local rank {local_rank}",
        rank0_only=False,
    )

    # Load processor (contains feature extractor for audio)
    logger.debug(
        config, f"Loading processor at local rank {local_rank}", rank0_only=False
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=str(config.hf_home) if config.hf_home else None,
    )
    logger.debug(
        config, f"Processor loaded at local rank {local_rank}", rank0_only=False
    )

    # Load dataset
    logger.debug(
        config, f"Start loading dataset at local rank {local_rank}", rank0_only=False
    )
    train_dataset = load_duplex_dataset(config)
    logger.debug(
        config, f"Finished loading dataset at local rank {local_rank}", rank0_only=False
    )

    # Create collator
    collator = FullDuplexCollator(
        processor=processor,
        audio_token_id=config.audio_token_id,
        silence_token_id=config.silence_token_id,
        pad_token_id=config.pad_token_id,
        max_length=config.max_length,
        max_segments_per_batch=config.max_segments_per_batch,
    )

    # Load model
    logger.debug(
        config, f"Start loading model at local rank {local_rank}", rank0_only=False
    )
    lora_config = config.lora_config
    bnb_config = None

    if lora_config.use_qlora:
        logger.info(config, "Using QLoRA 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=[
                "code_predictor",
                "mimi_model",
                "mimi_feature_extractor",
                "code2wav",
                "speaker_projection",
            ],
        )
        logger.debug(config, f"BitsAndBytesConfig: {bnb_config}")

    model_config = Qwen3OmniDuplexConfig.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=str(config.hf_home) if config.hf_home else None,
    )
    model_config.torch_dtype = torch.bfloat16
    model_config.train_mtp = config.train_mtp
    model_config.mtp_weight = config.mtp_weight

    if hasattr(model_config, "talker_config") and hasattr(
        model_config.talker_config, "text_config"
    ):
        if not hasattr(model_config.talker_config, "vocab_size"):
            model_config.talker_config.vocab_size = (
                model_config.talker_config.text_config.vocab_size
            )

    logger.debug(config, f"Model Config: {model_config}")

    model = Qwen3OmniDuplexModel.from_pretrained(
        config.model_id,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=config.attn_impl,
        cache_dir=str(config.hf_home) if config.hf_home else None,
    )
    logger.debug(
        config, f"Finished loading model at local rank {local_rank}", rank0_only=False
    )

    # CRITICAL: Load Mimi model AFTER from_pretrained() to avoid weight corruption
    logger.info(config, f"Loading Mimi model separately at local rank {local_rank}")
    model.load_mimi_model()
    logger.info(config, f"Mimi model loaded successfully at local rank {local_rank}")

    # Force bfloat16 for FSDP uniformity
    logger.debug(config, "Forcing model to bfloat16 to satisfy FSDP uniformity...")
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.bfloat16)

    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.bfloat16)

    # Freeze everything first
    model.requires_grad_(False)

    # Unfreeze specific modules
    if hasattr(model, "talker") and hasattr(model.talker, "code_predictor"):
        model.talker.code_predictor.requires_grad_(True)
        logger.debug(config, "Unfrozen talker.code_predictor (MTP)")

    if hasattr(model, "speaker_projection"):
        model.speaker_projection.requires_grad_(True)
        logger.debug(config, "Unfrozen speaker_projection for voice cloning")

    # Freeze codec embeddings to prevent NaN gradients
    if hasattr(model, "_freeze_codec_embeddings"):
        model._freeze_codec_embeddings()
        logger.debug(config, "Frozen codec embeddings (Mimi pretrained)")

    # Keep these frozen
    if hasattr(model, "mimi_model") and model.mimi_model is not None:
        model.mimi_model.requires_grad_(False)
    if hasattr(model, "code2wav"):
        model.code2wav.requires_grad_(False)
    if hasattr(model, "thinker"):
        if hasattr(model.thinker, "audio_tower"):
            model.thinker.audio_tower.requires_grad_(False)
        if hasattr(model.thinker, "visual"):
            model.thinker.visual.requires_grad_(False)

    # Prepare for k-bit training
    logger.debug(
        config, f"Preparing model for k-bit training, with grad_ckpt={grad_ckpt}"
    )
    model.config.thinker_config.text_config.use_cache = False
    model.config.talker_config.text_config.use_cache = False
    model.config.talker_config.code_predictor_config.use_cache = False
    prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_config.r,
        use_dora=lora_config.use_dora,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules_regex,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.lora_bias,
        task_type=lora_config.task_type,
        modules_to_save=lora_config.modules_to_save,
    )

    logger.debug(config, "Applying unified LoRA for thinker and talker...")
    logger.debug(config, f"  Target_modules regex: {lora_config.target_modules_regex}")
    logger.debug(config, f"  modules_to_save: {lora_config.modules_to_save}")
    model = get_peft_model(model, peft_config)

    logger.debug(config, f"Model type after PEFT: {type(model).__name__}")

    # Verify LoRA was created
    if get_local_rank() == 0:
        lora_count = sum(
            1 for n, p in model.named_parameters() if p.requires_grad and "lora_" in n
        )
        other_trainable_count = sum(
            1
            for n, p in model.named_parameters()
            if p.requires_grad and "lora_" not in n
        )

        logger.debug(config, f"Verification: LoRA parameters: {lora_count}")
        logger.debug(
            config,
            f"Verification: Other trainable parameters (modules_to_save): {other_trainable_count}",
        )

        if lora_count == 0:
            raise RuntimeError(
                "CRITICAL: LoRA was not created! Check get_peft_model call and regex."
            )

    if get_local_rank() == 0 and config.verbose >= config.verbose.INFO:
        model.print_trainable_parameters()

    # Clean up memory
    logger.debug(config, "Cleaning up memory")
    gc.collect()
    torch.cuda.empty_cache()

    # Training arguments
    training_args_config = config.training_args
    args = TrainingArguments(
        output_dir=str(config.train_output_dir),
        per_device_train_batch_size=training_args_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_args_config.gradient_accumulation_steps,
        warmup_ratio=training_args_config.warmup_ratio,
        num_train_epochs=training_args_config.num_train_epochs,
        max_steps=training_args_config.max_steps,
        learning_rate=training_args_config.learning_rate,
        max_grad_norm=training_args_config.max_grad_norm,
        fp16=training_args_config.fp16,
        bf16=training_args_config.bf16,
        logging_steps=training_args_config.logging_steps,
        save_steps=training_args_config.save_steps,
        optim=training_args_config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=training_args_config.remove_unused_columns,
        ddp_find_unused_parameters=training_args_config.ddp_find_unused_parameters,
        report_to=training_args_config.report_to,
        save_only_model=training_args_config.save_only_model,
        dataloader_pin_memory=training_args_config.dataloader_pin_memory,
        dataloader_num_workers=training_args_config.dataloader_num_workers,
        dataloader_prefetch_factor=training_args_config.dataloader_prefetch_factor,
        dataloader_persistent_workers=True,
    )
    logger.debug(config, f"TrainingArguments: {args}")

    # Create trainer
    trainer = QwenTrainer(
        config=config,
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # Start training
    logger.info(config, "Starting duplex training")
    trainer.train(resume_from_checkpoint=False)

    # Save final model
    if trainer.is_fsdp_enabled:
        logger.info(
            config, "Converting FSDP state dict to FULL_STATE_DICT for final save"
        )
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")  # type: ignore[union-attr]

    logger.info(config, f"Saving trained model to {config.train_output_dir}")
    trainer.save_model(str(config.train_output_dir / "final_model"))
    logger.info(config, "Duplex training completed")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SCA Duplex Training Script")
    parser.add_argument(
        "--config_file",
        "--config-file",
        type=str,
        required=True,
        help="Path to the training configuration file (YAML format)",
    )
    args = parser.parse_args()

    config_file = Path(args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    config = load_duplex_config(config_file)
    train_duplex(config)


if __name__ == "__main__":
    main()
