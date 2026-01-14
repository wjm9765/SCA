import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen3OmniMoeProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
)

from sca_data.dataset_utils import easy_load

from .data_collator import Qwen3OmniCollator
from . import logger
from .config import SCATrainingConfig
from .trainer import QwenTrainer
from .utils import is_fsdp, prepare_model_for_kbit_training, get_local_rank
from .config.loader import load_config
from .modeling import Qwen3OmniMoeWithProperForward, Qwen3OmniMoeWithProperForwardConfig


def train(config: SCATrainingConfig):
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
        if config.gradient_checkpointing:
            config.gradient_checkpointing = False
            logger.debug(
                config,
                "FSDP detected: Overriding config.gradient_checkpointing to False (Managed by FSDP Config)",
            )
    else:
        device_map = {"": local_rank}
    logger.debug(
        config,
        f"Using device map: {device_map} at local rank {local_rank}",
        rank0_only=False,
    )

    logger.debug(
        config, f"Start loading dataset at local rank {local_rank}", rank0_only=False
    )
    train_dataset = easy_load(
        format="talker_chat",
        cache_dir=config.dataset_cache_dir,
        system_prompt=config.system_prompt,
        instruction_prompt=config.instruction_prompt,
    )
    logger.debug(
        config, f"Finished loading dataset at local rank {local_rank}", rank0_only=False
    )

    logger.debug(
        config, f"Start loading processor at local rank {local_rank}", rank0_only=False
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.hf_home if config.hf_home else None,
    )
    logger.debug(
        config,
        f"Finished loading processor at local rank {local_rank}",
        rank0_only=False,
    )

    collator = Qwen3OmniCollator(
        processor=processor,
        max_length=config.max_length,
        mask_instruction=config.mask_instruction,
        train_talker=True,
    )

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

    model_config = Qwen3OmniMoeWithProperForwardConfig.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.hf_home if config.hf_home else None,
    )
    model_config.torch_dtype = torch.bfloat16
    model_config.train_mtp = config.train_mtp

    if hasattr(model_config, "talker_config") and hasattr(
        model_config.talker_config, "text_config"
    ):
        if not hasattr(model_config.talker_config, "vocab_size"):
            model_config.talker_config.vocab_size = (
                model_config.talker_config.text_config.vocab_size
            )

    logger.debug(config, f"Model Config: {model_config}")

    model = Qwen3OmniMoeWithProperForward.from_pretrained(
        config.model_id,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=config.attn_impl,
        cache_dir=config.hf_home if config.hf_home else None,
    )
    logger.debug(
        config, f"Finished loading model at local rank {local_rank}", rank0_only=False
    )

    # CRITICAL: Load Mimi model AFTER from_pretrained() to avoid weight corruption
    logger.info(config, f"Loading Mimi model separately at local rank {local_rank}")
    model.load_mimi_model()
    logger.info(config, f"Mimi model loaded successfully at local rank {local_rank}")

    logger.debug(config, "Forcing model to bfloat16 to satisfy FSDP uniformity...")
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.bfloat16)

    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(torch.bfloat16)

    model.requires_grad_(False)

    if hasattr(model, "talker") and hasattr(model.talker, "code_predictor"):
        model.talker.code_predictor.requires_grad_(True)
        logger.debug(config, "Unfrozen talker.code_predictor (MTP)")

    if hasattr(model, "speaker_projection"):
        model.speaker_projection.requires_grad_(True)
        logger.debug(config, "Unfrozen speaker_projection for voice cloning")

    if hasattr(model, "mimi_model"):
        model.mimi_model.requires_grad_(False)
    if hasattr(model, "code2wav"):
        model.code2wav.requires_grad_(False)
    if hasattr(model, "thinker"):
        if hasattr(model.thinker, "audio_tower"):
            model.thinker.audio_tower.requires_grad_(False)
        if hasattr(model.thinker, "visual"):
            model.thinker.visual.requires_grad_(False)

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

    # Debug: Log module names before PEFT wrapping (rank 0 only)
    if get_local_rank() == 0:
        logger.debug(config, "Sample module names BEFORE PEFT wrapping:")
        sample_modules = []
        for name, _ in model.named_modules():
            if (
                ("thinker" in name or "talker" in name)
                and "self_attn" in name
                and "proj" in name
            ):
                sample_modules.append(name)
                if len(sample_modules) >= 4:  # Show 2 thinker + 2 talker examples
                    break
        for name in sample_modules:
            logger.debug(config, f"  {name}")

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

    # Verify LoRA was created (fail-fast if something went wrong)
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

    if get_local_rank() == 0:
        lora_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_" in name:
                    lora_params.append(name)
                else:
                    other_params.append(name)

        entries_to_show = 20
        logger.debug(config, f"LoRA parameters: {len(lora_params)}")
        for name in lora_params[:entries_to_show]:
            logger.debug(config, f"  {name}")
        if len(lora_params) > entries_to_show:
            logger.debug(config, f"  ... and {len(lora_params) - entries_to_show} more")

        logger.debug(
            config, f"Other trainable parameters (modules_to_save): {len(other_params)}"
        )
        for name in other_params[:entries_to_show]:
            logger.debug(config, f"  {name}")
        if len(other_params) > entries_to_show:
            logger.debug(
                config, f"  ... and {len(other_params) - entries_to_show} more"
            )

    if get_local_rank() == 0 and config.verbose >= config.verbose.INFO:
        model.print_trainable_parameters()

    logger.debug(config, "Cleaning up memory")
    gc.collect()
    torch.cuda.empty_cache()

    training_args = config.training_args
    args = TrainingArguments(
        output_dir=config.train_output_dir.as_posix(),
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=training_args.warmup_ratio,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        learning_rate=training_args.learning_rate,
        max_grad_norm=training_args.max_grad_norm,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        optim=training_args.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=training_args.remove_unused_columns,
        ddp_find_unused_parameters=training_args.ddp_find_unused_parameters,
        report_to=training_args.report_to,
        save_only_model=training_args.save_only_model,
        dataloader_pin_memory=training_args.dataloader_pin_memory,
        dataloader_num_workers=training_args.dataloader_num_workers,
        dataloader_prefetch_factor=training_args.dataloader_prefetch_factor,
        dataloader_persistent_workers=True,
    )
    logger.debug(config, f"TrainingArguments: {args}")

    trainer = QwenTrainer(
        config=config,
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    logger.info(config, "Starting training")
    trainer.train(resume_from_checkpoint=False)

    if trainer.is_fsdp_enabled:
        logger.info(
            config, "Converting FSDP state dict to FULL_STATE_DICT for final save"
        )
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")  # type: ignore[union-attr]

    logger.info(config, f"Saving trained model to {config.train_output_dir.as_posix()}")
    trainer.save_model((config.train_output_dir / "final_model").as_posix())
    logger.info(config, "Training completed")


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="SCA Training Script")
    parser.add_argument(
        "--config_file",
        "--config-file",
        type=str,
        required=True,
        help="Path to the training configuration file (JSON or YAML format)",
    )
    args = parser.parse_args()

    config_file = Path(args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    config = load_config(config_file)

    train(config)


if __name__ == "__main__":
    main()
