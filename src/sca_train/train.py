import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)

from sca_data.dataset_utils import easy_load

from .data_collator import Qwen3OmniCollator
import logger
from .config import SCATrainingConfig
from .utils import is_fsdp, prepare_model_for_kbit_training, get_local_rank


def train(config: SCATrainingConfig):
    local_rank = get_local_rank()

    logger.debug(config, f"Detected FSDP mode as: {is_fsdp()} at local rank {local_rank}", rank0_only=False)
    if is_fsdp():
        device_map = None
    else:
        device_map = {"": local_rank}
    logger.debug(config, f"Using device map: {device_map} at local rank {local_rank}", rank0_only=False)

    logger.debug(config, f"Start loading dataset at local rank {local_rank}", rank0_only=False)
    train_dataset = easy_load(
        format="chat",
        cache_dir=config.dataset_cache_dir,
        system_prompt=config.system_prompt,
        instruction_prompt=config.instruction_prompt,
    )
    logger.debug(config, f"Finished loading dataset at local rank {local_rank}", rank0_only=False)

    logger.debug(config, f"Start loading processor at local rank {local_rank}", rank0_only=False)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.hf_home if config.hf_home else None,
    )
    logger.debug(config, f"Finished loading processor at local rank {local_rank}", rank0_only=False)

    collator = Qwen3OmniCollator(
        processor=processor,
        max_length=config.max_length,
        mask_instruction=config.mask_instruction,
    )

    logger.debug(config, f"Start loading model at local rank {local_rank}", rank0_only=False)
    lora_config = config.lora_config
    bnb_config = None
    if lora_config.use_qlora:
        logger.info(config, f"Using QLoRA 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        logger.debug(config, f"BitsAndBytesConfig: {bnb_config}")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=config.attn_impl,
        torch_dtype=torch.bfloat16,
        cache_dir=config.hf_home if config.hf_home else None,
    )
    logger.debug(config, f"Finished loading model at local rank {local_rank}", rank0_only=False)

    logger.debug(config, f"Freezing layers")
    if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
        model.thinker.audio_tower.requires_grad_(False)
        logger.debug(config, f"frozen thinker.audio_tower")
    elif hasattr(model, "audio_tower"):
        model.audio_tower.requires_grad_(False)
        logger.debug(config, f"frozen audio_tower")
    if hasattr(model, "talker"):
        model.talker.requires_grad_(False)
        logger.debug(config, f"frozen talker")
    if hasattr(model, "code2wav"):
        model.code2wav.requires_grad_(False)
        logger.debug(config, f"frozen code2wav")

    logger.debug(config, f"Preparing model for k-bit training")
    prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=None,
    )

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules_regex,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.lora_bias,
        task_type=lora_config.task_type,
    )
    logger.debug(config, f"PEFT LoraConfig: {peft_config}")

    logger.debug(config, f"Wrapping model with PEFT LoRA")
    model = get_peft_model(model, peft_config)
    if get_local_rank() == 0 and config.verbose >= config.verbose.INFO:
        model.print_trainable_parameters()

    training_args = config.training_args
    args = TrainingArguments(
        output_dir=config.train_output_dir.as_posix(),
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=training_args.warmup_ratio,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        learning_rate=training_args.learning_rate,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        optim=training_args.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=training_args.remove_unused_columns,
        ddp_find_unused_parameters=training_args.ddp_find_unused_parameters,
        report_to=training_args.report_to,
    )
    logger.debug(config, f"TrainingArguments: {args}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    logger.info(config, f"Starting training")
    trainer.train()

    logger.info(config, f"Saving trained model to {config.train_output_dir.as_posix()}")
    trainer.save_model(config.train_output_dir.as_posix())
    logger.info(config, f"Training completed")
