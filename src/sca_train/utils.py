import os


def prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool = True, gradient_checkpointing_kwargs = None):
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    # Automatic fallback for other models
    if hasattr(model, "thinker"):
        # Qwen3 Omni will use this
        embed_tokens = model.thinker.model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
    else:
        raise AttributeError("Could not locate embed_tokens in model structure.")

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    embed_tokens.register_forward_hook(make_inputs_require_grad)


def is_fsdp() -> bool:
    return os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))
