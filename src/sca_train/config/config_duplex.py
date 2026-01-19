"""Configuration for Full Duplex training."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .logger import SCATrainingVerbosityLevel
from .lora import SCALoraConfig
from .train import SCATrainingArguments


class SCADuplexTrainingConfig(BaseModel):
    """Configuration for full duplex training.

    This extends the base training config with duplex-specific settings.
    """

    # Paths
    hf_home: Optional[Path] = Path("./hf_models").absolute()
    dataset_cache_dir: Optional[Path] = Path("./hf_datasets").absolute()
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    train_output_dir: Path = Path("./SCA_duplex_finetune")
    verbose: SCATrainingVerbosityLevel = SCATrainingVerbosityLevel.INFO

    # Model settings
    gradient_checkpointing: bool = True
    attn_impl: str = "flash_attention_2"
    train_mtp: bool = True
    mtp_weight: float = 2.0
    max_length: int = 32768

    # Duplex-specific token IDs
    silence_token_id: int = 151646  # Silence token for Qwen3-Omni
    audio_token_id: int = 151675  # From config.thinker_config.audio_token_id
    pad_token_id: int = 151643  # Standard Qwen pad token

    # Duplex training settings
    max_segments_per_batch: int = 8

    # LoRA and training args
    lora_config: SCALoraConfig = SCALoraConfig()
    training_args: SCATrainingArguments = SCATrainingArguments()

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
