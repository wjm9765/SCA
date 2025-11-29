from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class SCATrainingVerbosityLevel(int, Enum):
    QUIET = 0
    NORMAL = 1
    DETAILED = 2
    DEBUG = 3


class SCALoraConfig(BaseModel):
    use_qlora: bool = True
    attn_impl: str = "flash_attention_2"
    r: int = 64
    lora_alpha: int = 128
    target_modules_regex = r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$"
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class SCATrainingConfig(BaseModel):
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    train_output_dir: Path = Path("./SCA_finetune")
    verbose: SCATrainingVerbosityLevel = SCATrainingVerbosityLevel.NORMAL
    gradient_checkpointing: bool = True
