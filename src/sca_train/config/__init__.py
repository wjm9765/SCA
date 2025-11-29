from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class SCATrainingVerbosityLevel(int, Enum):
    QUIET = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


class SCALoraConfig(BaseModel):
    use_qlora: bool = True
    r: int = 64
    lora_alpha: int = 128
    target_modules_regex = r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$"
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class SCATrainingArguments(BaseModel):



class SCATrainingConfig(BaseModel):
    hf_home: Optional[Path] = Path("./hf_models").absolute()
    dataset_cache_dir: Optional[Path] = Path("./hf_datasets").absolute()
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    train_output_dir: Path = Path("./SCA_finetune")
    verbose: SCATrainingVerbosityLevel = SCATrainingVerbosityLevel.NORMAL
    gradient_checkpointing: bool = True
    system_prompt: Optional[str] = None
    instruction_prompt: Optional[str] = None
    max_length: int = 16384
    mask_instruction: bool = True
    attn_impl: str = "flash_attention_2"
    lora_config: SCALoraConfig = SCALoraConfig()
