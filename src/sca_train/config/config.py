from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .logger import SCATrainingVerbosityLevel
from .lora import SCALoraConfig
from .train import SCATrainingArguments


class SCATrainingConfig(BaseModel):
    hf_home: Optional[Path] = Path("./hf_models").absolute()
    dataset_cache_dir: Optional[Path] = Path("./hf_datasets").absolute()
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    train_output_dir: Path = Path("./SCA_finetune")
    verbose: SCATrainingVerbosityLevel = SCATrainingVerbosityLevel.INFO
    gradient_checkpointing: bool = True
    system_prompt: Optional[str] = None
    instruction_prompt: Optional[str] = None
    max_length: int = 16384
    mask_instruction: bool = True
    attn_impl: str = "flash_attention_2"
    train_mtp: bool = True
    lora_config: SCALoraConfig = SCALoraConfig()
    training_args: SCATrainingArguments = SCATrainingArguments()
