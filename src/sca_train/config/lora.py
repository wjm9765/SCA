from typing import Literal

from pydantic import BaseModel


class SCALoraConfig(BaseModel):
    use_qlora: bool = True
    r: int = 64
    lora_alpha: int = 128
    target_modules_regex = r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$"
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"