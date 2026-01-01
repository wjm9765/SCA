from typing import Literal

from pydantic import BaseModel, Field


class ComponentLoraConfig(BaseModel):
    r: int = 64
    use_dora: bool = False
    lora_alpha: int = 128
    target_modules_regex: str
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class SCALoraConfig(BaseModel):
    use_qlora: bool = True
    thinker: ComponentLoraConfig = Field(
        default_factory=lambda: ComponentLoraConfig(
            target_modules_regex=r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$"
        )
    )
    talker: ComponentLoraConfig = Field(
        default_factory=lambda: ComponentLoraConfig(
            target_modules_regex=r"^talker\.model\.layers\.\d+\..*(q|k|v|o)_proj$"
        )
    )
