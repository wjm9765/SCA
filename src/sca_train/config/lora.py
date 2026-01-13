from typing import Literal

from pydantic import BaseModel, Field


class SCALoraConfig(BaseModel):
    use_qlora: bool = True
    r: int = 64
    use_dora: bool = False
    lora_alpha: int = 64
    target_modules_regex: str = (
        r".*(thinker|talker)\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
    )
    lora_dropout: float = 0.05
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] = Field(
        default_factory=lambda: ["talker.code_predictor", "speaker_projection"]
    )
