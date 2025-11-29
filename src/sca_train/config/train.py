from typing import List

from pydantic import BaseModel


class SCATrainingArguments(BaseModel):
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    optim: str = "paged_adamw_8bit"
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    report_to: List[str] = ["none"]