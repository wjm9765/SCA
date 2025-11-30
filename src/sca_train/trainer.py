import torch
from transformers import Trainer

from . import logger
from .config import SCATrainingConfig
from .utils import get_local_rank


class QwenTrainer(Trainer):
    def __init__(self, config: SCATrainingConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        local_rank = get_local_rank()
        logger.debug(self.config, f"[Rank {local_rank}] Custom compute_loss called with inputs keys: {list(inputs.keys())}", rank0_only=False)
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs_for_model = inputs.copy()
        if "num_items_in_batch" in inputs_for_model:
            del inputs_for_model["num_items_in_batch"]

        logger.debug(self.config, f"[Rank {local_rank}] Forward pass with inputs keys: {list(inputs_for_model.keys())}", rank0_only=False)
        outputs = model(**inputs_for_model)
        logger.debug(self.config, f"[Rank {local_rank}] Model forward pass completed.", rank0_only=False)

        logger.debug(self.config, f"[Rank {local_rank}] Outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}", rank0_only=False)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss

        elif labels is not None:
            logger.debug(self.config, f"[Rank {local_rank}] Calculating loss manually using CrossEntropyLoss.", rank0_only=False)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            raise ValueError("Model output did not contain loss, and no labels were provided.")

        logger.debug(self.config, f"[Rank {local_rank}] Computed loss: {loss.item()}", rank0_only=False)
        return (loss, outputs) if return_outputs else loss