from typing import Any, Dict, Optional

from transformers import Trainer

from . import logger
from .config import SCATrainingConfig
from .utils import get_local_rank


class QwenTrainer(Trainer):
    def __init__(self, config: SCATrainingConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.model_accepts_loss_kwargs = False
        self._current_loss_components: Dict[str, float] = {}

    def _get_base_model(self, model: Any) -> Any:
        """Unwrap model from PEFT/FSDP/DDP wrappers to get the underlying model."""
        base_model: Any = model
        while hasattr(base_model, "module"):
            base_model = base_model.module
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model
        return base_model

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        local_rank = get_local_rank()
        logger.debug(
            self.config,
            f"[Rank {local_rank}] compute_loss called with inputs: {list(inputs.keys())}",
            rank0_only=False,
        )

        inputs_for_model = inputs.copy()
        if "num_items_in_batch" in inputs_for_model:
            del inputs_for_model["num_items_in_batch"]

        outputs = model(**inputs_for_model)

        if not hasattr(outputs, "loss") or outputs.loss is None:
            raise ValueError(
                "Model output does not contain loss. Ensure 'labels' are provided in inputs."
            )

        loss = outputs.loss

        # Log individual loss components if available
        base_model = self._get_base_model(model)

        thinker_loss = getattr(base_model, "_last_thinker_loss", None)
        talker_loss = getattr(base_model, "_last_talker_loss", None)
        mtp_loss = getattr(base_model, "_last_mtp_loss", None)

        if (
            thinker_loss is not None
            and talker_loss is not None
            and mtp_loss is not None
        ):
            # Store for logging callback
            self._current_loss_components = {
                "loss/thinker": thinker_loss.item(),
                "loss/talker": talker_loss.item(),
                "loss/mtp": mtp_loss.item(),
            }
            logger.debug(
                self.config,
                f"[Rank {local_rank}] Loss breakdown - "
                f"Total: {loss.item():.4f}, "
                f"Thinker: {thinker_loss.item():.4f}, "
                f"Talker: {talker_loss.item():.4f}, "
                f"MTP: {mtp_loss.item():.4f}",
                rank0_only=False,
            )
        else:
            self._current_loss_components = {}
            logger.debug(
                self.config,
                f"[Rank {local_rank}] Loss: {loss.item():.4f}",
                rank0_only=False,
            )

        return (loss, outputs) if return_outputs else loss

    def log(
        self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs
    ) -> None:  # type: ignore[override]
        """Override log to include individual loss components."""
        # Add individual loss components to the logs
        if self._current_loss_components:
            logs.update(self._current_loss_components)

        super().log(logs, start_time, **kwargs)
