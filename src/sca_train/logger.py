from .config import (
    SCATrainingConfig,
    SCATrainingVerbosityLevel,
    SCADuplexTrainingConfig,
)
from .utils import get_local_rank


def debug(
    config: SCATrainingConfig | SCADuplexTrainingConfig,
    message: str,
    rank0_only: bool = True,
) -> None:
    if (
        not rank0_only or get_local_rank() == 0
    ) and config.verbose >= SCATrainingVerbosityLevel.DEBUG:
        print(f"[DEBUG] {message}")


def info(
    config: SCATrainingConfig | SCADuplexTrainingConfig,
    message: str,
    rank0_only: bool = True,
) -> None:
    if (
        not rank0_only or get_local_rank() == 0
    ) and config.verbose >= SCATrainingVerbosityLevel.INFO:
        print(f"[INFO] {message}")


def warning(
    config: SCATrainingConfig | SCADuplexTrainingConfig,
    message: str,
    rank0_only: bool = True,
) -> None:
    if (
        not rank0_only or get_local_rank() == 0
    ) and config.verbose >= SCATrainingVerbosityLevel.WARNING:
        print(f"[WARNING] {message}")


def error(
    config: SCATrainingConfig | SCADuplexTrainingConfig,
    message: str,
    rank0_only: bool = True,
) -> None:
    if (
        not rank0_only or get_local_rank() == 0
    ) and config.verbose >= SCATrainingVerbosityLevel.ERROR:
        print(f"[ERROR] {message}")
