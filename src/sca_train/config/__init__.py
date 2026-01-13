import importlib
import typing

from .config import SCATrainingConfig, SCATrainingVerbosityLevel
from .config_duplex import SCADuplexTrainingConfig

if typing.TYPE_CHECKING:
    from . import config, config_duplex, loader, logger, lora, train


def __getattr__(name: str):
    if name in {"config", "config_duplex", "loader", "logger", "lora", "train"}:
        return importlib.import_module("." + name, __package__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


__all__ = [
    "config",
    "config_duplex",
    "loader",
    "logger",
    "lora",
    "train",
    "SCATrainingConfig",
    "SCADuplexTrainingConfig",
    "SCATrainingVerbosityLevel",
]
