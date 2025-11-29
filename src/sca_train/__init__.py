import importlib
import typing

if typing.TYPE_CHECKING:
    from . import data_collator, utils, config


def __getattr__(name: str):
    if name in {"data_collator", "utils", "config"}:
        return importlib.import_module("." + name, __package__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return  __all__

__all__ = ["data_collator", "utils", "config"]
