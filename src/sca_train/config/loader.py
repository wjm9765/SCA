import json
from pathlib import Path
from typing import Optional

import yaml

from .config import SCATrainingConfig


def load_config(path: Optional[Path] = None) -> SCATrainingConfig:
    if path is None:
        return SCATrainingConfig()

    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    return SCATrainingConfig.model_validate(data)
