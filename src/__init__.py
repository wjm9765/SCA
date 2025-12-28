# src/__init__.py

__version__ = "1.0.0"
from .utils import setup_environment, load_train_args
from .template import register_custom_template
from .dataset import register_custom_dataset