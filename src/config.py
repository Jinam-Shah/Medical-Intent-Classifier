"""Config loading and reproducibility helpers."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    """Resolve a config path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed):
    """Seed every RNG the training run touches, and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
