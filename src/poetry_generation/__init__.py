"""Poetry Generation System - A modern poetry generation toolkit."""

__version__ = "0.1.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .data.dataset import PoetryDataset, PoetryDataModule, create_poetry_dataset
from .evaluation.metrics import PoetryEvaluator
from .models.generator import (
    PoetryGenerator,
    GPT2PoetryGenerator,
    AdvancedPoetryGenerator,
    create_poetry_generator,
)
from .utils.config import Config
from .utils.device import get_device, set_deterministic, get_device_info
from .utils.sampling import PoetrySampler, create_poetry_sampler, generate_poetry_samples

__all__ = [
    # Data
    "PoetryDataset",
    "PoetryDataModule", 
    "create_poetry_dataset",
    # Evaluation
    "PoetryEvaluator",
    # Models
    "PoetryGenerator",
    "GPT2PoetryGenerator",
    "AdvancedPoetryGenerator",
    "create_poetry_generator",
    # Utils
    "Config",
    "get_device",
    "set_deterministic",
    "get_device_info",
    "PoetrySampler",
    "create_poetry_sampler",
    "generate_poetry_samples",
]
