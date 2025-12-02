"""Device utilities for automatic device detection and management."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def set_deterministic(seed: int = 42) -> None:
    """
    Set deterministic behavior for reproducible results.
    
    Args:
        seed: Random seed for reproducibility.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic seed: {seed}")


def get_device_info() -> dict:
    """
    Get comprehensive device information.
    
    Returns:
        dict: Device information including type, memory, etc.
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "device_name": torch.cuda.get_device_name(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "cuda_version": torch.version.cuda,
        })
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon MPS"
    else:
        info["device_name"] = "CPU"
    
    return info
