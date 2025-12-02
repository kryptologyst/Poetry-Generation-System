"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the poetry generation system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self._config: Optional[DictConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if self.config_path and Path(self.config_path).exists():
            self._config = OmegaConf.load(self.config_path)
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        default_config = {
            "model": {
                "name": "gpt2",
                "max_length": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
                "do_sample": True,
                "pad_token_id": None,
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 1,
                "save_steps": 500,
                "eval_steps": 250,
                "logging_steps": 50,
            },
            "data": {
                "dataset_name": "poetry_dataset",
                "max_length": 128,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "cache_dir": "./data/cache",
            },
            "evaluation": {
                "metrics": ["perplexity", "bleu", "rouge", "bert_score"],
                "num_samples": 100,
                "max_eval_length": 100,
            },
            "paths": {
                "data_dir": "./data",
                "output_dir": "./outputs",
                "checkpoint_dir": "./checkpoints",
                "log_dir": "./logs",
            },
            "device": {
                "auto_detect": True,
                "device_type": "auto",  # auto, cuda, mps, cpu
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply.
        """
        self._config = OmegaConf.merge(self._config, updates)
    
    def save(self, path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration.
        """
        OmegaConf.save(self._config, path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying DictConfig object."""
        return self._config
