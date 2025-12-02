#!/usr/bin/env python3
"""Training script for poetry generation models."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poetry_generation import (
    Config,
    create_poetry_dataset,
    create_poetry_generator,
    get_device,
    set_deterministic,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train poetry generation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to poetry dataset file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=["gpt2", "advanced"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set deterministic behavior
    set_deterministic(args.seed)
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config.set("data.data_path", args.data_path)
    if args.output_dir:
        config.set("paths.checkpoint_dir", args.output_dir)
    
    logger.info("Starting poetry generation training")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    model_name = config.get("model.name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    logger.info("Loading dataset")
    data_module = create_poetry_dataset(
        tokenizer=tokenizer,
        data_path=config.get("data.data_path"),
        max_length=config.get("data.max_length", 128),
        batch_size=config.get("training.batch_size", 8),
        train_split=config.get("data.train_split", 0.8),
        val_split=config.get("data.val_split", 0.1),
        test_split=config.get("data.test_split", 0.1),
    )
    
    # Create model
    logger.info(f"Creating {args.model_type} model")
    generator = create_poetry_generator(
        model_type=args.model_type,
        model_name=model_name,
        device=device,
    )
    
    # Train model
    logger.info("Starting training")
    generator.train(
        train_dataset=data_module.train_dataset,
        val_dataset=data_module.val_dataset,
        output_dir=args.output_dir,
        num_epochs=config.get("training.num_epochs", 3),
        learning_rate=config.get("training.learning_rate", 5e-5),
        batch_size=config.get("training.batch_size", 8),
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
