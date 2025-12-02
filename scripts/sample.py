#!/usr/bin/env python3
"""Sampling script for poetry generation."""

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
    create_poetry_generator,
    create_poetry_sampler,
    get_device,
    set_deterministic,
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate poetry samples")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The moonlit sky is full of dreams, where",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save generated samples",
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
    if args.model_path:
        config.set("paths.checkpoint_dir", args.model_path)
    
    logger.info("Starting poetry generation")
    logger.info(f"Prompt: {args.prompt}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    model_name = config.get("model.name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info(f"Loading model: {model_name}")
    generator = create_poetry_generator(
        model_type="gpt2",
        model_name=model_name,
        device=device,
    )
    
    # Load trained model if path provided
    if args.model_path and Path(args.model_path).exists():
        logger.info(f"Loading trained model from {args.model_path}")
        generator.model = generator.model.from_pretrained(args.model_path)
        generator.model.to(device)
    
    # Create sampler
    sampler = create_poetry_sampler(generator, tokenizer, device)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples")
    samples = sampler.sample(
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    
    # Print samples
    print("\n" + "="*60)
    print("GENERATED POETRY SAMPLES")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("="*60)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print("-" * 40)
        print(sample)
        print("-" * 40)
    
    # Save samples if output path provided
    if args.output_path:
        sampler.save_samples(samples, args.output_path)
        logger.info(f"Saved samples to {args.output_path}")
    
    logger.info("Poetry generation completed")


if __name__ == "__main__":
    main()
