#!/usr/bin/env python3
"""Evaluation script for poetry generation models."""

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
    PoetryEvaluator,
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
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate poetry generation model")
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
        "--data_path",
        type=str,
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate for evaluation",
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
        "--output_path",
        type=str,
        help="Path to save evaluation results",
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
    if args.data_path:
        config.set("data.data_path", args.data_path)
    
    logger.info("Starting poetry generation evaluation")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    model_name = config.get("model.name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset for evaluation
    logger.info("Loading evaluation dataset")
    data_module = create_poetry_dataset(
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_length=config.get("data.max_length", 128),
        batch_size=1,  # Single sample generation
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,  # Use all data for testing
    )
    
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
    
    # Create evaluator
    evaluator = PoetryEvaluator(tokenizer)
    
    # Generate samples for evaluation
    logger.info(f"Generating {args.num_samples} samples for evaluation")
    generated_texts = []
    reference_texts = []
    
    # Use test dataset as reference
    test_poems = data_module.test_dataset.poems if data_module.test_dataset else []
    
    # Generate samples
    for i in range(min(args.num_samples, len(test_poems))):
        if i < len(test_poems):
            # Use first part of reference poem as prompt
            reference_poem = test_poems[i]
            prompt = reference_poem.split('\n')[0] if '\n' in reference_poem else reference_poem[:50]
            
            # Generate sample
            sample = generator.generate(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                num_return_sequences=1,
            )[0]
            
            generated_texts.append(sample)
            reference_texts.append(reference_poem)
        else:
            # Generate with random prompt if not enough references
            prompt = "The moonlit sky is full of dreams, where"
            sample = generator.generate(
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                num_return_sequences=1,
            )[0]
            generated_texts.append(sample)
    
    # Evaluate
    logger.info("Running evaluation metrics")
    results = evaluator.comprehensive_evaluation(
        generated_texts=generated_texts,
        reference_texts=reference_texts,
        include_diversity=True,
        include_length=True,
    )
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(
        generated_texts=generated_texts,
        reference_texts=reference_texts,
        model_name=model_name,
    )
    
    # Print results
    print("\n" + "="*60)
    print("POETRY GENERATION EVALUATION RESULTS")
    print("="*60)
    print(report)
    print("="*60)
    
    # Print detailed results
    print("\nDetailed Metrics:")
    print("-" * 40)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save results if output path provided
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nDetailed Metrics:\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value}\n")
        
        logger.info(f"Saved evaluation results to {args.output_path}")
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
