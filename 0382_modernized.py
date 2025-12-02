#!/usr/bin/env python3
"""
Modern Poetry Generation System - Refactored Version

This script demonstrates the modernized poetry generation system with:
- Clean, typed code with proper error handling
- Device detection and deterministic seeding
- Comprehensive configuration management
- Multiple model support
- Evaluation metrics
- Professional logging
"""

import logging
import sys
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poetry_generation import (
    Config,
    create_poetry_generator,
    create_poetry_sampler,
    PoetryEvaluator,
    get_device,
    set_deterministic,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    logger.info("Starting Poetry Generation System Demo")
    
    # Set deterministic behavior for reproducible results
    set_deterministic(42)
    
    # Load configuration
    config = Config("configs/default.yaml")
    logger.info(f"Loaded configuration: {config.get('model.name')}")
    
    # Get device information
    device = get_device()
    device_info = get_device_info()
    logger.info(f"Using device: {device}")
    logger.info(f"Device info: {device_info}")
    
    # Load tokenizer
    model_name = config.get("model.name", "gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create poetry generator
    logger.info(f"Creating poetry generator with model: {model_name}")
    generator = create_poetry_generator(
        model_type="gpt2",
        model_name=model_name,
        device=device,
    )
    
    # Create sampler
    sampler = create_poetry_sampler(generator, tokenizer, device)
    
    # Create evaluator
    evaluator = PoetryEvaluator(tokenizer)
    
    # Example poetry prompts
    prompts = [
        "The moonlit sky is full of dreams, where",
        "In the garden of my mind, I find",
        "Autumn leaves dance in the wind, as",
        "Beneath the ocean's endless blue,",
        "City lights at night so bright,",
    ]
    
    print("\n" + "="*80)
    print("POETRY GENERATION SYSTEM - DEMONSTRATION")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Configuration: {config.get('model.temperature')} temperature, {config.get('model.top_p')} top-p")
    print("="*80)
    
    all_generated_poems = []
    
    # Generate poetry for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}: {prompt} ---")
        
        # Generate multiple samples
        samples = sampler.sample(
            prompt=prompt,
            num_samples=2,
            max_length=config.get("model.max_length", 150),
            temperature=config.get("model.temperature", 0.7),
            top_p=config.get("model.top_p", 0.9),
            top_k=config.get("model.top_k", 50),
            repetition_penalty=config.get("model.repetition_penalty", 1.1),
            seed=42 + i,  # Different seed for each prompt
        )
        
        for j, sample in enumerate(samples, 1):
            print(f"\nSample {j}:")
            print("-" * 40)
            print(sample)
            print("-" * 40)
            all_generated_poems.append(sample)
    
    # Demonstrate style-based generation (if available)
    if hasattr(generator, 'generate_with_style'):
        print(f"\n--- Style-Based Generation ---")
        styles = ["romantic", "modern", "classical"]
        
        for style in styles:
            print(f"\n{style.title()} Style:")
            print("-" * 30)
            
            style_samples = sampler.sample_with_styles(
                prompt="The stars above whisper secrets",
                styles=[style],
                num_samples_per_style=1,
                max_length=100,
                temperature=0.7,
            )
            
            if style in style_samples:
                print(style_samples[style][0])
            print("-" * 30)
    
    # Evaluate generated poetry
    print(f"\n--- Evaluation Results ---")
    logger.info("Running comprehensive evaluation")
    
    evaluation_results = evaluator.comprehensive_evaluation(
        generated_texts=all_generated_poems,
        reference_texts=None,  # No reference texts for this demo
        include_diversity=True,
        include_length=True,
    )
    
    # Print evaluation summary
    print(f"Generated {len(all_generated_poems)} poems")
    print(f"Average length: {evaluation_results.get('length_mean', 0):.1f} words")
    print(f"Length range: {evaluation_results.get('length_min', 0)} - {evaluation_results.get('length_max', 0)} words")
    print(f"Diversity ratio: {evaluation_results.get('diversity_2gram_ratio', 0):.3f}")
    
    # Create detailed evaluation report
    report = evaluator.create_evaluation_report(
        generated_texts=all_generated_poems,
        reference_texts=None,
        model_name=model_name,
    )
    
    print(f"\nDetailed Evaluation Report:")
    print("-" * 50)
    print(report)
    
    # Save samples and report
    output_dir = Path("assets/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save generated samples
    sampler.save_samples(
        all_generated_poems,
        output_dir / "demo_samples.txt",
        format="txt"
    )
    
    # Save evaluation report
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Saved samples to {output_dir}")
    logger.info("Poetry generation demo completed successfully")
    
    print(f"\n--- Demo Complete ---")
    print(f"Samples saved to: {output_dir}")
    print(f"Evaluation report saved to: {output_dir / 'evaluation_report.txt'}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        sys.exit(1)
