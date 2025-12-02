#!/usr/bin/env python3
"""
Poetry Generation System - Complete Demonstration

This script demonstrates all features of the modernized poetry generation system:
- Model loading and configuration
- Poetry generation with different parameters
- Style-based generation
- Comprehensive evaluation
- Sample saving and reporting
"""

import logging
import sys
from pathlib import Path

# Add src to path
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


def demonstrate_basic_generation():
    """Demonstrate basic poetry generation."""
    print("\n" + "="*60)
    print("BASIC POETRY GENERATION")
    print("="*60)
    
    # Create generator
    generator = create_poetry_generator("gpt2", "gpt2")
    
    # Generate poetry
    prompt = "The moonlit sky is full of dreams, where"
    poems = generator.generate(
        prompt,
        max_length=100,
        temperature=0.7,
        num_return_sequences=3,
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated {len(poems)} poems:")
    
    for i, poem in enumerate(poems, 1):
        print(f"\nPoem {i}:")
        print("-" * 40)
        print(poem)
        print("-" * 40)
    
    return poems


def demonstrate_parameter_variation():
    """Demonstrate generation with different parameters."""
    print("\n" + "="*60)
    print("PARAMETER VARIATION DEMONSTRATION")
    print("="*60)
    
    generator = create_poetry_generator("gpt2", "gpt2")
    prompt = "In the garden of my mind"
    
    # Different temperature settings
    temperatures = [0.3, 0.7, 1.2]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 30)
        
        poem = generator.generate(
            prompt,
            max_length=80,
            temperature=temp,
            num_return_sequences=1,
        )[0]
        
        print(poem)
        print("-" * 30)


def demonstrate_style_generation():
    """Demonstrate style-based generation."""
    print("\n" + "="*60)
    print("STYLE-BASED GENERATION")
    print("="*60)
    
    generator = create_poetry_generator("advanced", "gpt2-medium")
    prompt = "The ocean waves crash against the shore"
    
    if hasattr(generator, 'generate_with_style'):
        styles = ["romantic", "modern", "classical"]
        
        for style in styles:
            print(f"\n{style.title()} Style:")
            print("-" * 30)
            
            poem = generator.generate_with_style(
                prompt,
                style=style,
                max_length=100,
            )[0]
            
            print(poem)
            print("-" * 30)
    else:
        print("Style-based generation not available with this model")


def demonstrate_evaluation():
    """Demonstrate evaluation metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS DEMONSTRATION")
    print("="*60)
    
    # Generate samples for evaluation
    generator = create_poetry_generator("gpt2", "gpt2")
    evaluator = PoetryEvaluator(generator.tokenizer)
    
    prompts = [
        "The stars above whisper secrets",
        "In the forest deep and dark",
        "Morning light breaks through",
    ]
    
    generated_poems = []
    for prompt in prompts:
        poem = generator.generate(
            prompt,
            max_length=80,
            temperature=0.7,
            num_return_sequences=1,
        )[0]
        generated_poems.append(poem)
    
    # Evaluate
    results = evaluator.comprehensive_evaluation(
        generated_poems,
        include_diversity=True,
        include_length=True,
    )
    
    print("Generated poems:")
    for i, poem in enumerate(generated_poems, 1):
        print(f"\nPoem {i}:")
        print("-" * 30)
        print(poem)
        print("-" * 30)
    
    print(f"\nEvaluation Results:")
    print(f"Average length: {results.get('length_mean', 0):.1f} words")
    print(f"Length range: {results.get('length_min', 0)} - {results.get('length_max', 0)} words")
    print(f"Diversity ratio: {results.get('diversity_2gram_ratio', 0):.3f}")
    
    return generated_poems


def demonstrate_sampling_utilities():
    """Demonstrate advanced sampling utilities."""
    print("\n" + "="*60)
    print("ADVANCED SAMPLING UTILITIES")
    print("="*60)
    
    generator = create_poetry_generator("gpt2", "gpt2")
    sampler = create_poetry_sampler(generator, generator.tokenizer)
    
    prompt = "The wind carries ancient stories"
    
    # Generate multiple samples with variation
    samples = sampler.sample(
        prompt,
        num_samples=3,
        max_length=100,
        temperature=0.8,
        seed=42,
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated {len(samples)} varied samples:")
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print("-" * 40)
        print(sample)
        print("-" * 40)
    
    # Save samples
    output_dir = Path("assets/samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sampler.save_samples(samples, output_dir / "demo_samples.txt")
    print(f"\nSamples saved to: {output_dir / 'demo_samples.txt'}")
    
    return samples


def main():
    """Main demonstration function."""
    print("🎭 POETRY GENERATION SYSTEM - COMPLETE DEMONSTRATION")
    print("="*80)
    
    # Set deterministic behavior
    set_deterministic(42)
    
    # Get device info
    device = get_device()
    device_info = get_device_info()
    print(f"Device: {device}")
    print(f"Device info: {device_info}")
    
    try:
        # Run all demonstrations
        basic_poems = demonstrate_basic_generation()
        demonstrate_parameter_variation()
        demonstrate_style_generation()
        eval_poems = demonstrate_evaluation()
        sample_poems = demonstrate_sampling_utilities()
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nSummary:")
        print(f"- Generated {len(basic_poems)} basic poems")
        print(f"- Demonstrated parameter variation")
        print(f"- Showed style-based generation")
        print(f"- Evaluated {len(eval_poems)} poems")
        print(f"- Generated {len(sample_poems)} samples with utilities")
        
        print("\nNext steps:")
        print("1. Run: streamlit run demo/streamlit_demo.py")
        print("2. Train: python scripts/train.py --data_path data/sample_poems.json")
        print("3. Evaluate: python scripts/evaluate.py")
        print("4. Sample: python scripts/sample.py")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
