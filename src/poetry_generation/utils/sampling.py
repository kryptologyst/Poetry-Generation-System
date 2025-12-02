"""Sampling utilities for poetry generation."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer

from ..models.generator import PoetryGenerator
from ..utils.device import get_device, set_deterministic

logger = logging.getLogger(__name__)


class PoetrySampler:
    """Sampling utilities for poetry generation."""
    
    def __init__(
        self,
        generator: PoetryGenerator,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize poetry sampler.
        
        Args:
            generator: Poetry generator model.
            tokenizer: Tokenizer for text processing.
            device: Device to run the model on.
        """
        self.generator = generator
        self.tokenizer = tokenizer
        self.device = device or get_device()
    
    def sample(
        self,
        prompt: str,
        num_samples: int = 5,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple poetry samples from a prompt.
        
        Args:
            prompt: Input prompt for generation.
            num_samples: Number of samples to generate.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repetition.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of generated poems.
        """
        if seed is not None:
            set_deterministic(seed)
        
        samples = []
        
        for i in range(num_samples):
            # Add slight variation to parameters for diversity
            current_temp = temperature + random.uniform(-0.1, 0.1)
            current_top_p = max(0.1, min(0.95, top_p + random.uniform(-0.05, 0.05)))
            
            sample = self.generator.generate(
                prompt,
                max_length=max_length,
                temperature=current_temp,
                top_p=current_top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                **kwargs
            )[0]
            
            samples.append(sample)
        
        return samples
    
    def sample_with_styles(
        self,
        prompt: str,
        styles: List[str] = None,
        num_samples_per_style: int = 2,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Generate samples with different styles.
        
        Args:
            prompt: Input prompt for generation.
            styles: List of styles to generate.
            num_samples_per_style: Number of samples per style.
            **kwargs: Additional generation parameters.
            
        Returns:
            Dictionary mapping styles to generated poems.
        """
        if styles is None:
            styles = ["romantic", "modern", "classical", "haiku", "sonnet"]
        
        results = {}
        
        for style in styles:
            if hasattr(self.generator, 'generate_with_style'):
                samples = []
                for _ in range(num_samples_per_style):
                    sample = self.generator.generate_with_style(
                        prompt, style, **kwargs
                    )[0]
                    samples.append(sample)
                results[style] = samples
            else:
                # Fallback to regular generation with style in prompt
                style_prompt = f"In {style} style: {prompt}"
                samples = self.sample(
                    style_prompt,
                    num_samples=num_samples_per_style,
                    **kwargs
                )
                results[style] = samples
        
        return results
    
    def interpolate_prompts(
        self,
        prompt1: str,
        prompt2: str,
        num_steps: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Generate interpolated samples between two prompts.
        
        Args:
            prompt1: First prompt.
            prompt2: Second prompt.
            num_steps: Number of interpolation steps.
            **kwargs: Additional generation parameters.
            
        Returns:
            List of interpolated poems.
        """
        interpolated_samples = []
        
        for i in range(num_steps):
            # Simple interpolation by combining prompts with weights
            weight = i / (num_steps - 1) if num_steps > 1 else 0
            
            if weight < 0.5:
                interpolated_prompt = prompt1
            elif weight > 0.5:
                interpolated_prompt = prompt2
            else:
                interpolated_prompt = f"{prompt1} and {prompt2}"
            
            sample = self.generator.generate(
                interpolated_prompt,
                num_return_sequences=1,
                **kwargs
            )[0]
            
            interpolated_samples.append(sample)
        
        return interpolated_samples
    
    def save_samples(
        self,
        samples: Union[List[str], Dict[str, List[str]]],
        output_path: str,
        format: str = "txt"
    ) -> None:
        """
        Save generated samples to file.
        
        Args:
            samples: Generated samples to save.
            output_path: Path to save the samples.
            format: Output format ("txt", "json").
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                if isinstance(samples, dict):
                    for style, style_samples in samples.items():
                        f.write(f"=== {style.upper()} STYLE ===\n\n")
                        for i, sample in enumerate(style_samples, 1):
                            f.write(f"Sample {i}:\n{sample}\n\n")
                        f.write("\n" + "="*50 + "\n\n")
                else:
                    for i, sample in enumerate(samples, 1):
                        f.write(f"Sample {i}:\n{sample}\n\n")
        
        elif format == "json":
            import json
            
            if isinstance(samples, dict):
                data = samples
            else:
                data = {"samples": samples}
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved samples to {output_path}")


def create_poetry_sampler(
    generator: PoetryGenerator,
    tokenizer: PreTrainedTokenizer,
    device: Optional[torch.device] = None,
) -> PoetrySampler:
    """
    Create a poetry sampler.
    
    Args:
        generator: Poetry generator model.
        tokenizer: Tokenizer for text processing.
        device: Device to run the model on.
        
    Returns:
        PoetrySampler instance.
    """
    return PoetrySampler(generator, tokenizer, device)


def generate_poetry_samples(
    generator: PoetryGenerator,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    num_samples: int = 5,
    output_path: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Generate poetry samples with a simple interface.
    
    Args:
        generator: Poetry generator model.
        tokenizer: Tokenizer for text processing.
        prompt: Input prompt for generation.
        num_samples: Number of samples to generate.
        output_path: Optional path to save samples.
        **kwargs: Additional generation parameters.
        
    Returns:
        List of generated poems.
    """
    sampler = create_poetry_sampler(generator, tokenizer)
    samples = sampler.sample(prompt, num_samples, **kwargs)
    
    if output_path:
        sampler.save_samples(samples, output_path)
    
    return samples
