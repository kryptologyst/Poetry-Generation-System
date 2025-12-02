"""Unit tests for poetry generation system."""

import pytest
import torch
from transformers import GPT2Tokenizer

from poetry_generation import (
    Config,
    create_poetry_dataset,
    create_poetry_generator,
    PoetryEvaluator,
    get_device,
    set_deterministic,
)


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        assert config.get("model.name") == "gpt2"
        assert config.get("training.batch_size") == 8
        assert config.get("data.train_split") == 0.8
    
    def test_config_set_get(self):
        """Test configuration set and get operations."""
        config = Config()
        config.set("test.value", 42)
        assert config.get("test.value") == 42
        assert config.get("test.nonexistent", "default") == "default"
    
    def test_config_update(self):
        """Test configuration update."""
        config = Config()
        updates = {"model.name": "gpt2-medium", "training.batch_size": 16}
        config.update(updates)
        assert config.get("model.name") == "gpt2-medium"
        assert config.get("training.batch_size") == 16


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_deterministic(self):
        """Test deterministic seeding."""
        set_deterministic(42)
        # This test mainly ensures no errors are raised
        assert True
    
    def test_get_device_info(self):
        """Test device information."""
        info = get_device_info()
        assert "device" in info
        assert "device_name" in info


class TestPoetryDataset:
    """Test poetry dataset functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return GPT2Tokenizer.from_pretrained("gpt2")
    
    @pytest.fixture
    def sample_poems(self):
        """Create sample poems for testing."""
        return [
            "The moonlit sky is full of dreams,\nWhere stars dance in silver streams.",
            "In shadows deep, the night unfolds,\nA story that the darkness holds.",
        ]
    
    def test_poetry_dataset_creation(self, tokenizer, sample_poems):
        """Test poetry dataset creation."""
        from poetry_generation.data.dataset import PoetryDataset
        
        dataset = PoetryDataset(sample_poems, tokenizer, max_length=64)
        assert len(dataset) == 2
        
        # Test getting an item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape[0] == 64
    
    def test_poetry_data_module(self, tokenizer):
        """Test poetry data module creation."""
        data_module = create_poetry_dataset(tokenizer)
        
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None
        
        # Test dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None


class TestPoetryGenerator:
    """Test poetry generator functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return GPT2Tokenizer.from_pretrained("gpt2")
    
    def test_gpt2_generator_creation(self, tokenizer):
        """Test GPT-2 generator creation."""
        generator = create_poetry_generator("gpt2", "gpt2")
        
        assert generator is not None
        assert generator.model is not None
        assert generator.tokenizer is not None
    
    def test_poetry_generation(self, tokenizer):
        """Test poetry generation."""
        generator = create_poetry_generator("gpt2", "gpt2")
        
        prompt = "The moonlit sky"
        poems = generator.generate(
            prompt,
            max_length=50,
            temperature=0.7,
            num_return_sequences=1,
        )
        
        assert len(poems) == 1
        assert isinstance(poems[0], str)
        assert len(poems[0]) > len(prompt)


class TestPoetryEvaluator:
    """Test poetry evaluation functionality."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer for testing."""
        return GPT2Tokenizer.from_pretrained("gpt2")
    
    @pytest.fixture
    def evaluator(self, tokenizer):
        """Create evaluator for testing."""
        return PoetryEvaluator(tokenizer)
    
    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for testing."""
        return [
            "The moonlit sky is full of dreams",
            "Where stars dance in silver streams",
            "And whispers float on gentle breeze",
        ]
    
    def test_evaluator_creation(self, evaluator):
        """Test evaluator creation."""
        assert evaluator is not None
        assert evaluator.tokenizer is not None
    
    def test_length_stats(self, evaluator, sample_texts):
        """Test length statistics evaluation."""
        results = evaluator.evaluate_length_stats(sample_texts)
        
        assert "length_mean" in results
        assert "length_std" in results
        assert "length_min" in results
        assert "length_max" in results
        assert results["length_mean"] > 0
    
    def test_diversity_evaluation(self, evaluator, sample_texts):
        """Test diversity evaluation."""
        results = evaluator.evaluate_diversity(sample_texts, n_gram=2)
        
        assert "diversity_2gram_ratio" in results
        assert "diversity_2gram_unique" in results
        assert "diversity_2gram_total" in results
        assert 0 <= results["diversity_2gram_ratio"] <= 1
    
    def test_comprehensive_evaluation(self, evaluator, sample_texts):
        """Test comprehensive evaluation."""
        results = evaluator.comprehensive_evaluation(sample_texts)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "perplexity_mean",
            "length_mean",
            "diversity_2gram_ratio",
        ]
        
        for metric in expected_metrics:
            assert metric in results


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_generation(self):
        """Test end-to-end poetry generation."""
        # Create generator
        generator = create_poetry_generator("gpt2", "gpt2")
        
        # Generate poetry
        prompt = "The moonlit sky is full of dreams, where"
        poems = generator.generate(
            prompt,
            max_length=100,
            temperature=0.7,
            num_return_sequences=2,
        )
        
        # Verify results
        assert len(poems) == 2
        for poem in poems:
            assert isinstance(poem, str)
            assert len(poem) > len(prompt)
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline."""
        # Create generator and evaluator
        generator = create_poetry_generator("gpt2", "gpt2")
        evaluator = PoetryEvaluator(generator.tokenizer)
        
        # Generate samples
        prompt = "The moonlit sky"
        poems = generator.generate(
            prompt,
            max_length=50,
            temperature=0.7,
            num_return_sequences=3,
        )
        
        # Evaluate
        results = evaluator.comprehensive_evaluation(poems)
        
        # Verify evaluation results
        assert "perplexity_mean" in results
        assert "length_mean" in results
        assert "diversity_2gram_ratio" in results


if __name__ == "__main__":
    pytest.main([__file__])
