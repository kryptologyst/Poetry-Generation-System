"""Poetry dataset and data loading utilities."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PoetryDataset(Dataset):
    """Dataset class for poetry generation."""
    
    def __init__(
        self,
        poems: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ):
        """
        Initialize poetry dataset.
        
        Args:
            poems: List of poem texts.
            tokenizer: Tokenizer for text processing.
            max_length: Maximum sequence length.
            add_special_tokens: Whether to add special tokens.
        """
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.poems)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index of the item.
            
        Returns:
            Dictionary containing input_ids and attention_mask.
        """
        poem = self.poems[idx]
        
        # Tokenize the poem
        encoding = self.tokenizer(
            poem,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        }


class PoetryDataModule:
    """Data module for poetry generation with train/val/test splits."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 8,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        """
        Initialize poetry data module.
        
        Args:
            tokenizer: Tokenizer for text processing.
            data_path: Path to poetry data file.
            max_length: Maximum sequence length.
            batch_size: Batch size for data loaders.
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
            num_workers: Number of workers for data loading.
            shuffle: Whether to shuffle data.
        """
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.train_dataset: Optional[PoetryDataset] = None
        self.val_dataset: Optional[PoetryDataset] = None
        self.test_dataset: Optional[PoetryDataset] = None
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and split poetry data."""
        if self.data_path and Path(self.data_path).exists():
            poems = self._load_from_file(self.data_path)
        else:
            poems = self._generate_sample_data()
        
        # Split data
        train_size = int(len(poems) * self.train_split)
        val_size = int(len(poems) * self.val_split)
        
        random.shuffle(poems)
        
        train_poems = poems[:train_size]
        val_poems = poems[train_size:train_size + val_size]
        test_poems = poems[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = PoetryDataset(
            train_poems, self.tokenizer, self.max_length
        )
        self.val_dataset = PoetryDataset(
            val_poems, self.tokenizer, self.max_length
        )
        self.test_dataset = PoetryDataset(
            test_poems, self.tokenizer, self.max_length
        )
        
        logger.info(f"Loaded {len(poems)} poems")
        logger.info(f"Train: {len(train_poems)}, Val: {len(val_poems)}, Test: {len(test_poems)}")
    
    def _load_from_file(self, path: str) -> List[str]:
        """Load poems from file."""
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "poems" in data:
                return data["poems"]
            else:
                raise ValueError("Invalid JSON format")
        
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            if "poem" in df.columns:
                return df["poem"].tolist()
            elif "text" in df.columns:
                return df["text"].tolist()
            else:
                raise ValueError("CSV must contain 'poem' or 'text' column")
        
        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _generate_sample_data(self) -> List[str]:
        """Generate sample poetry data for demonstration."""
        sample_poems = [
            "The moonlit sky is full of dreams,\nWhere stars dance in silver streams,\nAnd whispers float on gentle breeze,\nThrough ancient, rustling trees.",
            "In shadows deep, the night unfolds,\nA story that the darkness holds,\nOf mysteries that time conceals,\nAnd secrets that the heart reveals.",
            "Morning light breaks through the mist,\nWith golden rays that softly kiss,\nThe earth below in gentle glow,\nWhere flowers bloom and rivers flow.",
            "Autumn leaves in colors bright,\nPaint the world in pure delight,\nAs nature sings her final song,\nBefore the winter comes along.",
            "Beneath the ocean's endless blue,\nWhere coral reefs in colors grew,\nThe fish swim free in harmony,\nIn this vast, watery symphony.",
            "Mountains rise to touch the sky,\nTheir peaks so high they seem to fly,\nAbove the clouds in majesty,\nA sight of pure serenity.",
            "The forest whispers ancient tales,\nOf creatures that the night unveils,\nIn moonlight's silver, gentle beam,\nThey dance within the dreamer's dream.",
            "Spring awakens from her sleep,\nAs flowers bloom and rivers leap,\nThe world renews in vibrant hue,\nWith life that's fresh and bright and new.",
            "Desert sands stretch far and wide,\nWhere camels walk with steady stride,\nBeneath the sun's relentless gaze,\nIn this land of endless days.",
            "City lights at night so bright,\nIlluminate the urban sight,\nWhere people rush and traffic flows,\nIn this world that never slows.",
        ]
        return sample_poems
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def create_poetry_dataset(
    tokenizer: PreTrainedTokenizer,
    data_path: Optional[str] = None,
    **kwargs
) -> PoetryDataModule:
    """
    Create a poetry dataset.
    
    Args:
        tokenizer: Tokenizer for text processing.
        data_path: Path to poetry data file.
        **kwargs: Additional arguments for PoetryDataModule.
        
    Returns:
        PoetryDataModule instance.
    """
    return PoetryDataModule(tokenizer, data_path, **kwargs)
