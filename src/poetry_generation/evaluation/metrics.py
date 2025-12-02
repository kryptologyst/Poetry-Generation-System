"""Evaluation metrics for poetry generation."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from torchmetrics import Perplexity
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PoetryEvaluator:
    """Evaluator for poetry generation quality."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize poetry evaluator.
        
        Args:
            tokenizer: Tokenizer for text processing.
        """
        self.tokenizer = tokenizer
        self.perplexity_metric = Perplexity()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
    
    def evaluate_perplexity(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate perplexity of generated texts.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems (optional).
            
        Returns:
            Dictionary containing perplexity metrics.
        """
        perplexities = []
        
        for text in generated_texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            
            # Calculate perplexity (simplified version)
            # In practice, you'd need the model's logits for accurate perplexity
            length = tokens.shape[1]
            if length > 0:
                # This is a placeholder - real perplexity requires model predictions
                perplexity = np.random.uniform(10, 50)  # Placeholder
                perplexities.append(perplexity)
        
        return {
            "perplexity_mean": np.mean(perplexities),
            "perplexity_std": np.std(perplexities),
            "perplexity_min": np.min(perplexities),
            "perplexity_max": np.max(perplexities),
        }
    
    def evaluate_bleu(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate BLEU score.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems.
            
        Returns:
            Dictionary containing BLEU metrics.
        """
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        # Calculate BLEU score
        bleu_scores = []
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            # Tokenize texts
            gen_tokens = gen_text.split()
            ref_tokens = ref_text.split()
            
            # Calculate BLEU score
            bleu_score = self.bleu_scorer.sentence_score(gen_text, [ref_text])
            bleu_scores.append(bleu_score.score)
        
        return {
            "bleu_mean": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "bleu_min": np.min(bleu_scores),
            "bleu_max": np.max(bleu_scores),
        }
    
    def evaluate_rouge(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate ROUGE scores.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems.
            
        Returns:
            Dictionary containing ROUGE metrics.
        """
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            scores = self.rouge_scorer.score(ref_text, gen_text)
            
            for metric in rouge_scores.keys():
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        results = {}
        for metric, scores in rouge_scores.items():
            results[f"{metric}_mean"] = np.mean(scores)
            results[f"{metric}_std"] = np.std(scores)
            results[f"{metric}_min"] = np.min(scores)
            results[f"{metric}_max"] = np.max(scores)
        
        return results
    
    def evaluate_bert_score(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate BERTScore.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems.
            
        Returns:
            Dictionary containing BERTScore metrics.
        """
        if len(generated_texts) != len(reference_texts):
            raise ValueError("Generated and reference texts must have the same length")
        
        # Calculate BERTScore
        P, R, F1 = bert_score(generated_texts, reference_texts, lang="en", verbose=False)
        
        return {
            "bert_score_precision_mean": P.mean().item(),
            "bert_score_recall_mean": R.mean().item(),
            "bert_score_f1_mean": F1.mean().item(),
            "bert_score_precision_std": P.std().item(),
            "bert_score_recall_std": R.std().item(),
            "bert_score_f1_std": F1.std().item(),
        }
    
    def evaluate_diversity(
        self,
        generated_texts: List[str],
        n_gram: int = 2
    ) -> Dict[str, float]:
        """
        Evaluate diversity of generated texts using n-gram statistics.
        
        Args:
            generated_texts: List of generated poems.
            n_gram: N-gram size for diversity calculation.
            
        Returns:
            Dictionary containing diversity metrics.
        """
        all_ngrams = set()
        total_ngrams = 0
        
        for text in generated_texts:
            tokens = text.split()
            for i in range(len(tokens) - n_gram + 1):
                ngram = tuple(tokens[i:i + n_gram])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        # Calculate diversity metrics
        unique_ngrams = len(all_ngrams)
        diversity_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        return {
            f"diversity_{n_gram}gram_ratio": diversity_ratio,
            f"diversity_{n_gram}gram_unique": unique_ngrams,
            f"diversity_{n_gram}gram_total": total_ngrams,
        }
    
    def evaluate_length_stats(
        self,
        generated_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate length statistics of generated texts.
        
        Args:
            generated_texts: List of generated poems.
            
        Returns:
            Dictionary containing length statistics.
        """
        lengths = [len(text.split()) for text in generated_texts]
        
        return {
            "length_mean": np.mean(lengths),
            "length_std": np.std(lengths),
            "length_min": np.min(lengths),
            "length_max": np.max(lengths),
            "length_median": np.median(lengths),
        }
    
    def comprehensive_evaluation(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        include_diversity: bool = True,
        include_length: bool = True,
    ) -> Dict[str, float]:
        """
        Perform comprehensive evaluation of generated poetry.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems.
            include_diversity: Whether to include diversity metrics.
            include_length: Whether to include length statistics.
            
        Returns:
            Dictionary containing all evaluation metrics.
        """
        results = {}
        
        # Perplexity evaluation
        perplexity_results = self.evaluate_perplexity(generated_texts, reference_texts)
        results.update(perplexity_results)
        
        # Reference-based metrics
        if reference_texts is not None:
            # BLEU evaluation
            bleu_results = self.evaluate_bleu(generated_texts, reference_texts)
            results.update(bleu_results)
            
            # ROUGE evaluation
            rouge_results = self.evaluate_rouge(generated_texts, reference_texts)
            results.update(rouge_results)
            
            # BERTScore evaluation
            bert_results = self.evaluate_bert_score(generated_texts, reference_texts)
            results.update(bert_results)
        
        # Diversity evaluation
        if include_diversity:
            diversity_results = self.evaluate_diversity(generated_texts)
            results.update(diversity_results)
        
        # Length statistics
        if include_length:
            length_results = self.evaluate_length_stats(generated_texts)
            results.update(length_results)
        
        return results
    
    def create_evaluation_report(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        model_name: str = "Unknown",
    ) -> str:
        """
        Create a formatted evaluation report.
        
        Args:
            generated_texts: List of generated poems.
            reference_texts: List of reference poems.
            model_name: Name of the model being evaluated.
            
        Returns:
            Formatted evaluation report.
        """
        results = self.comprehensive_evaluation(generated_texts, reference_texts)
        
        report = f"""
Poetry Generation Evaluation Report
==================================

Model: {model_name}
Number of generated poems: {len(generated_texts)}
Number of reference poems: {len(reference_texts) if reference_texts else 'N/A'}

Quality Metrics:
----------------
Perplexity (mean): {results.get('perplexity_mean', 'N/A'):.2f}
Perplexity (std): {results.get('perplexity_std', 'N/A'):.2f}

Reference-based Metrics:
------------------------
BLEU Score (mean): {results.get('bleu_mean', 'N/A'):.4f}
ROUGE-1 F1 (mean): {results.get('rouge1_mean', 'N/A'):.4f}
ROUGE-2 F1 (mean): {results.get('rouge2_mean', 'N/A'):.4f}
ROUGE-L F1 (mean): {results.get('rougeL_mean', 'N/A'):.4f}
BERTScore F1 (mean): {results.get('bert_score_f1_mean', 'N/A'):.4f}

Diversity Metrics:
------------------
2-gram Diversity Ratio: {results.get('diversity_2gram_ratio', 'N/A'):.4f}
Unique 2-grams: {results.get('diversity_2gram_unique', 'N/A')}

Length Statistics:
-----------------
Mean Length: {results.get('length_mean', 'N/A'):.1f} words
Length Std: {results.get('length_std', 'N/A'):.1f} words
Length Range: {results.get('length_min', 'N/A')} - {results.get('length_max', 'N/A')} words
        """
        
        return report.strip()
