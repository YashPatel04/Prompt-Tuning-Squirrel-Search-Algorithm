from typing import List, Tuple, Dict
from src.evaluation.llm_interface import OllamaInterface
from src.evaluation.fitness import FitnessCalculator
from src.evaluation.cache import PromptCache
from src.ssa.squirrel import Squirrel
import pandas as pd

class Evaluator:
    """
    Evaluate squirrel prompts on a dataset and calculate fitness.
    """
    def __init__(self, llm_interface, fitness_calculator, cache=None, use_cache=True):
        """
        Initialize evaluator.
        Args:
            llm_interface: OllamaInterface instance
            fitness_calculator: FitnessCalculator instance
            cache: PromptCache instance
            use_cache: Whether to use caching
        """
        self.llm = llm_interface
        self.fitness_calc = fitness_calculator
        self.cache = cache or PromptCache()
        self.use_cache = use_cache
    
    def evaluate_prompt(self, prompt, dataset, sample_size=None):
        """
        Evaluate a single prompt on dataset and return fitness.
        Args:
            prompt: Prompt template with {input} placeholder
            dataset: List of dicts with 'text' and 'label' keys
            sample_size: Sample subset of dataset (for speed)
        Returns:
            Fitness score (lower is better)
        """
        # Sample dataset if specified
        if sample_size and sample_size < len(dataset):
            import random
            dataset = random.sample(dataset, sample_size)
        predictions = []
        ground_truth = []
        for example in dataset:
            text = example['text']
            label = example['label']
            # Create full prompt
            full_prompt = prompt.format(input=text)
            # Check cache
            if self.use_cache:
                cached_response = self.cache.get(full_prompt)
                if cached_response:
                    response = cached_response
                else:
                    response = self.llm.generate(full_prompt)
                    self.cache.set(full_prompt, response)
            else:
                response = self.llm.generate(full_prompt)
            # Extract label from response
            predicted_label = self.fitness_calc.extract_label_from_response(response)
            predictions.append(predicted_label)
            ground_truth.append(label)
        # Calculate fitness
        fitness = self.fitness_calc.calculate_combined_fitness(
            predictions, 
            ground_truth,
            accuracy_weight=0.6,
            f1_weight=0.4
        )
        return fitness
    
    def evaluate_squirrel(self, squirrel, decoder, dataset, sample_size=None):
        """
        Evaluate a squirrel (genome → prompt → fitness).
        Args:
            squirrel: Squirrel object with genome
            decoder: GenomeDecoder to convert genome to prompt
            dataset: Evaluation dataset
            sample_size: Sample subset of dataset
        Returns:
            Fitness score
        """
        # Decode genome to prompt
        prompt = decoder.decode(squirrel.genome)
        # Evaluate prompt
        fitness = self.evaluate_prompt(prompt, dataset, sample_size)
        return fitness
    
    def batch_evaluate(self, squirrels, decoder, dataset, sample_size=None, show_progress=True):
        """
        Evaluate multiple squirrels.
        Args:
            squirrels: List of Squirrel objects
            decoder: GenomeDecoder
            dataset: Evaluation dataset
            sample_size: Sample subset of dataset
            show_progress: Show progress
        Returns:
            List of fitness scores
        """
        fitnesses = []
        for i, squirrel in enumerate(squirrels):
            if show_progress:
                print(f"[{i+1}/{len(squirrels)}] Evaluating squirrel...")
            
            fitness = self.evaluate_squirrel(squirrel, decoder, dataset, sample_size)
            fitnesses.append(fitness)
        return fitnesses
    
    def evaluate_dataset_split(self, prompt, train_data, dev_data, test_data=None):
        """
        Evaluate prompt on multiple dataset splits.
        Args:
            prompt: Prompt to evaluate
            train_data: Training dataset
            dev_data: Development dataset
            test_data: Test dataset (optional)
        Returns:
            Dictionary with fitness for each split
        """
        results = {
            'train': self.evaluate_prompt(prompt, train_data),
            'dev': self.evaluate_prompt(prompt, dev_data),
        }
        
        if test_data:
            results['test'] = self.evaluate_prompt(prompt, test_data)
        return results