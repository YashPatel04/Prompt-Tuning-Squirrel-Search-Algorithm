import numpy as np
from typing import List, Tuple, Callable, Dict
from src.ssa.population import Population
from src.ssa.movement import Movement
from src.ssa.squirrel import Squirrel
from src.genome.genome import GenomeConfig
import json

class SSAOptimizer:
    """
    Squirrel Search Algorithm (SSA) for prompt optimization.
    Metaheuristic optimization adapted for discrete prompt space.
    """
    
    def __init__(self, 
                 population_size=20,
                 max_iterations=50,
                 Gc=1.9,
                 Pdp=0.1,
                 genome_config=None):
        """
        Initialize SSA optimizer.
        
        Args:
            population_size: Number of squirrels
            max_iterations: Maximum iterations
            Gc: Gravitational coefficient (attraction strength)
            Pdp: Predation presence probability (0-1)
            genome_config: Genome configuration
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.Gc = Gc
        self.Pdp = Pdp
        self.genome_config = genome_config or GenomeConfig()
        
        self.population = None
        self.best_squirrel = None
        self.iteration = 0
        self.history = {
            'best_fitness': [],
            'worst_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'population_snapshots': []
        }
    
    def initialize(self):
        """Initialize population with random squirrels"""
        self.population = Population(self.population_size, self.genome_config)
        self.population.initialize_random()
        self.iteration = 0
    
    def optimize(self, fitness_function, early_stopping_patience=10):
        """
        Run SSA optimization.
        
        Args:
            fitness_function: Function that takes squirrel and returns fitness score
                            Lower fitness is better
            early_stopping_patience: Stop if no improvement for N iterations
        
        Returns:
            (best_squirrel, evolution_history)
        """
        self.initialize()
        
        # Evaluate initial population
        for squirrel in self.population.squirrels:
            fitness = fitness_function(squirrel)
            squirrel.update_fitness(fitness, self.iteration)
        
        self.best_squirrel = self.population.get_best_squirrel()
        self._record_iteration()
        
        no_improvement_count = 0
        
        # Main optimization loop
        for self.iteration in range(1, self.max_iterations):
            # Update positions using SSA movement
            Movement.update_population(
                self.population,
                self.best_squirrel,
                Gc=self.Gc,
                Pdp=self.Pdp,
                iteration=self.iteration,
                max_iterations=self.max_iterations
            )
            
            # Evaluate population
            for squirrel in self.population.squirrels:
                if not squirrel.evaluated:
                    fitness = fitness_function(squirrel)
                    squirrel.update_fitness(fitness, self.iteration)
            
            # Update best squirrel
            current_best = self.population.get_best_squirrel()
            if current_best.is_better_than(self.best_squirrel):
                self.best_squirrel = current_best.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Record progress
            self._record_iteration()
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at iteration {self.iteration}")
                break
        
        return self.best_squirrel, self.history
    
    def _record_iteration(self):
        """Record statistics for current iteration"""
        stats = self.population.get_fitness_stats()
        
        self.history['best_fitness'].append(stats['best'])
        self.history['worst_fitness'].append(stats['worst'])
        self.history['mean_fitness'].append(stats['mean'])
        self.history['std_fitness'].append(stats['std'])
        
        # save population snapshot (memory intensive)
        if self.iteration % 10 == 0:
            snapshot = {
                'iteration': self.iteration,
                'best_fitness': self.best_squirrel.fitness,
                'population_size': len(self.population)
            }
            self.history['population_snapshots'].append(snapshot)
    
    def get_best_prompts(self, decoder, top_k=5):
        """
        Get top K best prompts found.
        
        Args:
            decoder: GenomeDecoder to convert genomes to prompts
            top_k: Number of prompts to return
        
        Returns:
            List of (prompt, fitness) tuples
        """
        evaluated = [s for s in self.population.squirrels if s.evaluated]
        evaluated.sort(key=lambda s: s.fitness)
        
        best_prompts = []
        for i, squirrel in enumerate(evaluated[:top_k]):
            prompt = decoder.decode(squirrel.genome)
            best_prompts.append((prompt, squirrel.fitness))
        
        return best_prompts
    
    def save_history(self, filepath):
        """Save evolution history to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'best_fitness': self.history['best_fitness'],
            'worst_fitness': self.history['worst_fitness'],
            'mean_fitness': self.history['mean_fitness'],
            'std_fitness': self.history['std_fitness'],
            'population_snapshots': self.history['population_snapshots']
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_json, f, indent=2)
    
    def __str__(self):
        return (f"SSAOptimizer(population_size={self.population_size}, "
                f"max_iterations={self.max_iterations}, "
                f"best_fitness={self.best_squirrel.fitness if self.best_squirrel else None})")