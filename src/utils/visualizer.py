import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

class Visualizer:
    """
    Visualization utilities for evolution history and results.
    """
    
    def __init__(self, output_dir="outputs/figures"):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_convergence(self, history, save_path):
        """
        Plot SSA convergence curves.
        
        Args:
            history: Evolution history dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SSA Convergence Analysis', fontsize=16, fontweight='bold')
        
        iterations = range(len(history['best_fitness']))
        
        # 1. Best Fitness
        axes[0, 0].plot(iterations, history['best_fitness'], 
                       'g-', linewidth=2, label='Best')
        axes[0, 0].set_title('Best Fitness Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Fitness (lower is better)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Mean ± Std
        mean_fitness = history['mean_fitness']
        std_fitness = history['std_fitness']
        axes[0, 1].plot(iterations, mean_fitness, 'b-', 
                       linewidth=2, label='Mean')
        axes[0, 1].fill_between(iterations,
                               np.array(mean_fitness) - np.array(std_fitness),
                               np.array(mean_fitness) + np.array(std_fitness),
                               alpha=0.2, color='blue', label='±1 Std')
        axes[0, 1].set_title('Population Mean ± Std Deviation', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Best vs Worst
        axes[1, 0].plot(iterations, history['best_fitness'], 
                       'g-', linewidth=2, label='Best')
        axes[1, 0].plot(iterations, history['worst_fitness'],
                       'r-', linewidth=2, label='Worst')
        axes[1, 0].fill_between(iterations,
                               history['best_fitness'],
                               history['worst_fitness'],
                               alpha=0.2, color='yellow')
        axes[1, 0].set_title('Population Range', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Fitness')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Improvement per iteration
        improvements = [history['best_fitness'][0] - history['best_fitness'][i] 
                       for i in range(len(history['best_fitness']))]
        axes[1, 1].bar(iterations, improvements, color='purple', alpha=0.7)
        axes[1, 1].set_title('Cumulative Improvement', fontweight='bold')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Total Improvement')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved convergence plot to {save_path}")
        
        plt.show()
    
    def plot_fitness_distribution(self, final_population_fitness, save_path):
        """
        Plot fitness distribution of final population.
        
        Args:
            final_population_fitness: Fitness scores of final population
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Final Population Fitness Distribution', 
                    fontsize=14, fontweight='bold')
        
        # Histogram
        axes[0].hist(final_population_fitness, bins=15, 
                    color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(final_population_fitness), 
                       color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(np.median(final_population_fitness),
                       color='green', linestyle='--', linewidth=2, label='Median')
        axes[0].set_title('Fitness Histogram')
        axes[0].set_xlabel('Fitness Score')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        axes[1].boxplot(final_population_fitness, vert=True)
        axes[1].set_title('Fitness Box Plot')
        axes[1].set_ylabel('Fitness Score')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distribution plot to {save_path}")
        
        plt.show()
    
    def plot_metric_comparison(self, metrics_dict, save_path):
        """
        Compare metrics across different prompts/runs.
        
        Args:
            metrics_dict: {prompt_name: {metric_name: score}}
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        prompts = list(metrics_dict.keys())
        metric_names = list(metrics_dict[prompts[0]].keys())
        x = np.arange(len(prompts))
        width = 0.2
        for i, metric in enumerate(metric_names):
            scores = [metrics_dict[p][metric] for p in prompts]
            ax.bar(x + i*width, scores, width, label=metric, alpha=0.8)
        ax.set_xlabel('Prompt/Run')
        ax.set_ylabel('Score')
        ax.set_title('Metric Comparison', fontweight='bold')
        ax.set_xticks(x + width * (len(metric_names)-1) / 2)
        ax.set_xticklabels(prompts, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metric comparison to {save_path}")
        plt.show()
    
    def plot_multi_run_comparison(self, runs_history, save_path):
        """
        Compare best fitness across multiple SSA runs.
        
        Args:
            runs_history: List of history dicts from multiple runs
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        max_iterations = max(len(h['best_fitness']) for h in runs_history)
        
        # Plot each run
        for i, history in enumerate(runs_history):
            iterations = range(len(history['best_fitness']))
            ax.plot(iterations, history['best_fitness'],
                   alpha=0.6, label=f'Run {i+1}')
        
        # Plot mean across runs
        mean_best = []
        for iter_idx in range(max_iterations):
            values = [h['best_fitness'][iter_idx] 
                     for h in runs_history
                     if iter_idx < len(h['best_fitness'])]
            if values:
                mean_best.append(np.mean(values))
        
        ax.plot(range(len(mean_best)), mean_best, 
               'r-', linewidth=3, label='Mean')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Multiple SSA Runs Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved multi-run comparison to {save_path}")
        
        plt.show()