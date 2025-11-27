import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

class MetricsCalculator:
    """
    Calculate and manage evaluation metrics.
    """
    
    @staticmethod
    def accuracy(predictions, ground_truth):
        """Calculate accuracy"""
        return accuracy_score(ground_truth, predictions)
    
    @staticmethod
    def precision(predictions, ground_truth, average='weighted'):
        """Calculate precision"""
        return precision_score(ground_truth, predictions, average=average, zero_division=0)
    
    @staticmethod
    def recall(predictions, ground_truth, average='weighted'):
        """Calculate recall"""
        return recall_score(ground_truth, predictions, average=average, zero_division=0)
    
    @staticmethod
    def f1(predictions, ground_truth, average='weighted'):
        """Calculate F1 score"""
        return f1_score(ground_truth, predictions, average=average, zero_division=0)
    
    @staticmethod
    def get_all_metrics(predictions, ground_truth):
        """
        Calculate all metrics at once.
        
        Returns:
            Dictionary with all metric values
        """
        return {
            'accuracy': MetricsCalculator.accuracy(predictions, ground_truth),
            'precision': MetricsCalculator.precision(predictions, ground_truth),
            'recall': MetricsCalculator.recall(predictions, ground_truth),
            'f1': MetricsCalculator.f1(predictions, ground_truth),
        }
    
    @staticmethod
    def confusion_matrix(predictions, ground_truth):
        """Get confusion matrix"""
        return confusion_matrix(ground_truth, predictions)
    
    @staticmethod
    def classification_report(predictions, ground_truth):
        """Get detailed classification report"""
        return classification_report(ground_truth, predictions)
    
    @staticmethod
    def per_class_metrics(predictions, ground_truth):
        """
        Calculate metrics for each class separately.
        
        Returns:
            Dictionary with per-class precision, recall, F1
        """
        unique_labels = set(ground_truth)
        results = {}
        
        for label in unique_labels:
            # Binary classification for this label
            binary_pred = [1 if p == label else 0 for p in predictions]
            binary_true = [1 if g == label else 0 for g in ground_truth]
            
            results[label] = {
                'precision': precision_score(binary_true, binary_pred, zero_division=0),
                'recall': recall_score(binary_true, binary_pred, zero_division=0),
                'f1': f1_score(binary_true, binary_pred, zero_division=0),
            }
        
        return results
    
    @staticmethod
    def consistency_score(predictions_list, ground_truth=None, include_accuracy=False):
        """
        Measure consistency across multiple predictions.
        Optionally combines with accuracy if ground truth provided.
        
        Args:
            predictions_list: List of prediction lists (multiple runs)
            ground_truth: Ground truth labels (optional)
            include_accuracy: If True and ground_truth provided, blend consistency with accuracy
        
        Returns:
            Consistency score (0-1, higher is better)
        """
        if len(predictions_list) < 2:
            return 1.0
        
        if not predictions_list or not predictions_list[0]:
            return 1.0
        
        # Normalize lengths
        min_length = min(len(preds) for preds in predictions_list)
        predictions_list = [preds[:min_length] for preds in predictions_list]
        
        consistency_scores = []
        num_runs = len(predictions_list)
        
        # For each example
        for i in range(min_length):
            predictions_for_example = [preds[i] for preds in predictions_list]
            
            # Majority voting
            from collections import Counter
            prediction_counts = Counter(predictions_for_example)
            max_count = max(prediction_counts.values())
            
            # Consistency: fraction that agree with majority
            consistency = max_count / num_runs
            consistency_scores.append(consistency)
        
        mean_consistency = np.mean(consistency_scores)
        
        # Optionally blend with accuracy
        if include_accuracy and ground_truth is not None:
            # Use majority vote as final prediction
            final_predictions = []
            for i in range(min_length):
                predictions_for_example = [preds[i] for preds in predictions_list]
                from collections import Counter
                prediction_counts = Counter(predictions_for_example)
                majority_pred = prediction_counts.most_common(1)[0][0]
                final_predictions.append(majority_pred)
            
            # Calculate accuracy
            accuracy = MetricsCalculator.accuracy(final_predictions, ground_truth[:min_length])
            
            # Blend: consistency shows agreement, accuracy shows correctness
            # Both matter for research
            blended_score = 0.6 * mean_consistency + 0.4 * accuracy
            
            return {
                'consistency': mean_consistency,
                'accuracy': accuracy,
                'blended_score': blended_score
            }
        
        return mean_consistency


class StatisticsCalculator:
    """
    Calculate statistics from evolution history.
    """
    
    @staticmethod
    def convergence_rate(fitness_history):
        """
        Calculate rate of convergence.
        How quickly does fitness improve?
        
        Returns:
            Average improvement per iteration
        """
        if len(fitness_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(fitness_history)):
            improvement = fitness_history[i-1] - fitness_history[i]
            improvements.append(max(0, improvement))  # Only positive improvements
        
        return np.mean(improvements) if improvements else 0.0
    
    @staticmethod
    def variance_reduction(fitness_history):
        """
        Measure how much population variance reduces over time.
        
        Returns:
            Reduction factor (higher = better convergence)
        """
        if len(fitness_history) < 2:
            return 0.0
        
        # Compare first half to second half
        mid = len(fitness_history) // 2
        first_half_var = np.var(fitness_history[:mid])
        second_half_var = np.var(fitness_history[mid:])
        
        if first_half_var == 0:
            return 0.0
        
        return (first_half_var - second_half_var) / first_half_var
    
    @staticmethod
    def improvement_summary(best_fitness_history):
        """
        Get summary of fitness improvements.
        
        Returns:
            Dictionary with improvement statistics
        """
        initial = best_fitness_history[0]
        final = best_fitness_history[-1]
        improvement = (initial - final) / initial * 100 if initial != 0 else 0
        
        return {
            'initial_fitness': initial,
            'final_fitness': final,
            'improvement_percent': improvement,
            'total_iterations': len(best_fitness_history),
            'iterations_to_improvement': StatisticsCalculator._iterations_to_first_improvement(best_fitness_history),
        }
    
    @staticmethod
    def _iterations_to_first_improvement(fitness_history):
        """How many iterations until first improvement?"""
        for i in range(1, len(fitness_history)):
            if fitness_history[i] < fitness_history[i-1]:
                return i
        return len(fitness_history)
    
    @staticmethod
    def plateau_detection(fitness_history, window_size=5, threshold=1e-4):
        """
        Detect if algorithm has plateaued.
        
        Args:
            fitness_history: List of fitness values
            window_size: Window for calculating moving average
            threshold: Improvement threshold to consider as plateau
        
        Returns:
            Dictionary with plateau detection info
        """
        if len(fitness_history) < window_size:
            return {'is_plateau': False, 'plateau_iteration': None}
        
        for i in range(window_size, len(fitness_history)):
            window = fitness_history[i-window_size:i]
            improvement = max(window) - min(window)
            
            if improvement < threshold:
                return {
                    'is_plateau': True,
                    'plateau_iteration': i - window_size,
                    'improvement_in_window': improvement
                }
        
        return {'is_plateau': False, 'plateau_iteration': None}