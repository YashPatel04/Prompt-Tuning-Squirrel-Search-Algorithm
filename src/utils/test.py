# ============================================
# LOGGING EXAMPLES
# ============================================
from src.utils.logger import get_logger

logger = get_logger()
logger.section("Starting SSA Optimization")
logger.info("Initialized population with 20 squirrels")
logger.debug("Debug information here")
logger.warning("This might be an issue")
logger.error("Error occurred!")


# ============================================
# METRICS EXAMPLES
# ============================================
from src.utils.metrics import MetricsCalculator, StatisticsCalculator

predictions = ['positive', 'negative', 'positive', 'neutral', 'positive']
ground_truth = ['positive', 'positive', 'positive', 'neutral', 'negative']

# Get all metrics
metrics = MetricsCalculator.get_all_metrics(predictions, ground_truth)
print(metrics)
# Output: {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 'f1': 0.77}

# Per-class metrics
per_class = MetricsCalculator.per_class_metrics(predictions, ground_truth)
print(per_class)

# Get classification report
report = MetricsCalculator.classification_report(predictions, ground_truth)
print(report)


# ============================================
# STATISTICS EXAMPLES
# ============================================
import json

# Load evolution history
with open('outputs/evolution_history.json', 'r') as f:
    history = json.load(f)

# Convergence analysis
convergence_rate = StatisticsCalculator.convergence_rate(history['best_fitness'])
print(f"Convergence rate: {convergence_rate}")

# Improvement summary
summary = StatisticsCalculator.improvement_summary(history['best_fitness'])
print(f"Improvement: {summary['improvement_percent']:.2f}%")

# Plateau detection
plateau = StatisticsCalculator.plateau_detection(history['best_fitness'])
print(f"Plateaued at iteration {plateau['plateau_iteration']}")


# ============================================
# VISUALIZATION EXAMPLES
# ============================================
from src.utils.visualizer import Visualizer

visualizer = Visualizer()

# Plot convergence
visualizer.plot_convergence(
    history,
    save_path="outputs/figures/convergence.png"
)

# Plot multiple runs comparison
runs_history = [history, history]  # In practice, load different runs
visualizer.plot_multi_run_comparison(
    runs_history,
    save_path="outputs/figures/multi_run_comparison.png"
)

# Metric comparison
metrics_comparison = {
    'Baseline Prompt': {'accuracy': 0.75, 'f1': 0.72},
    'SSA Optimized': {'accuracy': 0.92, 'f1': 0.90},
}
visualizer.plot_metric_comparison(
    metrics_comparison,
    save_path="outputs/figures/metric_comparison.png"
)   