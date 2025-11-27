from src.ssa.ssa_optimizer import SSAOptimizer
from src.genome.decoder import GenomeDecoder
from src.genome.genome import GenomeConfig
import random

# Initialize
config = GenomeConfig(dimensions=10)
optimizer = SSAOptimizer(
    population_size=20,
    max_iterations=50,
    Gc=1.9,
    Pdp=0.1,
    genome_config=config
)

# Define fitness function (mock for testing)
def fitness_function(squirrel):
    """
    In real scenario, this would:
    1. Decode genome to prompt
    2. Run prompt on dev dataset
    3. Calculate accuracy/F1
    """
    # Mock: random fitness (lower is better)
    return random.uniform(0, 1)

# Run optimization
decoder = GenomeDecoder()
best_squirrel, history = optimizer.optimize(fitness_function, early_stopping_patience=10)

print(f"Best fitness: {best_squirrel.fitness}")
print(f"Best prompt:\n{decoder.decode(best_squirrel.genome)}")

# Get top 5 prompts
top_prompts = optimizer.get_best_prompts(decoder, top_k=5)
for i, (prompt, fitness) in enumerate(top_prompts, 1):
    print(f"\n{i}. Fitness: {fitness}\n{prompt}")

# Save evolution history
optimizer.save_history('outputs/evolution_history.json')