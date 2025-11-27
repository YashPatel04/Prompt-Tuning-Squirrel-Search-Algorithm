import numpy as np
from src.ssa.squirrel import Squirrel
from src.genome.genome import Genome

class Movement:
    """
    SSA movement operators for different squirrel types.
    """

    @staticmethod
    def levy_flight(beta=1.5, dim=10):  
        """
        Generate Levy flight step.

        Args:
            beta: Levy parameter (typically 1.5)
            dim: Dimensionality

        Returns:
            Array of Levy flight steps
        """
        u = np.random.normal(0, 1, dim)
        v = np.random.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step
    
    @staticmethod
    def elite_movement(squirrel, best_squirrel, Gc=1.9, iteration=0, max_iterations=100):
        """
        Elite squirrels move towards the best acorn (best solution).
        Small step size for exploitation.

        Args:
            squirrel: Squirrel to move.
            best_squirrel: Best squirrel in population
            Gc: Gravitational Coefficient
            iteration: Current iteration
            max_iterations: Total iterations
        """
        # Seasonal value - decreases over iterations (exploitation phase)
        S = 2 * (1- iteration/max_iterations)

        # Movement towards best
        for i in range(len(squirrel.genome.vector)):
            direction = best_squirrel.genome.vector[i] - squirrel.genome.vector[i]
            noise = np.random.random()
            squirrel.genome.vector[i] += Gc * S * direction * noise
        
        # Clip to valid range
        squirrel.genome._validate()

    @staticmethod
    def exploratory_movement(squirrel, best_squirrel, Gc=1.9, iteration=0, max_iterations=100):
        """
        Exploratory squirrels use Levy flight towards the best.
        Balanced exploration and exploitation.

        Args:
            squirrel: Squirrel to move
            best_squirrel: Best squirrel in population
            Gc: Gravitational coefficient
            iteration: Current iteration
            max_iterations: Total iterations
        """

        # Seasonal value - gradual transition from exploration to exploitation
        S_max = 2 * (1-iteration/max_iterations)
        S_min = 0.5 * (iteration/max_iterations)
        S = S_min + np.random.random() * (S_max - S_min)

        # Levy flight towards the best
        levy_step = Movement.levy_flight(beta=1.5, dim=(len(squirrel.genome.vector)))

        for i in range(len(squirrel.genome.vector)):
            direction = best_squirrel.genome.vector[i] - squirrel.genome.vector[i]
            squirrel.genome.vector[i] += 0.01 * Gc * S * levy_step[i] * direction

        # Clip to valid range
        squirrel.genome._validate()

    @staticmethod
    def foraging_movement(squirrel, best_squirrel, Pdp=0.1):
        """
        Foraging squirrels random search with probability of following best.
        Exploration phase.

        Args: 
            squirrel: Squirrel to move
            best_squirrel: Best squirrel in the population
            Pdp: Predator probability
        """
        if np.random.random() < Pdp:
            # Escape predation: random mutation
            squirrel.genome.random_perturbation(perturbation_std=0.3)
        else:
            # Follow best with high variance
            for i in range(len(squirrel.genome.vector)):
                direction = best_squirrel.genome.vector[i] - squirrel.genome.vector[i]
                noise = np.random.normal(0, 0.5) # High variance
                squirrel.genome.vector[i] += direction * noise
            
        # Clip to valid range
        squirrel.genome._validate()
    
    @staticmethod
    def update_population(population, best_squirrel, Gc=1.9, Pdp=0.1, iteration=0, max_iterations=100):
        """
        Update all the squirrels in population using SSA movement rules.

        Args:
            population: Population object
            best_squirrel:  Best squirrel (target)
            Gc: Gravitational coefficient
            Pdp: Predation probability
            iteration: Current Iteration
            max_iterations: Total Iterations

        """
        for squirrel in population.squirrels:
            if squirrel.squirrel_type == 'elite':
                Movement.elite_movement(squirrel, best_squirrel, Gc, iteration, max_iterations)
            elif squirrel.squirrel_type == 'exploratory':
                Movement.exploratory_movement(squirrel, best_squirrel, Gc, iteration, max_iterations)
            elif squirrel.squirrel_type == 'foraging':
                Movement.foraging_movement(squirrel, best_squirrel, Pdp)

            # Squirrel becomes unevaluated after movement
            squirrel.evaluated = False

    