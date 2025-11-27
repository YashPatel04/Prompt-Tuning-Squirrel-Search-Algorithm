from src.genome.genome import Genome
from typing import Optional

class Squirrel:
    """
    Represents a single squirrel in the SSA population.
    Each Squirrel has a genome (prompt variant) and a fitness score.
    """

    def __init__(self, genome, squirrel_type='foraging'):
        """
        Initalize a squirrel.

        Args:
            genome: Genome object representing this squirrel's prompt
            squirrel_type: Type of squirrel - 'elite', 'exploratory', or 'foraging'
        """
        self.genome = genome
        self.squirrel_type = squirrel_type
        self.fitness = None
        self.evaluated = False
        self.iteration_created = 0
        self.last_improved_iteration = 0

    def update_fitness(self, fitness, iteration):
        """
        Update squirrel's fitness score.

        Args:
            fitness : New fitness rule
            iteration: Current iteration number
        """
        if self.fitness is None or fitness<self.fitness:
            self.fitness = fitness
            self.last_improved_iteration = iteration
        self.evaluated = True

    def copy(self):
        """Create a deep copy of this squirrel"""
        new_squirrel = Squirrel(self.genome.copy(), self.squirrel_type)
        new_squirrel.fitness = self.fitness
        new_squirrel.evaluated = self.evaluated
        return new_squirrel

    def is_better_than(self, other):
        """Check if this squirrel has better fitness than other"""
        if self.fitness is None or other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return (f"Squirrel(type={self.squirrel_type}, fitness={self.fitness}, "
                f"genome={self.genome})")
    
    def __repr__(self) -> str:
        return self.__str__()