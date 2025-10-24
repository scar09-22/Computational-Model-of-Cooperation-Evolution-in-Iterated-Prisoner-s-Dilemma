import random
from .agent import EvolvableAgent

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, tournament_size):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.population = [EvolvableAgent() for _ in range(population_size)]

    def _select_parent(self):
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda agent: agent.fitness)

    def _crossover(self, parent1, parent2):
        child_genome = parent1.genome[:2] + parent2.genome[2:]
        return EvolvableAgent(genome=child_genome)

    def _mutate(self, agent):
        for i in range(len(agent.genome)):
            if random.random() < self.mutation_rate:
                agent.genome[i] = 'C' if agent.genome[i] == 'D' else 'D'
        return agent

    def evolve(self):
        new_population = []
        
        best_agent = max(self.population, key=lambda agent: agent.fitness)
        new_population.append(EvolvableAgent(genome=best_agent.genome.copy()))

        while len(new_population) < self.population_size:
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)
        
        self.population = new_population