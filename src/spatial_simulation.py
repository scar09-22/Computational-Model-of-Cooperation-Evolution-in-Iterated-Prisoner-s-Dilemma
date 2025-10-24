import numpy as np
import random
from config import PAYOFF, GRID_WIDTH, GRID_HEIGHT, SPATIAL_MUTATION_RATE, ROUNDS_PER_MATCH
from .agent import EvolvableAgent


class SpatialSimulation:    
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.grid = np.empty((self.grid_height, self.grid_width), dtype=object)

    def populate_grid(self, agents):
        if len(agents) != self.grid_width * self.grid_height:
            raise ValueError("Number of agents must match grid size.")
        np.random.shuffle(agents) 
        it = iter(agents)
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                self.grid[i, j] = next(it)

    def _get_neighbor_coords(self, row, col):
        coords = []
        for i in range(max(0, row - 1), min(self.grid_height, row + 2)):
            for j in range(max(0, col - 1), min(self.grid_width, col + 2)):
                if (i, j) != (row, col):
                    coords.append((i, j))
        return coords

    def _mutate_genome(self, genome):
        new_genome = genome.copy()
        for i in range(len(new_genome)):
            if random.random() < SPATIAL_MUTATION_RATE:
                new_genome[i] = 'C' if new_genome[i] == 'D' else 'D'
        return new_genome

    def run_step(self):
        fitness_grid = np.zeros((self.grid_height, self.grid_width))
        
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                agent = self.grid[r, c]
                total_fitness = 0
                neighbor_coords = self._get_neighbor_coords(r, c)
                
                for nr, nc in neighbor_coords:
                    neighbor = self.grid[nr, nc]
                    
                    agent.history = []
                    neighbor.history = []
                    
                    match_fitness = 0
                    
                    for _ in range(ROUNDS_PER_MATCH):
                        move1 = agent.make_move(neighbor.history)
                        move2 = neighbor.make_move(agent.history)
                        
                        score1, _ = PAYOFF[(move1, move2)]
                        match_fitness += score1
                        
                        agent.history.append(move1)
                        neighbor.history.append(move2)
                    
                    total_fitness += match_fitness
                
                fitness_grid[r, c] = total_fitness

        new_grid = np.empty_like(self.grid)
        
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                neighborhood_coords = [(r, c)] + self._get_neighbor_coords(r, c)
                fittest_r, fittest_c = max(
                    neighborhood_coords, 
                    key=lambda coord: fitness_grid[coord[0], coord[1]]
                )
                fittest_agent = self.grid[fittest_r, fittest_c]
                
                new_genome = fittest_agent.genome.copy()
                mutated_genome = self._mutate_genome(new_genome)
                new_grid[r, c] = EvolvableAgent(genome=mutated_genome)

        self.grid = new_grid
        
        return fitness_grid
