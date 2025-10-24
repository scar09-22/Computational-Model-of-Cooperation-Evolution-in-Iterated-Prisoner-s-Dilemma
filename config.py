# Simulation Parameters
POPULATION_SIZE = 50
GENERATIONS = 150
ROUNDS_PER_MATCH = 200

# Simulation Type
SIMULATION_TYPE = 'tournament' # 'spatial' for spatial 

# Spatial Grid Parameters
GRID_WIDTH = 20
GRID_HEIGHT = 20

# Genetic Algorithm Parameters (for tournament simulation)
MUTATION_RATE = 0.01
TOURNAMENT_SIZE = 5

# Spatial Simulation Parameters
SPATIAL_MUTATION_RATE = 0.01

# Game Payoff Matrix
PAYOFF = {
    ('C', 'C'): (5, 5),
    ('D', 'D'): (1, 1),
    ('D', 'C'): (3, 0),
    ('C', 'D'): (0, 5)
}