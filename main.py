import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter
from config import *
from src.genetic_algorithm import GeneticAlgorithm
from src.simulation import Simulation
from src.agent import EvolvableAgent

if SIMULATION_TYPE == 'spatial':
    from src.spatial_simulation import SpatialSimulation

sim = None
im = None
avg_fitness_history = []
gen_text = None
fitness_text = None
diversity_text = None

tournament_ga = None
tournament_sim = None
tournament_fitness_history = []
tournament_best_history = []
bar_container = None
gen_text_tournament = None
fitness_text_tournament = None

STRATEGY_COLORS = {
    'CCCC': '#00FF00',  # Bright Green
    'CCCD': '#7FFF00',  # Chartreuse
    'CCDC': '#90EE90',  # Light Green
    'CCDD': '#32CD32',  # Lime Green
    'CDCC': '#00BFFF',  # Deep Sky Blue
    'CDCD': '#1E90FF',  # Dodger Blue (TFT)
    'CDDC': '#4169E1',  # Royal Blue
    'CDDD': '#0000CD',  # Medium Blue (Grim Trigger)
    'DCCC': '#FFD700',  # Gold
    'DCCD': '#FFA500',  # Orange
    'DCDC': '#FF8C00',  # Dark Orange
    'DCDD': '#FF4500',  # Orange Red
    'DDCC': '#FF0000',  # Red
    'DDCD': '#DC143C',  # Crimson
    'DDDC': '#8B0000',  # Dark Red
    'DDDD': '#000000'   # Black
}

STRATEGY_TO_NUM = {strategy: i for i, strategy in enumerate(sorted(STRATEGY_COLORS.keys()))}
COLORS = [STRATEGY_COLORS[s] for s in sorted(STRATEGY_COLORS.keys())]


def init_animation():
    global im, gen_text, fitness_text, diversity_text
    im.set_data(np.zeros((sim.grid_height, sim.grid_width)))
    gen_text.set_text('')
    fitness_text.set_text('')
    diversity_text.set_text('')
    return im, gen_text, fitness_text, diversity_text


def update_animation(frame):
    global sim, im, avg_fitness_history, gen_text, fitness_text, diversity_text
    
    fitness_grid = sim.run_step()
    
    avg_fitness = np.mean(fitness_grid)
    best_fitness = np.max(fitness_grid)
    avg_fitness_history.append(avg_fitness)
    
    strategies = [agent.get_name() for agent in sim.grid.flatten()]
    strategy_counts = Counter(strategies)
    num_unique_strategies = len(strategy_counts)
    top_3 = strategy_counts.most_common(3)
    most_common_strategy = top_3[0][0]
    
    print(f"Gen {frame+1:3}: Best={best_fitness:.0f}, Avg={avg_fitness:.0f}, "
          f"Dominant={most_common_strategy}, Unique={num_unique_strategies}")
    
    numerical_grid = np.zeros((sim.grid_height, sim.grid_width), dtype=int)
    for r in range(sim.grid_height):
        for c in range(sim.grid_width):
            strategy = sim.grid[r, c].get_name()
            numerical_grid[r, c] = STRATEGY_TO_NUM[strategy]
    
    im.set_data(numerical_grid)
    
    gen_text.set_text(f'Generation: {frame+1}')
    fitness_text.set_text(f'Avg: {avg_fitness:.0f} | Best: {best_fitness:.0f}')
    
    diversity_str = ', '.join([f"{s}: {c}" for s, c in top_3])
    diversity_text.set_text(f'Top 3: {diversity_str} | Unique: {num_unique_strategies}')
    
    return im, gen_text, fitness_text, diversity_text


def run_spatial_simulation():
    global sim, im, gen_text, fitness_text, diversity_text, avg_fitness_history
    
    avg_fitness_history = []
    
    grid_size = GRID_WIDTH * GRID_HEIGHT
    agents = []
    
    key_strategies = {
        'CCCC': 30,
        'CDCD': 30,
        'CDDC': 30,
        'CDDD': 30,
        'DCCC': 20,
        'DDDD': 20,
    }
    
    for strategy, count in key_strategies.items():
        for _ in range(count):
            genome = list(strategy)
            agents.append(EvolvableAgent(genome=genome))
    
    while len(agents) < grid_size:
        agents.append(EvolvableAgent())
    
    np.random.shuffle(agents)
    
    sim = SpatialSimulation()
    sim.populate_grid(agents)

    print(f"Starting spatial evolution with seeded diversity for {GENERATIONS} generations...")
    print(f"Initial strategies: {list(key_strategies.keys())}")
    
    fig = plt.figure(figsize=(14, 13))
    fig.suptitle('Spatial Evolution of Cooperation', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    ax = fig.add_subplot(111)
    
    cmap = mcolors.ListedColormap(COLORS)
    
    initial_grid = np.zeros((sim.grid_height, sim.grid_width), dtype=int)
    im = ax.imshow(initial_grid, cmap=cmap, interpolation='nearest', 
                   aspect='equal', vmin=0, vmax=15)
    
    ax.set_xticks(np.arange(-0.5, sim.grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sim.grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8, alpha=0.4)
    
    gen_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white', 
                       fontsize=16, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    fitness_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='white',
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    diversity_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, color='white',
                            fontsize=11, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(STRATEGY_TO_NUM)), 
                       shrink=0.5, pad=0.02)
    cbar.ax.set_yticklabels(sorted(STRATEGY_TO_NUM.keys()), fontsize=9)
    cbar.set_label('Strategy', rotation=270, labelpad=25, fontsize=11)

    legend_text = (
        "Strategy Colors:\n\n"
        "Green: Cooperative (CC**)\n"
        "Blue: Forgiving/TFT (CD**)\n"
        "Gold/Orange: Exploitative (DC**)\n"
        "Red/Black: Defectors (DD**)"
    )
    ax.text(1.18, 0.5, legend_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

    ani = animation.FuncAnimation(fig, update_animation, init_func=init_animation,
                                 frames=GENERATIONS, interval=100, blit=True, repeat=False)

    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.85)
    plt.show()
    
    plot_results(avg_fitness_history)


def init_tournament_animation():
    global bar_container, gen_text_tournament, fitness_text_tournament
    for rect in bar_container:
        rect.set_height(0)
    gen_text_tournament.set_text('')
    fitness_text_tournament.set_text('')
    return list(bar_container) + [gen_text_tournament, fitness_text_tournament]


def update_tournament_animation(frame):
    global tournament_ga, tournament_sim, bar_container, gen_text_tournament, fitness_text_tournament
    global tournament_fitness_history, tournament_best_history
    
    tournament_sim.run_tournament(tournament_ga.population)
    
    best_fitness = max(agent.fitness for agent in tournament_ga.population)
    avg_fitness = sum(agent.fitness for agent in tournament_ga.population) / POPULATION_SIZE
    
    tournament_fitness_history.append(avg_fitness)
    tournament_best_history.append(best_fitness)
    
    strategies = [agent.get_name() for agent in tournament_ga.population]
    strategy_counts = Counter(strategies)
    most_common = strategy_counts.most_common(1)[0]
    
    print(f"Gen {frame+1:3}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, "
          f"Dominant={most_common[0]} ({most_common[1]}/{POPULATION_SIZE})")
    
    all_strategies = sorted(STRATEGY_TO_NUM.keys())
    counts = [strategy_counts.get(s, 0) for s in all_strategies]
    
    for rect, count in zip(bar_container, counts):
        rect.set_height(count)
    
    gen_text_tournament.set_text(f'Generation: {frame+1}')
    fitness_text_tournament.set_text(
        f'Avg: {avg_fitness:.0f} | Best: {best_fitness:.0f} | Dominant: {most_common[0]} ({most_common[1]})')
    
    tournament_ga.evolve()
    
    return list(bar_container) + [gen_text_tournament, fitness_text_tournament]


def run_tournament_animation():
    global tournament_ga, tournament_sim, bar_container, gen_text_tournament, fitness_text_tournament
    global tournament_fitness_history, tournament_best_history
    
    tournament_fitness_history = []
    tournament_best_history = []
    
    tournament_ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        tournament_size=TOURNAMENT_SIZE
    )
    tournament_sim = Simulation(rounds_per_match=ROUNDS_PER_MATCH)
    
    print(f"Starting tournament evolution with animation for {GENERATIONS} generations...")
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Tournament Evolution: Strategy Distribution', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    ax = fig.add_subplot(111)
    
    all_strategies = sorted(STRATEGY_TO_NUM.keys())
    x_pos = np.arange(len(all_strategies))
    colors_list = [STRATEGY_COLORS[s] for s in all_strategies]
    
    bar_container = ax.bar(x_pos, np.zeros(len(all_strategies)), color=colors_list, 
                           alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_strategies, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_ylim(0, POPULATION_SIZE + 5)
    ax.set_xlim(-0.5, len(all_strategies) - 0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Strategy Count in Population', fontsize=14, pad=15)
    
    gen_text_tournament = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                                  fontsize=14, verticalalignment='top', fontweight='bold',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fitness_text_tournament = ax.text(0.02, 0.92, '', transform=ax.transAxes,
                                     fontsize=11, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ani = animation.FuncAnimation(fig, update_tournament_animation, 
                                 init_func=init_tournament_animation,
                                 frames=GENERATIONS, interval=100, blit=True, repeat=False)
    
    plt.subplots_adjust(top=0.93, bottom=0.15, left=0.08, right=0.95)
    plt.show()
    
    final_strategies = [agent.get_name() for agent in tournament_ga.population]
    final_counts = Counter(final_strategies)
    print("\nFinal Strategy Distribution:")
    for strategy, count in final_counts.most_common():
        percentage = (count / POPULATION_SIZE) * 100
        print(f"  {strategy}: {count} agents ({percentage:.1f}%)")
    
    plot_results(tournament_fitness_history, tournament_best_history)


def run_evolution():
    if SIMULATION_TYPE == 'spatial':
        run_spatial_simulation()
    else:
        run_tournament_animation()


def plot_results(avg_fitness, best_fitness=None):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(avg_fitness)), avg_fitness, label='Average Fitness', 
             color='blue', linewidth=2, alpha=0.7)
    if best_fitness is not None:
        plt.plot(range(len(best_fitness)), best_fitness, label='Best Fitness', 
                color='green', linewidth=2, alpha=0.7)
    plt.title('Fitness Evolution Over Generations', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evolution()
