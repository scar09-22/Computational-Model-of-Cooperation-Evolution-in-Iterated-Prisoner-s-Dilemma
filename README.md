# Computational Model of Cooperation Evolution in Iterated Prisoner's Dilemma

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> Agent-based evolutionary simulation demonstrating cooperation dynamics through spatial and tournament modes of the Iterated Prisoner's Dilemma with real-time visualization

## Overview

This project implements a sophisticated evolutionary game theory simulator that models how cooperation emerges and persists in competitive environments. By simulating agents playing the Iterated Prisoner's Dilemma across multiple generations, the system demonstrates how spatial structure and mutation rates fundamentally affect evolutionary outcomes.

### Key Features

- **Dual Simulation Modes**: Spatial (grid-based with Moore neighborhood) and Tournament (well-mixed population)
- **Real-Time Animated Visualization**: Live spatial grid and bar chart animations showing strategy evolution
- **Genetic Algorithm Implementation**: Tournament selection with configurable mutation rates
- **4-Bit Genome Encoding**: Reactive strategies based on outcome pairs (CC, CD, DC, DD)
- **16 Strategy Types**: From always-cooperate (CCCC) to always-defect (DDDD)
- **Configurable Parameters**: Mutation rates, population size, grid dimensions, payoff matrices

## Project Highlights

### Research Findings

Our simulations reveal three distinct evolutionary regimes:

| Condition | Result | Stability | Dominant Strategy |
|-----------|--------|-----------|-------------------|
| **Spatial + High Mutation (10%)** | 15-16 coexisting strategies | Stable diversity | CDDD/CDDC/CDCD cycling |
| **Tournament + Low Mutation (1%)** | Near-monoculture (90% CDDC) | Highly stable | CDDC (Forgiving) |
| **Tournament + High Mutation (10%)** | Unstable cycling | Chaotic | No convergence |

### Key Insights

- **Spatial structure enables cooperation**: Even with 10% mutation rate, spatial clustering maintains 15-16 unique strategies
- **CDDC dominates in tournaments**: Forgiving strategies outcompete strict Tit-for-Tat (CDCD) in low-mutation environments
- **Mutation rate is critical**: 1% vs 10% mutation completely changes evolutionary outcomes (stable vs chaotic)
- **Average fitness varies by mode**: Spatial ~6,500 | Tournament (low mutation) ~1,000 | Tournament (high mutation) ~100-1,000

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone git@github.com:scar09-22/Computational-Model-of-Cooperation-Evolution-in-Iterated-Prisoner-s-Dilemma.git
cd Computational-Model-of-Cooperation-Evolution-in-Iterated-Prisoner-s-Dilemma
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Quick Start

Run the default spatial simulation:
```bash
python3 main.py
```

### Configuration

Edit `config.py` to customize parameters:

```python
# Simulation mode: 'spatial' or 'tournament'
SIMULATION_TYPE = 'spatial'

# Population parameters
POPULATION_SIZE = 50
GRID_WIDTH = 20
GRID_HEIGHT = 20

# Evolution parameters
GENERATIONS = 150
ROUNDS_PER_MATCH = 200
MUTATION_RATE = 0.01  # For tournament mode
SPATIAL_MUTATION_RATE = 0.1  # For spatial mode

# Payoff matrix (T > R > P > S)
PAYOFF = {
    ('C', 'C'): (4, 4),  # Reward for mutual cooperation
    ('D', 'D'): (1, 1),  # Punishment for mutual defection
    ('D', 'C'): (5, 0),  # Temptation to defect / Sucker's payoff
    ('C', 'D'): (0, 5)
}
```

### Running Different Modes

**Spatial Mode** (default):
- Agents arranged on 20x20 grid
- Interact only with 8 neighbors (Moore neighborhood)
- High mutation rate (0.1) maintains diversity
- Real-time grid visualization with color-coded strategies

**Tournament Mode**:
- Well-mixed population (round-robin)
- Every agent plays every other agent
- Low mutation rate (0.01) for stability
- Real-time bar chart showing strategy distribution

## Strategy Encoding

Each agent has a 4-bit genome `[CC, CD, DC, DD]` representing responses to outcome pairs:

- **CCCC**: Always Cooperate
- **CDCD**: Tit-for-Tat variant
- **CDDC**: Forgiving strategy (tournament winner)
- **CDDD**: Grim Trigger
- **DDDD**: Always Defect

Color coding in visualizations:
- Green: Cooperative (CC**)
- Blue: Forgiving/TFT (CD**)
- Gold/Orange: Exploitative (DC**)
- Red/Black: Defectors (DD**)

## Results

### Spatial Evolution
- Maintains 15-16 unique strategies across 150 generations
- Average fitness: 6,200-6,900 (81-86% of theoretical maximum)
- Dynamic equilibrium with cycling dominant strategies
- Demonstrates Red Queen dynamics

### Tournament Evolution
- Converges to 90% CDDC (forgiving strategy)
- Average fitness: ~1,000 (near-optimal cooperation)
- Occasional defector invasions quickly suppressed
- Validates importance of forgiveness over strict retaliation

## Project Structure

```
.
├── main.py                    # Entry point with visualization
├── config.py                  # Configuration parameters
├── src/
│   ├── agent.py              # EvolvableAgent class with 4-bit genome
│   ├── simulation.py         # Tournament simulation (round-robin)
│   ├── spatial_simulation.py # Spatial grid simulation
│   └── genetic_algorithm.py  # GA with tournament selection
└── README.md
```

## Scientific Context

This project builds on:
- **Axelrod's Tournaments (1980)**: Original Tit-for-Tat findings in well-mixed populations
- **Nowak & May (1992)**: Spatial structure promoting cooperation
- **Modern evolutionary game theory**: Mutation-selection balance and spatial clustering

### Differences from Axelrod

| Aspect | Axelrod (1980) | This Project |
|--------|---------------|--------------|
| Population | Fixed, expert-designed strategies | Evolving, random initial genomes |
| Structure | Well-mixed only | Both spatial and well-mixed |
| Mutation | None | Configurable (0.01-0.1) |
| Winner | TFT (CDCD) | CDDC in tournaments, diversity in spatial |
| Encoding | Memory-based | Outcome-based (4-bit genome) |

## Technical Details

### Performance
- Processes 640,000+ interactions per generation (400 agents × 8 neighbors × 200 rounds)
- Real-time animation at 100ms intervals
- Efficient numpy-based grid operations

### Technologies
- **Python 3.8+**: Core language
- **NumPy**: Grid operations and numerical computing
- **Matplotlib**: Real-time animations and plotting
- **Object-Oriented Design**: Modular, extensible architecture

## Future Enhancements

- Network topologies (small-world, scale-free)
- Alternative payoff matrices and noise
- Memory-based strategies (Pavlov, Generous TFT)
- Performance optimization with Numba/Cython
- Web-based visualization with interactive controls
- Statistical analysis suite (Shannon diversity, spatial autocorrelation)

## References

1. Axelrod, R. (1984). *The Evolution of Cooperation*
2. Nowak, M. A., & May, R. M. (1992). Evolutionary games and spatial chaos. *Nature*
3. Hauert, C., & Doebeli, M. (2004). Spatial structure often inhibits the evolution of cooperation. *Proceedings of the Royal Society B*

## Author

**Shiva**
- GitHub: [@scar09-22](https://github.com/scar09-22)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Robert Axelrod's pioneering work on cooperation
- Built as a demonstration of evolutionary game theory principles
- Thanks to the open-source Python community

---

**If you find this project interesting, please star the repository!**

*For questions or suggestions, feel free to open an issue or reach out!*
