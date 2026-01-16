# Fog Shogi: Game Theory Research Framework

A research implementation of **Fog of War Shogi** (Mini-Shogi variant) with **Counterfactual Regret Minimization (CFR)** algorithms for computing Nash equilibria in imperfect information games.

## Overview

This project implements a game-theoretic framework for analyzing strategic decision-making under uncertainty. It combines:

- **Mini-Shogi (5×5)**: A simplified variant of Japanese chess with tractable state space
- **Fog of War Mechanics**: Imperfect information where players have limited visibility
- **CFR Algorithms**: State-of-the-art methods for computing optimal strategies

The framework is designed for quantitative finance applications, particularly for modeling:
- Strategic interaction under information asymmetry
- Optimal decision-making with uncertain state
- Nash equilibrium computation in competitive environments

## Features

### Game Implementation
- **Complete Mini-Shogi Rules**
  - 6 piece types: King, Gold, Silver, Bishop, Rook, Pawn
  - Piece promotion mechanics
  - Drop rules for captured pieces
  - Legal move generation with all Shogi constraints

- **Fog of War System**
  - Configurable visibility radius (default: 2 squares)
  - Players see only squares within radius of their pieces
  - Creates information asymmetry and strategic depth
  - Enables bluffing, inference, and deception

### Belief State Tracking
- **Bayesian Inference**: Probabilistic tracking of hidden pieces
- **Particle Filtering**: Monte Carlo sampling for belief updates
- **Information Metrics**: Entropy, expected information gain
- **Threat Assessment**: Probability-weighted threat evaluation

### CFR Algorithms

#### 1. Vanilla CFR
- Standard Counterfactual Regret Minimization
- Converges to ε-Nash equilibrium in O(1/√T) iterations
- Full game tree traversal with regret matching

#### 2. CFR+
- Enhanced variant with faster O(1/T) convergence
- Regret flooring and alternating updates
- Linear averaging for improved strategy quality

#### 3. Monte Carlo CFR (MCCFR)
- **External Sampling**: Sample opponent actions, explore own actions
- **Outcome Sampling**: Sample complete game trajectories
- O(|A|) per iteration vs. O(|A|^depth) for vanilla CFR
- Enables training on much larger state spaces

#### 4. Deep CFR (Skeleton)
- Neural network function approximation
- Generalizes across similar information sets
- Framework for future deep learning integration

### Analysis Tools
- **Regret Analysis**: Track regret bounds and convergence
- **Exploitability Computation**: Measure solution quality
- **Sample Complexity**: Theoretical iteration requirements
- **Training Statistics**: Nodes visited, strategy entropy, etc.

## Installation

### Prerequisites
- Python 3.8+
- NumPy

### Setup
```bash
# Clone the repository
git clone https://github.com/CharlesC63/fog_shogi.git
cd fog_shogi

# Install dependencies
pip install numpy

# Run tests
python tests/test_all.py

# Run example training
python examples/train_cfr.py
```

## Quick Start

### Basic Usage

```python
from fog_shogi.core.game import FogShogiGame
from fog_shogi.core.pieces import Player
from fog_shogi.algorithms.cfr import CFRPlus

# Create a new game
game = FogShogiGame(visibility_radius=2)

# Get legal actions
actions = game.get_legal_actions()

# Apply an action
action = actions[0]
game.apply_action(action)

# Get player's observation
observation = game.get_observation(Player.SENTE)
print(observation)  # Shows board with fog of war

# Get belief state
belief = game.get_belief(Player.SENTE)
print(f"Uncertainty: {belief.total_entropy():.2f} bits")
```

### Training a CFR Solver

```python
from fog_shogi.algorithms.cfr import CFRPlus

# Initialize solver
solver = CFRPlus(
    abstraction_level=2,    # Information set bucketing
    max_depth=4,            # Tree depth limit
    use_pruning=True        # Prune low-probability branches
)

# Train with progress tracking
def progress_callback(iteration, exploitability):
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: exploitability = {exploitability:.4f}")

solver.train(iterations=1000, progress_callback=progress_callback)

# Get trained strategy
state = game.get_full_state()
strategy = solver.get_strategy(state, game)
print(f"Action probabilities: {strategy}")

# Sample an action
action = solver.sample_action(state, game)
game.apply_action(action)
```

### Playing a Complete Game

```python
# Play game with trained strategy
game = FogShogiGame()
move_count = 0

while not game.is_terminal() and move_count < 50:
    state = game.get_full_state()
    player = game.get_current_player()

    # Sample action from trained strategy
    action = solver.sample_action(state, game)
    print(f"Move {move_count + 1}: {player} plays {action}")

    game.apply_action(action)
    move_count += 1

# Check result
if game.is_terminal():
    result = game.get_full_state().result
    print(f"Game Over: {result}")
```

## Project Structure

```
fog_shogi/
├── core/                   # Core game implementation
│   ├── pieces.py          # Piece types and movement patterns
│   ├── board.py           # Board with fog of war visibility
│   ├── game.py            # Game state and action logic
│   └── belief.py          # Bayesian belief state tracking
├── algorithms/            # CFR implementations
│   ├── cfr.py            # Vanilla CFR and CFR+
│   └── mccfr.py          # Monte Carlo CFR variants
├── analysis/             # Analysis and visualization
│   └── regret.py         # Regret analysis and metrics
├── examples/             # Example scripts
│   └── train_cfr.py      # Training demonstration
└── tests/                # Unit tests
    └── test_all.py       # Comprehensive test suite
```

## Game Rules: Mini-Shogi

### Board Setup
```
  0   1   2   3   4
0 -飛 -角 -銀 -金 -玉    GOTE's back rank
1  ▓  ▓  -歩  ▓  ▓     GOTE's pawn
2  ·   ·   ·   ·   ·    Empty row
3  ▓  ▓  +歩  ▓  ▓     SENTE's pawn
4 +玉 +金 +銀 +角 +飛    SENTE's back rank
```

### Piece Movement

| Piece | Symbol | Moves | Can Promote |
|-------|--------|-------|-------------|
| King (玉) | K | 1 square any direction | No |
| Gold (金) | G | 1 square orthogonal + forward diagonal | No |
| Silver (銀) | S | 1 square diagonal + forward | Yes → Gold |
| Bishop (角) | B | Diagonal any distance | Yes → +orthogonal |
| Rook (飛) | R | Orthogonal any distance | Yes → +diagonal |
| Pawn (歩) | P | 1 square forward | Yes → Gold |

### Special Rules
- **Promotion**: Pieces promote when entering opponent's back rank (row 0 for SENTE, row 4 for GOTE)
- **Drops**: Captured pieces can be dropped on empty squares
- **Two-Pawn Rule**: Can't have two unpromoted pawns in the same column
- **Win Condition**: Capture opponent's King

### Fog of War Rules
- Each player sees squares within **visibility radius** (default: 2) of their pieces
- Players always see their own pieces
- Hidden squares show fog (▓)
- Creates **imperfect information** game structure

## Theoretical Background

### Counterfactual Regret Minimization

CFR is an iterative algorithm that computes Nash equilibria in extensive-form games. Key concepts:

**Regret**: For each action, the difference between its value and the value of the current strategy.

```
R_i^T(a) = Σ_{t=1}^T [u_i(a, σ^t_{-i}) - u_i(σ^t)]
```

**Regret Matching**: Convert positive regrets to action probabilities:

```
σ^{t+1}(a) ∝ max(R_i^t(a), 0)
```

**Convergence**: Average regret decreases as O(1/√T), guaranteeing convergence to ε-Nash equilibrium.

### Information Sets

An **information set** groups all game states that are observationally equivalent to a player. In Fog Shogi:
- Same visible pieces and positions
- Same hand pieces
- Different only in hidden information

Players must use the same strategy at all states in an information set.

### Belief States

A **belief state** is a probability distribution over possible true game states, given observations. Updated via Bayes' rule:

```
P(state | obs) ∝ P(obs | state) × P(state)
```

We track beliefs using particle filtering for computational tractability.

## Performance Characteristics

### State Space Complexity
- **Perfect Information Mini-Shogi**: ~10^18 states
- **With Fog of War**: Exponentially larger due to information sets
- **Abstraction Required**: Bucketing similar information sets

### Computational Requirements

| Solver | Per-Iteration | Memory | Convergence |
|--------|--------------|--------|-------------|
| Vanilla CFR | O(\|A\|^depth) | O(\|I\| × \|A\|) | O(1/√T) |
| CFR+ | O(\|A\|^depth) | O(\|I\| × \|A\|) | O(1/T) |
| MCCFR | O(depth × \|A\|) | O(\|I\| × \|A\|) | O(1/√T) |

Where:
- |I| = number of information sets
- |A| = average number of actions
- depth = maximum game tree depth
- T = number of iterations

### Recommended Training

```python
# Quick testing (5 minutes)
solver = CFRPlus(max_depth=3, abstraction_level=2)
solver.train(iterations=500)

# Production quality (1-2 hours)
solver = CFRPlus(max_depth=5, abstraction_level=2)
solver.train(iterations=10000)

# Research quality (24+ hours)
solver = CFRPlus(max_depth=7, abstraction_level=1)
solver.train(iterations=100000)
```

## Applications

### Quantitative Finance
This framework models:
- **Market Making**: Information asymmetry between informed/uninformed traders
- **Adversarial Trading**: Strategic order placement in competitive environments
- **Risk Management**: Optimal decisions under uncertain market state

### Research Areas
- Game theory under imperfect information
- Bayesian inference and belief tracking
- Nash equilibrium computation
- Multi-agent reinforcement learning
- Information value and exploration strategies

## Testing

Run the test suite:

```bash
python tests/test_all.py
```

Tests cover:
- Piece movement and promotion
- Legal move generation
- Fog of war visibility
- Belief state updates
- CFR algorithm correctness
- Strategy convergence properties

## References

### Algorithms
1. **CFR**: Zinkevich et al., "Regret Minimization in Games with Incomplete Information" (NIPS 2007)
2. **CFR+**: Tammelin et al., "Solving Heads-Up Limit Texas Hold'em" (IJCAI 2015)
3. **MCCFR**: Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive Games" (NeurIPS 2009)
4. **Deep CFR**: Brown et al., "Deep Counterfactual Regret Minimization" (ICML 2019)

### Game Theory
- von Neumann & Morgenstern, "Theory of Games and Economic Behavior" (1944)
- Nash, "Non-Cooperative Games" (1951)
- Koller & Megiddo, "The Complexity of Two-Person Zero-Sum Games in Extensive Form" (1992)

### Applications
- Silver et al., "A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go" (Science 2018)
- Brown & Sandholm, "Superhuman AI for Multiplayer Poker" (Science 2019)

## License

This project is for research and educational purposes. See LICENSE for details.

## Contributing

Contributions welcome! Areas for improvement:
- Additional CFR variants (DCFR, LCFR)
- Neural network integration for Deep CFR
- Visualization tools for game trees
- Performance optimizations
- Extended analysis tools

## Contact

For questions or collaboration:
- GitHub: [@CharlesC63](https://github.com/CharlesC63)
- Repository: [fog_shogi](https://github.com/CharlesC63/fog_shogi)

## Acknowledgments

This implementation draws inspiration from:
- OpenSpiel (DeepMind)
- ReBeL (Facebook AI Research)
- Pluribus poker bot (CMU/Facebook)
- Game theory research at Stanford, CMU, and UC Berkeley

---

Built for game theory research and quantitative finance applications.
