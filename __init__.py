"""
Fog of War Shogi - Imperfect Information Game Theory Framework
==============================================================

A Python implementation of Shogi with fog of war mechanics and
game-theoretic equilibrium computation via Counterfactual Regret
Minimization (CFR).

Key Features:
-------------
1. Mini-Shogi (5x5 board) for tractable state space
2. Configurable fog of war (visibility radius)
3. Bayesian belief state tracking with particle filtering
4. Multiple CFR variants: Vanilla, CFR+, Monte Carlo CFR
5. Theoretical regret bounds and convergence analysis

Quant Finance Relevance:
------------------------
- Imperfect information → incomplete market information
- Belief tracking → Bayesian filtering in trading
- Regret minimization → online learning algorithms
- Nash equilibrium → market equilibrium concepts
- Exploitability analysis → adversarial robustness

Usage:
------
    from fog_shogi import FogShogiGame
    from fog_shogi.algorithms import CFRPlus
    from fog_shogi.analysis import RegretAnalyzer
    
    # Create and train solver
    solver = CFRPlus(abstraction_level=2, max_depth=4)
    solver.train(iterations=1000)
    
    # Analyze convergence
    analyzer = RegretAnalyzer(solver)
    print(analyzer.generate_report())
    
    # Play with fog of war
    game = FogShogiGame()
    state = game.get_full_state()
    action = solver.sample_action(state, game)
    game.apply_action(action)

Author: AI-Generated for Portfolio Demonstration
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Portfolio Project"

from fog_shogi.core.game import FogShogiGame
from fog_shogi.core.board import Board
from fog_shogi.core.pieces import Piece, PieceType, Player
from fog_shogi.core.belief import BeliefState, PieceDistribution

__all__ = [
    "FogShogiGame",
    "Board", 
    "Piece",
    "PieceType",
    "Player",
    "BeliefState",
    "PieceDistribution",
]
