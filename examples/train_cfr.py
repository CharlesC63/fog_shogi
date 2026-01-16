#!/usr/bin/env python3
"""
Example: Training a CFR Solver for Fog Shogi
============================================

This script demonstrates how to:
1. Train different CFR variants
2. Analyze convergence properties
3. Compare solver performance
4. Play games using trained strategies

Usage:
    python examples/train_cfr.py

Expected output: Training progress, regret bounds, and sample games.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fog_shogi.core.game import FogShogiGame
from fog_shogi.core.pieces import Player
from fog_shogi.algorithms.cfr import CFRSolver, CFRPlus
from fog_shogi.algorithms.mccfr import MCCFRSolver
from fog_shogi.analysis.regret import RegretAnalyzer, SampleComplexityAnalyzer


def train_and_analyze():
    """Train CFR solver and analyze results."""
    print("=" * 60)
    print("Fog of War Shogi - CFR Training")
    print("=" * 60)
    
    # Configuration
    iterations = 500
    max_depth = 4  # Limit tree depth for tractability
    
    print(f"\nTraining Configuration:")
    print(f"  Iterations: {iterations}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Abstraction Level: 2")
    
    # Initialize solver
    solver = CFRPlus(
        abstraction_level=2,
        max_depth=max_depth,
        use_pruning=True,
    )
    
    # Training with progress callback
    print("\nTraining Progress:")
    print("-" * 40)
    
    def progress_callback(iteration, exploitability):
        if iteration % 100 == 0:
            print(f"  Iteration {iteration:5d}: exploitability = {exploitability:.4f}")
    
    solver.train(iterations, progress_callback=progress_callback)
    
    # Analysis
    print("\n" + "=" * 60)
    print("Training Analysis")
    print("=" * 60)
    
    analyzer = RegretAnalyzer(solver)
    print(analyzer.generate_report())
    
    # Sample complexity analysis
    print("\nSample Complexity Analysis:")
    complexity = SampleComplexityAnalyzer.sample_complexity_analysis(solver)
    print(f"  Information Sets: {complexity['num_info_sets']}")
    print(f"  Max Actions: {complexity['max_actions']}")
    print("\n  Theoretical iterations for:")
    for level, iters in complexity['theoretical'].items():
        print(f"    {level}: {iters:,}")
    
    return solver


def play_sample_game(solver: CFRSolver):
    """Play a sample game using trained strategy."""
    print("\n" + "=" * 60)
    print("Sample Game")
    print("=" * 60)
    
    game = FogShogiGame()
    
    print("\nInitial State:")
    print(game)
    
    move_count = 0
    max_moves = 30
    
    while not game.is_terminal() and move_count < max_moves:
        # Get action from trained strategy
        state = game.get_full_state()
        action = solver.sample_action(state, game)
        
        player = game.get_current_player()
        print(f"\nMove {move_count + 1}: {player} plays {action}")
        
        game.apply_action(action)
        move_count += 1
        
        # Show belief state entropy
        belief = game.get_belief(player.opponent())
        print(f"  Opponent's uncertainty: {belief.total_entropy():.2f} bits")
    
    print("\n" + "-" * 40)
    if game.is_terminal():
        result = game.get_full_state().result
        print(f"Game Over: {result}")
    else:
        print(f"Game truncated at {max_moves} moves")
    
    print("\nFinal State:")
    print(game)


def compare_solvers():
    """Compare different CFR variants."""
    print("\n" + "=" * 60)
    print("Solver Comparison")
    print("=" * 60)
    
    iterations = 200
    max_depth = 3
    
    solvers = {
        "Vanilla CFR": CFRSolver(max_depth=max_depth),
        "CFR+": CFRPlus(max_depth=max_depth),
        "MCCFR": MCCFRSolver(max_depth=max_depth),
    }
    
    print(f"\nComparing with {iterations} iterations each...")
    print("-" * 40)
    
    results = {}
    for name, solver in solvers.items():
        print(f"\nTraining {name}...")
        solver.train(iterations)
        
        stats = solver.get_training_stats()
        results[name] = stats
        
        print(f"  Nodes created: {stats['nodes_created']}")
        print(f"  Est. exploitability: {stats['estimated_exploitability']:.4f}")
    
    print("\n" + "-" * 40)
    print("Comparison Summary:")
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Nodes: {stats['nodes_created']}")
        print(f"  Exploitability: {stats['estimated_exploitability']:.4f}")


def demonstrate_fog_of_war():
    """Demonstrate fog of war mechanics."""
    print("\n" + "=" * 60)
    print("Fog of War Demonstration")
    print("=" * 60)
    
    game = FogShogiGame(visibility_radius=1)  # Limited visibility
    
    print("\nTrue Board State (God's eye view):")
    print(game.get_full_state().board)
    
    print("\nSENTE's View (limited visibility):")
    print(game.get_observation(Player.SENTE))
    
    print("\nGOTE's View (limited visibility):")
    print(game.get_observation(Player.GOTE))
    
    print("\nSENTE's Belief State:")
    print(game.get_belief(Player.SENTE))


def main():
    """Main entry point."""
    print("Fog of War Shogi - Game Theory Research Framework")
    print("For Quantitative Finance Applications")
    print()
    
    # Demonstrate fog of war
    demonstrate_fog_of_war()
    
    # Train solver
    solver = train_and_analyze()
    
    # Play sample game
    play_sample_game(solver)
    
    # Compare solvers
    compare_solvers()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
