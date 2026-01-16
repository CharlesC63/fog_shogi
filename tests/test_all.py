"""
Test Suite for Fog of War Shogi
===============================

Run with: python -m pytest fog_shogi/tests/test_all.py -v
"""

import pytest
import numpy as np
from typing import List

# Core imports
from fog_shogi.core.pieces import Piece, PieceType, Player, get_movement_deltas
from fog_shogi.core.board import Board, BoardView, Square, BOARD_SIZE, create_initial_board
from fog_shogi.core.belief import BeliefState, InformationSet, PieceDistribution
from fog_shogi.core.game import (
    FogShogiGame, GameState, Action, ActionType, GameResult
)

# Algorithm imports
from fog_shogi.algorithms.cfr import CFRSolver, CFRPlus, CFRNode
from fog_shogi.algorithms.mccfr import MCCFRSolver

# Analysis imports
from fog_shogi.analysis.regret import RegretAnalyzer, SampleComplexityAnalyzer


class TestPieces:
    """Test piece mechanics."""
    
    def test_player_opponent(self):
        assert Player.SENTE.opponent() == Player.GOTE
        assert Player.GOTE.opponent() == Player.SENTE
    
    def test_piece_promotion(self):
        pawn = Piece(PieceType.PAWN, Player.SENTE, 0)
        promoted = pawn.promoted()
        
        assert promoted is not None
        assert promoted.piece_type == PieceType.TOKIN
        assert promoted.owner == Player.SENTE
        assert promoted.piece_id == 0
    
    def test_piece_demotion(self):
        tokin = Piece(PieceType.TOKIN, Player.SENTE, 0)
        demoted = tokin.demoted()
        
        assert demoted.piece_type == PieceType.PAWN
        assert demoted.owner == Player.GOTE  # Changes owner
    
    def test_king_cannot_promote(self):
        king = Piece(PieceType.KING, Player.SENTE, 0)
        assert king.promoted() is None
    
    def test_movement_deltas(self):
        pawn_sente = Piece(PieceType.PAWN, Player.SENTE, 0)
        pawn_gote = Piece(PieceType.PAWN, Player.GOTE, 1)
        
        sente_moves = get_movement_deltas(pawn_sente)
        gote_moves = get_movement_deltas(pawn_gote)
        
        # Pawns move in opposite directions
        assert sente_moves[0][0] == -1  # SENTE moves up
        assert gote_moves[0][0] == 1   # GOTE moves down


class TestBoard:
    """Test board mechanics."""
    
    def test_initial_board(self):
        board = create_initial_board()
        
        # Check kings are placed
        sente_king = board.find_king(Player.SENTE)
        gote_king = board.find_king(Player.GOTE)
        
        assert sente_king is not None
        assert gote_king is not None
        assert sente_king.row == 4
        assert gote_king.row == 0
    
    def test_visibility(self):
        board = create_initial_board()
        board.visibility_radius = 2
        
        visible_sente = board.get_visible_squares(Player.SENTE)
        visible_gote = board.get_visible_squares(Player.GOTE)
        
        # Each player should see different squares
        assert len(visible_sente) > 0
        assert len(visible_gote) > 0
    
    def test_fog_of_war(self):
        board = create_initial_board()
        board.visibility_radius = 1  # Limited visibility
        
        view = board.get_player_view(Player.SENTE)
        
        # Should have some fog
        fog_count = view.count_fog_squares()
        assert fog_count > 0
    
    def test_piece_capture(self):
        board = create_initial_board()
        
        # Move piece to hand
        pawn = Piece(PieceType.PAWN, Player.GOTE, 99)
        board.add_to_hand(Player.SENTE, pawn.demoted())
        
        hand = board.get_hand(Player.SENTE)
        assert len(hand) == 1
        assert hand[0].owner == Player.SENTE


class TestGame:
    """Test game mechanics."""
    
    def test_initial_state(self):
        game = FogShogiGame()
        
        assert game.get_current_player() == Player.SENTE
        assert not game.is_terminal()
    
    def test_legal_actions(self):
        game = FogShogiGame()
        actions = game.get_legal_actions()
        
        # Should have multiple legal moves at start
        assert len(actions) > 0
    
    def test_action_application(self):
        game = FogShogiGame()
        actions = game.get_legal_actions()
        
        initial_player = game.get_current_player()
        game.apply_action(actions[0])
        
        # Player should switch
        assert game.get_current_player() != initial_player
    
    def test_promotion_zone(self):
        sq_top = Square(0, 2)
        sq_bottom = Square(4, 2)
        
        assert sq_top.in_promotion_zone(Player.SENTE)
        assert sq_bottom.in_promotion_zone(Player.GOTE)
    
    def test_game_termination(self):
        # Play random game to test termination detection
        game = FogShogiGame()
        
        for _ in range(50):  # Limit moves
            if game.is_terminal():
                break
            
            actions = game.get_legal_actions()
            if actions:
                action = np.random.choice(actions)
                game.apply_action(action)
        
        # Game should either terminate or have actions
        assert game.is_terminal() or len(game.get_legal_actions()) > 0


class TestBelief:
    """Test belief state mechanics."""
    
    def test_initial_belief(self):
        board = create_initial_board()
        belief = BeliefState.from_initial_position(Player.SENTE, board)
        
        # Should track opponent pieces
        assert len(belief.opponent_piece_beliefs) > 0
    
    def test_belief_entropy(self):
        board = create_initial_board()
        belief = BeliefState.from_initial_position(Player.SENTE, board)
        
        # Initially entropy should be 0 (positions known)
        entropy = belief.total_entropy()
        assert entropy == 0.0  # Known positions
    
    def test_piece_distribution(self):
        dist = PieceDistribution(
            piece_id=0,
            piece_type=PieceType.PAWN,
            owner=Player.GOTE,
            location_probs={Square(1, 2): 0.5, Square(2, 2): 0.5}
        )
        
        assert dist.entropy() > 0  # Uncertainty
        
        loc, prob = dist.most_likely_location()
        assert prob == 0.5
    
    def test_information_set(self):
        game = FogShogiGame()
        obs = game.get_observation(Player.SENTE)
        
        info_set = InformationSet.from_observation(obs, [])
        
        # Same observation should give same info set
        info_set2 = InformationSet.from_observation(obs, [])
        assert hash(info_set) == hash(info_set2)


class TestCFR:
    """Test CFR algorithm."""
    
    def test_cfr_node_strategy(self):
        node = CFRNode(
            info_set=None,  # type: ignore
            num_actions=3,
        )
        
        # Initially uniform
        strategy = node.get_strategy()
        assert len(strategy) == 3
        assert np.allclose(strategy, [1/3, 1/3, 1/3])
    
    def test_cfr_regret_update(self):
        node = CFRNode(
            info_set=None,  # type: ignore
            num_actions=3,
        )
        
        # Simulate regret update
        action_utilities = np.array([1.0, -0.5, 0.0])
        strategy = node.get_strategy()
        node.update_regrets(action_utilities, strategy)
        
        # Strategy should shift toward high-utility action
        new_strategy = node.get_strategy()
        assert new_strategy[0] > new_strategy[1]
    
    def test_cfr_training(self):
        solver = CFRSolver(max_depth=3)
        solver.train(iterations=10)
        
        assert solver.iterations_completed == 10
        assert len(solver.nodes) > 0
    
    def test_cfr_strategy_retrieval(self):
        solver = CFRSolver(max_depth=3)
        solver.train(iterations=10)
        
        game = FogShogiGame()
        strategy = solver.get_strategy(game.get_full_state(), game)
        
        # Should return valid probability distribution
        assert len(strategy) > 0
        total_prob = sum(strategy.values())
        assert np.isclose(total_prob, 1.0)


class TestMCCFR:
    """Test Monte Carlo CFR."""
    
    def test_mccfr_training(self):
        solver = MCCFRSolver(max_depth=3)
        solver.train(iterations=20)
        
        assert solver.iterations_completed == 20
    
    def test_mccfr_sampling(self):
        solver = MCCFRSolver(max_depth=3)
        solver.train(iterations=20)
        
        game = FogShogiGame()
        action = solver.sample_action(game.get_full_state(), game)
        
        # Should return valid action
        assert action in game.get_legal_actions()


class TestAnalysis:
    """Test analysis tools."""
    
    def test_regret_bounds(self):
        solver = CFRSolver(max_depth=3)
        solver.train(iterations=50)
        
        analyzer = RegretAnalyzer(solver)
        bounds = analyzer.compute_regret_bounds()
        
        assert bounds.iterations == 50
        assert bounds.upper_bound > 0
    
    def test_sample_complexity(self):
        iters = SampleComplexityAnalyzer.iterations_for_epsilon(
            epsilon=0.1,
            num_info_sets=100,
            max_actions=10
        )
        
        assert iters > 0
    
    def test_analysis_report(self):
        solver = CFRSolver(max_depth=3)
        solver.train(iterations=50)
        
        analyzer = RegretAnalyzer(solver)
        report = analyzer.generate_report()
        
        assert "Regret" in report
        assert "Exploitability" in report


class TestIntegration:
    """Integration tests for full game play."""
    
    def test_full_game_with_cfr(self):
        # Train a quick solver
        solver = CFRSolver(max_depth=3)
        solver.train(iterations=20)
        
        # Play a game using trained strategy
        game = FogShogiGame()
        
        for _ in range(20):
            if game.is_terminal():
                break
            
            action = solver.sample_action(game.get_full_state(), game)
            game.apply_action(action)
        
        # Game should progress without errors
        assert game.state.move_count > 0
    
    def test_belief_tracking_during_game(self):
        game = FogShogiGame()
        
        for _ in range(10):
            if game.is_terminal():
                break
            
            # Check beliefs are maintained
            belief_sente = game.get_belief(Player.SENTE)
            belief_gote = game.get_belief(Player.GOTE)
            
            assert len(belief_sente.opponent_piece_beliefs) > 0
            assert len(belief_gote.opponent_piece_beliefs) > 0
            
            actions = game.get_legal_actions()
            if actions:
                game.apply_action(np.random.choice(actions))


def run_all_tests():
    """Run all tests manually."""
    print("Running Fog Shogi Tests...")
    
    test_classes = [
        TestPieces,
        TestBoard,
        TestGame,
        TestBelief,
        TestCFR,
        TestMCCFR,
        TestAnalysis,
        TestIntegration,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    total_failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
