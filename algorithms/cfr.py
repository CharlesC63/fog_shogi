"""
Counterfactual Regret Minimization (CFR)
========================================

Implements CFR and variants for computing Nash equilibria
in imperfect information games.

Theoretical Background:
- CFR converges to ε-Nash equilibrium in O(1/ε²) iterations
- Regret bound: Σᵢ Rᵢᵀ ≤ O(√T) where T is iterations
- For two-player zero-sum games, average strategy converges to minimax

Key Concepts:
- Counterfactual value: Expected utility assuming player tries to reach state
- Regret: Difference between action value and current strategy value
- Strategy update: Regret matching (positive regrets → probabilities)

References:
- Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod

from ..core.pieces import Player
from ..core.game import GameState, Action, FogShogiGame
from ..core.belief import InformationSet, compute_info_set_abstraction, BeliefState


@dataclass
class CFRNode:
    """
    Node in the CFR algorithm storing regrets and strategy.
    
    Each node corresponds to an information set.
    Tracks cumulative regrets for regret matching.
    """
    info_set: InformationSet
    num_actions: int
    
    # Cumulative regrets for each action
    regret_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Cumulative strategy (for averaging)
    strategy_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Visit count for diagnostics
    visit_count: int = 0
    
    def __post_init__(self):
        if len(self.regret_sum) == 0:
            self.regret_sum = np.zeros(self.num_actions)
        if len(self.strategy_sum) == 0:
            self.strategy_sum = np.zeros(self.num_actions)
    
    def get_strategy(self) -> np.ndarray:
        """
        Compute current strategy via regret matching.
        
        σ(a) ∝ max(R(a), 0) where R(a) is cumulative regret for action a
        If all regrets non-positive, use uniform strategy.
        """
        positive_regrets = np.maximum(self.regret_sum, 0)
        regret_total = np.sum(positive_regrets)
        
        if regret_total > 0:
            return positive_regrets / regret_total
        else:
            # Uniform strategy when no positive regrets
            return np.ones(self.num_actions) / self.num_actions
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Get the time-averaged strategy (converges to Nash equilibrium).
        
        This is what we actually use for play.
        """
        strategy_total = np.sum(self.strategy_sum)
        
        if strategy_total > 0:
            return self.strategy_sum / strategy_total
        else:
            return np.ones(self.num_actions) / self.num_actions
    
    def update_regrets(self, action_utilities: np.ndarray, 
                      strategy: np.ndarray) -> None:
        """
        Update cumulative regrets based on counterfactual values.
        
        R(a) += u(a) - Σ_a' σ(a') u(a')
        """
        strategy_value = np.dot(strategy, action_utilities)
        instant_regrets = action_utilities - strategy_value
        self.regret_sum += instant_regrets
    
    def update_strategy_sum(self, strategy: np.ndarray, 
                           reach_prob: float) -> None:
        """Update cumulative strategy weighted by reach probability."""
        self.strategy_sum += reach_prob * strategy
        self.visit_count += 1


class CFRSolver:
    """
    Vanilla CFR solver for Fog of War Shogi.
    
    Computes Nash equilibrium strategies through iterative
    self-play with regret minimization.
    
    Theoretical Guarantees:
    - Exploitability decreases as O(1/√T)
    - Average strategy converges to Nash equilibrium
    - Works for any two-player zero-sum game
    
    Usage:
        solver = CFRSolver()
        solver.train(iterations=10000)
        strategy = solver.get_strategy(info_set)
    """
    
    def __init__(self, 
                 abstraction_level: int = 2,
                 discount_factor: float = 1.0,
                 use_pruning: bool = True,
                 max_depth: Optional[int] = None):
        """
        Initialize CFR solver.
        
        Args:
            abstraction_level: Information set abstraction (0-3)
            discount_factor: Discount for older regrets (1.0 = no discount)
            use_pruning: Whether to prune low-probability branches
            max_depth: Maximum game tree depth (None = full tree)
        """
        self.abstraction_level = abstraction_level
        self.discount_factor = discount_factor
        self.use_pruning = use_pruning
        self.max_depth = max_depth
        
        # CFR node storage
        self.nodes: Dict[int, CFRNode] = {}
        
        # Training statistics
        self.iterations_completed = 0
        self.total_nodes_visited = 0
        
        # Regret bounds tracking
        self.cumulative_regret_bound = 0.0
        self.exploitability_history: List[float] = []
    
    def _get_or_create_node(self, 
                           info_set: InformationSet,
                           num_actions: int,
                           belief: Optional[BeliefState] = None) -> CFRNode:
        """Get existing node or create new one."""
        # Apply abstraction
        if belief:
            abstract_key = compute_info_set_abstraction(
                info_set, belief, self.abstraction_level
            )
        else:
            abstract_key = hash(info_set)
        
        if abstract_key not in self.nodes:
            self.nodes[abstract_key] = CFRNode(
                info_set=info_set,
                num_actions=num_actions
            )
        
        return self.nodes[abstract_key]
    
    def train(self, iterations: int, 
             progress_callback: Optional[Callable[[int, float], None]] = None) -> None:
        """
        Train the CFR solver for specified iterations.
        
        Each iteration performs one traversal of the game tree,
        updating regrets and strategy sums.
        
        Args:
            iterations: Number of training iterations
            progress_callback: Optional callback(iteration, exploitability)
        """
        for i in range(iterations):
            # Create fresh game for each iteration
            game = FogShogiGame()
            
            # CFR traversal from root
            self._cfr_traverse(
                game.state,
                Player.SENTE,
                reach_probs={Player.SENTE: 1.0, Player.GOTE: 1.0},
                depth=0,
                game=game
            )
            
            self.iterations_completed += 1
            
            # Apply discount to old regrets (CFR+ style)
            if self.discount_factor < 1.0:
                self._apply_discount()
            
            # Periodic exploitability computation
            if (i + 1) % 100 == 0:
                exploit = self._estimate_exploitability()
                self.exploitability_history.append(exploit)
                
                if progress_callback:
                    progress_callback(i + 1, exploit)
    
    def _cfr_traverse(self,
                     state: GameState,
                     traversing_player: Player,
                     reach_probs: Dict[Player, float],
                     depth: int,
                     game: FogShogiGame) -> float:
        """
        Recursive CFR traversal of game tree.
        
        Returns counterfactual value for the traversing player.
        
        This is the core algorithm:
        1. At terminal: return payoff
        2. At player's node: compute regrets for all actions
        3. At opponent's node: sample according to strategy
        """
        self.total_nodes_visited += 1
        
        # Terminal check
        if state.is_terminal():
            return state.get_payoff(traversing_player)
        
        # Depth limit
        if self.max_depth and depth >= self.max_depth:
            return state.evaluate(traversing_player) / 10000.0
        
        current_player = state.current_player
        actions = state.get_legal_actions()
        
        if not actions:
            return 0.0
        
        # Get information set and belief
        info_set = game.get_information_set(current_player)
        belief = game.get_belief(current_player)
        
        # Get or create CFR node
        node = self._get_or_create_node(info_set, len(actions), belief)
        strategy = node.get_strategy()
        
        # Compute counterfactual values
        action_values = np.zeros(len(actions))
        
        for i, action in enumerate(actions):
            # Pruning: skip low-probability actions
            if self.use_pruning and strategy[i] < 1e-6:
                if current_player != traversing_player:
                    continue
            
            # Apply action
            next_state = state.apply_action(action)
            
            # Update reach probabilities
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            
            # Recursive call
            action_values[i] = self._cfr_traverse(
                next_state,
                traversing_player,
                new_reach,
                depth + 1,
                game
            )
        
        # Node value under current strategy
        node_value = np.dot(strategy, action_values)
        
        # Update regrets (only for traversing player's nodes)
        if current_player == traversing_player:
            # Counterfactual reach probability
            cf_reach = reach_probs[current_player.opponent()]
            
            # Regret = cf_reach * (action_value - node_value)
            instant_regrets = cf_reach * (action_values - node_value)
            node.regret_sum += instant_regrets
        
        # Update average strategy
        node.update_strategy_sum(strategy, reach_probs[current_player])
        
        return node_value
    
    def _apply_discount(self) -> None:
        """Apply discount factor to all regrets (CFR+ enhancement)."""
        for node in self.nodes.values():
            node.regret_sum *= self.discount_factor
            node.strategy_sum *= self.discount_factor
    
    def _estimate_exploitability(self) -> float:
        """
        Estimate exploitability of current average strategy.
        
        Exploitability = max_opponent_response - nash_value
        Lower is better (0 = perfect Nash equilibrium)
        
        Note: Full computation is expensive, this is an approximation.
        """
        # Use regret bound as proxy
        total_positive_regret = sum(
            np.sum(np.maximum(node.regret_sum, 0))
            for node in self.nodes.values()
        )
        
        # Theoretical bound: exploitability ≤ 2 * max_regret / T
        if self.iterations_completed > 0:
            return total_positive_regret / self.iterations_completed
        return float('inf')
    
    def get_strategy(self, state: GameState, 
                    game: FogShogiGame) -> Dict[Action, float]:
        """
        Get the trained strategy for a game state.
        
        Returns action -> probability mapping.
        """
        actions = state.get_legal_actions()
        if not actions:
            return {}
        
        info_set = game.get_information_set(state.current_player)
        belief = game.get_belief(state.current_player)
        
        # Get abstracted key
        abstract_key = compute_info_set_abstraction(
            info_set, belief, self.abstraction_level
        )
        
        if abstract_key in self.nodes:
            node = self.nodes[abstract_key]
            avg_strategy = node.get_average_strategy()
            return {a: p for a, p in zip(actions, avg_strategy)}
        else:
            # Uniform strategy for unseen states
            uniform_prob = 1.0 / len(actions)
            return {a: uniform_prob for a in actions}
    
    def sample_action(self, state: GameState, 
                     game: FogShogiGame) -> Action:
        """Sample an action according to trained strategy."""
        strategy = self.get_strategy(state, game)
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        if not actions:
            raise ValueError("No legal actions")
        
        return np.random.choice(actions, p=probs)
    
    def get_regret_bound(self) -> float:
        """
        Compute the theoretical regret bound.
        
        For T iterations: R^T ≤ O(√T * |A| * Δ)
        where |A| is action count and Δ is game depth.
        """
        if self.iterations_completed == 0:
            return float('inf')
        
        # Approximate bound
        avg_actions = np.mean([n.num_actions for n in self.nodes.values()])
        depth_estimate = self.max_depth or 20
        
        bound = np.sqrt(self.iterations_completed) * avg_actions * depth_estimate
        return bound / self.iterations_completed
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "iterations": self.iterations_completed,
            "nodes_created": len(self.nodes),
            "total_visits": self.total_nodes_visited,
            "avg_visits_per_node": (
                self.total_nodes_visited / len(self.nodes) 
                if self.nodes else 0
            ),
            "regret_bound": self.get_regret_bound(),
            "estimated_exploitability": (
                self.exploitability_history[-1] 
                if self.exploitability_history else float('inf')
            ),
        }
    
    def save(self, filepath: str) -> None:
        """Save trained model to file."""
        import pickle
        data = {
            "nodes": self.nodes,
            "iterations": self.iterations_completed,
            "abstraction_level": self.abstraction_level,
            "exploitability_history": self.exploitability_history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'CFRSolver':
        """Load trained model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        solver = cls(abstraction_level=data["abstraction_level"])
        solver.nodes = data["nodes"]
        solver.iterations_completed = data["iterations"]
        solver.exploitability_history = data["exploitability_history"]
        
        return solver


class CFRPlus(CFRSolver):
    """
    CFR+ variant with faster convergence.
    
    Improvements over vanilla CFR:
    1. Regret matching+: floor regrets at 0
    2. Linear averaging: weight recent iterations more
    3. Alternating updates: update one player per iteration
    
    Converges as O(1/T) instead of O(1/√T).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_regret_floor = True
        self.use_linear_averaging = True
    
    def _cfr_traverse(self, state, traversing_player, reach_probs, 
                     depth, game) -> float:
        """CFR+ traversal with regret flooring."""
        result = super()._cfr_traverse(
            state, traversing_player, reach_probs, depth, game
        )
        
        # Floor regrets at 0 (CFR+ modification)
        if self.use_regret_floor:
            for node in self.nodes.values():
                node.regret_sum = np.maximum(node.regret_sum, 0)
        
        return result
    
    def train(self, iterations: int, 
             progress_callback: Optional[Callable[[int, float], None]] = None) -> None:
        """CFR+ training with linear averaging."""
        for i in range(iterations):
            game = FogShogiGame()
            
            # Alternating updates
            traverser = Player.SENTE if i % 2 == 0 else Player.GOTE
            
            self._cfr_traverse(
                game.state,
                traverser,
                reach_probs={Player.SENTE: 1.0, Player.GOTE: 1.0},
                depth=0,
                game=game
            )
            
            self.iterations_completed += 1
            
            # Linear averaging weight
            if self.use_linear_averaging:
                weight = self.iterations_completed
                for node in self.nodes.values():
                    node.strategy_sum *= weight / (weight + 1)
            
            if (i + 1) % 100 == 0:
                exploit = self._estimate_exploitability()
                self.exploitability_history.append(exploit)
                
                if progress_callback:
                    progress_callback(i + 1, exploit)
