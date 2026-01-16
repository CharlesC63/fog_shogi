"""
Monte Carlo CFR (MCCFR)
=======================

Monte Carlo sampling variants of CFR for scalability.

Key Variants:
1. Outcome Sampling: Sample single terminal path
2. External Sampling: Sample opponent actions only
3. Chance Sampling: Sample chance outcomes

Theoretical Properties:
- Same convergence guarantees as vanilla CFR
- Much lower per-iteration cost
- Variance increases but expectation preserved
- Sample complexity: O(|A|² |H| / ε²)

This implementation uses External Sampling MCCFR,
which samples opponent actions while exploring all
of the traversing player's actions.

Reference:
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in
  Extensive Games with Imperfect Information" (NeurIPS 2009)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from collections import defaultdict

from ..core.pieces import Player
from ..core.game import GameState, Action, FogShogiGame
from ..core.belief import InformationSet, compute_info_set_abstraction, BeliefState
from .cfr import CFRNode, CFRSolver


class MCCFRSolver(CFRSolver):
    """
    Monte Carlo CFR with External Sampling.
    
    Instead of traversing the full game tree, we:
    1. Sample opponent actions according to current strategy
    2. Explore ALL actions for the traversing player
    3. Update regrets based on sampled trajectories
    
    Benefits:
    - O(|A|) per iteration instead of O(|A|^depth)
    - Handles large action spaces efficiently
    - Maintains unbiased regret estimates
    
    Usage:
        solver = MCCFRSolver()
        solver.train(iterations=100000)  # Can train longer due to speed
        strategy = solver.get_strategy(state, game)
    """
    
    def __init__(self,
                 abstraction_level: int = 2,
                 epsilon: float = 0.6,
                 exploration_bonus: float = 0.1,
                 **kwargs):
        """
        Initialize MCCFR solver.
        
        Args:
            abstraction_level: Information set abstraction level
            epsilon: Exploration probability for ε-greedy sampling
            exploration_bonus: Bonus for rarely-visited nodes
        """
        super().__init__(abstraction_level=abstraction_level, **kwargs)
        self.epsilon = epsilon
        self.exploration_bonus = exploration_bonus
        
        # Sampling statistics
        self.samples_per_node: Dict[int, int] = defaultdict(int)
        self.variance_estimates: List[float] = []
    
    def train(self, iterations: int,
             progress_callback: Optional[Callable[[int, float], None]] = None) -> None:
        """
        Train using External Sampling MCCFR.
        
        Each iteration samples one game trajectory while computing
        counterfactual values for the traversing player.
        """
        for i in range(iterations):
            # Alternate traversing player
            traverser = Player.SENTE if i % 2 == 0 else Player.GOTE
            
            game = FogShogiGame()
            
            # External sampling traversal
            self._mccfr_traverse(
                game.state,
                traverser,
                sample_prob=1.0,
                depth=0,
                game=game
            )
            
            self.iterations_completed += 1
            
            # Track variance periodically
            if (i + 1) % 100 == 0:
                variance = self._estimate_variance()
                self.variance_estimates.append(variance)
                
                exploit = self._estimate_exploitability()
                self.exploitability_history.append(exploit)
                
                if progress_callback:
                    progress_callback(i + 1, exploit)
    
    def _mccfr_traverse(self,
                       state: GameState,
                       traverser: Player,
                       sample_prob: float,
                       depth: int,
                       game: FogShogiGame) -> float:
        """
        External Sampling MCCFR traversal.
        
        At traverser's nodes: explore all actions, compute regrets
        At opponent's nodes: sample one action, follow that path
        
        Returns sampled counterfactual value.
        """
        self.total_nodes_visited += 1
        
        # Terminal check
        if state.is_terminal():
            return state.get_payoff(traverser)
        
        # Depth limit
        if self.max_depth and depth >= self.max_depth:
            return state.evaluate(traverser) / 10000.0
        
        current_player = state.current_player
        actions = state.get_legal_actions()
        
        if not actions:
            return 0.0
        
        # Get information set
        info_set = game.get_information_set(current_player)
        belief = game.get_belief(current_player)
        node = self._get_or_create_node(info_set, len(actions), belief)
        
        # Track sampling
        abstract_key = compute_info_set_abstraction(
            info_set, belief, self.abstraction_level
        )
        self.samples_per_node[abstract_key] += 1
        
        strategy = node.get_strategy()
        
        if current_player == traverser:
            # TRAVERSER'S NODE: Explore all actions
            action_values = np.zeros(len(actions))
            
            for i, action in enumerate(actions):
                next_state = state.apply_action(action)
                action_values[i] = self._mccfr_traverse(
                    next_state,
                    traverser,
                    sample_prob * strategy[i],
                    depth + 1,
                    game
                )
            
            # Compute node value
            node_value = np.dot(strategy, action_values)
            
            # Update regrets
            # Importance sampling correction
            instant_regrets = action_values - node_value
            node.regret_sum += instant_regrets / max(sample_prob, 1e-6)
            
            # Update average strategy
            node.update_strategy_sum(strategy, 1.0)
            
            return node_value
        else:
            # OPPONENT'S NODE: Sample one action
            # Use epsilon-greedy for exploration
            if np.random.random() < self.epsilon:
                # Uniform random
                action_idx = np.random.randint(len(actions))
                sample_strategy = np.ones(len(actions)) / len(actions)
            else:
                # Sample from strategy
                action_idx = np.random.choice(len(actions), p=strategy)
                sample_strategy = strategy
            
            action = actions[action_idx]
            next_state = state.apply_action(action)
            
            # Recursive call with updated sample probability
            return self._mccfr_traverse(
                next_state,
                traverser,
                sample_prob * sample_strategy[action_idx],
                depth + 1,
                game
            )
    
    def _estimate_variance(self) -> float:
        """Estimate variance in regret estimates."""
        if not self.nodes:
            return 0.0
        
        variances = []
        for node in self.nodes.values():
            if node.visit_count > 1:
                # Use regret spread as variance proxy
                regret_range = np.max(node.regret_sum) - np.min(node.regret_sum)
                variances.append(regret_range ** 2)
        
        return np.mean(variances) if variances else 0.0
    
    def get_training_stats(self) -> Dict:
        """Extended statistics for MCCFR."""
        stats = super().get_training_stats()
        stats.update({
            "epsilon": self.epsilon,
            "avg_samples_per_node": np.mean(list(self.samples_per_node.values())),
            "variance_estimate": (
                self.variance_estimates[-1] if self.variance_estimates else 0.0
            ),
            "sample_efficiency": (
                self.iterations_completed / len(self.nodes) 
                if self.nodes else 0
            ),
        })
        return stats


class OutcomeSamplingMCCFR(MCCFRSolver):
    """
    Outcome Sampling MCCFR variant.
    
    Samples complete game outcomes (both players' actions)
    and uses importance sampling to correct regret estimates.
    
    Even faster than External Sampling but higher variance.
    Useful for very large games where External Sampling is still too slow.
    """
    
    def __init__(self, delta: float = 0.9, **kwargs):
        """
        Args:
            delta: Baseline probability for opponent actions
        """
        super().__init__(**kwargs)
        self.delta = delta
    
    def _mccfr_traverse(self,
                       state: GameState,
                       traverser: Player,
                       sample_prob: float,
                       depth: int,
                       game: FogShogiGame) -> float:
        """
        Outcome Sampling traversal.
        
        Sample actions for ALL players, update regrets with
        importance sampling correction.
        """
        self.total_nodes_visited += 1
        
        if state.is_terminal():
            return state.get_payoff(traverser)
        
        if self.max_depth and depth >= self.max_depth:
            return state.evaluate(traverser) / 10000.0
        
        current_player = state.current_player
        actions = state.get_legal_actions()
        
        if not actions:
            return 0.0
        
        info_set = game.get_information_set(current_player)
        belief = game.get_belief(current_player)
        node = self._get_or_create_node(info_set, len(actions), belief)
        
        strategy = node.get_strategy()
        
        # Sample action with exploration
        explore_prob = self.epsilon if current_player != traverser else 0.0
        
        if np.random.random() < explore_prob:
            action_idx = np.random.randint(len(actions))
            sample_p = explore_prob / len(actions) + (1 - explore_prob) * strategy[action_idx]
        else:
            action_idx = np.random.choice(len(actions), p=strategy)
            sample_p = strategy[action_idx]
        
        action = actions[action_idx]
        next_state = state.apply_action(action)
        
        # Recursive traversal
        child_value = self._mccfr_traverse(
            next_state,
            traverser,
            sample_prob * sample_p,
            depth + 1,
            game
        )
        
        # Importance sampling weight
        W = 1.0 / max(sample_prob * sample_p, 1e-6)
        
        if current_player == traverser:
            # Compute counterfactual regrets
            # For sampled action, we have the actual value
            # For other actions, use baseline estimate
            
            action_values = np.zeros(len(actions))
            action_values[action_idx] = child_value * W
            
            # Baseline for unsampled actions (pessimistic estimate)
            for i in range(len(actions)):
                if i != action_idx:
                    action_values[i] = child_value * self.delta
            
            node_value = np.dot(strategy, action_values)
            instant_regrets = action_values - node_value
            
            node.regret_sum += instant_regrets
            node.update_strategy_sum(strategy, 1.0)
            
            return child_value
        else:
            return child_value


class DeepCFR:
    """
    Deep CFR using neural networks for value function approximation.
    
    Instead of storing regrets per information set, we train neural networks to:
    1. Predict regrets from information set features
    2. Predict strategy from information set features
    
    Benefits:
    - Generalizes across similar information sets
    - Handles massive state spaces
    - Can incorporate domain knowledge in features
    
    Note: This is a skeleton implementation. Full version would require
    PyTorch/TensorFlow integration.
    """
    
    def __init__(self,
                 feature_dim: int = 128,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 3,
                 learning_rate: float = 1e-3):
        """
        Initialize Deep CFR.
        
        Args:
            feature_dim: Dimension of information set features
            hidden_dim: Hidden layer size
            num_hidden_layers: Number of hidden layers
            learning_rate: Optimizer learning rate
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        
        # Memory buffers for training
        self.advantage_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.strategy_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Placeholder for neural networks
        # In practice, these would be PyTorch/TF models
        self._regret_network = None
        self._strategy_network = None
    
    def extract_features(self, 
                        state: GameState,
                        belief: BeliefState) -> np.ndarray:
        """
        Extract feature vector from game state and belief.
        
        Features include:
        - Visible board positions (one-hot)
        - Hand pieces (counts)
        - Entropy of opponent piece beliefs
        - Turn indicator
        """
        features = []
        
        # Board features (visible pieces)
        observation = state.board.get_player_view(state.current_player)
        for r in range(5):  # BOARD_SIZE
            for c in range(5):
                if not observation.fog_mask[r][c]:
                    piece = observation.grid[r][c]
                    if piece:
                        # One-hot encoding: 10 piece types × 2 players
                        piece_feat = np.zeros(20)
                        piece_idx = piece.piece_type.value - 1
                        player_offset = 0 if piece.owner == Player.SENTE else 10
                        piece_feat[piece_idx + player_offset] = 1.0
                        features.extend(piece_feat)
                    else:
                        features.extend([0.0] * 20)
                else:
                    # Fog encoding
                    features.extend([-1.0] * 20)
        
        # Hand pieces
        hand = state.board.get_hand(state.current_player)
        hand_counts = np.zeros(10)  # One per piece type
        for p in hand:
            hand_counts[p.piece_type.value - 1] += 1
        features.extend(hand_counts)
        
        # Belief entropy
        features.append(belief.total_entropy())
        
        # Turn indicator
        features.append(1.0 if state.current_player == Player.SENTE else -1.0)
        
        return np.array(features[:self.feature_dim])
    
    def train(self, iterations: int) -> None:
        """
        Deep CFR training loop.
        
        1. CFR traversal to collect advantage samples
        2. Train advantage network on samples
        3. Use advantage network for strategy
        4. Repeat
        """
        raise NotImplementedError(
            "Deep CFR requires neural network implementation. "
            "See Facebook Research's OpenSpiel or ReBeL for reference."
        )
    
    def get_strategy(self, state: GameState, 
                    game: FogShogiGame) -> Dict[Action, float]:
        """Get strategy from trained network."""
        raise NotImplementedError("Requires trained neural network")
