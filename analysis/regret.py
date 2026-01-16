"""
Regret Analysis and Theoretical Bounds
======================================

Provides tools for analyzing the theoretical properties of
CFR-trained strategies.

Key Metrics:
- Cumulative Regret: Σᵢ [max_a u(a) - u(σ)]
- Exploitability: max_σ' u(σ', σ) - v*(game)
- Convergence Rate: How fast exploitability decreases

Theoretical Background:
- For CFR: R^T ≤ Δ√(|A|T) where Δ is payoff range
- This implies ε-Nash in O(Δ²|A|/ε²) iterations
- CFR+ achieves O(1/T) convergence rate

Applications to Quant Finance:
- Regret bounds → worst-case performance guarantees
- Exploitability → vulnerability to adversarial strategies
- Convergence → sample complexity for strategy learning
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from collections import defaultdict

from ..core.pieces import Player
from ..core.game import GameState, Action, FogShogiGame
from ..algorithms.cfr import CFRSolver, CFRNode


@dataclass
class RegretBound:
    """
    Theoretical regret bound for a CFR-trained strategy.
    
    Attributes:
        upper_bound: Proven upper bound on cumulative regret
        empirical_regret: Measured regret from training
        bound_type: Which theorem provides the bound
    """
    upper_bound: float
    empirical_regret: float
    bound_type: str
    iterations: int
    
    @property
    def tightness_ratio(self) -> float:
        """How tight is the bound (empirical / theoretical)."""
        if self.upper_bound > 0:
            return self.empirical_regret / self.upper_bound
        return 0.0
    
    def exploitability_bound(self) -> float:
        """
        Convert regret bound to exploitability bound.
        
        For two-player zero-sum: exploitability ≤ 2 * avg_regret
        """
        if self.iterations > 0:
            return 2 * self.upper_bound / self.iterations
        return float('inf')


class RegretAnalyzer:
    """
    Analyzer for CFR training dynamics and convergence.
    
    Provides:
    - Regret bound computation
    - Convergence rate estimation
    - Strategy quality assessment
    - Exploitability computation
    """
    
    def __init__(self, solver: CFRSolver):
        """
        Initialize analyzer with a trained solver.
        
        Args:
            solver: Trained CFRSolver instance
        """
        self.solver = solver
        self._cache = {}
    
    def compute_regret_bounds(self) -> RegretBound:
        """
        Compute theoretical regret bounds.
        
        Uses the standard CFR bound:
        R^T ≤ Δ * |I| * √(|A|_max * T)
        
        where:
        - Δ is the payoff range (2 for ±1 terminal payoffs)
        - |I| is number of information sets
        - |A|_max is maximum actions at any info set
        - T is iterations
        """
        T = self.solver.iterations_completed
        if T == 0:
            return RegretBound(
                upper_bound=float('inf'),
                empirical_regret=0.0,
                bound_type="N/A",
                iterations=0
            )
        
        # Game parameters
        delta = 2.0  # Payoff range [-1, 1]
        num_info_sets = len(self.solver.nodes)
        max_actions = max(
            (node.num_actions for node in self.solver.nodes.values()),
            default=1
        )
        
        # Zinkevich et al. bound
        theoretical_bound = delta * num_info_sets * np.sqrt(max_actions * T)
        
        # Empirical regret (sum of positive regrets)
        empirical = sum(
            np.sum(np.maximum(node.regret_sum, 0))
            for node in self.solver.nodes.values()
        )
        
        return RegretBound(
            upper_bound=theoretical_bound,
            empirical_regret=empirical,
            bound_type="Zinkevich2007",
            iterations=T
        )
    
    def estimate_exploitability(self, num_samples: int = 1000) -> float:
        """
        Estimate exploitability through best response sampling.
        
        Exploitability measures how much a perfect opponent could
        exploit the current strategy. Lower is better.
        
        For Nash equilibrium: exploitability = 0
        """
        total_exploit = 0.0
        
        for player in [Player.SENTE, Player.GOTE]:
            # Estimate best response value against strategy
            br_value = self._estimate_best_response_value(player, num_samples)
            
            # Nash value is 0 for symmetric game
            total_exploit += max(0, br_value)
        
        return total_exploit
    
    def _estimate_best_response_value(self, 
                                      player: Player,
                                      num_samples: int) -> float:
        """
        Estimate best response value for a player.
        
        Uses Monte Carlo sampling of game trajectories.
        """
        total_value = 0.0
        
        for _ in range(num_samples):
            game = FogShogiGame()
            
            # Play game with strategy vs best response
            while not game.is_terminal():
                state = game.get_full_state()
                
                if state.current_player == player:
                    # Best response: play greedily
                    action = self._get_greedy_action(state, game)
                else:
                    # Use trained strategy
                    action = self.solver.sample_action(state, game)
                
                game.apply_action(action)
            
            total_value += game.get_payoff(player)
        
        return total_value / num_samples
    
    def _get_greedy_action(self, state: GameState, 
                          game: FogShogiGame) -> Action:
        """Get greedy action (simple heuristic best response)."""
        actions = state.get_legal_actions()
        if not actions:
            raise ValueError("No legal actions")
        
        player = state.current_player
        best_action = actions[0]
        best_value = float('-inf')
        
        for action in actions:
            next_state = state.apply_action(action)
            value = next_state.evaluate(player)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def convergence_rate(self, 
                        window_size: int = 10) -> float:
        """
        Estimate convergence rate from exploitability history.
        
        Fits exploitability ~ C/T^α and returns α.
        α = 0.5 for vanilla CFR, α = 1.0 for CFR+
        """
        history = self.solver.exploitability_history
        if len(history) < window_size:
            return 0.0
        
        # Use recent window
        recent = history[-window_size:]
        iterations = list(range(
            self.solver.iterations_completed - window_size * 100,
            self.solver.iterations_completed,
            100
        ))
        
        if len(iterations) != len(recent):
            iterations = list(range(len(recent)))
        
        # Log-log regression: log(exploit) = log(C) - α*log(T)
        log_t = np.log(np.array(iterations) + 1)
        log_e = np.log(np.array(recent) + 1e-10)
        
        # Simple linear regression
        slope, _ = np.polyfit(log_t, log_e, 1)
        
        return -slope  # Convergence rate
    
    def strategy_entropy(self) -> Dict[str, float]:
        """
        Compute entropy statistics for the learned strategy.
        
        Higher entropy = more randomization in strategy.
        Pure strategies have 0 entropy.
        """
        entropies = []
        
        for node in self.solver.nodes.values():
            strategy = node.get_average_strategy()
            # Shannon entropy
            probs = strategy[strategy > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
        
        return {
            "mean": np.mean(entropies),
            "std": np.std(entropies),
            "min": np.min(entropies),
            "max": np.max(entropies),
            "total_nodes": len(entropies),
        }
    
    def action_distribution(self) -> Dict[str, float]:
        """
        Analyze action selection distribution across nodes.
        
        Returns statistics about how concentrated strategies are.
        """
        concentrations = []  # Gini coefficient for each node
        
        for node in self.solver.nodes.values():
            strategy = node.get_average_strategy()
            sorted_probs = np.sort(strategy)
            n = len(sorted_probs)
            
            # Gini coefficient
            index = np.arange(1, n + 1)
            gini = (np.sum((2 * index - n - 1) * sorted_probs)) / (n * np.sum(sorted_probs) + 1e-10)
            concentrations.append(gini)
        
        return {
            "mean_concentration": np.mean(concentrations),
            "std_concentration": np.std(concentrations),
            "highly_concentrated": sum(1 for c in concentrations if c > 0.7),
            "near_uniform": sum(1 for c in concentrations if c < 0.3),
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        bounds = self.compute_regret_bounds()
        entropy = self.strategy_entropy()
        actions = self.action_distribution()
        conv_rate = self.convergence_rate()
        
        lines = [
            "=" * 60,
            "CFR Training Analysis Report",
            "=" * 60,
            "",
            "Training Statistics:",
            f"  Iterations: {self.solver.iterations_completed}",
            f"  Information Sets: {len(self.solver.nodes)}",
            f"  Total Node Visits: {self.solver.total_nodes_visited}",
            "",
            "Regret Bounds:",
            f"  Theoretical Upper Bound: {bounds.upper_bound:.2f}",
            f"  Empirical Regret: {bounds.empirical_regret:.2f}",
            f"  Bound Tightness: {bounds.tightness_ratio:.2%}",
            f"  Bound Type: {bounds.bound_type}",
            "",
            "Exploitability:",
            f"  Theoretical Bound: {bounds.exploitability_bound():.4f}",
            f"  Last Measured: {self.solver.exploitability_history[-1] if self.solver.exploitability_history else 'N/A':.4f}",
            "",
            "Convergence:",
            f"  Estimated Rate (α): {conv_rate:.3f}",
            f"  Expected (CFR): 0.5, Expected (CFR+): 1.0",
            "",
            "Strategy Entropy:",
            f"  Mean: {entropy['mean']:.3f} bits",
            f"  Std: {entropy['std']:.3f}",
            f"  Range: [{entropy['min']:.3f}, {entropy['max']:.3f}]",
            "",
            "Action Distribution:",
            f"  Mean Concentration (Gini): {actions['mean_concentration']:.3f}",
            f"  Highly Concentrated Nodes: {actions['highly_concentrated']}",
            f"  Near-Uniform Nodes: {actions['near_uniform']}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)


class SampleComplexityAnalyzer:
    """
    Analyzes sample complexity for learning.
    
    Key Questions:
    - How many iterations to reach ε-Nash?
    - What's the variance in value estimates?
    - How does abstraction affect convergence?
    """
    
    @staticmethod
    def iterations_for_epsilon(epsilon: float,
                              num_info_sets: int,
                              max_actions: int,
                              delta: float = 2.0) -> int:
        """
        Compute iterations needed for ε-Nash equilibrium.
        
        Based on: ε ≤ 2 * Δ * |I| * √(|A|/T)
        
        Solving for T: T ≥ 4 * Δ² * |I|² * |A| / ε²
        """
        return int(np.ceil(
            4 * delta**2 * num_info_sets**2 * max_actions / epsilon**2
        ))
    
    @staticmethod
    def sample_complexity_analysis(solver: CFRSolver) -> Dict:
        """
        Analyze sample complexity of the training.
        
        Returns theoretical and empirical complexity measures.
        """
        num_info_sets = len(solver.nodes)
        max_actions = max(
            (n.num_actions for n in solver.nodes.values()),
            default=1
        )
        
        # Theoretical iterations for different ε levels
        epsilon_levels = [0.1, 0.05, 0.01, 0.005]
        theoretical_iters = {
            f"ε={eps}": SampleComplexityAnalyzer.iterations_for_epsilon(
                eps, num_info_sets, max_actions
            )
            for eps in epsilon_levels
        }
        
        # Empirical estimate based on exploitability history
        if solver.exploitability_history:
            current_exploit = solver.exploitability_history[-1]
            current_iters = solver.iterations_completed
            
            # Extrapolate assuming O(1/√T) convergence
            empirical_iters = {
                f"ε={eps}": int(current_iters * (current_exploit / eps) ** 2)
                for eps in epsilon_levels
                if eps < current_exploit
            }
        else:
            empirical_iters = {}
        
        return {
            "theoretical": theoretical_iters,
            "empirical": empirical_iters,
            "num_info_sets": num_info_sets,
            "max_actions": max_actions,
        }


def compare_solvers(solvers: Dict[str, CFRSolver],
                   iterations: int = 1000) -> Dict:
    """
    Compare multiple CFR solver variants.
    
    Trains each solver and compares convergence properties.
    """
    results = {}
    
    for name, solver in solvers.items():
        # Train solver
        exploitability_trace = []
        
        def callback(i, e):
            exploitability_trace.append((i, e))
        
        solver.train(iterations, progress_callback=callback)
        
        # Analyze
        analyzer = RegretAnalyzer(solver)
        bounds = analyzer.compute_regret_bounds()
        
        results[name] = {
            "final_exploitability": (
                exploitability_trace[-1][1] if exploitability_trace else None
            ),
            "convergence_rate": analyzer.convergence_rate(),
            "regret_bound": bounds.upper_bound,
            "empirical_regret": bounds.empirical_regret,
            "nodes_created": len(solver.nodes),
            "trace": exploitability_trace,
        }
    
    return results
