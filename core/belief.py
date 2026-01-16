"""
Bayesian Belief State Module
============================

Implements probabilistic tracking of hidden information.

Key Concepts:
- Belief state: Probability distribution over possible true game states
- Information set: Set of game states consistent with observations
- Bayesian update: Condition beliefs on new observations

This module enables:
1. Tracking where opponent pieces might be
2. Reasoning about what opponent knows about our position
3. Optimal play under uncertainty

Theoretical Foundation:
- Each observation partitions the state space
- Bayes' rule: P(state|obs) ∝ P(obs|state) × P(state)
- Belief complexity grows exponentially with hidden information
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import defaultdict
import numpy as np
from copy import deepcopy

from .pieces import Piece, PieceType, Player, PIECE_VALUES
from .board import Board, BoardView, Square, BOARD_SIZE


@dataclass
class PieceDistribution:
    """
    Probability distribution for a single hidden piece's location.
    
    Tracks where an opponent's piece might be, given observations.
    Uses particle-based representation for tractability.
    """
    piece_id: int
    piece_type: PieceType
    owner: Player
    
    # Location probabilities: square -> probability
    location_probs: Dict[Square, float] = field(default_factory=dict)
    
    # Whether this piece might be captured (in opponent's hand)
    capture_prob: float = 0.0
    
    def normalize(self) -> None:
        """Normalize probabilities to sum to 1."""
        total = sum(self.location_probs.values()) + self.capture_prob
        if total > 0:
            for sq in self.location_probs:
                self.location_probs[sq] /= total
            self.capture_prob /= total
    
    def update_observation(self, 
                          visible_squares: Set[Square],
                          observed_pieces: Dict[Square, Optional[Piece]]) -> None:
        """
        Bayesian update given new visibility information.
        
        If we can see a square and the piece isn't there, set P(there) = 0.
        If we see the piece somewhere, set P(there) = 1.
        """
        # If piece is seen at a location
        for sq, piece in observed_pieces.items():
            if piece and piece.piece_id == self.piece_id:
                # Certain location found
                self.location_probs = {sq: 1.0}
                self.capture_prob = 0.0
                return
        
        # Remove probability mass from visible empty squares
        for sq in visible_squares:
            if sq in observed_pieces and observed_pieces[sq] is None:
                # Square is visible and empty - piece can't be there
                self.location_probs.pop(sq, None)
        
        self.normalize()
    
    def entropy(self) -> float:
        """
        Shannon entropy of the location distribution.
        
        Higher entropy = more uncertainty about piece location.
        H = -Σ p(x) log p(x)
        """
        probs = list(self.location_probs.values()) + [self.capture_prob]
        probs = [p for p in probs if p > 0]
        if not probs:
            return 0.0
        return -sum(p * np.log2(p) for p in probs)
    
    def most_likely_location(self) -> Tuple[Optional[Square], float]:
        """Return the most likely location and its probability."""
        if not self.location_probs:
            return None, self.capture_prob
        
        best_sq = max(self.location_probs, key=self.location_probs.get)
        return best_sq, self.location_probs[best_sq]
    
    def expected_value_weighted_position(self) -> Optional[Tuple[float, float]]:
        """Expected (row, col) position weighted by probability."""
        if not self.location_probs:
            return None
        
        exp_row = sum(sq.row * p for sq, p in self.location_probs.items())
        exp_col = sum(sq.col * p for sq, p in self.location_probs.items())
        total_prob = sum(self.location_probs.values())
        
        if total_prob > 0:
            return exp_row / total_prob, exp_col / total_prob
        return None


@dataclass
class BeliefState:
    """
    Complete belief state for a player.
    
    Tracks probability distributions over:
    - Opponent piece locations
    - Game state history
    - Inferred opponent beliefs (level-2 reasoning)
    
    The belief state represents what a rational player should believe
    given their observation history.
    """
    player: Player
    
    # Distributions for each opponent piece
    opponent_piece_beliefs: Dict[int, PieceDistribution] = field(default_factory=dict)
    
    # History of observations (for filtering)
    observation_history: List[BoardView] = field(default_factory=list)
    
    # Particles for Monte Carlo methods (sampled consistent states)
    particles: List[Board] = field(default_factory=list)
    num_particles: int = 100
    
    # Information set statistics
    _info_set_size: int = 0
    _last_entropy: float = 0.0
    
    @classmethod
    def from_initial_position(cls, player: Player, board: Board) -> 'BeliefState':
        """
        Create initial belief state from game start.
        
        At game start, players know the initial setup,
        so beliefs are certain about initial positions.
        """
        belief = cls(player=player)
        
        # Initialize beliefs for opponent pieces
        opponent = player.opponent()
        for square, piece in board.get_all_pieces(opponent):
            dist = PieceDistribution(
                piece_id=piece.piece_id,
                piece_type=piece.piece_type,
                owner=piece.owner,
                location_probs={square: 1.0}
            )
            belief.opponent_piece_beliefs[piece.piece_id] = dist
        
        # Generate initial particles (all same since position is known)
        belief.particles = [board.copy() for _ in range(belief.num_particles)]
        
        return belief
    
    def update(self, observation: BoardView, 
               own_action: Optional['Action'] = None,
               opponent_action_visible: bool = False) -> None:
        """
        Update beliefs based on new observation.
        
        This is the core Bayesian update step:
        1. Process visibility changes
        2. Update piece distributions
        3. Resample particles
        4. Compute information-theoretic metrics
        """
        self.observation_history.append(observation)
        
        # Build observed pieces dict
        observed_pieces = {}
        for sq in observation.visible_squares:
            observed_pieces[sq] = observation.get_visible_piece(sq)
        
        # Update each piece distribution
        for piece_id, dist in self.opponent_piece_beliefs.items():
            dist.update_observation(observation.visible_squares, observed_pieces)
        
        # Particle filter update
        self._resample_particles(observation)
        
        # Update metrics
        self._info_set_size = self._estimate_info_set_size()
        self._last_entropy = self.total_entropy()
    
    def _resample_particles(self, observation: BoardView) -> None:
        """
        Resample particles to maintain consistent state samples.
        
        Uses Sequential Importance Resampling (SIR):
        1. Weight particles by observation likelihood
        2. Resample proportional to weights
        """
        if not self.particles:
            return
        
        weights = []
        for particle in self.particles:
            weight = self._observation_likelihood(particle, observation)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            # All particles inconsistent - reinitialize
            self._reinitialize_particles(observation)
            return
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Resample with replacement
        indices = np.random.choice(
            len(self.particles), 
            size=self.num_particles,
            p=weights
        )
        self.particles = [self.particles[i].copy() for i in indices]
    
    def _observation_likelihood(self, state: Board, 
                                observation: BoardView) -> float:
        """
        Compute P(observation | state).
        
        Observation is consistent if all visible squares match.
        """
        view = state.get_player_view(self.player)
        
        for sq in observation.visible_squares:
            obs_piece = observation.get_visible_piece(sq)
            state_piece = view.get_visible_piece(sq)
            
            # Check consistency
            if obs_piece is None and state_piece is not None:
                return 0.0
            if obs_piece is not None and state_piece is None:
                return 0.0
            if obs_piece is not None and state_piece is not None:
                if obs_piece.piece_id != state_piece.piece_id:
                    return 0.0
        
        return 1.0
    
    def _reinitialize_particles(self, observation: BoardView) -> None:
        """Reinitialize particles when resampling fails."""
        # Sample from piece distributions
        self.particles = []
        for _ in range(self.num_particles):
            particle = self._sample_consistent_state(observation)
            if particle:
                self.particles.append(particle)
    
    def _sample_consistent_state(self, observation: BoardView) -> Optional[Board]:
        """Sample a state consistent with current observation."""
        # This is a simplified version - full implementation would use
        # more sophisticated constraint satisfaction
        from .board import create_initial_board
        board = create_initial_board()
        
        # Place opponent pieces according to distributions
        for piece_id, dist in self.opponent_piece_beliefs.items():
            if dist.location_probs:
                probs = list(dist.location_probs.values())
                squares = list(dist.location_probs.keys())
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                    idx = np.random.choice(len(squares), p=probs)
                    sq = squares[idx]
                    piece = Piece(dist.piece_type, dist.owner, piece_id)
                    board.set_piece(sq, piece)
        
        return board
    
    def _estimate_info_set_size(self) -> int:
        """
        Estimate the size of the information set.
        
        This is the number of game states consistent with observations.
        Exact computation is intractable; we use particle diversity.
        """
        # Use particle diversity as proxy
        unique_states = len(set(
            self._state_hash(p) for p in self.particles
        ))
        return unique_states
    
    def _state_hash(self, board: Board) -> int:
        """Hash a board state for uniqueness checking."""
        items = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = board.grid[r][c]
                if piece:
                    items.append((r, c, piece.piece_id))
        return hash(tuple(sorted(items)))
    
    def total_entropy(self) -> float:
        """
        Total entropy across all piece distributions.
        
        Measures overall uncertainty in the belief state.
        """
        return sum(d.entropy() for d in self.opponent_piece_beliefs.values())
    
    def get_expected_opponent_positions(self) -> Dict[int, Tuple[float, float]]:
        """Get expected positions for each opponent piece."""
        positions = {}
        for pid, dist in self.opponent_piece_beliefs.items():
            pos = dist.expected_value_weighted_position()
            if pos:
                positions[pid] = pos
        return positions
    
    def sample_state(self) -> Optional[Board]:
        """Sample a state from the belief distribution."""
        if self.particles:
            return np.random.choice(self.particles).copy()
        return None
    
    def get_threat_probabilities(self, 
                                 target_square: Square) -> Dict[int, float]:
        """
        Compute probability that each opponent piece threatens a square.
        
        Useful for evaluating king safety under uncertainty.
        """
        threats = {}
        
        for piece_id, dist in self.opponent_piece_beliefs.items():
            threat_prob = 0.0
            for sq, prob in dist.location_probs.items():
                # Check if piece at sq could threaten target
                # (simplified - would need full move generation)
                if sq.distance_to(target_square) <= 2:  # Approximation
                    threat_prob += prob
            threats[piece_id] = min(threat_prob, 1.0)
        
        return threats
    
    def information_value(self, reveal_square: Square) -> float:
        """
        Expected information gain from revealing a square.
        
        Uses mutual information: I(X; Y) = H(X) - H(X|Y)
        Useful for active information gathering strategies.
        """
        current_entropy = self.total_entropy()
        
        # Expected entropy after observation
        expected_entropy = 0.0
        
        # Case 1: Square is empty
        p_empty = 1.0
        for dist in self.opponent_piece_beliefs.values():
            p_empty *= (1.0 - dist.location_probs.get(reveal_square, 0.0))
        
        # Case 2: Square contains each piece
        for pid, dist in self.opponent_piece_beliefs.items():
            p_piece = dist.location_probs.get(reveal_square, 0.0)
            if p_piece > 0:
                # Entropy would be reduced significantly
                expected_entropy += p_piece * (current_entropy - dist.entropy())
        
        expected_entropy += p_empty * current_entropy * 0.95  # Small reduction
        
        return current_entropy - expected_entropy
    
    def __repr__(self) -> str:
        lines = [f"BeliefState for {self.player}"]
        lines.append(f"  Info set size: ~{self._info_set_size}")
        lines.append(f"  Total entropy: {self.total_entropy():.2f} bits")
        lines.append(f"  Particles: {len(self.particles)}")
        
        lines.append("  Opponent pieces:")
        for pid, dist in self.opponent_piece_beliefs.items():
            loc, prob = dist.most_likely_location()
            lines.append(f"    {dist.piece_type.symbol}: {loc} (p={prob:.2f}), H={dist.entropy():.2f}")
        
        return "\n".join(lines)


@dataclass(frozen=True)
class InformationSet:
    """
    Abstract representation of an information set for CFR.
    
    An information set groups all game states that are
    observationally equivalent to a player.
    
    Key properties:
    - Same legal actions available
    - Same action history from player's perspective
    - Different only in hidden information
    """
    player: Player
    visible_board_hash: int  # Hash of visible board state
    hand_hash: int  # Hash of player's hand
    action_sequence_hash: int  # Hash of known action sequence
    
    @classmethod
    def from_observation(cls, observation: BoardView,
                        action_history: List[Tuple[Player, 'Action']]) -> 'InformationSet':
        """Create information set from current observation."""
        # Hash visible board
        board_items = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if not observation.fog_mask[r][c]:
                    piece = observation.grid[r][c]
                    if piece:
                        board_items.append((r, c, piece.piece_id))
                    else:
                        board_items.append((r, c, None))
        
        visible_hash = hash(tuple(sorted(board_items)))
        hand_hash = hash(tuple(p.piece_id for p in sorted(
            observation.own_hand, key=lambda x: x.piece_id)))
        
        # Hash action sequence (only own actions fully known)
        own_actions = [(i, a) for i, (p, a) in enumerate(action_history)
                      if p == observation.player]
        seq_hash = hash(tuple(str(a) for _, a in own_actions))
        
        return cls(
            player=observation.player,
            visible_board_hash=visible_hash,
            hand_hash=hand_hash,
            action_sequence_hash=seq_hash
        )
    
    def __hash__(self) -> int:
        return hash((self.player, self.visible_board_hash, 
                    self.hand_hash, self.action_sequence_hash))


def compute_info_set_abstraction(info_set: InformationSet,
                                 belief: BeliefState,
                                 abstraction_level: int = 2) -> int:
    """
    Compute an abstracted information set for tractability.
    
    Abstraction reduces the number of distinct information sets
    while preserving strategic similarity. Common approaches:
    
    Level 0: Full information set (no abstraction)
    Level 1: Bucket by entropy ranges
    Level 2: Bucket by threat levels + entropy
    Level 3: Bucket by material + simple features
    
    Returns an integer bucket ID.
    """
    if abstraction_level == 0:
        return hash(info_set)
    
    # Entropy buckets
    entropy = belief.total_entropy()
    entropy_bucket = int(entropy / 2.0)  # Bucket every 2 bits
    
    if abstraction_level == 1:
        return hash((info_set.player, info_set.hand_hash, entropy_bucket))
    
    # Add threat assessment
    if abstraction_level >= 2:
        # Simplified threat level
        total_threat = sum(
            max(d.location_probs.values()) if d.location_probs else 0
            for d in belief.opponent_piece_beliefs.values()
        )
        threat_bucket = int(total_threat * 4)
        
        return hash((info_set.player, info_set.hand_hash, 
                    entropy_bucket, threat_bucket))
    
    return hash(info_set)
