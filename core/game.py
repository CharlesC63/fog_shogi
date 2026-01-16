"""
Game Module: Actions, States, and Rules
=======================================

Implements the core game logic for Fog of War Shogi.

Game Flow:
1. Players alternate turns
2. Each turn: move a piece OR drop a captured piece
3. Win by capturing opponent's king
4. Additional rules: promotion, drop restrictions

Fog of War Mechanics:
- Actions outside opponent's view are hidden
- Creates information asymmetry
- Enables deception and inference
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Union, Iterator
from enum import Enum, auto
from copy import deepcopy

from .pieces import Piece, PieceType, Player, get_movement_deltas, PIECE_VALUES
from .board import Board, BoardView, Square, BOARD_SIZE, create_initial_board
from .belief import BeliefState, InformationSet


class ActionType(Enum):
    """Types of actions in Shogi."""
    MOVE = auto()        # Move a piece on the board
    DROP = auto()        # Drop a captured piece
    PROMOTE = auto()     # Move with promotion (combined)


@dataclass(frozen=True)
class Action:
    """
    Immutable action representation.
    
    Actions are the edges in the game tree.
    Each action transforms one game state to another.
    """
    action_type: ActionType
    
    # For MOVE/PROMOTE actions
    from_square: Optional[Square] = None
    to_square: Optional[Square] = None
    
    # For DROP actions
    drop_piece_type: Optional[PieceType] = None
    
    # Whether to promote (for MOVE actions that allow it)
    promote: bool = False
    
    def __repr__(self) -> str:
        if self.action_type == ActionType.DROP:
            return f"DROP {self.drop_piece_type.symbol}@{self.to_square.row},{self.to_square.col}"
        elif self.promote:
            return f"{self.from_square.row},{self.from_square.col}->{self.to_square.row},{self.to_square.col}+"
        else:
            return f"{self.from_square.row},{self.from_square.col}->{self.to_square.row},{self.to_square.col}"


class GameResult(Enum):
    """Possible game outcomes."""
    ONGOING = auto()
    SENTE_WIN = auto()
    GOTE_WIN = auto()
    DRAW = auto()


@dataclass
class GameState:
    """
    Complete game state including history.
    
    The game state tracks everything needed to:
    - Determine legal actions
    - Evaluate terminal states
    - Support undo operations
    """
    board: Board
    current_player: Player
    move_count: int = 0
    action_history: List[Tuple[Player, Action]] = field(default_factory=list)
    result: GameResult = GameResult.ONGOING
    
    # Repetition tracking for draw detection
    position_counts: dict = field(default_factory=dict)
    
    def copy(self) -> 'GameState':
        """Deep copy the game state."""
        new_state = GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            move_count=self.move_count,
            action_history=list(self.action_history),
            result=self.result,
            position_counts=dict(self.position_counts),
        )
        return new_state
    
    def get_legal_actions(self) -> List[Action]:
        """
        Generate all legal actions for the current player.
        
        Legal actions include:
        1. All valid piece moves
        2. All valid drops
        3. Promotion options where applicable
        """
        if self.result != GameResult.ONGOING:
            return []
        
        actions = []
        player = self.current_player
        
        # Generate move actions
        for from_sq, piece in self.board.get_all_pieces(player):
            for to_sq in self._get_piece_moves(from_sq, piece):
                # Check if promotion is possible/mandatory
                can_promote = (piece.piece_type.can_promote() and 
                             (from_sq.in_promotion_zone(player) or 
                              to_sq.in_promotion_zone(player)))
                
                must_promote = self._must_promote(piece, to_sq)
                
                if must_promote:
                    actions.append(Action(
                        action_type=ActionType.MOVE,
                        from_square=from_sq,
                        to_square=to_sq,
                        promote=True
                    ))
                elif can_promote:
                    # Add both options
                    actions.append(Action(
                        action_type=ActionType.MOVE,
                        from_square=from_sq,
                        to_square=to_sq,
                        promote=False
                    ))
                    actions.append(Action(
                        action_type=ActionType.MOVE,
                        from_square=from_sq,
                        to_square=to_sq,
                        promote=True
                    ))
                else:
                    actions.append(Action(
                        action_type=ActionType.MOVE,
                        from_square=from_sq,
                        to_square=to_sq,
                        promote=False
                    ))
        
        # Generate drop actions
        hand = self.board.get_hand(player)
        seen_types = set()
        for piece in hand:
            if piece.piece_type in seen_types:
                continue
            seen_types.add(piece.piece_type)
            
            for to_sq in self._get_drop_squares(piece.piece_type, player):
                actions.append(Action(
                    action_type=ActionType.DROP,
                    to_square=to_sq,
                    drop_piece_type=piece.piece_type
                ))
        
        return actions
    
    def _get_piece_moves(self, from_sq: Square, piece: Piece) -> Iterator[Square]:
        """Generate valid destination squares for a piece."""
        deltas = get_movement_deltas(piece)
        
        for dr, dc, is_sliding in deltas:
            if is_sliding:
                # Sliding pieces: move multiple squares in direction
                dist = 1
                while True:
                    to_sq = Square(from_sq.row + dr * dist, 
                                   from_sq.col + dc * dist)
                    if not to_sq.is_valid():
                        break
                    
                    target = self.board.get_piece(to_sq)
                    if target is None:
                        yield to_sq
                        dist += 1
                    elif target.owner != piece.owner:
                        yield to_sq  # Capture
                        break
                    else:
                        break  # Blocked by own piece
            else:
                # Non-sliding: single step
                to_sq = Square(from_sq.row + dr, from_sq.col + dc)
                if to_sq.is_valid():
                    target = self.board.get_piece(to_sq)
                    if target is None or target.owner != piece.owner:
                        yield to_sq
    
    def _get_drop_squares(self, piece_type: PieceType, 
                         player: Player) -> Iterator[Square]:
        """Generate valid drop squares for a piece type."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                sq = Square(row, col)
                if self.board.get_piece(sq) is not None:
                    continue
                
                # Pawn restrictions
                if piece_type == PieceType.PAWN:
                    # Can't drop pawn on last rank
                    if sq.in_promotion_zone(player):
                        continue
                    
                    # Two-pawn rule: can't have two pawns in same column
                    has_pawn = False
                    for r in range(BOARD_SIZE):
                        p = self.board.get_piece(Square(r, col))
                        if (p and p.owner == player and 
                            p.piece_type == PieceType.PAWN):
                            has_pawn = True
                            break
                    if has_pawn:
                        continue
                
                yield sq
    
    def _must_promote(self, piece: Piece, to_square: Square) -> bool:
        """Check if promotion is mandatory (pawn on last rank)."""
        if piece.piece_type == PieceType.PAWN:
            return to_square.in_promotion_zone(piece.owner)
        return False
    
    def apply_action(self, action: Action) -> 'GameState':
        """
        Apply an action and return the new game state.
        
        This is a pure function - returns new state without modifying self.
        """
        new_state = self.copy()
        
        if action.action_type == ActionType.DROP:
            # Drop a piece from hand
            piece = new_state.board.remove_from_hand(
                new_state.current_player, 
                action.drop_piece_type
            )
            if piece:
                new_state.board.set_piece(action.to_square, piece)
        else:
            # Move a piece
            piece = new_state.board.get_piece(action.from_square)
            if piece:
                # Handle capture
                captured = new_state.board.get_piece(action.to_square)
                if captured:
                    # Add demoted piece to captor's hand
                    demoted = captured.demoted()
                    new_state.board.add_to_hand(
                        new_state.current_player, 
                        demoted
                    )
                
                # Clear source square
                new_state.board.set_piece(action.from_square, None)
                
                # Place piece (possibly promoted)
                if action.promote:
                    piece = piece.promoted() or piece
                new_state.board.set_piece(action.to_square, piece)
        
        # Update state
        new_state.action_history.append((new_state.current_player, action))
        new_state.move_count += 1
        new_state.current_player = new_state.current_player.opponent()
        
        # Check for game end
        new_state._check_game_end()
        
        # Update position count for repetition
        pos_hash = new_state._position_hash()
        new_state.position_counts[pos_hash] = \
            new_state.position_counts.get(pos_hash, 0) + 1
        
        return new_state
    
    def _check_game_end(self) -> None:
        """Check for king capture (immediate win)."""
        sente_king = self.board.find_king(Player.SENTE)
        gote_king = self.board.find_king(Player.GOTE)
        
        if sente_king is None:
            self.result = GameResult.GOTE_WIN
        elif gote_king is None:
            self.result = GameResult.SENTE_WIN
    
    def _position_hash(self) -> int:
        """Hash current position for repetition detection."""
        items = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board.grid[r][c]
                if piece:
                    items.append((r, c, piece.piece_type, piece.owner))
        return hash(tuple(sorted(items)))
    
    def is_terminal(self) -> bool:
        """Check if game has ended."""
        return self.result != GameResult.ONGOING
    
    def get_payoff(self, player: Player) -> float:
        """
        Get terminal payoff for a player.
        
        Returns:
            +1.0 for win
            -1.0 for loss
            0.0 for draw or ongoing
        """
        if self.result == GameResult.ONGOING:
            return 0.0
        elif self.result == GameResult.DRAW:
            return 0.0
        elif (self.result == GameResult.SENTE_WIN and player == Player.SENTE) or \
             (self.result == GameResult.GOTE_WIN and player == Player.GOTE):
            return 1.0
        else:
            return -1.0
    
    def get_observation(self, player: Player) -> BoardView:
        """Get the observable board state for a player."""
        return self.board.get_player_view(player)
    
    def evaluate(self, player: Player) -> float:
        """
        Heuristic evaluation for non-terminal states.
        
        Returns a score from the player's perspective.
        Higher is better for the player.
        """
        if self.is_terminal():
            return self.get_payoff(player) * 10000
        
        score = 0.0
        
        # Material advantage
        score += self.board.material_score(player) / 100.0
        
        # Piece activity (center control)
        center = Square(BOARD_SIZE // 2, BOARD_SIZE // 2)
        for sq, piece in self.board.get_all_pieces():
            dist = sq.distance_to(center)
            activity = (BOARD_SIZE - dist) / BOARD_SIZE
            if piece.owner == player:
                score += activity * 0.1
            else:
                score -= activity * 0.1
        
        # King safety (simplified)
        my_king = self.board.find_king(player)
        opp_king = self.board.find_king(player.opponent())
        
        if my_king:
            # Prefer king safety (corner/edge in endgame)
            score += 0.05 * min(my_king.row, BOARD_SIZE - 1 - my_king.row)
        
        return score


class FogShogiGame:
    """
    High-level game manager for Fog of War Shogi.
    
    Manages:
    - Game state
    - Belief states for both players
    - Information set computation
    - Game tree traversal utilities
    """
    
    def __init__(self, visibility_radius: int = 2):
        """Initialize a new game."""
        board = create_initial_board()
        board.visibility_radius = visibility_radius
        
        self.state = GameState(
            board=board,
            current_player=Player.SENTE
        )
        
        # Initialize belief states
        self.beliefs = {
            Player.SENTE: BeliefState.from_initial_position(Player.SENTE, board),
            Player.GOTE: BeliefState.from_initial_position(Player.GOTE, board),
        }
    
    def get_legal_actions(self) -> List[Action]:
        """Get legal actions for current player."""
        return self.state.get_legal_actions()
    
    def apply_action(self, action: Action) -> None:
        """Apply an action and update beliefs."""
        prev_player = self.state.current_player
        self.state = self.state.apply_action(action)
        
        # Update beliefs for both players
        for player in [Player.SENTE, Player.GOTE]:
            observation = self.state.get_observation(player)
            self.beliefs[player].update(observation)
    
    def get_current_player(self) -> Player:
        """Get the player to act."""
        return self.state.current_player
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.state.is_terminal()
    
    def get_payoff(self, player: Player) -> float:
        """Get terminal payoff."""
        return self.state.get_payoff(player)
    
    def get_observation(self, player: Player) -> BoardView:
        """Get player's observable state."""
        return self.state.get_observation(player)
    
    def get_belief(self, player: Player) -> BeliefState:
        """Get player's belief state."""
        return self.beliefs[player]
    
    def get_information_set(self, player: Player) -> InformationSet:
        """Get the information set for current position."""
        observation = self.get_observation(player)
        return InformationSet.from_observation(
            observation, 
            self.state.action_history
        )
    
    def get_full_state(self) -> GameState:
        """Get the true game state (for analysis/debugging)."""
        return self.state
    
    def reset(self) -> None:
        """Reset to initial position."""
        self.__init__(self.state.board.visibility_radius)
    
    def __repr__(self) -> str:
        lines = [
            f"=== Fog Shogi Game ===",
            f"Move: {self.state.move_count}",
            f"To play: {self.state.current_player}",
            f"Result: {self.state.result}",
            "",
            "True Board State:",
            repr(self.state.board),
            "",
            f"SENTE's View:",
            repr(self.get_observation(Player.SENTE)),
            "",
            f"GOTE's View:", 
            repr(self.get_observation(Player.GOTE)),
        ]
        return "\n".join(lines)


# Utility functions for game tree analysis

def count_legal_actions(state: GameState) -> int:
    """Count legal actions without generating full list."""
    return len(state.get_legal_actions())


def minimax_value(state: GameState, depth: int, 
                  player: Player, alpha: float = float('-inf'),
                  beta: float = float('inf')) -> float:
    """
    Minimax with alpha-beta pruning for perfect information analysis.
    
    Note: This operates on true game state, not belief state.
    Used for benchmarking against optimal play.
    """
    if state.is_terminal() or depth == 0:
        return state.evaluate(player)
    
    if state.current_player == player:
        value = float('-inf')
        for action in state.get_legal_actions():
            child = state.apply_action(action)
            value = max(value, minimax_value(child, depth - 1, player, alpha, beta))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for action in state.get_legal_actions():
            child = state.apply_action(action)
            value = min(value, minimax_value(child, depth - 1, player, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
