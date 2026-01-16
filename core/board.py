"""
Board Module with Fog of War
============================

Implements the game board with visibility mechanics. Key concepts:

Visibility Model:
- Each player can see squares within a radius of their pieces
- Visibility radius is configurable (default: 2 squares)
- Players always see their own pieces
- Hidden squares show fog symbol

Information Asymmetry:
- Creates imperfect information game structure
- Each player has a different view of the same board state
- Enables Bayesian inference about hidden pieces
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple, List, Dict
from copy import deepcopy
import numpy as np

from .pieces import Piece, PieceType, Player, get_movement_deltas, PIECE_VALUES


# Board dimensions for Mini-Shogi
BOARD_SIZE = 5


@dataclass
class Square:
    """Represents a single square on the board."""
    row: int
    col: int
    
    def __hash__(self) -> int:
        return hash((self.row, self.col))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Square):
            return False
        return self.row == other.row and self.col == other.col
    
    def is_valid(self) -> bool:
        """Check if square is within board bounds."""
        return 0 <= self.row < BOARD_SIZE and 0 <= self.col < BOARD_SIZE
    
    def in_promotion_zone(self, player: Player) -> bool:
        """Check if square is in promotion zone for given player."""
        if player == Player.SENTE:
            return self.row == 0  # Top row for SENTE
        else:
            return self.row == BOARD_SIZE - 1  # Bottom row for GOTE
    
    def distance_to(self, other: 'Square') -> int:
        """Chebyshev distance (max of row/col difference)."""
        return max(abs(self.row - other.row), abs(self.col - other.col))


@dataclass
class Board:
    """
    Game board with fog of war visibility mechanics.
    
    The board maintains the true game state and can generate
    observational views for each player based on visibility rules.
    
    Attributes:
        grid: 2D array of pieces (None for empty squares)
        visibility_radius: How far each piece can see
        sente_hand: Captured pieces available for SENTE to drop
        gote_hand: Captured pieces available for GOTE to drop
    """
    grid: List[List[Optional[Piece]]] = field(default_factory=lambda: [
        [None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
    ])
    visibility_radius: int = 2
    sente_hand: List[Piece] = field(default_factory=list)
    gote_hand: List[Piece] = field(default_factory=list)
    
    def copy(self) -> 'Board':
        """Deep copy the board."""
        new_board = Board(
            grid=[[self.grid[r][c] for c in range(BOARD_SIZE)] 
                  for r in range(BOARD_SIZE)],
            visibility_radius=self.visibility_radius,
            sente_hand=list(self.sente_hand),
            gote_hand=list(self.gote_hand),
        )
        return new_board
    
    def get_piece(self, square: Square) -> Optional[Piece]:
        """Get piece at a square."""
        if square.is_valid():
            return self.grid[square.row][square.col]
        return None
    
    def set_piece(self, square: Square, piece: Optional[Piece]) -> None:
        """Place or remove a piece."""
        if square.is_valid():
            self.grid[square.row][square.col] = piece
    
    def get_hand(self, player: Player) -> List[Piece]:
        """Get the hand (captured pieces) for a player."""
        return self.sente_hand if player == Player.SENTE else self.gote_hand
    
    def add_to_hand(self, player: Player, piece: Piece) -> None:
        """Add a captured piece to player's hand."""
        hand = self.get_hand(player)
        hand.append(piece)
    
    def remove_from_hand(self, player: Player, piece_type: PieceType) -> Optional[Piece]:
        """Remove a piece of given type from hand (for drops)."""
        hand = self.get_hand(player)
        for i, piece in enumerate(hand):
            if piece.piece_type == piece_type:
                return hand.pop(i)
        return None
    
    def get_visible_squares(self, player: Player) -> Set[Square]:
        """
        Calculate all squares visible to a player.
        
        Visibility rules:
        - Player sees all squares within visibility_radius of their pieces
        - Player always sees their hand pieces
        - Fog covers everything else
        """
        visible = set()
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.grid[row][col]
                if piece and piece.owner == player:
                    piece_square = Square(row, col)
                    # Add all squares within visibility radius
                    for dr in range(-self.visibility_radius, self.visibility_radius + 1):
                        for dc in range(-self.visibility_radius, self.visibility_radius + 1):
                            target = Square(row + dr, col + dc)
                            if target.is_valid():
                                visible.add(target)
        
        return visible
    
    def get_player_view(self, player: Player) -> 'BoardView':
        """
        Generate the observable board state for a player.
        
        This is the information set - squares outside visibility show fog.
        """
        visible = self.get_visible_squares(player)
        
        view_grid = [[None for _ in range(BOARD_SIZE)] 
                     for _ in range(BOARD_SIZE)]
        fog_mask = [[True for _ in range(BOARD_SIZE)] 
                    for _ in range(BOARD_SIZE)]
        
        for square in visible:
            fog_mask[square.row][square.col] = False
            view_grid[square.row][square.col] = self.grid[square.row][square.col]
        
        return BoardView(
            grid=view_grid,
            fog_mask=fog_mask,
            player=player,
            own_hand=list(self.get_hand(player)),
            visible_squares=visible,
        )
    
    def find_king(self, player: Player) -> Optional[Square]:
        """Find the king's position for a player."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.grid[row][col]
                if piece and piece.owner == player and piece.piece_type == PieceType.KING:
                    return Square(row, col)
        return None
    
    def get_all_pieces(self, player: Optional[Player] = None) -> List[Tuple[Square, Piece]]:
        """Get all pieces on board, optionally filtered by player."""
        pieces = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.grid[row][col]
                if piece and (player is None or piece.owner == player):
                    pieces.append((Square(row, col), piece))
        return pieces
    
    def material_score(self, player: Player) -> int:
        """Calculate material advantage for a player."""
        score = 0
        for _, piece in self.get_all_pieces():
            value = PIECE_VALUES[piece.piece_type]
            if piece.owner == player:
                score += value
            else:
                score -= value
        
        # Add hand pieces
        for piece in self.get_hand(player):
            score += PIECE_VALUES[piece.piece_type]
        for piece in self.get_hand(player.opponent()):
            score -= PIECE_VALUES[piece.piece_type]
        
        return score
    
    def __repr__(self) -> str:
        """ASCII representation of the board."""
        lines = ["  " + " ".join(str(c) for c in range(BOARD_SIZE))]
        for row in range(BOARD_SIZE):
            row_str = f"{row} "
            for col in range(BOARD_SIZE):
                piece = self.grid[row][col]
                if piece:
                    row_str += repr(piece) + " "
                else:
                    row_str += " ·  "
            lines.append(row_str)
        
        lines.append(f"SENTE hand: {self.sente_hand}")
        lines.append(f"GOTE hand: {self.gote_hand}")
        return "\n".join(lines)


@dataclass
class BoardView:
    """
    Observable board state from a player's perspective.
    
    This represents what a player can actually see - their information set.
    Hidden squares show fog, which requires Bayesian inference to reason about.
    """
    grid: List[List[Optional[Piece]]]
    fog_mask: List[List[bool]]  # True = fogged (hidden)
    player: Player
    own_hand: List[Piece]
    visible_squares: Set[Square]
    
    def is_visible(self, square: Square) -> bool:
        """Check if a square is visible."""
        return square in self.visible_squares
    
    def get_visible_piece(self, square: Square) -> Optional[Piece]:
        """Get piece at square if visible, None otherwise."""
        if self.is_visible(square):
            return self.grid[square.row][square.col]
        return None
    
    def count_fog_squares(self) -> int:
        """Count number of hidden squares."""
        return sum(1 for row in self.fog_mask for is_fog in row if is_fog)
    
    def __repr__(self) -> str:
        """ASCII representation with fog."""
        lines = ["  " + " ".join(str(c) for c in range(BOARD_SIZE))]
        for row in range(BOARD_SIZE):
            row_str = f"{row} "
            for col in range(BOARD_SIZE):
                if self.fog_mask[row][col]:
                    row_str += " ▓  "  # Fog symbol
                else:
                    piece = self.grid[row][col]
                    if piece:
                        row_str += repr(piece) + " "
                    else:
                        row_str += " ·  "
            lines.append(row_str)
        
        lines.append(f"Hand: {self.own_hand}")
        lines.append(f"Fog squares: {self.count_fog_squares()}")
        return "\n".join(lines)


def create_initial_board() -> Board:
    """
    Create the standard Mini-Shogi starting position.
    
    Mini-Shogi Layout (5x5):
    Row 0: GOTE's back rank (King, Gold, Silver, Bishop, Rook)
    Row 1: GOTE's pawn
    Row 2: Empty
    Row 3: SENTE's pawn
    Row 4: SENTE's back rank
    """
    board = Board()
    piece_id = 0
    
    # GOTE back rank (row 0)
    gote_pieces = [
        (0, 0, PieceType.ROOK),
        (0, 1, PieceType.BISHOP),
        (0, 2, PieceType.SILVER),
        (0, 3, PieceType.GOLD),
        (0, 4, PieceType.KING),
    ]
    
    for row, col, ptype in gote_pieces:
        board.set_piece(
            Square(row, col),
            Piece(ptype, Player.GOTE, piece_id)
        )
        piece_id += 1
    
    # GOTE pawn (row 1)
    board.set_piece(
        Square(1, 2),
        Piece(PieceType.PAWN, Player.GOTE, piece_id)
    )
    piece_id += 1
    
    # SENTE pawn (row 3)
    board.set_piece(
        Square(3, 2),
        Piece(PieceType.PAWN, Player.SENTE, piece_id)
    )
    piece_id += 1
    
    # SENTE back rank (row 4)
    sente_pieces = [
        (4, 0, PieceType.KING),
        (4, 1, PieceType.GOLD),
        (4, 2, PieceType.SILVER),
        (4, 3, PieceType.BISHOP),
        (4, 4, PieceType.ROOK),
    ]
    
    for row, col, ptype in sente_pieces:
        board.set_piece(
            Square(row, col),
            Piece(ptype, Player.SENTE, piece_id)
        )
        piece_id += 1
    
    return board
