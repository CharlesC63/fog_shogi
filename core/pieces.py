"""
Shogi Pieces Module
===================

Defines piece types, movement patterns, and promotion rules for Shogi.
Uses a simplified 5x5 Mini-Shogi variant for tractable CFR computation.

Mini-Shogi is strategically rich while keeping the state space manageable
for equilibrium computation (full 9x9 Shogi has ~10^71 game states).
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Optional


class Player(Enum):
    """Two-player zero-sum game."""
    SENTE = auto()  # First player (Black in Western terms)
    GOTE = auto()   # Second player (White in Western terms)
    
    def opponent(self) -> 'Player':
        return Player.GOTE if self == Player.SENTE else Player.SENTE
    
    def __repr__(self) -> str:
        return "☗" if self == Player.SENTE else "☖"


class PieceType(Enum):
    """
    Mini-Shogi piece types with their movement characteristics.
    
    Standard Mini-Shogi uses: King, Gold, Silver, Bishop, Rook, Pawn
    Each piece (except King and Gold) can promote when reaching promotion zone.
    """
    KING = auto()           # Gyoku/Ou - moves one square any direction
    GOLD = auto()           # Kin - moves one square orthogonally + forward diagonal
    SILVER = auto()         # Gin - moves one square diagonally + forward
    BISHOP = auto()         # Kaku - moves diagonally any distance
    ROOK = auto()           # Hisha - moves orthogonally any distance
    PAWN = auto()           # Fu - moves one square forward
    
    # Promoted pieces
    PROMOTED_SILVER = auto()  # Narigin - moves like Gold
    PROMOTED_BISHOP = auto()  # Uma - Bishop + one square orthogonally
    PROMOTED_ROOK = auto()    # Ryu - Rook + one square diagonally
    TOKIN = auto()            # Promoted Pawn - moves like Gold
    
    def can_promote(self) -> bool:
        """Check if this piece type can promote."""
        return self in {
            PieceType.SILVER, PieceType.BISHOP, 
            PieceType.ROOK, PieceType.PAWN
        }
    
    def promoted_form(self) -> Optional['PieceType']:
        """Get the promoted form of this piece."""
        promotions = {
            PieceType.SILVER: PieceType.PROMOTED_SILVER,
            PieceType.BISHOP: PieceType.PROMOTED_BISHOP,
            PieceType.ROOK: PieceType.PROMOTED_ROOK,
            PieceType.PAWN: PieceType.TOKIN,
        }
        return promotions.get(self)
    
    def demoted_form(self) -> 'PieceType':
        """Get the unpromoted form (for captured pieces)."""
        demotions = {
            PieceType.PROMOTED_SILVER: PieceType.SILVER,
            PieceType.PROMOTED_BISHOP: PieceType.BISHOP,
            PieceType.PROMOTED_ROOK: PieceType.ROOK,
            PieceType.TOKIN: PieceType.PAWN,
        }
        return demotions.get(self, self)
    
    def is_promoted(self) -> bool:
        return self in {
            PieceType.PROMOTED_SILVER, PieceType.PROMOTED_BISHOP,
            PieceType.PROMOTED_ROOK, PieceType.TOKIN
        }
    
    @property
    def symbol(self) -> str:
        """Japanese character representation."""
        symbols = {
            PieceType.KING: "玉",
            PieceType.GOLD: "金",
            PieceType.SILVER: "銀",
            PieceType.BISHOP: "角",
            PieceType.ROOK: "飛",
            PieceType.PAWN: "歩",
            PieceType.PROMOTED_SILVER: "全",
            PieceType.PROMOTED_BISHOP: "馬",
            PieceType.PROMOTED_ROOK: "龍",
            PieceType.TOKIN: "と",
        }
        return symbols[self]


@dataclass(frozen=True)
class Piece:
    """
    Immutable piece representation.
    
    Attributes:
        piece_type: The type of piece
        owner: The player who owns this piece
        piece_id: Unique identifier for tracking through fog of war
    """
    piece_type: PieceType
    owner: Player
    piece_id: int  # Unique ID for belief state tracking
    
    def __repr__(self) -> str:
        owner_mark = "+" if self.owner == Player.SENTE else "-"
        return f"{owner_mark}{self.piece_type.symbol}"
    
    def promoted(self) -> Optional['Piece']:
        """Return a new promoted piece, or None if cannot promote."""
        promoted_type = self.piece_type.promoted_form()
        if promoted_type:
            return Piece(promoted_type, self.owner, self.piece_id)
        return None
    
    def demoted(self) -> 'Piece':
        """Return demoted piece (for capture)."""
        return Piece(
            self.piece_type.demoted_form(),
            self.owner.opponent(),  # Changes ownership on capture
            self.piece_id
        )


# Movement deltas for each piece type
# Format: (row_delta, col_delta, is_sliding)
MOVEMENT_PATTERNS: dict[PieceType, List[Tuple[int, int, bool]]] = {
    PieceType.KING: [
        (-1, -1, False), (-1, 0, False), (-1, 1, False),
        (0, -1, False),                  (0, 1, False),
        (1, -1, False),  (1, 0, False),  (1, 1, False),
    ],
    PieceType.GOLD: [
        (-1, -1, False), (-1, 0, False), (-1, 1, False),
        (0, -1, False),                  (0, 1, False),
                         (1, 0, False),
    ],
    PieceType.SILVER: [
        (-1, -1, False), (-1, 0, False), (-1, 1, False),
        (1, -1, False),                  (1, 1, False),
    ],
    PieceType.BISHOP: [
        (-1, -1, True), (-1, 1, True),
        (1, -1, True),  (1, 1, True),
    ],
    PieceType.ROOK: [
        (-1, 0, True), (0, -1, True), (0, 1, True), (1, 0, True),
    ],
    PieceType.PAWN: [
        (-1, 0, False),
    ],
    PieceType.PROMOTED_SILVER: [
        (-1, -1, False), (-1, 0, False), (-1, 1, False),
        (0, -1, False),                  (0, 1, False),
                         (1, 0, False),
    ],
    PieceType.TOKIN: [
        (-1, -1, False), (-1, 0, False), (-1, 1, False),
        (0, -1, False),                  (0, 1, False),
                         (1, 0, False),
    ],
    PieceType.PROMOTED_BISHOP: [
        (-1, -1, True), (-1, 1, True), (1, -1, True), (1, 1, True),
        (-1, 0, False), (0, -1, False), (0, 1, False), (1, 0, False),
    ],
    PieceType.PROMOTED_ROOK: [
        (-1, 0, True), (0, -1, True), (0, 1, True), (1, 0, True),
        (-1, -1, False), (-1, 1, False), (1, -1, False), (1, 1, False),
    ],
}


def get_movement_deltas(piece: Piece) -> List[Tuple[int, int, bool]]:
    """
    Get movement deltas adjusted for player orientation.
    
    SENTE moves "up" the board (decreasing row index).
    GOTE moves "down" the board (increasing row index).
    """
    base_moves = MOVEMENT_PATTERNS[piece.piece_type]
    
    if piece.owner == Player.GOTE:
        return [(-dr, dc, sliding) for dr, dc, sliding in base_moves]
    return base_moves


# Material values for position evaluation (centipawn scale)
PIECE_VALUES: dict[PieceType, int] = {
    PieceType.KING: 10000,
    PieceType.ROOK: 1000,
    PieceType.BISHOP: 800,
    PieceType.GOLD: 500,
    PieceType.SILVER: 450,
    PieceType.PAWN: 100,
    PieceType.PROMOTED_ROOK: 1200,
    PieceType.PROMOTED_BISHOP: 1000,
    PieceType.PROMOTED_SILVER: 500,
    PieceType.TOKIN: 500,
}
