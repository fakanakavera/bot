from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class SeatState:
    has_glow: bool = False
    last_action: Optional[str] = None
    bet: Optional[str] = None


@dataclass
class GameState:
    dealer_seat: Optional[int] = None
    next_to_act_seat: Optional[int] = None
    hole_cards: List[Optional[str]] = field(default_factory=lambda: [None, None])
    board_cards: List[Optional[str]] = field(default_factory=lambda: [None, None, None, None, None])
    pot: Optional[str] = None
    seats: Dict[int, SeatState] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
