"""
Module C: Dock Contention Adversarial Game Environment.
5x5 Grid World where Robot 1 (MAX) and Robot 2 (MIN) race/block for the last charging dock.
Branching factor >= 3 (Up, Down, Left, Right, Wait).
"""

from typing import Tuple, List, Optional, Dict, Any
import copy


class DockGameState:
    DOCK_POS = (2, 2)
    OBSTACLES = {(1, 1), (3, 3), (1, 3), (3, 1)}
    GRID_SIZE = 5

    def __init__(
        self,
        r1_pos: Tuple[int, int] = (0, 0),
        r2_pos: Tuple[int, int] = (4, 4),
        turn: int = 1,  # 1 = MAX (Robot 1), -1 = MIN (Robot 2)
        step_count: int = 0,
        max_steps: int = 30
    ):
        self.r1_pos = r1_pos
        self.r2_pos = r2_pos
        self.turn = turn
        self.step_count = step_count
        self.max_steps = max_steps

    def is_terminal(self) -> Tuple[bool, Optional[float]]:
        """Returns (is_terminal, utility). Utility: +100 for R1 win, -100 for R2 win, 0 for draw."""
        if self.r1_pos == self.DOCK_POS:
            return True, 100.0
        if self.r2_pos == self.DOCK_POS:
            return True, -100.0
        if self.step_count >= self.max_steps:
            # Step limit reached -> evaluate by dock proximity
            d1 = abs(self.r1_pos[0] - self.DOCK_POS[0]) + abs(self.r1_pos[1] - self.DOCK_POS[1])
            d2 = abs(self.r2_pos[0] - self.DOCK_POS[0]) + abs(self.r2_pos[1] - self.DOCK_POS[1])
            if d1 < d2:
                return True, 50.0
            elif d2 < d1:
                return True, -50.0
            return True, 0.0
        return False, None

    def get_legal_moves(self, player: int) -> List[Tuple[int, int]]:
        """
        Legal actions: delta (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)].
        Branching factor: typically 3 to 5 options per state.
        """
        curr_pos = self.r1_pos if player == 1 else self.r2_pos
        other_pos = self.r2_pos if player == 1 else self.r1_pos

        deltas = [
            (-1, 0),  # North
            (1, 0),   # South
            (0, -1),  # West
            (0, 1),   # East
            (0, 0)    # Wait / Block
        ]
        legal = []
        for dr, dc in deltas:
            nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
            # Bounds check
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                # Obstacle check
                if (nr, nc) not in self.OBSTACLES:
                    # Cannot bump into opponent unless moving into dock
                    if (nr, nc) != other_pos or (nr, nc) == self.DOCK_POS:
                        legal.append((dr, dc))

        # Always at least (0, 0) Wait
        if not legal:
            legal.append((0, 0))
        return legal

    def apply_move(self, move: Tuple[int, int]) -> 'DockGameState':
        dr, dc = move
        new_state = copy.deepcopy(self)
        new_state.step_count += 1

        if self.turn == 1:
            new_r1 = (self.r1_pos[0] + dr, self.r1_pos[1] + dc)
            new_state.r1_pos = new_r1
            new_state.turn = -1
        else:
            new_r2 = (self.r2_pos[0] + dr, self.r2_pos[1] + dc)
            new_state.r2_pos = new_r2
            new_state.turn = 1

        return new_state

    def evaluate_heuristic(self) -> float:
        """
        Evaluation function for non-terminal states.
        Favors closer distance to Dock and higher mobility.
        """
        d1 = abs(self.r1_pos[0] - self.DOCK_POS[0]) + abs(self.r1_pos[1] - self.DOCK_POS[1])
        d2 = abs(self.r2_pos[0] - self.DOCK_POS[0]) + abs(self.r2_pos[1] - self.DOCK_POS[1])

        # Mobility
        m1 = len(self.get_legal_moves(1))
        m2 = len(self.get_legal_moves(-1))

        score = (d2 - d1) * 10.0 + (m1 - m2) * 1.5
        return score

    def render_ascii(self) -> str:
        lines = []
        lines.append("  0 1 2 3 4")
        for r in range(self.GRID_SIZE):
            row = [f"{r}"]
            for c in range(self.GRID_SIZE):
                pos = (r, c)
                if pos == self.r1_pos and pos == self.DOCK_POS:
                    row.append("1*")
                elif pos == self.r2_pos and pos == self.DOCK_POS:
                    row.append("2*")
                elif pos == self.r1_pos:
                    row.append("R1")
                elif pos == self.r2_pos:
                    row.append("R2")
                elif pos == self.DOCK_POS:
                    row.append(" D")
                elif pos in self.OBSTACLES:
                    row.append("##")
                else:
                    row.append(" .")
            lines.append(" ".join(row))
        return "\n".join(lines)
