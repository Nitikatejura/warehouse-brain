"""
Module C: Plain MiniMax Algorithm for Dock Contention.
Performs full tree search up to a depth limit, tracking total nodes visited.
"""

from typing import Tuple, Optional
from .dock_game_env import DockGameState


class MiniMaxSolver:
    def __init__(self):
        self.nodes_explored = 0

    def get_best_move(self, state: DockGameState, depth: int) -> Tuple[Optional[Tuple[int, int]], float]:
        self.nodes_explored = 0
        best_move = None
        player = state.turn

        if player == 1:
            best_val = -float('inf')
            for move in state.get_legal_moves(1):
                next_state = state.apply_move(move)
                val = self._minimax(next_state, depth - 1, False)
                if val > best_val:
                    best_val = val
                    best_move = move
            return best_move, best_val
        else:
            best_val = float('inf')
            for move in state.get_legal_moves(-1):
                next_state = state.apply_move(move)
                val = self._minimax(next_state, depth - 1, True)
                if val < best_val:
                    best_val = val
                    best_move = move
            return best_move, best_val

    def _minimax(self, state: DockGameState, depth: int, is_maximizing: bool) -> float:
        self.nodes_explored += 1
        is_term, utility = state.is_terminal()
        if is_term:
            return utility
        if depth == 0:
            return state.evaluate_heuristic()

        if is_maximizing:
            max_eval = -float('inf')
            for move in state.get_legal_moves(1):
                next_state = state.apply_move(move)
                eval_val = self._minimax(next_state, depth - 1, False)
                max_eval = max(max_eval, eval_val)
            return max_eval
        else:
            min_eval = float('inf')
            for move in state.get_legal_moves(-1):
                next_state = state.apply_move(move)
                eval_val = self._minimax(next_state, depth - 1, True)
                min_eval = min(min_eval, eval_val)
            return min_eval
