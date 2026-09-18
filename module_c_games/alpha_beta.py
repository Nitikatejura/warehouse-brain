"""
Module C: Alpha-Beta Pruning Algorithm for Dock Contention.
Maintains [alpha, beta] window to prune provably suboptimal branches.
Logs visited node counts to quantitatively demonstrate search space reduction.
"""

from typing import Tuple, Optional
from .dock_game_env import DockGameState


class AlphaBetaSolver:
    def __init__(self):
        self.nodes_explored = 0
        self.pruning_cutoffs = 0

    def get_best_move(self, state: DockGameState, depth: int) -> Tuple[Optional[Tuple[int, int]], float]:
        self.nodes_explored = 0
        self.pruning_cutoffs = 0
        best_move = None
        player = state.turn

        alpha = -float('inf')
        beta = float('inf')

        if player == 1:
            best_val = -float('inf')
            for move in state.get_legal_moves(1):
                next_state = state.apply_move(move)
                val = self._alphabeta(next_state, depth - 1, alpha, beta, False)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    self.pruning_cutoffs += 1
                    break
            return best_move, best_val
        else:
            best_val = float('inf')
            for move in state.get_legal_moves(-1):
                next_state = state.apply_move(move)
                val = self._alphabeta(next_state, depth - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    self.pruning_cutoffs += 1
                    break
            return best_move, best_val

    def _alphabeta(self, state: DockGameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
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
                eval_val = self._alphabeta(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    self.pruning_cutoffs += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in state.get_legal_moves(-1):
                next_state = state.apply_move(move)
                eval_val = self._alphabeta(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    self.pruning_cutoffs += 1
                    break
            return min_eval
