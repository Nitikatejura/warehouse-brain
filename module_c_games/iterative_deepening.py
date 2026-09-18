"""
Module C: Iterative Deepening Search (IDS) with Move-Ordering Heuristics.
Refines search by prioritizing most promising candidate moves first,
maximizing Alpha-Beta cutoffs and enabling anytime decision making.
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from .dock_game_env import DockGameState


class IterativeDeepeningSolver:
    def __init__(self):
        self.total_nodes_explored = 0
        self.depth_profile: List[Dict[str, Any]] = []

    def _order_moves(self, state: DockGameState, moves: List[Tuple[int, int]], player: int) -> List[Tuple[int, int]]:
        """
        Move-ordering heuristic:
        Orders moves by immediate one-step heuristic evaluation to maximize alpha-beta cutoffs.
        """
        scored_moves = []
        for m in moves:
            nxt = state.apply_move(m)
            h = nxt.evaluate_heuristic()
            scored_moves.append((h, m))

        # MAX wants highest heuristic first; MIN wants lowest heuristic first
        reverse_order = (player == 1)
        scored_moves.sort(key=lambda x: x[0], reverse=reverse_order)
        return [m for _, m in scored_moves]

    def search(self, state: DockGameState, max_depth: int) -> Tuple[Optional[Tuple[int, int]], float, int, float]:
        """
        Runs Iterative Deepening from depth 1 to max_depth.
        Returns:
            - best_move: Tuple[int, int]
            - best_value: float
            - total_nodes: int
            - elapsed_time: float (seconds)
        """
        start_time = time.perf_counter()
        self.total_nodes_explored = 0
        self.depth_profile.clear()

        overall_best_move = None
        overall_best_val = 0.0

        for current_depth in range(1, max_depth + 1):
            d_start = time.perf_counter()
            d_nodes_before = self.total_nodes_explored

            move, val = self._alphabeta_ordered_root(state, current_depth)
            if move is not None:
                overall_best_move = move
                overall_best_val = val

            d_elapsed = time.perf_counter() - d_start
            d_nodes = self.total_nodes_explored - d_nodes_before

            self.depth_profile.append({
                "depth": current_depth,
                "nodes_at_depth": d_nodes,
                "cumulative_nodes": self.total_nodes_explored,
                "time_sec": d_elapsed,
                "best_move": overall_best_move,
                "eval": overall_best_val
            })

        total_elapsed = time.perf_counter() - start_time
        return overall_best_move, overall_best_val, self.total_nodes_explored, total_elapsed

    def _alphabeta_ordered_root(self, state: DockGameState, depth: int) -> Tuple[Optional[Tuple[int, int]], float]:
        player = state.turn
        legal_moves = state.get_legal_moves(player)
        ordered_moves = self._order_moves(state, legal_moves, player)

        best_move = None
        alpha = -float('inf')
        beta = float('inf')

        if player == 1:
            best_val = -float('inf')
            for move in ordered_moves:
                nxt = state.apply_move(move)
                val = self._alphabeta(nxt, depth - 1, alpha, beta, False)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_move, best_val
        else:
            best_val = float('inf')
            for move in ordered_moves:
                nxt = state.apply_move(move)
                val = self._alphabeta(nxt, depth - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_move, best_val

    def _alphabeta(self, state: DockGameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        self.total_nodes_explored += 1
        is_term, utility = state.is_terminal()
        if is_term:
            return utility
        if depth == 0:
            return state.evaluate_heuristic()

        if is_maximizing:
            max_eval = -float('inf')
            legal = state.get_legal_moves(1)
            ordered = self._order_moves(state, legal, 1)
            for move in ordered:
                nxt = state.apply_move(move)
                eval_val = self._alphabeta(nxt, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            legal = state.get_legal_moves(-1)
            ordered = self._order_moves(state, legal, -1)
            for move in ordered:
                nxt = state.apply_move(move)
                eval_val = self._alphabeta(nxt, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval
