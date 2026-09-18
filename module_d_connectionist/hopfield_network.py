"""
Module D: Discrete Hopfield Network from Scratch (Pure NumPy).
Stores 4-6 bipolar binary patterns (8x8 = 64-bit box label bitmaps).
Demonstrates auto-associative memory and error correction under 10-15% bit-flip corruption.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class HopfieldNetwork:
    def __init__(self, size: int = 64):
        self.size = size  # 64 neurons (8x8 grid)
        self.weights = np.zeros((size, size), dtype=np.float64)

    def train(self, patterns: List[np.ndarray]):
        """
        Hebbian Learning rule:
        W = (1 / N) * sum_p (x^p (x^p)^T)
        Set diagonal W_ii = 0 (no self-connections).
        """
        self.weights = np.zeros((self.size, self.size), dtype=np.float64)
        num_patterns = len(patterns)

        for p in patterns:
            vec = p.flatten().astype(np.float64)
            assert len(vec) == self.size, f"Pattern size must be {self.size}"
            self.weights += np.outer(vec, vec)

        self.weights /= self.size
        np.fill_diagonal(self.weights, 0.0)

    def compute_energy(self, state: np.ndarray) -> float:
        """Energy function: E = -0.5 * s^T * W * s"""
        s = state.flatten().astype(np.float64)
        return float(-0.5 * np.dot(s, np.dot(self.weights, s)))

    def recall(self, corrupted_state: np.ndarray, max_iter: int = 20) -> Tuple[np.ndarray, List[float], int]:
        """
        Asynchronous/Synchronous update until convergence to minimum energy attractor.
        s_i(t+1) = sign(sum_j W_ij * s_j(t))
        Returns:
            - final_pattern: np.ndarray
            - energy_history: List[float]
            - iterations: int
        """
        s = corrupted_state.flatten().astype(np.float64).copy()
        energy_history = [self.compute_energy(s)]

        for it in range(max_iter):
            # Synchronous update
            activations = np.dot(self.weights, s)
            new_s = np.where(activations >= 0, 1.0, -1.0)

            energy = self.compute_energy(new_s)
            energy_history.append(energy)

            # Check convergence
            if np.array_equal(new_s, s):
                return new_s.reshape(corrupted_state.shape), energy_history, it + 1
            s = new_s

        return s.reshape(corrupted_state.shape), energy_history, max_iter

    @staticmethod
    def corrupt_pattern(pattern: np.ndarray, flip_ratio: float = 0.12) -> Tuple[np.ndarray, int]:
        """Flips a fraction of bits randomly (-1 -> 1 or 1 -> -1)."""
        corrupted = pattern.flatten().copy()
        num_flips = int(len(corrupted) * flip_ratio)
        flip_indices = np.random.choice(len(corrupted), size=num_flips, replace=False)
        corrupted[flip_indices] *= -1
        return corrupted.reshape(pattern.shape), num_flips

    @staticmethod
    def render_ascii_8x8(pattern: np.ndarray) -> str:
        """Renders an 8x8 pattern as readable ASCII characters ('#' for +1, '.' for -1)."""
        grid = pattern.reshape((8, 8))
        lines = []
        for r in range(8):
            row_str = "".join([" #" if grid[r, c] > 0 else " ." for c in range(8)])
            lines.append(row_str)
        return "\n".join(lines)


def generate_warehouse_label_patterns() -> Dict[str, np.ndarray]:
    """Generates 4 distinct 8x8 warehouse label patterns."""
    patterns = {}

    # Pattern 1: 'Cross / X' Hazard Box Label
    x_mat = -np.ones((8, 8), dtype=np.int8)
    for i in range(8):
        x_mat[i, i] = 1
        x_mat[i, 7 - i] = 1
    patterns["HAZARD_CROSS"] = x_mat

    # Pattern 2: 'Square / Perimeter' Standard Pallet Box Label
    sq_mat = -np.ones((8, 8), dtype=np.int8)
    sq_mat[0, :] = 1
    sq_mat[7, :] = 1
    sq_mat[:, 0] = 1
    sq_mat[:, 7] = 1
    patterns["PALLET_BORDER"] = sq_mat

    # Pattern 3: 'Plus / Medical' Fragile Kit Box Label
    plus_mat = -np.ones((8, 8), dtype=np.int8)
    plus_mat[3:5, :] = 1
    plus_mat[:, 3:5] = 1
    patterns["FRAGILE_PLUS"] = plus_mat

    # Pattern 4: 'Chevron / Arrow' Directional Box Label
    arr_mat = -np.ones((8, 8), dtype=np.int8)
    for i in range(4):
        arr_mat[i, 3 - i:5 + i] = 1
    arr_mat[4:7, 3:5] = 1
    patterns["ARROW_UP"] = arr_mat

    return patterns
