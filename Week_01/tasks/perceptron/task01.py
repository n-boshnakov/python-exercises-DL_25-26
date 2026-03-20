from __future__ import annotations
import numpy as np

def create_dataset(n: int) -> list[tuple[int, int]]:
    return [(n, n * 2) for n in range(n)]

def initialize_weights(x: int, y: int) -> float:
    return np.random.uniform(x, y)
