from typing import Tuple

import numpy as np
from findiff import FinDiff


def gradient(data: np.ndarray, spacing: Tuple[np.ndarray, np.ndarray, np.ndarray], dim: int) -> np.ndarray:
    d_dx = [FinDiff(dim, spacing[dim][s]) for s in range(data.shape[0])]
    return np.stack([
        np.stack([
            d_dx[s](data[s, ..., t]) for t in range(data.shape[-1])
        ], axis=-1) for s in range(data.shape[0])
    ])


def divergence(grad: np.ndarray, grid_basis: np.ndarray) -> np.ndarray:
    _grad = np.einsum('bij,bjkxyzt->bikxyzt', grid_basis, grad)
    return _grad[:, 0, 0] + _grad[:, 1, 1] + _grad[:, 2, 2]