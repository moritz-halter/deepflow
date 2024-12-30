from typing import Tuple

import numpy as np
from findiff import FinDiff

from algebra.util import get_grid_sides, change_of_base_matrix


def gradient(matrix: np.ndarray, dim: int) -> np.ndarray:
    d_dx = FinDiff(dim, 1)
    return np.stack([
        np.stack([
            d_dx(matrix[s, ..., t]) for t in range(matrix.shape[-1])
        ], axis=-1) for s in range(matrix.shape[0])
    ], axis=0)

def angular_error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_norm = np.linalg.norm(y, axis=1)
    y_pred_norm = np.linalg.norm(y_pred, axis=1)
    dot = np.einsum('bixyzt,bixyzt->bxyzt', y, y_pred)
    norm_prod = y_norm * y_pred_norm

    return np.arccos(np.clip(np.divide(dot, norm_prod, where=norm_prod != 0, out=np.ones_like(dot)), -1, 1))

def divergence(matrix: np.ndarray) -> np.ndarray:
    return gradient(matrix[:, 0, ...], 0) + gradient(matrix[:, 1, ...], 1) + gradient(matrix[:, 2, ...], 2)


def divergence_error(y: np.ndarray, y_pred: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_x, d_y, d_z = get_grid_sides(coords)
    d_x /= np.stack([np.linalg.norm(d_x, axis=1)] * 3, 1)
    d_y /= np.stack([np.linalg.norm(d_y, axis=1)] * 3, 1)
    d_z /= np.stack([np.linalg.norm(d_z, axis=1)] * 3, 1)
    P = change_of_base_matrix(d_x, d_y, d_z, np.array([[1, 0, 0]] * y.shape[0]), np.array([[0, 1, 0]] * y.shape[0]), np.array([[0, 0, 1]] * y.shape[0]))
    div_y = divergence(np.einsum('bic,bchwdt->bihwdt', P, y))
    div_y_pred = divergence(np.einsum('bic,bchwdt->bihwdt', P, y_pred))
    return div_y, div_y_pred, np.abs(div_y - div_y_pred)

def mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    diff = y_pred - y
    return np.linalg.norm(diff, axis=1)