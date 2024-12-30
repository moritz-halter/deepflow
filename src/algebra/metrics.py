from typing import List

import numpy as np

from algebra.interpolation import interpolation
from algebra.util import get_grid_basis, lies_within


def lp_diff(x: np.ndarray, y: np.ndarray):
    diff = x - y
    return np.linalg.norm(diff, axis=1)


def angular_error(x: np.ndarray, y: np.ndarray):
    res = np.ones((x.shape[0], *x.shape[2:]))
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    dot = np.einsum('bixyzt,bixyzt->bxyzt', x, y)
    norm_prod = y_norm * x_norm
    return np.arccos(np.clip(np.divide(dot, norm_prod, where=norm_prod != 0, out=res), a_min=-1, a_max=1)) * 180 / np.pi


def wall_shear_stress(grad: np.ndarray, grid_points: np.ndarray, normals: List[np.ndarray], normal_points: List[np.ndarray]):
    grid_basis = get_grid_basis(grid_points)
    grad_stacked = grad.reshape((grad.shape[0], 9, *grad.shape[3:]))
    viscosity = 3.5e-3
    corners = [
        grid_points[:, :, 0, 0, 0, 0],
        grid_points[:, :, -1, 0, 0, 0],
        grid_points[:, :, 0, -1, 0, 0],
        grid_points[:, :, 0, 0, -1, 0]
    ]
    valid_indices = [lies_within(normal_points[i], [c[i] for c in corners]) for i in range(len(normal_points))]
    normal_points = [normal_points[i][valid_indices[i]] for i in range(len(valid_indices))]
    delta = 1 / np.stack([
        np.linalg.norm(corners[1] - corners[0], axis=1),
        np.linalg.norm(corners[2] - corners[0], axis=1),
        np.linalg.norm(corners[3] - corners[0], axis=1)
    ], axis=1)
    _grid_basis = np.einsum('bij,bi->bij', grid_basis, delta)
    _normal_points = [np.einsum('ik,nk->ni', _grid_basis[i], normal_points[i] - corners[0][i]) for i in range(len(normal_points))]
    normals = [-normals[i][valid_indices[i]] for i in range(len(valid_indices))]
    wss = []
    for s in range(grad.shape[0]):
        t_w = np.empty((*normal_points[s].shape, grad_stacked.shape[-1]))
        for t in range(grad_stacked.shape[-1]):
            surface_grad = interpolation(grad_stacked[s, ..., t],
                                         _normal_points[s]).reshape(3, 3, -1)
            I_minus_n = np.repeat(np.eye(3)[np.newaxis, :, :], normals[s].shape[0], 0) - np.einsum('ni,nj->nij', normals[s], normals[s])
            t_w[..., t] = viscosity * np.einsum('nik,nj,jl,lkn->ni', I_minus_n, normals[s], grid_basis[s], surface_grad)
        wss.append(t_w)
    return wss, valid_indices

