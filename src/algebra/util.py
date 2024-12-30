from typing import List, Union, Tuple, Dict

import numpy as np


def get_transformation_matrix(corner: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    a = np.stack([x / np.linalg.norm(x),
                  y / np.linalg.norm(y),
                  z / np.linalg.norm(z)], axis=-1)
    t = corner + x / 2 + y / 2 + z / 2
    s = np.diag([np.linalg.norm(x),
                 np.linalg.norm(y),
                 np.linalg.norm(z)])

    scaling_mat = np.identity(4)
    scaling_mat[:3, :3] = s

    translate_mat = np.identity(4)
    translate_mat[:3, 3] = t

    rotate_mat = np.identity(4)
    rotate_mat[:3, :3] = a

    return translate_mat @ rotate_mat @ scaling_mat


def flat_to_grid(param_coords: np.ndarray, data: np.ndarray) -> np.ndarray:
    print(data.shape)
    dims = tuple(np.max(param_coords, axis=0) + 1)
    if data.ndim == 2:
        dims += (data.shape[-1],)
    res = np.empty(dims, dtype=data.dtype)
    if data.ndim == 2:
        for i, pc in enumerate(param_coords):
            res[*pc, :] = data[i, :]
    else:
        for i, pc in enumerate(param_coords):
            res[*pc] = data[i]
    return res


def flat_to_grid_dict(param_coords: np.ndarray, data_dict: Dict[str, np.ndarray]) -> dict:
    for k, v in data_dict.items():
        data_dict[k] = flat_to_grid(param_coords, v)
    return data_dict


def expand_boundary(boundary: np.ndarray, mask: np.ndarray) -> np.ndarray:
    res = np.copy(boundary)
    boundary_pad = np.pad(boundary, (1, 1), constant_values=0)
    res[np.where(np.logical_and(np.any(np.stack([
            boundary_pad[2:, 1:-1, 1:-1] == 1,
            boundary_pad[:-2, 1:-1, 1:-1] == 1,
            boundary_pad[1:-1, 2:, 1:-1] == 1,
            boundary_pad[1:-1, :-2, 1:-1] == 1,
            boundary_pad[1:-1, 1:-1, 2:] == 1,
            boundary_pad[1:-1, 1:-1, :-2] == 1
    ], axis=0), axis=0), mask == 1))] = 1
    return res


def down_sample(data: np.ndarray, sample_rate: int) -> np.ndarray:
    if data.ndim == 6:
        return data[:, :, ::sample_rate, ::sample_rate, ::sample_rate, :]
    else:
        return data[::sample_rate, ::sample_rate, ::sample_rate, ...]


def get_grid_sides(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_x = points[:, :, -1, 0, 0, 0] - points[:, :, 0, 0, 0, 0]
    delta_y = points[:, :, 0, -1, 0, 0] - points[:, :, 0, 0, 0, 0]
    delta_z = points[:, :, 0, 0, -1, 0] - points[:, :, 0, 0, 0, 0]
    return delta_x, delta_y, delta_z


def get_grid_basis(points: np.ndarray, with_spacing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    delta_x, delta_y, delta_z = get_grid_sides(points)
    grid_basis = np.stack([
        np.stack([
            delta_x[i, :] / np.linalg.norm(delta_x[i, :]),
            delta_y[i, :] / np.linalg.norm(delta_y[i, :]),
            delta_z[i, :] / np.linalg.norm(delta_z[i, :])
        ], axis=0) for i in range(delta_x.shape[0])
    ], axis=0)
    if with_spacing:
        spacing = (np.linalg.norm(delta_x, axis=-1) / (points.shape[2] - 1),
                   np.linalg.norm(delta_x, axis=-1) / (points.shape[3] - 1),
                   np.linalg.norm(delta_x, axis=-1) / (points.shape[4] - 1))
        return grid_basis, spacing
    else:
        return grid_basis


def rotation_matrix(old_basis: np.ndarray, new_basis: np.ndarray):
    old_basis_t = old_basis.swapaxes(1, 2)
    r = np.einsum('bxy,byz->bxz', new_basis, old_basis_t)
    return r


def rotate_vector_field_flat(field: np.ndarray, grid_basis: np.ndarray, inverse=False):
    basis = np.array([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]])
    rm = rotation_matrix(grid_basis.reshape(1, 3, 3), basis) if inverse else rotation_matrix(basis, grid_basis.reshape(1, 3, 3))
    return np.einsum('bic,bnc->bni', rm, field.reshape(1, *field.shape))[0, ...]


def rotate_vector_field(field: np.ndarray, grid_basis: np.ndarray, inverse=False):
    basis = np.array([[
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]] * grid_basis.shape[0])
    rm = rotation_matrix(grid_basis, basis)
    if inverse:
        rm = rm.swapaxes(1, 2)
    return np.einsum('bic,bchwdt->bihwdt', rm, field)


def lies_within(points: np.ndarray, corners: List[np.ndarray]):
    origin = corners[0]
    edge_vectors = [
        corners[1] - origin,
        corners[2] - origin,
        corners[3] - origin
    ]
    res = np.empty((points.shape[0], 3))
    try:
        for i in range(points.shape[0]):
            res[i] = np.linalg.solve(np.column_stack(edge_vectors), points[i] - origin)
    except np.linalg.LinAlgError:
        raise ValueError("The provided corners do not form a valid parallelepiped.")
    return np.all(np.logical_and(res <= 1, res >= 0), axis=-1)


def normalize(vectors: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vectors)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors,
                     norm,
                     where=norm != 0,
                     out=out)


def change_of_base_matrix(x_new: np.ndarray,
                          y_new: np.ndarray,
                          z_new: np.ndarray,
                          x_old: np.ndarray,
                          y_old: np.ndarray,
                          z_old: np.ndarray):
    b_new = np.stack([x_new, y_new, z_new], axis=2)
    b_old = np.stack([x_old, y_old, z_old], axis=2)
    b_old_inv = np.empty_like(b_old)
    for i in range(b_old.shape[0]):
        b_old_inv[i] = np.linalg.inv(b_old[i])
    return np.einsum('bxi,biy->bxy', b_new, b_old_inv)
