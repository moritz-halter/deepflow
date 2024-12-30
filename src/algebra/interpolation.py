import numpy as np
import Divergence_Free_Interpolant as dfi
from scipy.interpolate import Rbf, RegularGridInterpolator

def dfi_interpolate(matrix: np.ndarray, old_coords, new_coords):
    new_grid = np.meshgrid(*new_coords)
    old_grid = np.stack(np.meshgrid(*old_coords), axis=0).reshape(3, -1)
    interpolator = dfi.interpolant()
    interpolator.condition(old_grid.T, matrix.reshape(3, -1).T)
    return np.permute_dims(interpolator(*new_grid), (3, 0, 1, 2))


def rbf_interpolation(matrix: np.ndarray, old_coords, new_coords):
    x_new, y_new, z_new = np.meshgrid(*new_coords)
    old_grid = np.meshgrid(*old_coords)
    res = np.empty((matrix.shape[0], new_coords[0].shape[0], new_coords[1].shape[0], new_coords[2].shape[0]))
    for c in range(matrix.shape[0]):
        interpolator = Rbf(*old_grid, matrix[c].reshape(-1))
        res[c, ...] = interpolator(x_new, y_new, z_new)
    return res


def linear_interpolation(matrix: np.ndarray, old_coords, new_coords, sampling_rate):
    oob_limit = - sampling_rate + 1
    new_grid = np.stack(np.meshgrid(*new_coords), axis=-1)[:oob_limit, :oob_limit, :oob_limit, :].reshape(-1, 3)
    res = np.zeros((matrix.shape[0], matrix.shape[1] * sampling_rate, matrix.shape[2] * sampling_rate, matrix.shape[3] * sampling_rate))
    res[:, oob_limit:, oob_limit:, oob_limit:] = np.tile(matrix[:, oob_limit, oob_limit, oob_limit, np.newaxis, np.newaxis, np.newaxis], (1, -oob_limit, -oob_limit, -oob_limit))
    for c in range(matrix.shape[0]):
        interpolator = RegularGridInterpolator(old_coords, matrix[c, ...])
        res[c, :oob_limit, :oob_limit, :oob_limit] = np.reshape(interpolator(new_grid), (res.shape[1] + oob_limit, res.shape[2] + oob_limit, res.shape[3] + oob_limit))
    return np.permute_dims(res, (0, 2, 1, 3))

def interpolation_time(matrix: np.ndarray, sampling_rate):
    n_t = matrix.shape[-1] * sampling_rate
    d = 1 / (n_t - 1)
    t_old = np.linspace(0, 1 - (sampling_rate - 1) * d, n_t // sampling_rate)
    t_new = np.linspace(0, 1, n_t)
    res = np.empty((matrix.shape[0], matrix.shape[1], matrix.shape[2], matrix.shape[3], matrix.shape[4], n_t))
    for s in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            for x in range(matrix.shape[2]):
                for y in range(matrix.shape[3]):
                    for z in range(matrix.shape[4]):
                        res[s, c, x, y, z, :] = np.interp(t_new, t_old, matrix[s, c, x, y, z, :])
    return res

def interpolation_grid(matrix: np.ndarray, resolution, sampling_rate, interpolation_mode: str = 'linear'):
    d = 1 / (resolution - 1)
    x_old = np.linspace(0, 1 - (sampling_rate - 1) * d, resolution // sampling_rate)
    y_old = np.linspace(0, 1 - (sampling_rate - 1) * d, resolution // sampling_rate)
    z_old = np.linspace(0, 1 - (sampling_rate - 1) * d, resolution // sampling_rate)
    x_new = np.linspace(0, 1, resolution)
    y_new = np.linspace(0, 1, resolution)
    z_new = np.linspace(0, 1, resolution)
    res = np.empty((matrix.shape[0], matrix.shape[1], resolution, resolution, resolution, matrix.shape[-1]))
    for t in range(matrix.shape[-1]):
        for s in range(matrix.shape[0]):
            if interpolation_mode == 'rbf':
                res[s, ..., t] = rbf_interpolation(matrix[s, ..., t], (x_old, y_old, z_old), (x_new, y_new, z_new))
            elif interpolation_mode == 'dfi':
                res[s, ..., t] = dfi_interpolate(matrix[s, ..., t], (x_old, y_old, z_old), (x_new, y_new, z_new))
            else:
                res[s, ..., t] = linear_interpolation(matrix[s, ..., t], (x_old, y_old, z_old), (x_new, y_new, z_new), sampling_rate)
    return res


def interpolation(data: np.ndarray, interpolation_points: np.ndarray):
    res = np.zeros((data.shape[0], interpolation_points.shape[0]))
    for c in range(data.shape[0]):
        interpolator = RegularGridInterpolator((np.linspace(0, 1, data.shape[1]),
                                                np.linspace(0, 1, data.shape[2]),
                                                np.linspace(0, 1, data.shape[3])), data[c])
        res[c, ...] = interpolator(interpolation_points)
    return res


def interpolate_flat(data: np.ndarray, points: np.ndarray, interpolation_points: np.ndarray):
    res = np.zeros((interpolation_points.shape[0], data.shape[1]))
    for c in range(data.shape[-1]):
        interpolator = Rbf(points[:, 0], points[:, 1], points[:, 2], data[:, c])
        res[:, c] = interpolator(interpolation_points[:, 0], interpolation_points[:, 1], interpolation_points[:, 2])
    return res
