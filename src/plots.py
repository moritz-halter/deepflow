from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from matplotlib.pyplot import colormaps
from mpl_toolkits.axes_grid1 import ImageGrid


# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter, vtkRenderWindowInteractor, vtkRenderWindow, vtkRenderer, \
    vtkActor, vtkPolyDataMapper

from algebra.gradient import gradient
from algebra.metrics import wall_shear_stress
from algebra.util import get_grid_basis, normalize
from file_io import get_surface_data, get_run_data

def create_mse_plots(mse: np.ndarray, save_path: Path = None):
    mean_time = np.mean(mse, axis=(1, 2, 3))
    mean_space = np.mean(mse, axis=-1)
    markers = ['o', 'v', '^', '<', '>', 's', '*', 'D', 'P']

    fig = plt.figure(figsize=(60, 20))
    ax = fig.add_subplot(111)
    ax.set_ylabel('average error [m/s]')
    ax.set_xlabel('t [s]')
    fontsize=50
    for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    t = np.linspace(0, mse.shape[-1] * 0.01, mse.shape[-1])
    for i in range(mse.shape[0]):
            ax.plot(t, mean_time[i], color='blue', marker=markers[i])
    if save_path is not None:
        plt.savefig(save_path.joinpath('flow_err_vs_time.png'),
                    transparent=True,
                    dpi='figure',
                    format='png')
    else:
        plt.show()


    fig = plt.figure(figsize=(60, 20))
    grid = ImageGrid(fig,
                     rect=111,
                     nrows_ncols=(1, 3),
                     axes_pad=(0, 0),
                     share_all=True,
                     label_mode='L',
                     cbar_pad=0.5,
                     cbar_location='right')
    x_mean_space = mean_space[6, 16, :, :]
    y_mean_space = mse[6, :, 16, :]
    z_mean_space = mse[6, :, :, 16]
    norm = Normalize(vmin=min([np.min(x_mean_space),
                               np.min(y_mean_space),
                               np.min(z_mean_space)]),
                     vmax=max([np.max(x_mean_space),
                               np.max(y_mean_space),
                               np.max(z_mean_space)]))
    ax = grid[0]
    ax.set_axis_off()
    ax.imshow(mean_space[6, :, :, 16], norm=norm, cmap='binary')
    ax = grid[1]
    ax.set_axis_off()
    ax.imshow(mean_space[6, :, 16, :], norm=norm, cmap='binary')
    ax = grid[2]
    ax.set_axis_off()
    im = ax.imshow(mean_space[6, 16, :, :], norm=norm, cmap='binary')
    grid.cbar_axes[0].colorbar(im)

    if save_path is not None:
        plt.savefig(save_path.joinpath('flow_err_vs_space.png'),
                    transparent=True,
                    dpi='figure',
                    format='png')
    else:
        plt.show()

def gather_samples_data(geometries_dir: Path, sample_names):
    surface_points = []
    surface_normals = []
    eigenvectors = []
    surface_wss = []
    surface_u = []
    for sample_name in sample_names:
        sp, sn, eig, wss, u = get_surface_data(sample_name,
                                               geometries_dir)
        surface_points.append(sp)
        surface_normals.append(sn)
        eigenvectors.append(eig)
        surface_wss.append(wss)
        surface_u.append(u)
    return surface_points, surface_normals, eigenvectors, surface_wss, surface_u

def eigenvectors_plot(eigenvectors: np.ndarray, points: np.ndarray,
                      s: int = 1, colormap: str = 'coolwarm',
                      elev: float = 90, azim: float = -90, roll: float = 0,
                      save_path: Path = None, wspace: float = 0):
    if colormap not in list(colormaps):
        print('Unknown colormap {colormap}')
        colormap = 'coolwarm'

    offset = 8
    fig = plt.figure(figsize=(300, 50))
    ax = fig.add_subplot(161, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 0 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(162, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 1 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(163, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 2 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(164, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 3 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(165, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 4 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(166, projection='3d')
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=eigenvectors[:, 5 + offset], cmap=colormap, s=s)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.subplots_adjust(wspace=wspace)
    if save_path is not None:
        fig.savefig(save_path.as_posix(),
                    dpi='figure',
                    format='png',
                    transparent=True)
    else:
        plt.show()


def flow_plot(points: np.ndarray, flow: np.ndarray, save_path: Path = None):
    points_flat = points[:, ::4, :, :].reshape(3, -1).swapaxes(0, 1)
    flow_flat = flow[:, ::4, :, :, 0].reshape(3, -1).swapaxes(0, 1)
    norm_adjusted = Normalize(vmin=flow_flat.min(), vmax=flow_flat.max())
    indices = np.where(np.linalg.norm(flow_flat, axis=-1) > 0.01)
    points_flat = points_flat[*indices, :]
    flow_flat = flow_flat[*indices, :]

    elev = -90
    azim = 0
    roll = -30

    s = 500
    colormap = 'coolwarm'

    fig = plt.figure(figsize=(150,50))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], c=flow_flat[:, 0], s=s, cmap=colormap, norm=norm_adjusted)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], c=flow_flat[:, 1], s=s, cmap=colormap, norm=norm_adjusted)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], c=flow_flat[:, 2], s=s, cmap=colormap, norm=norm_adjusted)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.subplots_adjust(wspace=-0.3)
    if save_path is not None:
        plt.savefig(save_path.joinpath('flow_scatter.png'),
                    dpi='figure',
                    format='png',
                    transparent=True)

    flow_grid = flow[:, :, :, :, 0].copy()
    flow_grid[:, *np.where(np.linalg.norm(flow_grid, axis=0) < 0.01)] = np.nan
    norm = Normalize(vmin=np.min(flow_grid), vmax=np.max(flow_grid))
    fig = plt.figure(figsize=(150,150))
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(3, 3),
                     axes_pad=(0, 0),
                     share_all=True,
                     label_mode='L',
                     cbar_mode='single',
                     cbar_pad=0.5,
                     cbar_location='right')
    ax = grid[0]
    ax.imshow(flow_grid[0, ..., flow_grid.shape[3] // 2], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[1]
    ax.imshow(flow_grid[1, ..., flow_grid.shape[3] // 2], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[2]
    ax.imshow(flow_grid[2, ..., flow_grid.shape[3] // 2], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[3]
    ax.imshow(flow_grid[0, ..., flow_grid.shape[2] // 2, :], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[4]
    ax.imshow(flow_grid[1, ..., flow_grid.shape[2] // 2, :], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[5]
    ax.imshow(flow_grid[2, ..., flow_grid.shape[2] // 2, :], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[6]
    ax.imshow(flow_grid[0, flow_grid.shape[1] // 2, ...], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[7]
    ax.imshow(flow_grid[1, flow_grid.shape[1] // 2, ...], cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax = grid[8]
    pos = ax.imshow(flow_grid[2, flow_grid.shape[1] // 2, ...], cmap=colormap, norm=norm)
    ax.set_axis_off()
    grid.cbar_axes[0].colorbar(pos)
    grid.cbar_axes[0].tick_params(labelsize=10)
    if save_path is not None:
        plt.savefig(save_path.joinpath('flow_imshow.png'),
                    dpi='figure',
                    format='png',
                    transparent=True)
    else:
        plt.show()


def create_flow_plot(ds_file: Path, save_dir: Union[Path, None] = None):
    data = torch.load(ds_file)
    u = data['u']
    noise = data['noise']
    coords = data['coords'].permute(0, 5, 1, 2, 3, 4)
    coords_flat = coords[24, :, ::4, :, :, 0].reshape(3, -1)
    u_noise = u + noise
    u_flat = u[24, :, ::4, :, :, 0].reshape(3, -1)
    u_noise_flat = u_noise[24, :, ::4, :, :, 0].reshape(3, -1)
    vmin = min(u_flat.min(), u_noise_flat.min())
    vmax = max(u_flat.max(), u_noise_flat.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    indices = np.where(np.linalg.norm(u_flat, axis=0) > 0.01)
    coords_flat = coords_flat[:, *indices]
    u_flat = u_flat[:, *indices]
    u_noise_flat = u_noise_flat[:, *indices]

    elev = -80
    azim = 0
    roll = -30

    s = 1500
    colormap = 'coolwarm'

    fig = plt.figure(figsize=(150,100))
    ax = fig.add_subplot(231, projection='3d')
    ax.scatter(coords_flat[0, :], coords_flat[1, :], coords_flat[2, :], c=u_flat[0, :], s=s, cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(232, projection='3d')
    ax.scatter(coords_flat[0, :], coords_flat[1, :], coords_flat[2, :], c=u_flat[1, :], s=s, cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(233, projection='3d')
    ax.scatter(coords_flat[0, :], coords_flat[1, :], coords_flat[2, :], c=u_flat[2, :], s=s, cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(234, projection='3d')
    ax.scatter(coords_flat[0, :], coords_flat[1, :], coords_flat[2, :], c=u_noise_flat[0, :], s=s, cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(235, projection='3d')
    ax.scatter(coords_flat[0, :], coords_flat[1, :], coords_flat[2, :], c=u_noise_flat[1, :], s=s, cmap=colormap, norm=norm)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax = fig.add_subplot(236, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    # cbar_ax = fig.add_axes((0.95, 0.25, 0.025, 0.5))
    # fig.colorbar(pos, cax=cbar_ax)
    fig.subplots_adjust(hspace=-0.4,
                        wspace=-0.2)
    if save_dir is not None:
        plt.savefig(save_dir.joinpath('flow_scatter_with_noise.png'),
                    transparent=True,
                    format='png',
                    dpi='figure')
    else:
        plt.show()


def create_p_v_t_plot(samples_dir: Path, save_dir: Union[Path, None] = None):
    markers = ['o', 'v', '^', '<', '>', 's', '*', 'D', 'P']

    fig = plt.figure(figsize=(60, 20))
    ax = fig.add_subplot(111)
    ax.set_ylabel('u [m/s]')
    ax.set_xlabel('t [s]')
    i = 0
    offset = 21
    for sample in samples_dir.iterdir():
        if sample.as_posix().endswith('.npz') and sample.stem in ['C0045', 'C0048', 'C0020_1', 'C0056', 'C0043',
                                                                  'C0079', 'C0069', 'C0080_1']:
            data = np.load(sample)
            u = data['u']
            t = data['t'] - data['t'][0]
            u_mag = np.linalg.norm(u, axis=-1)
            n_points = np.where(u_mag > 0)[0].shape[0] / 102
            u_mag_avg = np.sum(u_mag, axis=(0, 1, 2)) / n_points
            ax.plot(t, u_mag_avg[:-1], color='blue', marker=markers[i])
            if i == 0:
                ax.vlines(t[offset], 0, 0.55, color='red')
                ax.vlines(t[offset + 25 - 1], 0, 0.55, color='red')
            i += 1
    fontsize=50
    for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize)
    if save_dir is not None:
        plt.savefig(save_dir.joinpath('u_vs_t.png').as_posix(),
                    transparent=True,
                    format='png',
                    dpi='figure')
    else:
        plt.show()


def create_comparison_plot(task_dir: Path, geometries_dir: Path, runs: List[str], sample_names: List[str], save_dir: Union[Path, None] = None):

    surface_points, surface_normals, eigenvectors, surface_wss, surface_u = gather_samples_data(geometries_dir, sample_names)

    truth = []
    prediction = []
    wss_truth = []
    wss_prediction = []
    indices = []

    for run in runs:
        print(run)
        run_dir = task_dir.joinpath(run)
        points, _prediction, _truth = get_run_data(run_dir, 32)
        truth.append(_truth)
        prediction.append(_prediction)

        grid_basis, spacing = get_grid_basis(points, with_spacing=True)
        grad_prediction = np.stack([gradient(_prediction, spacing=spacing, dim=i) for i in range(3)], axis=1)
        grad_truth = np.stack([gradient(_truth, spacing=spacing, dim=i) for i in range(3)], axis=1)

        _wss_truth, _ = wall_shear_stress(grad_truth, points, surface_normals, surface_points)
        _wss_prediction, _indices = wall_shear_stress(grad_prediction, points, surface_normals, surface_points)

        wss_truth.append(_wss_truth)
        wss_prediction.append(_wss_prediction)
        indices.append(_indices)

    u_truth = [x[6, :, :, :, 16, 10] for x in truth]
    u_prediction = [x[6, :, :, :, 16, 10] for x in prediction]
    wss_truth = [x[6][..., 10] for x in wss_truth]
    wss_prediction = [x[6][..., 10] for x in wss_prediction]
    _u_truth = [np.copy(x) for x in u_truth]
    _u_prediction = [np.copy(x) for x in u_prediction]
    for r in range(len(_u_truth)):
        n = np.linalg.norm(_u_truth[r], axis=0)
        _u_truth[r][:, *np.where(n == 0)] = np.nan
        _u_prediction[r][:, *np.where(n == 0)] = np.nan
    v_max_truth = max([np.max(x) for x in u_truth])
    v_max_prediction = max(np.max(x) for x in u_prediction)
    v_min_truth = max([np.min(x) for x in u_truth])
    v_min_prediction = max([np.min(x) for x in u_prediction])
    v_max_wss_truth = max([np.max(x) for x in wss_truth])
    v_max_wss_prediction = max([np.max(x) for x in wss_prediction])
    v_min_wss_truth = min([np.min(x) for x in wss_truth])
    v_min_wss_prediction = min([np.min(x) for x in wss_prediction])
    norm = Normalize(vmin=min(v_min_truth, v_min_prediction), vmax=max(v_max_truth, v_max_prediction))
    norm_wss = Normalize(vmin=min(v_min_wss_truth, v_min_wss_prediction),
                         vmax=max(v_max_wss_truth, v_max_wss_prediction))
    wss_truth_norm = [np.linalg.norm(x, axis=1) for x in wss_truth]
    wss_prediction_norm = [np.linalg.norm(x, axis=1) for x in wss_prediction]
    fontsize = 100
    length = 2e-4
    colormap = cm.plasma
    linewidth = 4.0

    fig = plt.figure(figsize=(150, 75))

    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(3, 6),
                     axes_pad=(0, 0),
                     cbar_location='right',
                     cbar_mode='single')

    ax = grid[0]
    ax.set_title('ground truth', fontsize=fontsize)
    ax.set_ylabel('u [m/s]', fontsize=fontsize, rotation=0, labelpad=200)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_truth[0][0], norm=norm, cmap='coolwarm')
    ax = grid[6]
    ax.set_ylabel('v [m/s]', fontsize=fontsize, rotation=0, labelpad=200)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_truth[0][1], norm=norm, cmap='coolwarm')
    ax = grid[12]
    ax.set_ylabel('w [m/s]', fontsize=fontsize, rotation=0, labelpad=200)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_truth[0][2], norm=norm, cmap='coolwarm')

    # lep-fno
    ax = grid[1]
    ax.set_title('LEP-FNO', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[0][0], norm=norm, cmap='coolwarm')
    ax = grid[7]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[0][1], norm=norm, cmap='coolwarm')
    ax = grid[13]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[0][2], norm=norm, cmap='coolwarm')

    # fno
    ax = grid[2]
    ax.set_title('FNO', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[1][0], norm=norm, cmap='coolwarm')
    ax = grid[8]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[1][1], norm=norm, cmap='coolwarm')
    ax = grid[14]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[1][2], norm=norm, cmap='coolwarm')

    # edsr
    ax = grid[3]
    ax.set_title('EDSR', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[2][0], norm=norm, cmap='coolwarm')
    ax = grid[9]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[2][1], norm=norm, cmap='coolwarm')
    ax = grid[15]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[2][2], norm=norm, cmap='coolwarm')

    # srcnn
    ax = grid[4]
    ax.set_title('SRCNN', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[3][0], norm=norm, cmap='coolwarm')
    ax = grid[10]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[3][1], norm=norm, cmap='coolwarm')
    ax = grid[16]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[3][2], norm=norm, cmap='coolwarm')

    # linear interpolation
    ax = grid[5]
    ax.set_title('linear interpolation', fontsize=fontsize)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[4][0], norm=norm, cmap='coolwarm')
    ax = grid[11]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(_u_prediction[4][1], norm=norm, cmap='coolwarm')
    ax = grid[17]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    im_flow = ax.imshow(_u_prediction[0][2], norm=norm, cmap='coolwarm')
    cbar = plt.colorbar(im_flow, cax=grid[0].cax)
    tick_labels = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(tick_labels, fontsize=fontsize)

    if save_dir is not None:
        plt.savefig(save_dir.joinpath('comparison_flow.png'),
                    transparent=True,
                    format='png',
                    dpi='figure')
    else:
        plt.show()

    elev = -90
    azim = 0
    roll = -30

    fig_wss = plt.figure(figsize=(150, 25))
    ax = fig_wss.add_subplot(1, 6, 1, projection='3d')
    ax.set_ylabel('wss', fontsize=fontsize, rotation=0, labelpad=65)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_truth[0][:, 0],
              wss_truth[0][:, 1],
              wss_truth[0][:, 2],
              length=length,
              colors=colormap(wss_truth_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')
    ax = fig_wss.add_subplot(1, 6, 2, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_prediction[0][:, 0],
              wss_prediction[0][:, 1],
              wss_prediction[0][:, 2],
              length=length,
              colors=colormap(wss_prediction_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')
    ax = fig_wss.add_subplot(1, 6, 3, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_prediction[1][:, 0],
              wss_prediction[1][:, 1],
              wss_prediction[1][:, 2],
              length=length,
              colors=colormap(wss_prediction_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')
    ax = fig_wss.add_subplot(1, 6, 4, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_prediction[2][:, 0],
              wss_prediction[2][:, 1],
              wss_prediction[2][:, 2],
              length=length,
              colors=colormap(wss_prediction_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')
    ax = fig_wss.add_subplot(1, 6, 5, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_prediction[3][:, 0],
              wss_prediction[3][:, 1],
              wss_prediction[3][:, 2],
              length=length,
              colors=colormap(wss_prediction_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')
    ax = fig_wss.add_subplot(1, 6, 6, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.quiver(surface_points[6][indices[0][6], 0],
              surface_points[6][indices[0][6], 1],
              surface_points[6][indices[0][6], 2],
              wss_prediction[4][:, 0],
              wss_prediction[4][:, 1],
              wss_prediction[4][:, 2],
              length=length,
              colors=colormap(wss_prediction_norm[0]),
              norm=norm_wss,
              linewidth=linewidth,
              pivot='middle')

    plt.subplots_adjust(wspace=-0.7)
    if save_dir is not None:
        plt.savefig(save_dir.joinpath('comparison_wss.png'),
                    transparent=True,
                    format='png',
                    dpi='figure')
    else:
        plt.show()


def surface_mesh_plot(sample_name: str, surface_dir: Path, save_dir: Path):
    reader = vtkPolyDataReader()

    reader.SetFileName(surface_dir.joinpath(f'{sample_name}_surface.vtk'))
    reader.Update()

    colors = vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)

    polyMapper = vtkPolyDataMapper()
    polyMapper.SetInputConnection(reader.GetOutputPort())
    polyMapper.ScalarVisibilityOff()

    polyActor = vtkActor()
    polyActor.SetMapper(polyMapper)
    polyActor.GetProperty().SetColor(colors.GetColor3d("Beige"))
    polyActor.GetProperty().EdgeVisibilityOn()
    polyActor.RotateX(15)
    polyActor.RotateY(0)
    polyActor.RotateZ(130)

    axesWidget = vtkOrientationMarkerWidget()

    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    axesActor = vtkAxesActor()
    axesWidget.SetOrientationMarker(axesActor)
    axesWidget.SetInteractor(iren)
    axesWidget.EnabledOff()

    ren.AddActor(polyActor)
    ren.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(300, 300)
    renWin.SetWindowName('Surface')
    renWin.SetAlphaBitPlanes(1)

    iren.Initialize()

    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.85)
    renWin.Render()

    iren.Start()

    imgFilter = vtkWindowToImageFilter()
    imgFilter.SetInput(renWin)
    imgFilter.SetInputBufferTypeToRGBA()

    pngWriter = vtkPNGWriter()
    pngWriter.SetInputConnection(imgFilter.GetOutputPort())
    pngWriter.SetFileName(save_dir.joinpath('surface.png').as_posix())
    pngWriter.Write()


def surface_plot(points, normals, wss, save_dir: Union[Path, None] = None):
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=0, roll=-30)
    ax.quiver(points[:, 0],
              points[:, 1],
              points[:, 2],
              normals[:, 0],
              normals[:, 1],
              normals[:, 2],
              length=0.0004,
              linewidth=2)
    ax.set_axis_off()
    if save_dir is not None:
        plt.savefig(save_dir.joinpath('surface_normals.png'))
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=0, roll=-30)
    ax.quiver(points[:, 0],
              points[:, 1],
              points[:, 2],
              wss[:, 0],
              wss[:, 1],
              wss[:, 2],
              length=0.01)
    ax.set_axis_off()
    if save_dir is not None:
        plt.savefig(save_dir.joinpath('surface_wss.png'))
    else:
        plt.show()


def wall_shear_stress_plot(wss, points, length=1.0, norm=True, save_path: Union[str, Path] = None):
    colormap = cm.inferno
    wss_norm = normalize(wss[..., 0]) if norm else wss[..., 0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=0, roll=-30)
    ax.quiver(points[:, 0],
              points[:, 1],
              points[:, 2],
              wss_norm[:, 0],
              wss_norm[:, 1],
              wss_norm[:, 2],
              length=length,
              colors=colormap(np.linalg.norm(wss[..., 0], axis=1)))
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path.joinpath('wall_shear_stress_quiver.png'))
    else:
        plt.show()
