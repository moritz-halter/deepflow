from typing import List, Union, Tuple

import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def get_axis(fig: plt.Figure,
             nrows: int = 1,
             ncols: int = 1,
             index: Union[int, Tuple[int, int]] = 1,
             ndim: int = 2,
             title: str = '') -> plt.Axes:
    if ndim == 2:
        ax = fig.add_subplot(nrows, ncols, index)
    elif ndim == 3:
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
    else:
        raise NotImplementedError('Only 2D and 3D plots are supported')
    ax.set_title(title)
    return ax

def plot(data: np.ndarray,
         ax: plt.Axes,
         norm: Union[None, Normalize]):
    if data.ndim == 2:
        im = ax.imshow(data, norm=norm)
    else:
        raise NotImplementedError('Only 2D plots are supported')
    return im

def plot_grid(data: np.ndarray,
              top_labels: List[str],
              left_labels: List[str],
              normalize: bool = True):
    norm = Normalize(vmin=np.min(data), vmax=np.max(data)) if normalize else None
    nrows = data.shape[0]
    ncols = data.shape[1]
    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=(0, 0),
                     share_all=True,
                     label_mode='L',
                     cbar_mode='single',
                     cbar_pad=0.5,
                     cbar_location='right')
    i = 0
    j = 0
    im = None
    for ax in grid:
        im = ax.imshow(data[i, j], norm=norm)
        if j == 0:
            ax.set_ylabel(left_labels[i])
        if i == 0:
            ax.set_title(top_labels[j])
        j += 1
        if j == ncols:
            j = 0
            i += 1
    if im is not None:
        grid.cbar_axes[0].colorbar(im)
