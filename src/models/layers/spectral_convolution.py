from typing import Optional, Tuple, Union

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from models.layers.einsum_utils import einsum_complexhalf
from models.layers.base_spectral_conv import BaseSpectralConv
from models.layers.resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    weight_syms.insert(1, einsum_symbols[order])  # outputs
    out_syms = list(weight_syms)
    out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)


def get_contract_fun(weight):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    return _contract_dense


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    n_modes : None or int tuple, default is None
        Number of modes to use for contraction in Fourier domain during training.
 
        .. warning::
            
            We take care of the redundancy in the Fourier modes, therefore, for an input 
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the 
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes 
            as the real FFT is redundant along that last dimension.

            
        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.  

        This can be updated dynamically during training.
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        bias=True,
        n_layers=1,
        device=None,
        dtype=None,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        self.n_layers = n_layers

        init_std = (2 / (in_channels + out_channels))**0.5

        weight_shape = (in_channels, out_channels, *self.n_modes)

        tensor_kwargs = {}

        self.weight = nn.ModuleList(
            [
                FactorizedTensor.new(
                    weight_shape,
                    factorization='ComplexDense',
                    rank=0.5,
                    **tensor_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        for w in self.weight:
            w.normal_(0, init_std)
        self._contract = get_contract_fun(self.weight[0])

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _get_weight(self, index):
        return self.weight[index]

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(
                x,
                1.0,
                list(range(2, x.ndim)),
                output_shape=out_shape,
            )
    
    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1
        output_shape: Optional Tuple[int]

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))


        x = torch.fft.rfftn(x, norm="forward", dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.n_modes)]
        slices_w =  [slice(None), slice(None)] # Batch_size, channels
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels
        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm='forward')

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            Warning("A single convolution is parametrized, directly use the main class.")

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SpectralConv4D(BaseSpectralConv):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            bias=True,
            n_layers=1,
            device=None,
            dtype=None
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_modes = n_modes
        self.n_layers = n_layers

        init_std = (2 / (in_channels + out_channels)) ** 0.5

        weight_shape = (in_channels, out_channels, *self.n_modes)

        tensor_kwargs = {}

        self.weight = nn.ModuleList(
            [
                FactorizedTensor.new(
                    weight_shape,
                    factorization='ComplexDense',
                    rank=0.5,
                    **tensor_kwargs
                ) for _ in range(n_layers)
            ]
        )

        for w in self.weight:
            w.normal_(0, init_std)
        self._contract = get_contract_fun(self.weight[0])

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _get_weight(self, index):
        return self.weight[index]

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(
                x,
                1.0,
                list(range(2, x.ndim)),
                output_shape=out_shape,
            )

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
            self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm='forward', dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:1])

        out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.n_modes)]
        slices_w = [slice(None), slice(None)]
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels
        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm='forward')

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SubConv(nn.Module):
    """Class representing one of the convolutions from the mother joint
    factorized convolution.

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to
    the same data, which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)

    def transform(self, x, **kwargs):
        return self.main_conv.transform(x, self.indices, **kwargs)

    @property
    def weight(self):
        return self.main_conv.get_weight(indices=self.indices)
