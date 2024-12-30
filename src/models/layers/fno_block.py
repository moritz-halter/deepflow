from typing import Union

import torch
from torch import nn

from models.layers.skip_connections import skip_connection
from models.layers.spectral_convolution import SpectralConv
from models.layers.convolutional_neural_network import CNN


Number = Union[int, float]


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers
    as described in "Fourier Neural Operator for Parametric
    Partial Differential Equations (Li et al., 2021).
    Parameters
    ----------
    Parameters
        ----------
        in_channels : int
            input channels to Fourier layers
        out_channels : int
            output channels after Fourier layers
        n_modes : int, List[int]
            number of modes to keep along each dimension 
            in frequency space. Can either be specified as
            an int (for all dimensions) or an iterable with one
            number per dimension
        output_scaling_factor : Optional[Union[Number, List[Number]]], optional
            factor by which to scale outputs for super-resolution, by default None
        n_layers : int, optional
            number of Fourier layers to apply in sequence, by default 1
        max_n_modes : int, List[int], optional
            maximum number of modes to keep along each dimension, by default None
        fno_block_precision : str, optional
            floating point precision to use for computations, by default "full"
        use_mlp : bool, optional
            whether to use mlp layers to parameterize skip connections, by default False
        mlp_dropout : int, optional
            dropout parameter for self.mlp, by default 0
        mlp_expansion : float, optional
            expansion parameter for self.mlp, by default 0.5
        non_linearity : torch.nn.F module, optional
            nonlinear activation function to use between layers, by default F.gelu
        stabilizer : Literal["tanh"], optional
            stabilizing module to use between certain layers, by default None
            if "tanh", use tanh
        norm : Literal["ada_in", "group_norm", "instance_norm"], optional
            Normalization layer to use, by default None
        ada_in_features : int, optional
            number of features for adaptive instance norm above, by default None
        preactivation : bool, optional
            whether to call forward pass with pre-activation, by default False
            if True, call nonlinear activation and norm before Fourier convolution
            if False, call activation and norms after Fourier convolutions
        fno_skip : str, optional
            module to use for FNO skip connections, by default "linear"
            see layers.skip_connections for more details
        mlp_skip : str, optional
            module to use for MLP skip connections, by default "soft-gating"
            see layers.skip_connections for more details
        SpectralConv Params
        -------------------
        separable : bool, optional
            separable parameter for SpectralConv, by default False
        factorization : str, optional
            factorization parameter for SpectralConv, by default None
        rank : float, optional
            rank parameter for SpectralConv, by default 1.0
        SpectralConv : BaseConv, optional
            module to use for SpectralConv, by default SpectralConv
        joint_factorization : bool, optional
            whether to factorize all spectralConv weights as one tensor, by default False
        fixed_rank_modes : bool, optional
            fixed_rank_modes parameter for SpectralConv, by default False
        implementation : str, optional
            implementation parameter for SpectralConv, by default "factorized"
        decomposition_kwargs : _type_, optional
            kwargs for tensor decomposition in SpectralConv, by default dict()
        fft_norm : str, optional
            how to normalize discrete fast Fourier transform, by default "forward"
            if "forward", normalize just the forward direction F(v(x)) by 1/n (number of total modes)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        n_layers=1,
        preactivation=False,
        geo_feature=False,
        geo_embedding=False,
        geo_channels=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.preactivation = preactivation

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels if not geo_feature or geo_embedding else (self.in_channels + geo_channels),
                    self.out_channels,
                    skip_type='linear',
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.geo_embedding = CNN(geo_channels, geo_channels, geo_channels, 2) if geo_embedding else None
        self.non_linearity = torch.nn.functional.gelu

    def forward(self, x, geo_feature=None, chi=None, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, geo_feature, chi, index, output_shape)
        else:
            return self.forward_with_postactivation(x, geo_feature, chi, index, output_shape)

    def forward_with_postactivation(self, x, geo_feature=None, chi=None, index=0, output_shape=None):

        x_skip_fno = self.fno_skips[index](x if geo_feature is None or self.geo_embedding else torch.cat([x, geo_feature.expand(x.size())], dim=1))
        x_skip_fno = self.convs.transform(x_skip_fno, layer_index=index, output_shape=output_shape)

        if chi is None:
            x_fno = self.convs(x, indices=index, output_shape=output_shape)
            x = x_fno + x_skip_fno
        else:
            chi = chi.expand(x.size())
            conv_chi = self.convs(chi, indices=index, output_shape=output_shape)
            conv_chi_x = self.convs(chi * x, indices=index, output_shape=output_shape)
            x_conv_chi = x * conv_chi
            x_fno = conv_chi_x - x_conv_chi
            x = chi * (x_fno + x_skip_fno)

        if geo_feature is not None:
            if self.geo_embedding is None:
                x = x + geo_feature
            else:
                x = x + self.geo_embedding(geo_feature)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, geo_feature=None, chi=None, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        x_skip_fno = self.fno_skips[index](x if geo_feature is None else torch.cat([x, geo_feature], dim=1))
        x_skip_fno = self.convs.transform(x_skip_fno, layer_index=index, output_shape=output_shape)

        if chi is None:
            x_fno = self.convs[index](x, output_shape=output_shape)
            x = x_fno + x_skip_fno
        else:
            chi = chi.expand(x.size())
            conv_chi = self.convs[index](chi, output_shape=output_shape)
            conv_chi_x = self.convs[index](chi * x, output_shape=output_shape)
            x_conv_chi = x * conv_chi
            x = chi * (conv_chi_x - x_conv_chi + x_skip_fno)

        if geo_feature is not None:
            if self.geo_embedding is None:
                x = x + geo_feature
            else:
                x = x + self.geo_embedding(geo_feature)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
