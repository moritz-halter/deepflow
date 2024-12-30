import torch.nn.functional as F

from models.layers.upsampler import UpsamplerTime
from models.srcnn import SRCNNBlock
from models.layers import FNOBlocks, MLP
from models.layers import ResidualDenseNetwork, Upsampler, EnhancedDeepResidualNetwork
from models.base_model import BaseModel


class FNO4D(BaseModel, name='FNO4D'):
    def __init__(self,
                 n_modes,
                 hidden_channels,
                 data_channels=3,
                 out_channels=3,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 implicit=True,
                 geo_feature=False,
                 geo_channels=1,
                 super_resolution_rate=None,
                 super_resolution_dim='space',
                 super_resolution_layer='edsr',
                 n_super_resolution_layers=1,
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = data_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.implicit = implicit

        if super_resolution_rate is not None:
            if super_resolution_dim == 'space':
                if super_resolution_layer == 'plain':
                    self.super_resolution_layer = Upsampler(super_resolution_rate, self.in_channels)
                elif super_resolution_layer == 'edsr':
                    self.super_resolution_layer = EnhancedDeepResidualNetwork(in_channels=self.in_channels,
                                                                              out_channels=self.in_channels,
                                                                              hidden_channels=hidden_channels,
                                                                              scale=super_resolution_rate,
                                                                              n_res_blocks=n_super_resolution_layers)
                elif super_resolution_layer == 'srcnn':
                    self.super_resolution_layer = SRCNNBlock(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             scale=super_resolution_rate)
                else:
                    self.super_resolution_layer = ResidualDenseNetwork(in_channels=self.in_channels,
                                                                       out_channels=self.in_channels,
                                                                       hidden_channels=self.in_channels,
                                                                       scale=super_resolution_rate,
                                                                       n_res_blocks=n_super_resolution_layers)
            else:
                self.super_resolution_layer = UpsamplerTime(super_resolution_rate, self.in_channels)
        else:
            self.super_resolution_layer = None


        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            geo_feature=geo_feature,
            geo_channels=geo_channels,
            n_modes=self.n_modes,
            n_layers=self.n_layers if not implicit else 1,
            **kwargs
        )

        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim
            )
        else:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=F.gelu
        )

    def forward(self, x, geo_feature=None, chi=None, output_shape=None, **kwargs):
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        x = self.super_resolution_layer(x)

        x = self.lifting(x)
        for layer_idx in range(self.n_layers):
            if self.implicit:
                x = self.fno_blocks(x, chi=chi, geo_feature=geo_feature, index=0, output_shape=output_shape[layer_idx])
            else:
                x = self.fno_blocks(x, chi=chi, geo_feature=geo_feature, index=layer_idx, output_shape=output_shape[layer_idx])

        x = self.projection(x)

        return x
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes
