from models.base_model import BaseModel
from models.layers.enhanced_deep_residual_network import EnhancedDeepResidualNetwork, EnhancedDeepResidualNetworkT


class EDSR(BaseModel, name='EDSR'):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 hidden_channels=32,
                 layers=1,
                 super_resolution_rate=2,
                 super_resolution_dim='space',
                 **kwargs):
        super().__init__()
        if super_resolution_dim == 'space':
            self.model = EnhancedDeepResidualNetwork(in_channels=in_channels,
                                                     out_channels=out_channels,
                                                     hidden_channels=hidden_channels,
                                                     scale=super_resolution_rate,
                                                     n_res_blocks=layers)
        else:
            self.model = EnhancedDeepResidualNetworkT(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      hidden_channels=hidden_channels,
                                                      scale=super_resolution_rate,
                                                      n_res_blocks=layers)

    def forward(self, x, **kwargs):
        return self.model(x)
