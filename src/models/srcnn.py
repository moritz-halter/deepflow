from models.base_model import BaseModel
from models.layers.convolutional_neural_network import SRCNNBlock, SRCNNBlockT


class SRCNN(BaseModel, name='SRCNN'):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 super_resolution_rate=2,
                 super_resolution_dim='space',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if super_resolution_dim == 'space':
            self.srcnn_block = SRCNNBlock(in_channels=self.in_channels,
                                          out_channels=self.out_channels,
                                          scale=super_resolution_rate)
        else:
            self.srcnn_block = SRCNNBlockT(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           scale=super_resolution_rate)

    def forward(self, x, **kwargs):
        return self.srcnn_block(x)
