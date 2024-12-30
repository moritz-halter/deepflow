import torch
import math
from models.layers.shuffle import PixelShuffle, TimeShuffle
from models.layers.conv4d import Conv4d


class Upsampler(torch.nn.Sequential):
    def __init__(self, scale, n_feats, act=False):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(torch.nn.Conv3d(n_feats, 8 * n_feats, 3, padding=1, bias=act))
                m.append(PixelShuffle(2))
        elif scale == 3:
            m.append(torch.nn.Conv3d(n_feats, 27 * n_feats, 3, padding=1))
            m.append(PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class UpsamplerTime(torch.nn.Sequential):
    def __init__(self, scale, n_feats, act=False):
        m = [
            Conv4d(in_channels=n_feats, out_channels=scale * n_feats, kernel_size=3, padding=1),
            TimeShuffle(scale)
        ]
        super(UpsamplerTime, self).__init__(*m)
