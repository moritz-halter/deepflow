from typing import Union, List

import torch
from torch import nn

from models.layers.upsampler import Upsampler
from models.layers.residual_block import ResBlock

class EnhancedDeepResidualNetwork(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 scale,
                 n_res_blocks=2,
                 res_scale=1):
        super().__init__()
        kernel_size = 3
        act = torch.nn.ReLU(inplace=True)

        self.head = torch.nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2, bias=True)

        m_body: List[Union[ResBlock, torch.nn.Conv3d]] = [
            ResBlock(hidden_channels, kernel_size, act=act, res_scale=res_scale) for _ in range(n_res_blocks)
        ]
        m_body.append(torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2, bias=True))

        self.body = torch.nn.Sequential(*m_body)

        m_tail = [
            Upsampler(scale, hidden_channels, act=False),
            torch.nn.Conv3d(hidden_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True)
        ]

        self.tail = torch.nn.Sequential(*m_tail)

    def forward(self, x):
        t = x.size(-1) if x.ndim == 6 else 0
        if t > 0:
            c, h, w, d = x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        if t > 0:
            c, h, w, d = x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.reshape(-1, t, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        return x

    def to(self, device: Union[int, torch.device, None]):
        self.head.to(device)
        self.body.to(device)
        self.tail.to(device)
        return self

    def cuda(self, device: Union[int, torch.device, None] = None):
        self.head.cuda(device)
        self.body.cuda(device)
        self.tail.cuda(device)
        return self

    def cpu(self):
        self.head.cpu()
        self.body.cpu()
        self.tail.cpu()
        return self

class EnhancedDeepResidualNetworkT(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 scale,
                 n_res_blocks=2,
                 res_scale=1):
        super().__init__()
        kernel_size = 3
        act = torch.nn.ReLU(inplace=True)

        self.head = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2, bias=True)

        m_body: List[Union[ResBlock, torch.nn.Conv3d]] = [
            ResBlock(hidden_channels, kernel_size, act=act, res_scale=res_scale) for _ in range(n_res_blocks)
        ]
        m_body.append(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2, bias=True))

        self.body = torch.nn.Sequential(*m_body)

        self.tail_1 = nn.Conv3d(hidden_channels, hidden_channels * scale, kernel_size, padding=kernel_size // 2, bias=True)
        self.tail_2 = nn.Conv3d(hidden_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True)
        self.scale = scale

    def forward(self, x):
        _, c, h, w, d, t = x.size()
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
        x = self.head(x)
        res = self.body(x)
        res += x

        c = res.size(1)
        x = self.tail_1(res)
        x = x.reshape(-1, t, c, self.scale, h, w, d).permute(0, 1, 3, 2, 4, 5, 6)
        x = x.reshape(-1, c, h, w, d)
        x = self.tail_2(x)
        c = x.size(1)
        x = x.reshape(-1, t * self.scale, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        return x