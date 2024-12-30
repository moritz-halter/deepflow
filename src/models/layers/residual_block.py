import torch

class ResBlock(torch.nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 bias=True,
                 bn=False,
                 act=torch.nn.ReLU(True),
                 res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(torch.nn.Conv3d(channels, channels, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(torch.nn.BatchNorm3d(channels))
            if i == 0:
                m.append(act)

        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
