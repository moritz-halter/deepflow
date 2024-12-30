import torch
from models.layers.shuffle import PixelShuffle

class RDBConv(torch.nn.Module):
    def __init__(self, in_channels, grow_rate, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Sequential(*[
            torch.nn.Conv3d(in_channels, grow_rate, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
            torch.nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), dim=1)


class RDB(torch.nn.Module):
    def __init__(self, grow_rate_0, grow_rate, n_layers):
        super().__init__()
        convs = []
        for c in range(n_layers):
            convs.append(RDBConv(grow_rate_0 + n_layers * grow_rate, grow_rate))
        self.convs = torch.nn.Sequential(*convs)

        self.LFF = torch.nn.Conv3d(grow_rate_0 + n_layers * grow_rate, grow_rate, 1, padding=1, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class ResidualDenseNetwork(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 scale=2,
                 n_res_blocks=4,
                 n_layers=2,
                 grow_rate_0=64,
                 kernel_size=3):
        super().__init__()
        self.n_res_blocks = n_res_blocks
        self.SFENet1 = torch.nn.Conv3d(in_channels, grow_rate_0, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        self.SFENet2 = torch.nn.Conv3d(grow_rate_0, grow_rate_0, kernel_size, padding=(kernel_size - 1) // 2, stride=1)

        self.RDBs = torch.nn.ModuleList()
        for i in range(n_res_blocks):
            self.RDBs.append(
                RDB(grow_rate_0=grow_rate_0, grow_rate=hidden_channels, n_layers=n_layers)
            )

        self.GFF = torch.nn.Sequential(*[
            torch.nn.Conv3d(n_res_blocks * grow_rate_0, grow_rate_0, 1, padding=0, stride=1),
            torch.nn.Conv3d(grow_rate_0, grow_rate_0, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        ])

        self.out_dim = out_channels
        if scale == 2 or scale == 3:
            self.UPNet = torch.nn.Sequential(*[
                torch.nn.Conv3d(grow_rate_0, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                PixelShuffle(scale),
                torch.nn.Conv3d(hidden_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = torch.nn.Sequential(*[
                torch.nn.Conv3d(grow_rate_0, hidden_channels * 8, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                PixelShuffle(2),
                torch.nn.Conv3d(hidden_channels, 8 * hidden_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                PixelShuffle(2),
                torch.nn.Conv3d(hidden_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f_1 = self.SFENet1(x)
        x = self.SFENet2(f_1)

        rdb_out = []
        for i in range(self.n_res_blocks):
            x = self.RDBs[i](x)
            rdb_out.append(x)

        x = self.GFF(torch.cat(rdb_out, 1))
        x += f_1

        x = self.UPNet(x)
        return x
