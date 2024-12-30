from torch import nn

from models.layers.shuffle import PixelShuffle

class SRCNNBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scale=2):
        super(SRCNNBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv3d(32, out_channels * scale ** 3, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = PixelShuffle(scale=scale)

    def forward(self, x):
        t = x.size(-1) if x.ndim == 6 else 0
        if t > 0:
            _, c, h, w, d, _ = x.size()
            x = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        if t > 0:
            _, c, h, w, d = x.size()
            x = x.reshape(-1, t, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        return x

class SRCNNBlockT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scale=5):
        super(SRCNNBlockT, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv3d(32, out_channels * scale, kernel_size=5, padding=5 // 2)
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w, d, t = x.size()

        x = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.reshape(-1, t, c, self.scale, x.size(2), x.size(3), x.size(4)).permute(0, 2, 4, 5, 6, 1, 3)
        x = x.reshape(-1, c, x.size(2), x.size(3), x.size(4), t * self.scale)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=1, n_layers=2):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        ] + [
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        ] * n_layers)
        self.out_conv = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        t = x.size(-1) if x.ndim == 6 else 0
        if t > 0:
            _, c, h, w, d, _ = x.size()
            x = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = self.out_conv(x)
        if t > 0:
            _, c, h, w, d = x.size()
            x = x.reshape(-1, t, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
        return x
