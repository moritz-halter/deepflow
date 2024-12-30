import torch


class PixelShuffle(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, in_channels, in_height, in_width, in_depth = x.size()
        out_channels = in_channels // (self.scale ** 3)
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        out_depth = in_depth * self.scale
        x_view = x.contiguous().view(batch_size, out_channels, self.scale, self.scale, self.scale, in_height, in_width, in_depth)
        output = x_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, out_channels, out_height, out_width, out_depth)


class TimeShuffle(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, in_channels, in_height, in_width, in_depth, in_time = x.size()
        x_view = x.contiguous().view(batch_size, in_channels // self.scale, self.scale, in_height, in_width, in_depth, in_time)
        output = x_view.permute(0, 1, 3, 4, 5, 6, 2).contiguous()
        return output.view(batch_size, in_channels // self.scale, in_height, in_width, in_depth, in_time * self.scale)