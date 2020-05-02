import torch
import torch.nn as nn
from Hw4.utils import PrintLayerShape

class UpsamplingDepthToSpace(nn.Module):
    def __init__(self, block_size=2):
        super(UpsamplingDepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = int(block_size ** 2)

    def forward(self, x):
        # bunch of dimension stuff
        out = x.permute(0, 2, 3, 1)
        (bs, or_height, or_width, or_channels) = out.shape
        up_height = int(or_height * self.block_size)
        up_width = int(or_width * self.block_size)
        up_channels = int(or_channels / self.block_size_sq)
        out_expanded = out.reshape(bs, or_height, or_width, self.block_size_sq, up_channels)  # 4 copies
        split = out_expanded.split(self.block_size, dim=3)  # split in 2
        stack = [x.reshape(bs, or_height, up_width, up_channels) for x in split] # reshape to double h and w
        out = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(
            bs, up_height, up_width, up_channels)  # Stack, transpose, and reshape to [N, H, W, C]
        out = out.permute(0, 3, 1, 2)
        return out.contiguous()  # to easy backprop


class UpsampleConv2d(nn.Module):
    def __init__(self, c_in, c_out, ks=4, padding=1):
        super(UpsampleConv2d, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = ks
        self.padding = padding

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=ks,stride=1, padding=padding)
        self.depth_to_space = UpsamplingDepthToSpace(2)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)  # Prep for upsampling method.
        x = self.depth_to_space(x) # special upsampling method.
        x = self.conv(x)
        return x


class UpResnetBlock(nn.Module):
    def __init__(self, c_in, filters=128):
        super(UpResnetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            UpsampleConv2d(filters, filters, ks=3, padding=1),
        )
        self.upsample_x = UpsampleConv2d(c_in, filters, ks=1, padding=0)

    def forward(self, x):
        res = self.layers(x)
        x = self.upsample_x(x)
        return res + x


class Generator(nn.Module):
    def __init__(self, noise_dim=128, n_filters=128):
        super(Generator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim
        self.filters = n_filters
        self.dense_init = nn.Linear(noise_dim, 4 * 4 * n_filters)
        self.layers = nn.Sequential(
            UpResnetBlock(c_in=n_filters, filters=n_filters),
            UpResnetBlock(c_in=n_filters, filters=n_filters),
            UpResnetBlock(c_in=n_filters, filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, bs):
        z = torch.randn(bs, self.noise_dim).to(self.device)
        out = self.dense_init(z)
        out = out.reshape(-1, 128, 4, 4)
        out = self.layers(out)
        return out


class DownsamplingSpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super(DownsamplingSpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = int(block_size ** 2)

    def forward(self, x):
        # bunch of dimension stuff
        out = x.permute(0, 2, 3, 1)
        (bs, or_height, or_width, or_channels) = out.shape
        down_height = int(or_height / self.block_size)
        down_width = int(or_width / self.block_size)
        down_channels = int(or_channels * self.block_size_sq)
        split = x.split(self.block_size, dim=2)
        stack = [x.reshape(bs, down_height, down_channels) for x in split]
        output = torch.stack(stack, dim=1)
        output = output.permute(0, 3, 2, 1)
        return output.contiguous()

class Downsample_Conv2d(nn.Module):
    def __init__(self, c_in, c_out, ks=3, stride=1, padding=1):
        super(Downsample_Conv2d, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = ks
        self.padding = padding

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=ks, stride=1, padding=padding,bias=True)
        self.space_to_depth = DownsamplingSpaceToDepth(2)

    def forward(self, x):
        x = self.space_to_depth(x)
        # TODO: ASK RASMUS
        x = sum(x.chunk(4, dim=1)) / 4.0
        x = self.conv(x)
        return x


class DownResnetBlock(nn.Module):
    def __init__(self, c_in, filters=128):
        super(DownResnetBlock, self).__init__()
        self.c_in = c_in
        self.filters = filters

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(c_in, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            Downsample_Conv2d(filters, filters, ks=3, padding=1)
        )
        self.downsample_x = Downsample_Conv2d(c_in, filters, ks=1, padding=0)

    def forward(self, x):
        res = self.layers(x)
        x = self.downsample_x(x)
        return res + x

class ResnetBlock(nn.Module):
    def __init__(self, c_in, filters=128):
        super(ResnetBlock, self).__init__()
        self.c_in = c_in
        self.filters = filters

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(c_in, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1,padding=1)
        )

    def forward(self, x):
        res = self.layers(x)
        return res + x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            DownResnetBlock(3, filters=128),
            DownResnetBlock(128, 128),
            ResnetBlock(128, 128),
            ResnetBlock(128, 128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layers(x)
        x = torch.sum(x, dim=[2, 3])  # TODO: WHY ARE THESE SUMMED?
        x = self.fc(x)
        return x


