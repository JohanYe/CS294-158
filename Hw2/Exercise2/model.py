import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()

        self.n_channels = n_channels
        self.layers = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels // 2, kernel_size=1),
            # nn.BatchNorm2d(self.n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_channels // 2, self.n_channels // 2, kernel_size=3),
            # nn.BatchNorm2d(self.n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_channels // 2, self.n_channels),
        )

    def forward(self, x):
        return self.layers(x) + x


class ResNet(nn):
    def __init__(self, channels_in, n_channels, n_blocks=4):
        super(ResNet, self).__init__()

        self.n_blocks = n_blocks
        self.n_channels = n_channels

        self.first_conv = nn.Conv2d(channels_in, n_channels, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResidualBlock(n_channels) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(self.n_channels, channels_in*2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.first_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_conv(torch.relu(out))
        return out


class AffineCouplingLayer(nn.Module):
    """ Very similar to Ex1, but with Resnet block """
    def __init__(self, mask, channels_in, n_channels=32, n_blocks=4):
        super(AffineCouplingLayer, self).__init__()

        self.register_buffer('mask', mask)
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.ResNet = ResNet(channels_in, self.n_channels, self.n_blocks)

    def forward(self, x, Forward=True):
        if Forward:
            x_masked = self.mask * x
            # Half of Resnet is log_scale and other half is transform
            log_scale, translate = torch.split(self.ResNet(x_masked), x_masked.shape[1], dim=1)
            log_scale = torch.tanh(log_scale)
            z = x_masked + (1 - self.mask) * (x * torch.exp(log_scale) + translate)
            log_determinant = torch.sum(torch.flatten(log_scale * (1 - self.mask), start_dim=1), dim=1, keepdim=True)
            return z, log_determinant
        else:
            x_masked = self.mask * x
            log_scale, translate = torch.split(self.ResNet(x_masked), x_masked.shape[1], dim=1)
            log_scale = torch.tanh(log_scale)
            z = x_masked + (1 - self.mask) * (x - translate) / torch.exp(log_scale)
            return z

class ActNorm(nn.Module):
    """ Same as Ex1 except added flatten in log_determinant calculation"""

    def __init__(self, c, h, w):
        super(ActNorm, self).__init__()
        self.scale = torch.ones([1, c, h, w])
        self.translate = torch.ones([1, c, h, w])

    def initialize(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        # Copy is used to save us for formatting tensor
        self.scale.data.copy_(1.0 / std)
        self.translate.data.copy_(-torch.mean(x * self.scale, dim=0, keepdim=True))

    def forward(self, x, Forward=True):
        if Forward:
            log_determinant = torch.sum(torch.Flatten(
                torch.log(torch.abs(self.scale) + 1e-6), start_dim=1), dim=1, keepdim=True)
            y = x * self.scale + self.translate
            return y, log_determinant
        else:
            y = (x - self.translate) / self.scale
            return y



# class SimpleRealNVP(nn.Module):
#     def __init__(self, mask, channels, n_layers=):












