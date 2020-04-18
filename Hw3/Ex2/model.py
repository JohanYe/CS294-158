import torch
import torch.nn as nn
import numpy as np


class GatedResidualBlock(nn.Module):
    """
    Gated residual block
    """
    def __init__(self, channels, kernel_size=3):

        self.channels = channels
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, 2*channels, kernel_size=kernel_size, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = torch.sigmoid(gate)
        return x*gate


class ResidualStack(nn.Module):
    def __init__(self, ):
        super(ResidualStack, self).__init__()

        layers = []

        for _ in range(5):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(128, 64, 3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(64, 128, 3, stride=1, padding=1))
            layers.append(GatedResidualBlock(channels=128))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out + x

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.layers = [nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)] +\
            [nn.ReLU()] +\
            [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)]+\
            [nn.Relu()]+\
            [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)]
        self.layers.append(ResidualStack)
        self.layers = nn.Sequential(*self.layers)
        conv_out_dim = (32 // 8)**2 * 128
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.conv_in_size = (64, 32 // 8, 32 // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.conv_in_size))
        self.layers = [nn.ReLU()] +\
        [nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2)] +\
        [ResidualStack] +\
        [nn.ReLU()] + \
        [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)] +\
        [nn.ReLU()] +\
        [nn.ConvTranspose2d(64, 6, kernel_size=3, stride=2, padding=1)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.conv_in_size)
        out = self.layers(out)
        return out

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def calc_loss(self, x):





















