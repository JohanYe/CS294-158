import torch
import torch.nn as nn
import numpy as np
from Hw3.Ex2.Utils import *


class GatedResidualBlock(nn.Module):
    """
    Gated residual block
    """

    def __init__(self, channels, kernel_size=3):
        super(GatedResidualBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, 2 * channels, kernel_size=kernel_size, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = torch.sigmoid(gate)
        return x * gate


class ResidualStack(nn.Module):
    def __init__(self):
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
    def __init__(self, latent_dim=64):
        super(ConvEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.layers = [nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)] + \
                      [nn.ReLU()] + \
                      [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)] + \
                      [nn.ReLU()] + \
                      [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)]
        self.layers.append(ResidualStack())
        self.layers = nn.Sequential(*self.layers)
        conv_out_dim = (32 // 8) ** 2 * 128
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], -1)
        mu, log_var = self.fc(out).chunk(2, dim=1)
        return mu, log_var


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.conv_in_size = (64, 32 // 8, 32 // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.conv_in_size))
        self.layers = [nn.ReLU()] + [nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2)] + \
                      [ResidualStack()] + [nn.ReLU()] + \
                      [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)] + [nn.ReLU()] + \
                      [nn.ConvTranspose2d(64, 6, kernel_size=4, stride=2, padding=2)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.conv_in_size)
        mu, log_var = self.layers(out).chunk(2, dim=1)
        return mu, log_var


class ConvVAE(nn.Module):
    """ Was lazy with this model so hardcoded everything """

    def __init__(self, latent_dim=16):
        super(ConvVAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu, lv):
        std = lv.mul(0.5).exp()
        z = mu + torch.randn_like(mu).mul(std)
        return z

    def TopDown(self, z):
        mu_x, lv_x = self.decoder(z)
        x_recon = self.reparameterize(mu_x, lv_x)
        return x_recon

    def forward(self, x, noise=True):
        mu_z, lv_z = self.encoder(x)
        z = self.reparameterize(mu_z, lv_z)
        mu_x, lv_x = self.decoder(z)
        if noise:
            x_recon = self.reparameterize(mu_x, lv_x)
        else:
            x_recon = mu_x
        return x_recon

    def calc_loss(self, x):
        mu_z, lv_z = self.encoder(x)
        z = self.reparameterize(mu_z, lv_z)
        mu_x, lv_x = self.decoder(z)
        x_recon = self.reparameterize(mu_x, lv_x)

        # reconstruction_loss = nn.functional.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(1).mean()
        # kl = -(lv_z*0.5) - 0.5 + 0.5 * (torch.exp(lv_z) + mu_z ** 2)
        # kl = kl.sum(1).mean()

        reconstruction_loss = log_normal_pdf(x, mu_x, lv_x).mean()
        zeros = torch.zeros_like(z).to(self.device)
        ones = torch.ones_like(z).to(self.device)
        kl = (log_normal_pdf(z, mu_z, lv_z) - log_normal_pdf(z, zeros, ones)).mean()

        ELBO = reconstruction_loss + np.max((kl, 1))

        return ELBO, kl, reconstruction_loss

    def sample(self, num_samples):
        z = torch.randn([num_samples, self.latent_dim]).to(self.device)
        mu_x, lv_x = self.decoder(z)
        x_recon = self.reparameterize(mu_x, lv_x)
        return x_recon

    def interpolations(self, x):
        with torch.no_grad():
            mu_z, lv_z = self.encoder(x)
            z = self.reparameterize(mu_z, lv_z)
            z1, z2 = z.chunk(2, dim=0)
            interpolations = [self.TopDown(z1 * (1 - alpha) + z2 * alpha) for alpha in np.linspace(0, 1, 10)]
            interpolations = torch.stack(interpolations, dim=1).view(-1, 3, 32, 32)
            interpolations_mu = [self.decoder(z1 * (1 - alpha) + z2 * alpha)[0] for alpha in np.linspace(0, 1, 10)]
            interpolations_mu = torch.stack(interpolations_mu, dim=1).view(-1, 3, 32, 32)

        return interpolations, interpolations_mu


class ConvVAE2(nn.Module):
    """ Was lazy with this model so hardcoded everything """

    def __init__(self, latent_dim=16):
        super(ConvVAE2, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim

        self.encoder = [nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)] + [nn.ReLU()] + \
                       [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)] + [nn.ReLU()] + \
                       [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)] + [nn.ReLU()] + \
                       [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        self.encoder = nn.Sequential(*self.encoder)
        conv_out_dim = (32 // 16) ** 2 * 256
        self.fc1 = nn.Linear(conv_out_dim, 2 * latent_dim)

        self.conv_in_size = (64, 32 // 16, 32 // 16)
        self.fc2 = nn.Linear(latent_dim, np.prod(self.conv_in_size))
        self.decoder = [nn.ReLU()] + [nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)] + \
                       [nn.ReLU()] + [nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)] + \
                       [nn.ReLU()] + [nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)] + \
                       [nn.ReLU()] + [nn.ConvTranspose2d(32, 6, kernel_size=4, stride=2, padding=1)]
        self.decoder = nn.Sequential(*self.decoder)

    def reparameterize(self, mu, lv):
        std = lv.mul(0.5).exp()
        z = mu + torch.randn_like(mu).mul(std)
        return z

    def BottomUp(self, x):
        out = self.encoder(x)
        out = out.view(out.shape[0], -1)
        mu_z, lv_z = self.fc1(out).chunk(2, dim=1)
        return mu_z, lv_z

    def TopDown(self, z):
        out = self.fc2(z)
        out = out.view(out.shape[0], *self.conv_in_size)
        mu_x, lv_x = self.decoder(out).chunk(2, dim=1)
        return mu_x, lv_x

    def forward(self, x):
        mu_z, lv_z = self.BottomUp(x)
        z = self.reparameterize(mu_z, lv_z)
        mu_x, lv_x = self.TopDown(z)
        x_recon = self.reparameterize(mu_x, lv_x)
        return x_recon

    def calc_loss(self, x, beta):
        mu_z, lv_z = self.BottomUp(x)
        z = self.reparameterize(mu_z, lv_z)
        mu_x, lv_x = self.TopDown(z)
        x_recon = self.reparameterize(mu_x, lv_x)

        reconstruction_loss = nn.functional.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(1).mean()

        kl = -(lv_z * 0.5) - 0.5 + 0.5 * (torch.exp(lv_z) + mu_z ** 2)
        kl = kl.sum(1).mean()

        ELBO = reconstruction_loss + np.max((kl, 1))

        return ELBO, kl, reconstruction_loss

    def sample(self, num_samples):
        z = torch.randn([num_samples, self.latent_dim]).to(self.device)
        mu_x, lv_x = self.TopDown(z)
        x_recon = self.reparameterize(mu_x, lv_x)
        return x_recon
