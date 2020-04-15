import torch
import torch.nn as nn
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, n_layers=4, n_hidden=64, vector=True):
        super(VariationalAutoEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_variance = vector

        self.encode = nn.ModuleList()
        self.encode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.encode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.encode.append(nn.Linear(self.n_hidden, 4))

        self.decode = nn.ModuleList()
        self.decode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.decode.append(nn.Linear(self.n_hidden, self.n_hidden))
        if vector:
            self.decode.append(nn.Linear(self.n_hidden, 4))
        else:
            self.mu_x = nn.Linear(self.n_hidden, 2)
            self.lv_x = nn.Linear(self.n_hidden, 1)

    def encoder(self, x):
        for layer in self.encode[:-1]:
            x = torch.relu(layer(x))
        x = self.encode[-1](x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std) * std + mu
        return z

    def decoder(self, x):
        for layer in self.decode[:-1]:
            x = torch.relu(layer(x))
        if self.vector_variance:
            x = self.decode[-1](x)
            mu_dec, lv_dec = x.chunk(2, dim=1)
        else:
            mu_dec = self.mu_x(x)
            lv_dec = self.lv_x(x)
            lv_dec = torch.ones_like(mu_dec) * lv_dec
        return mu_dec, lv_dec

    def forward(self, x, noise=True):
        mu_enc, log_var_enc = self.encoder(x)
        z = self.reparameterize(mu_enc, log_var_enc)
        if noise:
            mu_dec, lv_dec = self.decoder(z)
            return self.reparameterize(mu_dec, lv_dec)
        else:
            mu_dec, lv_dec = self.decoder(z)
            return mu_dec

    def calc_loss(self, x, beta):
        mu_z, log_var_z = self.encoder(x)
        # log_var_z = nn.Softplus()(log_var_z)
        z_Gx = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decoder(z_Gx)
        # log_var_x = nn.Softplus()(log_var_x)

        # KL
        # kl = -0.5 * log_var_z + 0.5 * (log_var_z.exp() + mu_z ** 2) - 0.5
        # kl = kl.sum(1).mean()
        kl = torch.mean(-0.5 * torch.sum(1 + log_var_z - mu_z ** 2 - torch.exp(log_var_z), dim=1))

        # Recon_loss - negative 1 is multiplied due to optimizing on the negative reconstruction error.
        reconstruction_loss = - 0.5 * np.log(2*np.pi) - 0.5*log_var_x - (x - mu_x)**2 / (2 * log_var_x.exp() + 1e-5)
        reconstruction_loss = torch.sum(-reconstruction_loss, dim=1).mean() / np.log(2) / 2
        # reconstruction_loss = 0.5 * np.log(2 * np.pi) + 0.5 * log_var_x + (x - mu_x) ** 2 * log_var_x.exp() * 0.5
        # reconstruction_loss = reconstruction_loss.sum(1).mean()

        return reconstruction_loss + kl * beta, kl, reconstruction_loss

    def sample(self, n_samples, decoder_noise=True):
        z = torch.distributions.Normal(0, 1).sample([n_samples, 2]).to(self.device)
        mu_x, log_var_x = self.decoder(z)
        # log_var_x = nn.Softplus()(log_var_x)
        print(mu_x.shape, log_var_x.shape)
        if decoder_noise:
            std_x = (0.5 * log_var_x).exp()
            x_recon = torch.randn_like(mu_x) * std_x + mu_x
        else:
            x_recon = mu_x
        return x_recon

class IWAE(nn.Module):
    def __init__(self, n_layers=4, n_hidden=64):
        super(IWAE, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encode = nn.ModuleList()
        self.encode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.encode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.encode.append(nn.Linear(self.n_hidden, 4))

        self.decode = nn.ModuleList()
        self.decode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.decode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.decode.append(nn.Linear(self.n_hidden, 4))

    def encoder(self, x):
        for layer in self.encode[:-1]:
            x = torch.relu(layer(x))
        x = self.encode[-1](x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var, n_samples):

        std = torch.exp(0.5 * log_var)
        if len(mu.shape) < 3:
            mu = mu.unsqueeze(1)
            std = std.unsqueeze(1)
        eps = torch.randn(mu.shape[0], n_samples, mu.shape[2]).to(self.device).detach()
        z = eps * std + mu
        return z

    def decoder(self, x):
        for layer in self.decode[:-1]:
            x = torch.relu(layer(x))
        x = self.decode[-1](x)
        mu_dec, lv_dec = x.chunk(2, dim=2)
        return mu_dec, lv_dec

    def forward(self, x, ):
        mu_enc, log_var_enc = self.encoder(x)
        z = self.reparameterize(mu_enc, log_var_enc, n_samples=1)
        mu_dec, lv_dec = self.decoder(z)
        x_recon = self.reparameterize(mu_dec, lv_dec, n_samples=1).squeeze(1)
        return x_recon

    def calc_loss(self, x, beta, num_samples=1):
        mu_z, log_var_z = self.encoder(x)
        z_Gx = self.reparameterize(mu_z, log_var_z, num_samples)
        mu_x, log_var_x = self.decoder(z_Gx)
        x = x.unsqueeze(1)

        reconstruction_loss = - 0.5 * np.log(2*np.pi) - 0.5*log_var_x - (x - mu_x)**2 / (2 * log_var_x.exp() + 1e-5)
        reconstruction_loss = torch.sum(-reconstruction_loss, dim=1).mean() / np.log(2) / 2

        # IWAE KL:
        mu_z, log_var_z = mu_z.unsqueeze(1), log_var_z.unsqueeze(1)
        log_qz = - 0.5 * np.log(2*np.pi) - 0.5*log_var_z - (z_Gx - mu_z)**2 / (2 * log_var_z.exp() + 1e-5)
        log_pz = - 0.5 * np.log(2*np.pi) - 0.5*1 - (z_Gx - 0)**2 / (2 * np.exp(1) + 1e-5)  # evaluation in N(0,1)
        kl = torch.mean(torch.sum(log_qz - log_pz, dim=2))/np.log(2)/2  # logsumexp

        return reconstruction_loss + kl * beta, kl, reconstruction_loss

    def sample(self, n_samples):
        z = torch.distributions.Normal(0, 1).sample([n_samples, 2]).to(self.device)
        for layer in self.decode:
            z = torch.relu(layer(z))
        mu_dec, lv_dec = z.chunk(2, dim=1)
        std_x = (0.5 * lv_dec).exp()
        x_sampled = torch.randn_like(mu_dec) * std_x + mu_dec
        return x_sampled


