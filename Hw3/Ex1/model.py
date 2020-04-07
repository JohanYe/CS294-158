import torch
import torch.nn as nn
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, n_layers=4, n_hidden=100, vector=True):
        super(VariationalAutoEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_features = 2 if vector else 1

        self.encode = nn.ModuleList()
        self.encode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 1):
            self.encode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.encode.append(nn.Linear(self.n_hidden, 4))

        self.decode = nn.ModuleList()
        self.decode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 1):
            self.decode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.decode.append(nn.Linear(self.n_hidden, self.out_features*2))

    def encoder(self, x):
        for layer in self.encode[:-1]:
            x = torch.relu(layer(x))
        x = self.encode[-1](x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        z = torch.randn_like(std) * std + mu
        return z


    def decoder(self, x):
        for layer in self.decode[:-1]:
            x = torch.relu(layer(x))
        x = self.decode[-1](x)
        mu_dec, log_var = x.chunk(2, dim=1)
        return mu_dec, log_var

    def forward(self, x):
        mu_enc, log_var_enc = self.encoder(x)
        z = self.reparameterize(mu_enc, log_var_enc)
        mu_dec, std_dec = self.decoder(z)
        return mu_dec, std_dec

    def calc_loss(self, x, beta):
        mu_z, log_var_z = self.encoder(x)
        log_var_z = nn.Softplus()(log_var_z)
        z_Gx = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decoder(z_Gx)
        log_var_x = nn.Softplus()(log_var_x)

        # KL
        kl = -0.5*log_var_z + 0.5 * (log_var_z.exp() + mu_z**2) - 0.5
        kl = kl.sum(dim=1).mean()

        # Recon_loss
        reconstruction_loss = 0.5 * np.log(2 * np.pi) + 0.5*log_var_x + (x - mu_x)**2 * log_var_x.exp() * 0.5
        reconstruction_loss = reconstruction_loss.sum(dim=1).mean()
        # print(log_var_x)

        return reconstruction_loss + kl, kl, reconstruction_loss

    def sample(self, n_samples, decoder_noise=True):
        z = torch.distributions.Normal(0, 1).sample([n_samples, 2]).to(self.device)
        mu_x, log_var_x = self.decoder(z)
        log_var_x = nn.Softplus()(log_var_x)
        std_x = (0.5*log_var_x).exp()
        if decoder_noise:
            x_recon = torch.randn_like(mu_x) * std_x + mu_x
        else:
            x_recon = mu_x
        return x_recon


