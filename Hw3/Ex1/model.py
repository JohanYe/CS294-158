import torch
import torch.nn as nn
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, n_layers=3, n_hidden=100, vector=True):
        super(VariationalAutoEncoder, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encode = nn.ModuleList()
        self.out_features = 2 if vector else 1

        self.encode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.encode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.mu_enc = nn.Linear(self.n_hidden, 2)
        self.log_var_enc = nn.Linear(self.n_hidden, 2)

        self.decode = nn.ModuleList()
        self.decode.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.decode.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.mu_dec = nn.Linear(self.n_hidden, 2)
        self.log_var_dec = nn.Linear(self.n_hidden, self.out_features)

    def encoder(self, x):
        for layer in self.encode:
            x = torch.relu(layer(x))
        mu = self.mu_enc(x)
        log_var = self.log_var_enc(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = 0.5 * log_var.exp()
        qz_Gx_dist = torch.distributions.Normal(loc=mu, scale=std)
        z_Gx = qz_Gx_dist.rsample()
        return qz_Gx_dist, z_Gx

    def decoder(self, x):
        for layer in self.decode:
            x = torch.relu(layer(x))
        mu_dec = torch.sigmoid(self.mu_dec(x))
        log_var = torch.sigmoid(self.log_var_dec(x))
        return mu_dec, log_var

    def forward(self, x):
        mu_enc, log_var = self.encoder(x)
        z = self.reparameterize(mu_enc, log_var)
        mu_dec, std_dec = self.decoder(z)
        return mu_dec, std_dec

    def calc_loss(self, x, beta):
        mu_enc, log_var_enc = self.encoder(x)
        qz_Gx_dist, z_Gx = self.reparameterize(mu_enc, log_var_enc)
        mu_dec, log_var_dec = self.decoder(z_Gx)

        # Find q(z|x)
        # log_QhGx = qz_Gx_dist.log_prob(z_Gx).sum(dim=-1)
        # log_QhGx = torch.sum(log_QhGx, -1)
        log_QhGx = torch.sum(-0.5 * np.log(2 * np.pi) - x ** 2 / 2, dim=-1)

        # Find p(z)
        # p_z = torch.distributions.Normal(
        #     loc=torch.zeros([2]).to(self.device), scale=torch.ones([2]).to(self.device)).log_prob(z_Gx)
        # log_Pz = torch.sum(p_z, -1)
        log_Pz = torch.sum(-0.5 * z_Gx ** 2 - 0.5 * torch.log(2 * z_Gx.new_tensor(np.pi)), -1)

        # Find p(x|z)
        # px_Gz = torch.distributions.Normal(loc=mu_dec, scale=std_dec).log_prob(x)
        # log_PxGz = torch.sum(px_Gz, -1)
        log_PxGz = torch.sum(-0.5 * np.log(2 * np.pi) - log_var_dec / 2 - (x - mu_dec) ** 2 / (2 * torch.exp(
            log_var_dec)), -1)


        # nll and kl
        print(log_PxGz.mean(), log_Pz.mean(), log_QhGx.mean())
        nll = torch.mean(log_PxGz)
        kl = torch.mean(log_Pz - log_QhGx)
        ELBO = nll + kl*beta

        return ELBO, kl, nll
