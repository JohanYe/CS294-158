import torch
import torch.nn as nn

class VAE1(nn.Module):
    def __init__(self, n_layers=3, n_hidden=100, vector=True):
        super(VAE1, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vector = vector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.ModuleList()

        self.encoder.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.encoder.append(nn.Linear(self.n_hidden, self.n_hidden))

        self.mu_enc = nn.Linear(self.n_hidden, 2)
        self.log_var_enc = nn.Linear(self.n_hidden, 2)

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(2, self.n_hidden))
        for _ in range(n_layers - 2):
            self.decoder.append(nn.Linear(self.n_hidden, self.n_hidden))
        self.mu_dec = nn.Linear(self.n_hidden, 2)

        self.log_var_dec_vector = nn.Linear(self.n_hidden, 2)
        self.log_var_dec_scalar = nn.Linear(self.n_hidden, 1)

    def encoder(self, x):
        for layer in self.encoder:
            x = torch.relu(layer(x))
        mu = self.mu_enc(x)
        log_var = self.log_var_enc(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = 0.5*log_var.exp()
        qz_Gx_dist = torch.distributions.Normal(loc=mu, scale=std)
        qz_Gx = qz_Gx_dist.rsample()
        return qz_Gx_dist, qz_Gx

    def VectorCovariance(self, x):
        for layer in self.decoder:
            x = torch.relu(layer(x))
        mu_dec = self.mu_dec(x)
        log_var = self.log_var_dec_vector(x)
        return mu_dec, log_var

    def ScalarCovariance(self, x):
        for layer in self.decoder:
            x = layer(x)
        mu_dec = self.mu_dec(x)
        log_var = self.log_var_dec_scalar(x)
        std = 0.5*log_var.exp()
        return mu_dec, std

    def forward(self, x):


    def calc_loss(self, x):
        mu_enc, std_enc = self.encoder(x)
        qz_Gx_dist, qz_Gx = self.reparameterize(mu_enc, std_enc)

        












