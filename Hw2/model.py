import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        # Register buffer are weights that are not trained - this is gonna be used for masking, i believe buffer
        # stops backpropagation
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('mask', torch.ones([out_features, in_features]))  # .double())
        self.reset_parameters()

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

    def set_mask(self, mask):
        self.mask.data = mask.data

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class flow(nn.Module):
    """
    MADE one hidden layer
    """
    def __init__(self, k, hidden_size = 100):
        super(flow, self).__init__()

        self.k = k
        mask_1 = torch.ones([hidden_size, 2]).to(device)
        mask_1[:, 1] = 0  # masking x2
        mask_2 = torch.ones([2*self.k, hidden_size]).to(device)
        mask_2[:k, :] = 0  # Not entirely sure what this does

        # x1
        self.pi_1 = MaskedLinear(2, hidden_size)
        self.pi_1.set_mask(mask_1)
        self.mu_1 = MaskedLinear(2, hidden_size)
        self.mu_1.set_mask(mask_1)
        self.log_var_1 = MaskedLinear(2, hidden_size)
        self.log_var_1.set_mask(mask_1)

        # x2
        self.pi_2 = MaskedLinear(hidden_size, 2 * self.k)
        self.pi_2.set_mask(mask_2)
        self.mu_2 = MaskedLinear(hidden_size, 2 * self.k)
        self.mu_2.set_mask(mask_2)
        self.log_var_2 = MaskedLinear(hidden_size, 2 * self.k)
        self.log_var_2.set_mask(mask_2)

    def forward(self, x):

        pi = self.pi_2(F.relu(self.pi_1(x)))
        pi = F.softmax(pi.view(-1,2,self.k),dim=2)

        mu = self.mu_2(F.relu(self.mu_1(x)))
        mu = mu.view(-1,2,self.k)

        log_var = self.log_var_2(F.relu(self.log_var_1(x)))
        var = torch.exp(log_var).view(-1,2,self.k)

        return pi, mu, var

    def sampling(self, pixel=100):
        axis = np.linspace(-4, 4, pixel)
        samples = np.array(np.meshgrid(axis, axis)).T.reshape([-1, 2])
        samples = torch.from_numpy(samples).to(device).float()
        with torch.no_grad():
            pi, mu, var = self(samples)

        # calc loss
        weighted = pi * (torch.exp(- torch.pow(samples.unsqueeze(2) - mu, 2) / (2 * var)) / torch.sqrt(2 * np.pi * var))
        density = torch.sum(weighted, dim=2)
        joint = density[:, 0] * density[:, 1]
        pdf = torch.exp(-1 * -torch.log(torch.abs(joint))).reshape(pixel, pixel)
        return pdf

    def Latent(self, x, pi, mu, var):

        z = torch.distributions.normal.Normal(mu, var).cdf(x.unsqueeze(2))
        z = torch.sum(pi * z, dim=2)
        return z


class AffineCoupling(nn.Module):
    """ Implementation inspired by 02456 Deep Learning """

    def __init__(self, in_features=2, hidden_features=100):
        super(AffineCoupling, self).__init__()

        self.scale = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Tanh(),
        )

        self.translate = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x, mask=None):
        """ forward propagation, in this case going from data to noise """
        x_masked = mask * x
        s = self.scale(x_masked) * (1 - mask)
        t = self.translate(x_masked) * (1 - mask)

        y = x_masked + (1 - mask) * (x * torch.exp(s) + t)  # Eq 9 RealNVP
        jacobian = torch.sum(s, dim=-1)
        return y, jacobian

    def inverse(self, x, mask=None):
        """ Inverse propagation, going from noise to data """
        x_masked = mask * x
        s = self.scale(x_masked) * (1 - mask)
        t = self.translate(x_masked) * (1 - mask)

        x = x_masked + (1 - mask) * (x - t) * torch.exp(-s)  # Inverse propagation
        # jacobian = torch.sum(s, dim=-1)
        return x #, jacobian


class RealNVP(nn.Module):
    def __init__(self, in_features=2, hidden_features=100, AC_layers=3):
        super(RealNVP, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList([AffineCoupling(in_features, hidden_features) for _ in range(AC_layers)])

        # 2D mask - change which variable is masked each iteration to learn from both
        mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * (AC_layers // 2)).astype(np.float32))
        self.register_buffer('mask', mask)

    def forward(self, x):
        """ Forward through all affine coupling layers"""
        jacobian = torch.zeros(x.shape[0]).to(self.device)
        z = x
        # for i in range(len(self.layers)):
        #     z, log_determinant = self.layers[i](z, self.mask[i])
        #     jacobian += log_determinant  # jacobian, log det(ab) = log det(a) + log det(b)
        for mask, layer in zip(self.mask, self.layers):
            x, log_determinant_j = layer.forward(x, mask)
            jacobian += log_determinant_j

        return z, jacobian

    def sample(self, num_samples, prior):
        x = prior.sample((num_samples, 1)).to(self.device)  # z is sampled, but then converted to x
        for i in range(len(self.layers)):
            x = self.layers[i].inverse(x, self.mask[i])

        return x