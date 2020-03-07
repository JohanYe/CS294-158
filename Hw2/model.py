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


# gave up on debugging this setup, should be close to working.
# class AffineCoupling(nn.Module):
#     """ Implementation inspired by 02456 Deep Learning """
#
#     def __init__(self, in_features=2, hidden_features=100):
#         super(AffineCoupling, self).__init__()
#
#         self.scale = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_features, hidden_features),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_features, in_features),
#             nn.Tanh(),
#         )
#
#         self.translate = nn.Sequential(
#             nn.Linear(in_features, hidden_features),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_features, hidden_features),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_features, in_features),
#         )
#
#     def forward(self, x, mask=None):
#         """ forward propagation, in this case going from data to noise """
#         x_masked = mask * x
#         s = self.scale(x_masked) * (1 - mask)
#         t = self.translate(x_masked) * (1 - mask)
#
#         z = (1 - mask) * (x - t) * torch.exp(-s) + x_masked
#         # z = x_masked + (1 - mask) * (x * torch.exp(s) + t)  # Eq 9 RealNVP
#         jacobian = torch.sum(s, dim=-1)#, keepdim=True)
#         return z, jacobian
#
#     def inverse(self, x, mask=None):
#         """ Inverse propagation, going from noise to data """
#         x_masked = mask * x
#         s = self.scale(x_masked) * (1 - mask)
#         t = self.translate(x_masked) * (1 - mask)
#
#         # x = x_masked + (1 - mask) * (x * torch.exp(s) + t)
#         x = x_masked + (1 - mask) * (x - t) * torch.exp(-s)  # Inverse propagation
#         # jacobian = torch.sum(s, dim=-1)
#         return x #, jacobian
#
#
# class RealNVP(nn.Module):
#     def __init__(self, in_features=2, hidden_features=100, AC_layers=3):
#         super(RealNVP, self).__init__()
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.layers = nn.ModuleList([AffineCoupling(in_features, hidden_features) for _ in range(AC_layers)])
#
#         # 2D mask - change which variable is masked each iteration to learn from both
#         mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * (AC_layers // 2)).astype(np.float32))
#         self.register_buffer('mask', mask)
#
#     def forward(self, x):
#         """ Forward through all affine coupling layers"""
#         jacobian = torch.zeros(x.shape[0]).to(self.device)
#         z = x
#         # for i in range(len(self.layers)):
#         #     z, log_determinant = self.layers[i](z, self.mask[i])
#         #     jacobian += log_determinant  # jacobian, log det(ab) = log det(a) + log det(b)
#         for mask, layer in zip(self.mask, self.layers):
#             z, log_determinant_j = layer.forward(z, mask)
#             jacobian -= log_determinant_j
#
#         return z, jacobian
#
#     def sample(self, num_samples, prior):
#         x = prior.sample((num_samples, 1)).to(self.device)  # z is sampled, but then converted to x
#         for i in range(len(self.layers)):
#             x = self.layers[i].inverse(x, self.mask[i])
#
#         return x

class TransformNet(nn.Module):
    """
    Network that learns the scale and translate functions of the RealNVP paper
    """

    def __init__(self, hidden_size=100):
        super(TransformNet, self).__init__()

        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2), )

    def forward(self, x):
        return self.layers(x)


class ActNorm(nn.Module):
    """ Performs affine transformation using scale and bias """

    def __init__(self, dimensions):
        super(ActNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones([1, dimensions]))
        self.translate = nn.Parameter(torch.zeros([1, dimensions]))

    def initialize(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        # Copy is used to save us for formatting tensor
        self.scale.data.copy_(1.0 / std)
        self.translate.data.copy_(-torch.mean(x * self.scale, dim=0, keepdim=True))

    def forward(self, x, forward=True):
        if forward:
            log_determinant = torch.sum(torch.log(torch.abs(self.scale) + 1e-5), dim=1, keepdim=True)
            # log_determinant = torch.sum(torch.log(self.scale), dim=-1, keepdim=True)
            out = x * self.scale + self.translate
            return out, log_determinant
        else:
            out = (x - self.translate) / self.scale
            return out


class CouplingLayer(nn.Module):

    def __init__(self, mask, hidden_size=100, n_layers=1):
        super(CouplingLayer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("mask", mask)
        self.TranslateNet = TransformNet(hidden_size).to(device)
        self.ScaleNet = TransformNet(hidden_size).to(device)

    def forward(self, x, forward=True):
        z = self.mask * x
        scale = torch.tanh(self.ScaleNet(z))
        translate = self.TranslateNet(z)

        if forward:  # Forward propagation (data -> noise)
            z = z + (1 - self.mask) * (x * torch.exp(scale) + translate)  # Eq 9 RealNVP
            log_determinant = torch.sum((1 - self.mask) * scale, dim=1, keepdim=True)
            return z, log_determinant
        else:  # inverse propagation  (noise -> data)
            z = z + (1 - self.mask) * (x - translate) * torch.exp(-scale)
            return z


class RealNVP(nn.Module):

    def __init__(self, n_layers=5):
        super(RealNVP, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_layers = n_layers
        mask = torch.Tensor([0, 1]).float()
        self.layers = nn.ModuleList()

        for i in range(self.n_layers - 1):  # All but final layer
            mask = 1 - mask
            self.layers.append(CouplingLayer(mask))
            self.layers.append(ActNorm(2))
        self.layers.append(CouplingLayer(mask))
        self.layers.to(self.device)

    def initialize(self, x):  # Initialize Actnorm layers
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, ActNorm):
                    layer.initialize(x)
                x, _ = layer(x)

    def forward(self, x, forward=True):
        if forward:
            log_determinant_sum = torch.zeros([x.shape[0], 1]).float().to(self.device)
            for layer in self.layers:
                x, log_determinant = layer(x)
                log_determinant_sum += log_determinant
            out = torch.sigmoid(x)
            # maybe add small constant before log to avoid undefined error.
            log_determinant_sigmoid = torch.sum(torch.log(out * (1 - out + 1e-5)), dim=1, keepdim=True)
            log_determinant_sum += log_determinant_sigmoid
            return out, log_determinant_sum
        else:
            z = - torch.log(1 / x - 1)  # Reversing the sigmoid
            for layer in reversed(self.layers):
                z = layer(z, forward=False)
            return z