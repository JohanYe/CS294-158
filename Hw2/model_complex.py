import torch.nn as nn
import torch

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
        self.TranslateNet = TransformNet(hidden_size).to(self.device)
        self.ScaleNet = TransformNet(hidden_size).to(self.device)

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