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
        mask_2[:k, :] = 0 # Not sure what this does?

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

