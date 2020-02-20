import itertools
from typing import Any, Union

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import itertools

# loss function
from Hw1.Utils import one_hot_cat, get_mean_NLL, save_checkpoint
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_loss(theta, data_batch):
    # gather picks indexes of the data we send in
    nominator = torch.gather(theta, dim=0, index=data_batch)

    # Softmax of theta
    px = torch.exp(nominator) / torch.sum(torch.exp(theta))

    # average NLL
    a_NLL = torch.sum(-1 * torch.log2(px)) / px.shape[0]

    return a_NLL

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.theta = nn.Parameter(torch.zeros([200]).float())  # Like in exercise 1 (x1)
        self.prob2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # torch doesn't understand it if i don't int64 it, 32 does not work...
        index_tensor = x[:,0].long().to(device)
        x1hot = F.one_hot(index_tensor, num_classes=200).float() #needs to be float for nn.Linear later

        # nominator = torch.gather(self.theta, dim=0, index=index_tensor)
        px1 = torch.exp(self.theta) / torch.sum(torch.exp(self.theta)) #Exactly like exercise 1
        #px1 = torch.gather(px1, dim=0, index=index_tensor)
        px2Gx1 = self.prob2(x1hot) #p(x1|x2)
        px2Gx1 = torch.gather(px2Gx1, dim=1, index=x[:, 1:].long().to(device))
        return px1, px2Gx1

    def get_distribution(self):
        px1 = torch.softmax(self.theta, dim=0)
        x1hot = torch.eye(200).to(device) # to get dist for each
        px2Gx1 = self.prob2(x1hot)
        return (px2Gx1*px1).detach().cpu().data.numpy()

def train(num_epochs, train_loader, val_loader, theta, optimizer, train_losses, val_losses, k):
    """
    Training loop
    :param num_epochs: number of epochs training
    :return:
    """
    best_NLL = np.inf
    for epoch in range(num_epochs):
        for idx, train_batch in enumerate(train_loader):
            tr_loss = calc_loss(theta, train_batch)

            optimizer.zero_grad()

            # This calculated 'backward' direction of gradient
            tr_loss.backward()

            # This uses the the calculated gradient to change the theta values in the correction direction
            optimizer.step()

            # train_loss
            train_losses[k] = tr_loss.item()
            k += 1

        print('[Epoch %d/%d][Step: %d] Train Loss: %s' \
              % (epoch, num_epochs, k, tr_loss.item()))

        # Perform validation after each epoch
        for idx, test_batch in enumerate(val_loader):
            val_losses[k] = calc_loss(theta, test_batch)

        if calc_loss(theta, test_batch) < best_NLL:
            best_NLL = calc_loss(theta, test_batch)

    return train_losses, val_losses, best_NLL


class MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        # Register buffer are weights that are not trained - this is gonna be used for masking, i believe buffer
        # stops backpropagation
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('mask', torch.ones([out_features, in_features]))  # .double())
        self.reset_parameters()

    def forward(self, x):
        # print(self.weight)
        # print(self.weight.size())
        # print(self.mask.size())
        return F.linear(x, self.mask * self.weight, self.bias)

    def set_mask(self, mask):
        self.mask.data = mask.data

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MADE(nn.Module):

    def __init__(self, d=2, n_classes=200, hidden_size=200):
        super(MADE, self).__init__()
        self.d = d
        self.n_classes = n_classes
        self.input_size = d * n_classes
        self.hidden_size = hidden_size

        self.layer1 = MaskedLinear(self.input_size, self.hidden_size)
        self.outlayer = MaskedLinear(self.hidden_size, self.input_size)

        # generating the numbers in the nodes for MADE
        self.m = []
        m_0 = []
        for i in range(d):
            m_0.extend([i + 1] * n_classes)

        # Appending numbered nodes to attribute 'm'
        self.m.append(torch.from_numpy(np.array(m_0)).requires_grad_(False))  # requires_grad is redundant?
        self.m.append(torch.randint(torch.min(self.m[-1]), d, [hidden_size]).requires_grad_(False))

        # determining when to mask
        mask = (self.m[1].view(-1, 1) >= self.m[1 - 1].view(1, -1).long()).float()
        self.layer1.set_mask(mask)

        output_mask = (self.m[0].view(-1, 1).float() > self.m[-1].view(1, -1).float()).float()
        self.outlayer.set_mask(output_mask)

    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.outlayer(out)
        out = out.view(-1, self.d, self.n_classes)
        out = F.softmax(out, dim=2)
        return out

    #sample
    def get_distribution(self):
        shell = []
        for i in range(self.d):
            shell.append(np.array(np.arange(0, self.n_classes)).reshape([self.n_classes]))
        x_long = torch.from_numpy(np.array(list(itertools.product(*shell)))).long()
        x1hot = one_hot_cat(x_long)
        p = self.forward(x1hot)
        batch_size = x1hot.shape[0]
        p = torch.gather(p, 2, x_long.view([batch_size, self.d, 1])).view(batch_size, 2)
        p = p.detach().numpy()
        ans = np.ones([p.shape[0]]) #[40000,]
        for i in range(p.shape[1]):
            ans = ans * p[:, i]
        return ans.reshape([self.n_classes] * self.d), p


def train_MADE(num_epochs, train_loader, val_loader, model, optimizer, train_losses, val_losses, save_dir, n_iter_per_epoch):
    best_nll = np.inf
    k = 0
    for epoch in range(num_epochs):
        for idx, train_batch in enumerate(train_loader):
            x_train_long = train_batch.long()
            x_train = one_hot_cat(x_train_long)
            p_train = model(x_train)

            loss = get_mean_NLL(p_train, x_train_long)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nll_train = loss.clone().detach() / (2 * np.log(2.0))
            train_losses[k] = nll_train.item()
            k += 1

        for idx, val_batch in enumerate(val_loader):
            x_val_long = val_batch.long()
            x_val = one_hot_cat(x_val_long)
            p_val = model(x_val)

            nll_val = get_mean_NLL(p_val, x_val_long).detach() / (2 * np.log(2.0))

            val_losses[k] = nll_val.item()

            print("Epoch-{:d}/{:d} Iter-{:d}/{:d} loss: {:.4f}, train_nll: {:.3f}, val_nll: {:.3f}"
                  .format(epoch, num_epochs, idx+1, n_iter_per_epoch, loss.item(), nll_train.item(), nll_val.item()))

            if nll_val < best_nll:
                best_nll = nll_val
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict()}, save_dir)

    save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()}, save_dir, ckpt_name='last.pth.tar')

    return model, train_losses, val_losses