# Homework 1.2_MLP Deep unsupervised learning
# Spring 2020
# Johan Ye
# CS294-158-SP19 Spring 2019
# I forgot this part - so some code practices may differ from  Ex1 and Ex2.2 - still learning pycharm setup

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data
from Hw1.DataGenerator import *
from Hw1.Utils import load_checkpoint, one_hot_cat
from Hw1.model import *
import torch

X = sample_data2d(2)
X_train, X_val = train_test_split(X, test_size=0.2)
ground_truth = np.load('Hw1/distribution.npy')

# plot the data of the three sets
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(ground_truth)
axes[0].set_title("Ground truth")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].hist2d(X_train[:, 1], X_train[:, 0], bins=[200, 200])
axes[1].invert_yaxis()
axes[1].set_title("Training set")
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].hist2d(X_val[:, 1], X_val[:, 0], bins=[200, 200])
axes[2].invert_yaxis()
axes[2].set_title("Validation set")
axes[2].set_xticks([])
axes[2].set_yticks([])

fig.suptitle("Data")
plt.savefig('./Hw1/Figures/Figure_4.pdf', bbox_inches='tight')
plt.close()

# Hyperparameters:
batch_size = 250
lr = 1e-3
n_epochs = 20
train_log = []
val_log = {}
k = 0
best_nll = np.inf
save_dir = './checkpoints/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders

train_loader = torch.utils.data.DataLoader(torch.Tensor(X_train), batch_size=batch_size, shuffle=True)
X_val = torch.Tensor(X_val).to(device)

# Initialize model
net = MLP().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)


def calc_loss(px1, px2Gx1):
    # log to get NLL, mean to get per dim, divide by log2 to get in bits
    loss = -torch.mean(torch.log2(px1*px2Gx1))/2
    return loss


# Training loop
for epoch in range(n_epochs):
    for mini_batch in train_loader:
        mini_batch.to(device) #To save memory we send one batch to cuda at the time
        px1, px2Gx1 = net(mini_batch)
        loss = calc_loss(px1, px2Gx1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1

    with torch.no_grad():
        px1, px2Gx1 = net(X_val)
        loss = calc_loss(px1, px2Gx1)
        val_log[k] = loss.item()

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch,
                         'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch, n_epochs, k, np.mean(train_log[-10:]), val_log[k]))

# Plotting each minibatch step
x_val = list(val_log.keys())
y_val = list(val_log.values())

# Plot the loss graph
train_x_vals = np.arange(len(train_log))
train_y_vals = train_log

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(train_x_vals, train_y_vals, label='Training Error')
ax[0].plot(x_val, y_val, label='Validation Error')
ax[0].legend(loc='best')
ax[0].set_title('Training Curve')
ax[0].set_xlabel('Num Steps')
ax[0].set_ylabel('Negative Log Likelihood')

# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
dist_best = net.get_distribution()
ax[1].imshow(dist_best,aspect= 'auto')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Best distribution on validation set")

plt.savefig('./Hw1/Figures/Figure_5.pdf',bbox_inches='tight')
plt.close()
