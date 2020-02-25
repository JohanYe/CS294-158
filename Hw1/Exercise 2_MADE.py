# Homework 1.2_MADE Deep unsupervised learning
# Spring 2020
# Johan Ye
# CS294-158-SP19 Spring 2019

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data
from Hw1.DataGenerator import *
from Hw1.Utils import load_checkpoint,one_hot_cat
from Hw1.model import *
import torch

X = sample_data2d(2)
X_train, X_val = train_test_split(X, test_size=0.2)
ground_truth = np.load('Hw1/distribution.npy')

# plot the data of the three sets
fig, axes = plt.subplots(1, 3,figsize=(15,5))
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
axes[2].set_title("Test set")
axes[2].set_xticks([])
axes[2].set_yticks([])

fig.suptitle("Data")
plt.show()

# Hyperparams:
lr = 5e-4
d=2
n_classes=200
hidden_size=200
n_epochs = 20
batch_size = 250
n_iter_per_epoch = X_val.shape[0] // batch_size
m_0, m = [], []
train_log, val_log = {}, {}

#Mask
for i in range(2):
    m_0.extend([i + 1] * n_classes)

#Data loaders
train_loader = torch.utils.data.DataLoader(torch.Tensor(X_train), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.Tensor(X_val), batch_size=batch_size, shuffle=False)

# MADE
net = MADE(2, 200, 256)
optimizer = optim.Adam(net.parameters(), lr=lr)

net,train_log,val_log = train_MADE(n_epochs, train_loader, val_loader, net, optimizer, train_log, val_log, './checkpoints/', n_iter_per_epoch)


# Plotting each minibatch step
x_val = list(val_log.keys())
y_val = list(val_log.values())

# Plotting
fig, ax = plt.subplots(1,2,figsize=(10,5))
plt.figure(2)
train_x_vals = sorted(train_log.keys())
train_y_vals = [train_log[k] for k in train_x_vals]
ax[0].plot(train_x_vals, train_y_vals, label='Training Error')
ax[0].plot(x_val, y_val, label='Validation Error')
ax[0].legend(loc='best')
ax[0].set_title('MADE Training Curve')
ax[0].set_xlabel('Num Steps')
ax[0].set_ylabel('Negative Log Likelihood')

# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
dist_best, p = net.get_distribution()
ax[1].imshow(dist_best,aspect= 'auto')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Best distribution on validation set")

plt.savefig('./Hw1/Figures/Figure_6.pdf',bbox_inches='tight')