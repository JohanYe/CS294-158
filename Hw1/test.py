import numpy as np
import torch
import torch.nn
import torch.utils.data
import matplotlib.pyplot as plt
import math


def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))


def compute_loss(theta, x):
    # softmax
    _theta = torch.exp(theta) / torch.sum(torch.exp(theta))

    # instead of having a one hot encoded vector x, just use gather instead:
    prob = torch.gather(_theta, dim=0, index=x)

    loss = torch.sum(-torch.log2(prob)) / prob.shape[0]
    return loss

sampled_data = sample_data()
n = len(sampled_data)
train_data = sampled_data[:int(0.6 * n)]
val_data = sampled_data[int(0.6 * n):int(0.8 * n)]
test_data = sampled_data[int(0.8 * n):]
batch_size = 512
theta = torch.zeros(100, requires_grad=True)
optimizer = torch.optim.Adam([theta], lr=3e-4)
epochs = 1000

train_iter = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_iter = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
losses = []
val_losses = []

for epoch in range(epochs):
    train_loss = 0

    for train_batch in train_iter:
        loss = compute_loss(theta, train_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)

    with torch.set_grad_enabled(False):

        val_loss = compute_loss(theta, torch.from_numpy(val_data))
        val_losses.append(val_loss)

    if epoch % 100 == 0:
        print('Epoch {}: loss{} val_loss{} '.format(epoch, loss, val_loss))

import matplotlib.pyplot as plt

plt.plot(losses, label="train_loss")
plt.plot(np.arange(0, len(losses), int(len(losses) / len(val_losses))), val_losses, label="val_loss")
plt.legend()
plt.show()

