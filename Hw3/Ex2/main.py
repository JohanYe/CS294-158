import numpy as np
import torch
import matplotlib.pyplot as plt
from Hw3.Ex2.Utils import *
from Hw3.Ex2.model import *
import seaborn as sns
import torch.optim as optim
import pickle as pkl
import torchvision

sns.set_style("darkgrid")

k = 0
beta = 0
batch_size = 64
n_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = IWAE1(n_hidden=128).to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)
train_log = {}
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'

# Loading data
train_loader = torch.utils.data.DataLoader(SVHNDataset('train'), batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(SVHNDataset('valid'), batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(SVHNDataset('test'), batch_size, shuffle=True)


quick_plot = next(iter(train_loader))
img_grid = torchvision.utils.make_grid(quick_plot.cpu(), nrow=8).numpy()
plt.figure(figsize=(12, 12))
plt.imshow(np.transpose(img_grid, (1, 2, 0)))
plt.axis('off')
plt.savefig('./Hw3/Figures/Figure_7.pdf')

# Training loop
for epoch in range(n_epochs):
    batch_loss = []
    for batch in train_loader:
        net.train()
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch, beta, 100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log[k] = [loss.item(), kl.item(), nll.item()]
        batch_loss.append(loss.item())

        k += 1
        if 1 > beta:
            beta += 0.001

    with torch.no_grad():
        net.eval()
        loss, kl, nll = net.calc_loss(X_val, 1)
        val_log[k] = [loss.item(), kl.item(), nll.item()]

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch + 1, n_epochs, k, np.mean(batch_loss), val_log[k][0]))

# Plotting each minibatch step
x_val = list(val_log.keys())
values = np.array(list(val_log.values()))
loss_val, kl_val, recon_val = values[:, 0], values[:, 1], values[:, 2]

# Plot the loss graph
train_x_vals = np.arange(len(train_log))
values = np.array(list(train_log.values()))
loss_train, kl_train, recon_train = values[:, 0], values[:, 1], values[:, 2]

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].plot(train_x_vals, loss_train, label='Training ELBO')
ax[0].plot(x_val, loss_val, label='Validation ELBO')
ax[0].set_title('Training Curve')
ax[1].plot(train_x_vals, kl_train, label='Training KL')
ax[1].plot(x_val, kl_val, label='Validation KL')
ax[1].set_title('Kullback-Leibler Divergence Curve')
ax[2].plot(train_x_vals, recon_train, label='Training Reconstruction Error')
ax[2].plot(x_val, recon_val, label='Validation Reconstruction Error')
ax[2].set_title('Reconstruction Error Curve')
plt.legend(loc='best')

plt.xlabel('Num Steps')
plt.ylabel('NLL in bits per dim')
# plt.savefig('./Hw3/Figures/Figure_5.pdf', bbox_inches='tight')
# plt.close()

# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
plt.figure(3)
reconstructions = net(X_val[:1000]).detach().cpu().numpy()
plt.scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
plt.scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon with noise')

samples = net.sample(1000).detach().cpu().numpy()
plt.scatter(samples[:, 0], samples[:, 1], label='Sampled samples')
plt.legend(loc='best')
# plt.savefig('./Hw3/Figures/Figure_6.pdf', bbox_inches='tight')

# Latent visualization
z = net.get_latent(X_val[:1000]).detach().cpu().numpy()
plt.figure(4)
plt.scatter(z[:, 0], z[:, 1], c=y_val[:1000])
# plt.savefig('./Hw2/Figures/Figure_7.pdf', bbox_inches='tight')
