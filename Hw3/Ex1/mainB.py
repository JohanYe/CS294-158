import numpy as np
import torch
import matplotlib.pyplot as plt
from Hw3.Ex1.Utils import *
from Hw3.Ex1.model import *
import seaborn as sns
import torch.optim as optim

sns.set_style("darkgrid")

k = 0
beta = 0
batch_size = 128
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = IWAE1(n_hidden=64).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net2 = IWAE1(n_hidden=64).to(device)
optimizer2 = optim.Adam(net2.parameters(), lr=1e-3)
train_log_IWAE = {}
val_log_IWAE = {}
train_log_VAE = {}
val_log_VAE = {}
best_nll = np.inf
save_dir = './checkpoints/'

# Data set 1
x3, y3 = sample_data_3()
# x3, y3 = x3[:10000], y3[:10000]
plt.figure(1)
plt.scatter(x3[:, 0], x3[:, 1])
plt.title('Data set 3')
plt.savefig('./Hw3/Figures/Figure_4.pdf')
train_loader = torch.utils.data.DataLoader(torch.from_numpy(x3[:int(len(x3) * 0.8)]).float(), batch_size=batch_size,
                                           shuffle=True)
X_val = torch.from_numpy(x3[int(len(x3) * 0.8):]).to(device).float()
y_val = y3[int(len(y3) * 0.8):]
# Training loop
for epoch in range(n_epochs):
    batch_loss_IWAE, batch_loss_VAE = [], []
    for batch in train_loader:
        net.train()
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch, beta, num_samples=100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log_IWAE[k] = [loss.item(), kl.item(), nll.item()]
        batch_loss_IWAE.append(loss.item())

        net2.train()
        loss2, kl2, nll2 = net2.calc_loss(batch, beta, num_samples=1)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        train_log_VAE[k] = [loss2.item(), kl2.item(), nll2.item()]
        batch_loss_VAE.append(loss2.item())

        k += 1
        if 1 > beta:
            beta += 0.0005

    with torch.no_grad():
        net.eval()
        loss, kl, nll = net.calc_loss(X_val, 1)
        val_log_IWAE[k] = [loss.item(), kl.item(), nll.item()]

        net2.eval()
        loss2, kl2, nll2 = net2.calc_loss(X_val, 1)
        val_log_VAE[k] = [loss2.item(), kl2.item(), nll2.item()]


    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] IWAE Train Loss: %s IWAE Test Loss: %s --- VAE Train Loss: %s VAE Test Loss: %s' \
          % (epoch + 1, n_epochs, k, round(np.mean(batch_loss_IWAE), 4), round(val_log_IWAE[k][0], 4),
          round(np.mean(batch_loss_VAE),4), round(val_log_VAE[k][0],4)))

# Plotting each minibatch step
x_val = list(val_log_IWAE.keys())
values = np.array(list(val_log_IWAE.values()))
loss_val_IWAE, kl_val_IWAE, recon_val_IWAE = values[:, 0], values[:, 1], values[:, 2]

# Plot the loss graph
train_x_vals = np.arange(len(train_log_IWAE))
values = np.array(list(train_log_IWAE.values()))
loss_train_IWAE, kl_train_IWAE, recon_train_IWAE = values[:, 0], values[:, 1], values[:, 2]

# Plot variables
values = np.array(list(val_log_VAE.values()))
loss_val_VAE, kl_val_VAE, recon_val_VAE = values[:, 0], values[:, 1], values[:, 2]
values = np.array(list(train_log_VAE.values()))
loss_train_VAE, kl_train_VAE, recon_train_VAE = values[:, 0], values[:, 1], values[:, 2]


fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
ax[0].plot(train_x_vals, loss_train_VAE, label='VAE Training ELBO')
ax[0].plot(train_x_vals, loss_train_IWAE, label='IWAE Training ELBO')
ax[0].plot(x_val, loss_val_VAE, label='VAE Training ELBO')
ax[0].plot(x_val, loss_val_IWAE, label='IWAE Validation ELBO')
ax[0].set_title('ELBO Training Curve')
ax[0].set_xlabel('Num steps')
ax[0].legend(loc='best')

ax[1].plot(train_x_vals, kl_train_VAE, label='VAE Training KL')
ax[1].plot(train_x_vals, kl_train_IWAE, label='IWAE Training KL')
ax[1].plot(x_val, kl_val_VAE, label='VAE Validation KL')
ax[1].plot(x_val, kl_val_IWAE, label='IWAE Validation KL')
ax[1].set_title('Kullback-Leibler Divergence Curve')
ax[1].set_xlabel('Num steps')
ax[1].legend(loc='best')

ax[2].plot(train_x_vals, recon_train_VAE, label='VAE Training')
ax[2].plot(train_x_vals, recon_train_IWAE, label='IWAE Training')
ax[2].plot(x_val, recon_val_VAE, label='VAE Validation')
ax[2].plot(x_val, recon_val_IWAE, label='IWAE Validation')
ax[2].set_title('Reconstruction Error Curve')
ax[2].set_xlabel('Num steps')
ax[2].legend(loc='best')

plt.savefig('./Hw3/Figures/Figure_5.pdf', bbox_inches='tight')
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
plt.savefig('./Hw3/Figures/Figure_6.pdf', bbox_inches='tight')

# Latent visualization
z = net.get_latent(X_val[:1000]).detach().cpu().numpy()
plt.figure(4)
plt.scatter(z[:, 0], z[:, 1], c=y_val[:1000])
plt.savefig('./Hw3/Figures/Figure_7.pdf', bbox_inches='tight')


net.calc_loss(X_val[:100], 1)
net2.calc_loss(X_val[:100], 1)
