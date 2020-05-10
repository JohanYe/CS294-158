import numpy as np
import torch
import matplotlib.pyplot as plt
from Hw3.Ex1.Utils import *
from Hw3.Ex1.model import *
import seaborn as sns
import torch.optim as optim

sns.set_style("darkgrid")

ds1 = True
ds2 = False
k = 0
beta = 0
batch_size = 256
n_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_vector = VariationalAutoEncoder(vector=True).to(device)
optimizer_vector = optim.Adam(net_vector.parameters(), lr=0.001)
net_scalar = VariationalAutoEncoder(vector=False).to(device)
optimizer_scalar = optim.Adam(net_scalar.parameters(), lr=0.001)
train_log_vector = {}
train_log_scalar = {}
val_log_vector = {}
val_log_scalar = {}
best_nll = np.inf
best_nll2 = np.inf
save_dir = './checkpoints/'

fig, ax = plt.subplots(1, 2)
# Data set 1
x1 = sample_data_1()
ax[0].scatter(x1[:10000, 0], x1[:10000, 1])
ax[0].set_title('Data set 1')
ax[0].set_xlim(-25, 25)
ax[0].set_ylim(-25, 25)

# Data set 2
x2 = sample_data_2()
ax[1].scatter(x2[:10000, 0], x2[:10000, 1])
ax[1].set_title('Data set 2')
plt.savefig('./Hw3/Figures/Figure_1.pdf', bbox_inches='tight')
plt.close()

if ds1:
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(x1[:int(len(x1) * 0.8)]).float(), batch_size=batch_size,
                                               shuffle=True)
    X_val = torch.from_numpy(x1[int(len(x1) * 0.8):]).to(device).float()
else:
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(x2[:int(len(x2) * 0.8)]).float(), batch_size=batch_size,
                                               shuffle=True)
    X_val = torch.from_numpy(x2[int(len(x2) * 0.8):]).to(device).float()

# Training loop
for epoch in range(n_epochs):
    batch_loss_vector, batch_loss_scalar = [], []
    for batch in train_loader:
        net_vector.train()
        net_scalar.train()
        batch = batch.to(device)

        # vector
        loss, kl, nll = net_vector.calc_loss(batch, 1)
        optimizer_vector.zero_grad()
        loss.backward()
        optimizer_vector.step()
        train_log_vector[k] = [loss.item(), kl.item(), nll.item()]
        batch_loss_vector.append(loss.item())  # for print only

        # scalar
        loss2, kl2, nll2 = net_scalar.calc_loss(batch, 1)
        optimizer_scalar.zero_grad()
        loss2.backward()
        optimizer_scalar.step()
        train_log_scalar[k] = [loss2.item(), kl2.item(), nll2.item()]
        batch_loss_scalar.append(loss2.item())  # for print only

        k += 1
        if 1 > beta:
            beta += 0.0002

    with torch.no_grad():
        net_vector.eval()
        loss, kl, nll = net_vector.calc_loss(X_val, 1)
        val_log_vector[k] = [loss.item(), kl.item(), nll.item()]

        net_scalar.eval()
        loss2, kl2, nll2 = net_scalar.calc_loss(X_val, 1)
        val_log_scalar[k] = [loss2.item(), kl2.item(), nll2.item()]

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net_vector.state_dict()}, save_dir)

    if loss2.item() < best_nll2:
        best_nll = loss2.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net_scalar.state_dict()}, save_dir, ckpt_name='best2.pth.tar')

    print('[Epoch %d/%d][Step: %d] Vector Train Loss: %s Vector Test Loss: %s --- '
          'Scalar Train Loss: %s Scalar Test Loss: %s' \
          % (epoch + 1, n_epochs, k,
             round(np.mean(batch_loss_vector), 4), round(val_log_vector[k][0], 4),
             round(np.mean(batch_loss_scalar), 4), round(val_log_scalar[k][0], 4)))

# SCALAR
x_val = list(val_log_scalar.keys())
values = np.array(list(val_log_scalar.values()))
sloss_val, skl_val, srecon_val = values[:, 0], values[:, 1], values[:, 2]

# Plot the loss graph
train_x_vals = np.arange(len(train_log_scalar))
values = np.array(list(train_log_scalar.values()))
sloss_train, skl_train, srecon_train = values[:, 0], values[:, 1], values[:, 2]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(train_x_vals, sloss_train, label='Training ELBO')
ax[0].plot(x_val, sloss_val, label='Validation ELBO')
ax[0].plot(train_x_vals, skl_train, label='Training KL')
ax[0].plot(x_val, skl_val, label='Validation KL')
ax[0].plot(train_x_vals, srecon_train, label='Training Reconstruction Error')
ax[0].plot(x_val, srecon_val, label='Validation Reconstruction Error')
ax[0].legend(loc='best')
ax[0].set_title('Scalar variance Training Curve')
ax[0].set_xlabel('Num Steps')
ax[0].set_ylabel('NLL in bits per dim')

# VECTOR
x_val = list(val_log_vector.keys())
v_values = np.array(list(val_log_vector.values()))
vloss_val, vkl_val, vrecon_val = v_values[:, 0], v_values[:, 1], v_values[:, 2]

train_x_vals = np.arange(len(train_log_vector))
v_values = np.array(list(train_log_vector.values()))
vloss_train, vkl_train, vrecon_train = v_values[:, 0], v_values[:, 1], v_values[:, 2]

ax[1].plot(train_x_vals, vloss_train, label='Training ELBO')
ax[1].plot(x_val, vloss_val, label='Validation ELBO')
ax[1].plot(train_x_vals, vkl_train, label='Training KL')
ax[1].plot(x_val, vkl_val, label='Validation KL')
ax[1].plot(train_x_vals, vrecon_train, label='Training Reconstruction Error')
ax[1].plot(x_val, vrecon_val, label='Validation Reconstruction Error')
ax[1].legend(loc='best')
ax[1].set_title('Vector Variance Training Curve')
ax[1].set_xlabel('Num Steps')
ax[1].set_ylabel('NLL in bits per dim')
# plt.savefig('./Hw3/Figures/Figure_2.pdf', bbox_inches='tight')
# plt.close()


# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net_vector)
samples = net_vector.sample(1000, decoder_noise=False).detach().cpu().numpy()
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0, 0].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[0, 0].scatter(samples[:, 0], samples[:, 1], label='Sampled')
ax[0, 0].set_title("Vector no decoder noise")
ax[0, 0].set_xlim(-17.5, 17.5)
ax[0, 0].set_ylim(-17.5, 17.5)
ax[0, 0].legend(loc='best')

samples = net_vector.sample(1000, decoder_noise=True).detach().cpu().numpy()
ax[0, 1].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[0, 1].scatter(samples[:, 0], samples[:, 1], label='Sampled')
ax[0, 1].set_title("Vector decoder noise")
ax[0, 1].set_xlim(-17.5, 17.5)
ax[0, 1].set_ylim(-17.5, 17.5)
ax[0, 1].legend(loc='best')

load_checkpoint('./checkpoints/best2.pth.tar', net_scalar)
samples = net_scalar.sample(1000, decoder_noise=False).detach().cpu().numpy()
ax[1, 0].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[1, 0].scatter(samples[:, 0], samples[:, 1], label='Sampled')
ax[1, 0].set_title("Scalar no decoder noise")
ax[1, 0].set_xlim(-17.5, 17.5)
ax[1, 0].set_ylim(-17.5, 17.5)
ax[1, 0].legend(loc='best')

samples = net_scalar.sample(1000, decoder_noise=True).detach().cpu().numpy()
ax[1, 1].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[1, 1].scatter(samples[:, 0], samples[:, 1], label='Sampled')
ax[1, 1].set_title("Scalar decoder noise")
ax[1, 1].set_xlim(-17.5, 17.5)
ax[1, 1].set_ylim(-17.5, 17.5)
ax[1, 1].legend(loc='best')

if ds1:
    plt.savefig('./Hw3/Figures/Figure_3ds1.pdf', bbox_inches='tight')
else:
    plt.savefig('./Hw3/Figures/Figure_3ds2.pdf', bbox_inches='tight')

fig, ax = plt.subplots(2, 2, figsize=(10, 5))
reconstructions = net_vector(X_val[:1000], noise=False).detach().cpu().numpy()
ax[0, 0].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[0, 0].scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon without noise')
ax[0, 0].set_title("Vector No decoder noise")
ax[0, 0].set_xlim(-17.5, 17.5)
ax[0, 0].set_ylim(-17.5, 17.5)
ax[0, 0].legend(loc='best')

reconstructions = net_vector(X_val[:1000], noise=True).detach().cpu().numpy()
ax[0, 1].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[0, 1].scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon with noise')
ax[0, 1].set_title("Vector Decoder noise")
ax[0, 1].set_xlim(-17.5, 17.5)
ax[0, 1].set_ylim(-17.5, 17.5)
ax[0, 1].legend(loc='best')

reconstructions = net_scalar(X_val[:1000], noise=False).detach().cpu().numpy()
ax[1, 0].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[1, 0].scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon without noise')
ax[1, 0].set_title("Scalar No decoder noise")
ax[1, 0].set_xlim(-17.5, 17.5)
ax[1, 0].set_ylim(-17.5, 17.5)
ax[1, 0].legend(loc='best')

reconstructions_val = net_scalar(X_val[:1000], noise=True).detach().cpu().numpy()
ax[1, 1].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
ax[1, 1].scatter(reconstructions_val[:, 0], reconstructions_val[:, 1], label='Recon with noise')
ax[1, 1].set_title("Scalar Decoder noise")
ax[1, 1].set_xlim(-17.5, 17.5)
ax[1, 1].set_ylim(-17.5, 17.5)
ax[1, 1].legend(loc='best')
if ds1:
    plt.savefig('./Hw3/Figures/Figure_4ds1.pdf', bbox_inches='tight')
else:
    plt.savefig('./Hw3/Figures/Figure_4ds2.pdf', bbox_inches='tight')

y = np.zeros(1000)
for i in range(1000):
    if X_val[i, 0] > 0:
        if X_val[i, 1] > 2:
            y[i] = 3
        else:
            y[i] = 4
    else:
        if X_val[i, 1] > 2:
            y[i] = 5
        else:
            y[i] = 6

latent, mu_z = net_vector.get_latent(X_val[:1000])
prior = torch.distributions.Normal(0, 1).sample([1000, 2])
latent = latent.detach().cpu().numpy()
mu_z = mu_z.detach().cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(prior[:, 0], prior[:, 1], label='prior')
ax[0].scatter(latent[:, 0], latent[:, 1], c=y, label='latent_z')
ax[0].scatter(mu_z[:, 0], mu_z[:, 1], label='mean')
ax[0].legend(loc='best')
ax[0].set_title('Latent visualization of Vector Variance Network')

latent2, mu_z2 = net_scalar.get_latent(X_val[:1000])
prior = torch.distributions.Normal(0, 1).sample([1000, 2])
latent2 = latent2.detach().cpu().numpy()
mu_z2 = mu_z2.detach().cpu().numpy()
ax[1].scatter(prior[:, 0], prior[:, 1], label='prior')
ax[1].scatter(latent2[:, 0], latent2[:, 1], c=y, label='latent_z')
ax[1].scatter(mu_z2[:, 0], mu_z2[:, 1], label='mean')
ax[1].legend(loc='best')
ax[1].set_title('Latent visualization of Scalar Variance Network')

plt.figure(6)
plt.scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], c=y)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
reconstructions = net_vector(X_val[:1000], noise=False).detach().cpu().numpy()
ax[0].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], c=y, label='X original')
ax[0].scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon without noise')
ax[0].set_title("Vector No decoder noise")
ax[0].set_xlim(-17.5, 17.5)
ax[0].set_ylim(-17.5, 17.5)
ax[0].legend(loc='best')

reconstructions = net_vector(X_val[:1000], noise=True).detach().cpu().numpy()
ax[1].scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1],c=y , label='X original')
ax[1].scatter(reconstructions[:, 0], reconstructions[:, 1], c=y , label='Recon with noise')
ax[1].set_title("Vector Decoder noise")
ax[1].set_xlim(-17.5, 17.5)
ax[1].set_ylim(-17.5, 17.5)
ax[1].legend(loc='best')