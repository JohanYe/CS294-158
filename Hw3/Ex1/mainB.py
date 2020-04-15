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
batch_size = 125
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = IWAE(n_hidden=128).to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)
train_log = {}
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'


# Data set 1
x3, y3 = sample_data_3()
plt.figure(1)
plt.scatter(x3[:, 0], x3[:, 1])
plt.title('Data set 3')
plt.savefig('./Hw3/Figures/Figure_4.pdf')
train_loader = torch.utils.data.DataLoader(torch.from_numpy(x3[:int(len(x3) * 0.8)]).float(), batch_size=batch_size,
                                           shuffle=True)
X_val = torch.from_numpy(x3[int(len(x3) * 0.8):]).to(device).float()

# Training loop
for epoch in range(n_epochs):
    batch_loss = []
    for batch in train_loader:
        net.train()
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch, beta, num_samples=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log[k] = [loss.item(), kl.item(), nll.item()]
        batch_loss.append(loss.item())

        k += 1
        if 1 > beta:
            beta += 0.0005

    with torch.no_grad():
        net.eval()
        loss, kl, nll = net.calc_loss(X_val, beta)
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

plt.figure(2)
plt.plot(train_x_vals, loss_train, label='Training ELBO')
plt.plot(x_val, loss_val, label='Validation ELBO')
plt.plot(train_x_vals, kl_train, label='Training KL')
plt.plot(x_val, kl_val, label='Validation KL')
plt.plot(train_x_vals, recon_train, label='Training Reconstruction Error')
plt.plot(x_val, recon_val, label='Validation Reconstruction Error')
plt.legend(loc='best')
plt.title('Training Curve')
plt.xlabel('Num Steps')
plt.ylabel('NLL in bits per dim')
plt.savefig('./Hw3/Figures/Figure_5.pdf', bbox_inches='tight')
# plt.close()

# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
plt.figure(3)
reconstructions = net(X_val[:1000]).detach().cpu().numpy()
plt.scatter(X_val.cpu()[:1000, 0], X_val.cpu()[:1000, 1], label='X original')
plt.scatter(reconstructions[:, 0], reconstructions[:, 1], label='Recon with noise')

samples = net.sample(1000).detach().cpu().numpy()
# plt.scatter(samples[:, 0], samples[:, 1], label='Sampled samples')
plt.legend(loc='best')
plt.savefig('./Hw3/Figures/Figure_6.pdf', bbox_inches='tight')



# # Latent visualization
# x, y = sample_data()
# x = torch.from_numpy(x).float().to(device)
# pi, mu, var = net(x)
# z = net.Latent(x, pi, mu, var).cpu().detach().numpy()
#
# plt.figure(3)
# plt.scatter(z[:, 0], z[:, 1], c=y)
# plt.savefig('./Hw2/Figures/Figure_3.pdf', bbox_inches='tight')
