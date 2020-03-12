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
batch_size = 125
n_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VariationalAutoEncoder(vector=True).to(device)
optimizer = optim.Adam(net.parameters(), lr=2e-4)
train_log = []
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'


fig, ax = plt.subplots(1, 2)
# Data set 1
x1 = sample_data_1()
ax[0].scatter(x1[:, 0], x1[:, 1])
ax[0].set_title('Data set 1')

# Data set 2
x2 = sample_data_2()
ax[1].scatter(x2[:, 0], x2[:, 1])
ax[1].set_title('Data set 2')
plt.savefig('./Hw3/Ex1/Figure_1.pdf', bbox_inches='tight')

if ds1:
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(x1[:int(len(x1) * 0.8)]).float(), batch_size=batch_size, shuffle=True)
    X_val = torch.from_numpy(x1[int(len(x1) * 0.8):]).to(device).float()
else:
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(x2[:int(len(x2) * 0.8)]).float(), batch_size=batch_size, shuffle=True)
    X_val = torch.from_numpy(x2[int(len(x2) * 0.8):]).to(device).float()



# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        loss, kl, nll = net.calc_loss(batch,beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1
        if 1 > beta:
            beta += 0.001

    with torch.no_grad():
        loss, kl, nll = net.calc_loss(X_val, beta)
        val_log[k] = loss.item()

    # if loss.item() < best_nll:
    #     best_nll = loss.item()
    #     save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch + 1, n_epochs, k, np.mean(train_log[-10:]), val_log[k]))

# Plotting each minibatch step
x_val = list(val_log.keys())
y_val = list(val_log.values())

# Plot the loss graph
train_x_vals = np.arange(len(train_log))
train_y_vals = train_log

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(train_x_vals, train_y_vals, label='Training Error')
ax[0].plot(x_val, y_val, label='Validation Error')
ax[0].legend(loc='best')
ax[0].set_title('Training Curve')
ax[0].set_xlabel('Num Steps')
ax[0].set_ylabel('NLL in bits per dim')

# # Load best and generate
# load_checkpoint('./checkpoints/best.pth.tar', net)
# pdf = net.sampling(pixel=200)
#
# ax[1].imshow(np.rot90(pdf.cpu().numpy()))
# ax[1].set_xticks([])
# ax[1].set_yticks([])
# ax[1].set_title("Best distribution on validation set")
#
# plt.savefig('./Hw2/Figures/Figure_2.pdf', bbox_inches='tight')
# plt.close()
#
# # Latent visualization
# x, y = sample_data()
# x = torch.from_numpy(x).float().to(device)
# pi, mu, var = net(x)
# z = net.Latent(x, pi, mu, var).cpu().detach().numpy()
#
# plt.figure(3)
# plt.scatter(z[:, 0], z[:, 1], c=y)
# plt.savefig('./Hw2/Figures/Figure_3.pdf', bbox_inches='tight')
#
