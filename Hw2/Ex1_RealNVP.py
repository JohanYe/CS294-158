import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from Hw2.Utils import *
from Hw2.model import *
sns.set_style("darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 15
net = RealNVP(in_features=2,hidden_features=100,AC_layers=8).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
batch_size = 125
n_epochs = 2
train_log = []
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'

plt.figure(1)
x, y = sample_data()
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.savefig('./Hw2/Figures/Figure_1.pdf')
plt.close(1)
train_loader = torch.utils.data.DataLoader(torch.from_numpy(x[:int(len(x) * 0.8)]).float(), batch_size=batch_size, shuffle=True)
X_val = torch.from_numpy(x[int(len(x) * 0.8):]).float().to(device)


def calc_loss(z, jacobian):
    """
    Evaluate loss in prior distirbution
    :param z: affine_coupled data from RealNVP
    :param log_determinant: Log_determinant from RealNVP
    :return: loss
    """

    # Evaluation in normal dist
    prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    loss = prior.log_prob(z.cpu()) + jacobian.cpu().squeeze()
    loss = -loss.mean() / np.log(2)
    return loss

# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        z, jacobian = net(batch)
        loss = calc_loss(z, jacobian)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1

    with torch.no_grad():
        z, jacobian = net(X_val)
        loss = calc_loss(z, jacobian)
        val_log[k] = loss.item()

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d]\t[Step: %d]\tTrain Loss: %s\tTest Loss: %s' \
          % (epoch+1, n_epochs, k, np.mean(train_log[-10:]), val_log[k]))

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
ax[0].set_ylabel('NLL in bits per dim')

# # Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
pdf = net.sampling(pixel=200)

ax[1].imshow(np.rot90(pdf.cpu().numpy()))
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Best distribution on validation set")

plt.savefig('./Hw2/Figures/Figure_4.pdf', bbox_inches='tight')
plt.close()

# Latent visualization
x, y = sample_data()
x = torch.from_numpy(x).float().to(device)
pi, mu, var = net(x)
z = net.Latent(x, pi, mu, var).cpu().detach().numpy()

plt.figure(3)
plt.scatter(z[:, 0], z[:, 1], c=y)
plt.savefig('./Hw2/Figures/Figure_5.pdf', bbox_inches='tight')

plt.figure(4)
axis = np.linspace(-4, 4, 100)
samples = np.array(np.meshgrid(axis, axis)).T.reshape([-1, 2])
samples = torch.from_numpy(samples).to(device).float()
with torch.no_grad():
    z, jacobian = net(samples)

# calc loss
pdf = torch.exp(jacobian).cpu().numpy().reshape(100,100)
plt.imshow(np.rot90(pdf))
plt.set_xticks([])
plt.set_yticks([])
plt.set_title("Best distribution on validation set")


prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
z = prior.sample((125, 1))

