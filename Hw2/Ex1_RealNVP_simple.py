import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Hw2.Utils import *
from Hw2.model import *

sns.set_style("darkgrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
batch_size = 125
x, y = sample_data()
x_train = x[:int(len(x) * 0.8)]
train_loader = torch.utils.data.DataLoader(
    torch.from_numpy(x_train).float(), batch_size=batch_size, shuffle=True)
X_val = torch.from_numpy(x[int(len(x) * 0.8):]).float().to(device)

k = 0
net = RealNVP(in_features=2, hidden_features=100, AC_layers=8).to(device)
# optimizer = optim.Adam(net.parameters(), lr=5e-4)
optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad == True], lr=1e-4)
n_epochs = 25
train_log = []
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'
prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))


def calc_loss(z, jacobian, prior):
    """
    Evaluate loss in prior distirbution
    :param z: affine_coupled data from RealNVP
    :param jacobina: Jacobian from RealNVP
    :return: loss
    """
    z = z.cpu()
    jacobian = jacobian.cpu().squeeze()

    # Evaluation in normal dist
    loss = prior.log_prob(z) + jacobian
    loss = -loss.mean() / np.log(2)
    return loss


# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        z, jacobian = net(batch)
        loss = calc_loss(z, jacobian, prior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1

    with torch.no_grad():
        z, jacobian = net(X_val)
        loss = calc_loss(z, jacobian, prior)
        val_log[k] = loss.item()

    if loss.item() < best_nll:  # since loss curve is very flat near bottom, we will neglect this
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d]\t[Step: %d]\tTrain Loss: %s\tTest Loss: %s' \
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

# Latent visualization
load_checkpoint('./checkpoints/best.pth.tar', net)
with torch.no_grad():
    z, _ = net(torch.from_numpy(x).to(device).float())
z = z.cpu().detach().numpy()
ax[1].scatter(z[:, 0], z[:, 1], c=y)
ax[1].set_title("Latent space")
plt.savefig('./Hw2/Figures/Figure_4.pdf', bbox_inches='tight')

# # Load best and generate + visualize latent space
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 5000)
ax[0].scatter(z[:, 0], z[:, 1])
ax[0].set_title("p(z)")


x_plot = net.sample(5000, prior)
x_plot = x_plot.detach().cpu()
ax[1].scatter(x_plot[:, :, 0], x_plot[:, :, 1])
ax[1].set_title('Generated samples from p(z)')
plt.savefig('./Hw2/Figures/Figure_5.pdf', bbox_inches='tight')
plt.close()




