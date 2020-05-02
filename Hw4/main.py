import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from Hw4.utils import *
from Hw4.model import *

sns.set_style("darkgrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
batch_size = 256
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Show some samples
x_plot, labels = next(iter(trainloader))
imshow(torchvision.utils.make_grid(x_plot[:100], nrow=10))
plt.savefig('./Hw4/figures/figure_1.pdf',bbox_inches='tight')

k = 0
g = Generator()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0, 0.9))
n_epochs = 2
train_log = []
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'


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
ax[1].scatter(x_plot[:, :, 0], x_plot[:, :, 1], s=9)
ax[1].set_title('Generated samples from p(z)')
plt.savefig('./Hw2/Figures/Figure_5.pdf', bbox_inches='tight')
plt.close()




