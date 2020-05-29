import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Hw4.utils import *
from Hw4.model import *

sns.set_style("darkgrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: SKAL JEG GENERERE INCEPTION SCORES?

# Data Loading
batch_size = 256
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# Show some samples
x_plot, labels = next(iter(train_loader))
imshow(torchvision.utils.make_grid(x_plot[:100], nrow=10))
plt.savefig('./Hw4/figures/figure_1.pdf', bbox_inches='tight')
plt.close()

# Hyperparams
k = 0
epoch = 0
n_epochs = 60
g_train_log = []
d_train_log = []
best_nll = np.inf
best_nll_negative = np.inf
save_dir = './checkpoints/'
critic_iter = 0
n_critic = 3

# Generator
g = Generator().to(device)
g_optimizer = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0, 0.9))
g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda epoch: (n_epochs - epoch) / n_epochs, last_epoch=-1)

# Discriminator
d = Discriminator().to(device)
d_optimizer = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0, 0.9))
d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda epoch: (n_epochs - epoch) / n_epochs, last_epoch=-1)


def gradient_penalty(x_real, x_fake, LAMBDA=10):
    # Gradient penalty according to improving WGAN training
    alpha = torch.rand(x_real.size(0), 1, 1, 1).expand_as(x_real).to(device)
    interpolates = (alpha * x_real + ((1 - alpha) * x_fake)).requires_grad_(True)
    disc_interpolates = d(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean() * LAMBDA


# Training loop
for epoch in range(n_epochs):
    g_train_batch_loss, d_train_batch_loss = [], []
    for batch, _ in tqdm(train_loader):
        critic_iter += 1
        x_real = batch.to(device)

        d_optimizer.zero_grad()
        x_fake = g(bs=x_real.shape[0])
        gp = gradient_penalty(x_real, x_fake)
        d_loss = d(x_fake).mean() - d(x_real).mean() + gp
        d_loss.backward()
        d_optimizer.step()
        d_train_batch_loss.append(d_loss.item())

        if critic_iter % n_critic == 0:  # How does this not result in too powerful discriminator?
            g_optimizer.zero_grad()
            x_fake = g(bs=x_real.shape[0])
            g_loss = -d(x_fake).mean()
            g_loss.backward()
            g_optimizer.step()

            # g_scheduler.step()
            # d_scheduler.step()

            g_train_batch_loss.append(g_loss.item())

    g_train_log.append(np.mean(g_train_batch_loss))
    d_train_log.append(np.mean(d_train_batch_loss))

    if g_train_log[-1] < best_nll and g_train_log[-1] > 0:  # since loss curve is very flat near bottom, we will neglect this
        best_nll = g_train_log[-1]
        save_checkpoint({'g_state_dict': g.state_dict(), 'd_state_dict': d.state_dict(),
                         'g_optimizer': g_optimizer.state_dict(), 'd_optimizer': d_optimizer.state_dict(),
                         # 'g_scheduler': g_scheduler.state_dict(), 'd_scheduler': d_scheduler.state_dict(),
                         'g_train_log': g_train_log, 'd_train_log': d_train_log,
                         'epoch': epoch, 'best_nll': best_nll}, save_dir)
        imshow(torchvision.utils.make_grid(x_fake[:100].detach().cpu(), nrow=10))
        filename = './Hw4/figures/epoch' + str(epoch) + '_LL.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    elif epoch % 10 == 0:
        imshow(torchvision.utils.make_grid(x_fake[:100].detach().cpu(), nrow=10))
        filename = './Hw4/figures/epoch' + str(epoch) + '.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        imshow(torchvision.utils.make_grid(x_fake[:100].detach().cpu(), nrow=10))
        plt.savefig('./Hw4/figures/latest.pdf', bbox_inches='tight')
        plt.close()

    if epoch == 30:  # The loss curve needs to stabilize, we give it 30 epochs
        best_nll = np.inf

    print('[Epoch %d/%d]\tTrain Loss: %s' % (epoch + 1, n_epochs, g_train_log[-1]))

# Plot the loss graph
train_x_vals = np.arange(len(g_train_log))
train_y_vals = g_train_log

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(train_x_vals, train_y_vals, label='Training Error')
ax[0].legend(loc='best')
ax[0].set_title('Training Curve')
ax[0].set_xlabel('Num Steps')
ax[0].set_ylabel('NLL in bits per dim')

# Latent visualization
load_checkpoint('./checkpoints/best.pth.tar', g, d)

imshow(torchvision.utils.make_grid(x_fake[:100].detach().cpu(), nrow=10))
