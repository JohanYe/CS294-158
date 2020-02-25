import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from Hw2.Utils import *
from Hw2.model import *
sns.set_style("darkgrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k = 10
net = flow(k, 100).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
batch_size = 250
n_epochs = 5
train_log = []
val_log = {}
best_nll = np.inf
save_dir = './checkpoints/'

plt.figure(1)
x, y = sample_data()
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.savefig('Figure_1.pdf')
plt.close(1)
train_loader = torch.utils.data.DataLoader(torch.from_numpy(x[:int(len(x) * 0.8)]).float(), batch_size=batch_size, shuffle=True)
X_val = torch.from_numpy(x[int(len(x) * 0.8):]).float().to(device)

def calc_loss(x, pi, mu, var):
    """
    calculates by evaluation in weighted normal distribution
    :param pi: torch.Tensor network output
    :param mu: torch.Tensor network output
    :param var: torch.Tensor network output
    :return: loss
    """

    # Evaluation in normal dist
    weighted = pi * (torch.exp(- torch.pow(x.unsqueeze(2) - mu, 2) / (2 * var)) / torch.sqrt(2 * np.pi * var))
    density = torch.sum(weighted, dim=2)
    loss = -torch.mean(torch.log2(density))/2
    return loss





# Training loop
for epoch in range(n_epochs):
    for batch in train_loader:
        batch = batch.to(device)
        pi, mu, var = net(batch)
        loss = calc_loss(batch, pi, mu, var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1

    with torch.no_grad():
        pi, mu, var = net(X_val)
        loss = calc_loss(X_val, pi, mu, var)
        val_log[k] = loss.item()

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

    print('[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
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
ax[0].set_ylabel('Negative Log Likelihood')

# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
pdf = net.sampling()


ax[1].imshow(pdf.cpu().detach(),aspect= 'auto')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Best distribution on validation set")

plt.savefig('Figure_5.pdf',bbox_inches='tight')
plt.close()

axis = np.linspace(-2, 3.5, 100)
samples = np.array(np.meshgrid(axis, axis)).reshape([-1, 2])
samples = torch.from_numpy(samples).to(device).float()
with torch.no_grad():
    pi, mu, var = net(samples)

# calc loss
weighted = pi * (torch.exp(- torch.pow(samples.unsqueeze(2) - mu, 2) / (2 * var)) / torch.sqrt(2 * np.pi * var))
density = torch.sum(weighted, dim=2)
pdf = (2 ** torch.log2(density)).reshape(100,100)
ax[1].imshow(pdf.cpu().detach(),aspect= 'auto')