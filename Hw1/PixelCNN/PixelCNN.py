# Needed more computational power so moved to colab at least temporarily
import torch
import numpy as np
import matplotlib.pyplot as plt   
import pickle
import seaborn as sns
import time
import torch.optim as optim
from Hw1.PixelCNN.model import *
from Hw1.PixelCNN.Utils import *
sns.set_style("darkgrid")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
n_epochs = 2
batch_size = 128
lr = 1e-3
train_log = []
val_log = {}
k = 0
best_nll = np.inf
save_dir = './checkpoints/'

# Loading data
pickle_file = './mnist-hw1.pkl'
#path = '/content/drive/My Drive/mnist-hw1.pkl'
mnist = pickle.load(open(pickle_file,'rb'))

X_train = mnist['train'].astype('float32')[:20000]
train_loader = torch.utils.data.DataLoader(torch.from_numpy(X_train), batch_size=batch_size, shuffle=True)

X_val = mnist['test'].astype('float32')[:1000]
val_loader = torch.utils.data.DataLoader(torch.from_numpy(X_val), batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots(4,4,figsize=(5,5))
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(X_val[4*i+j]/3)
        ax[i, j].axis('off')
plt.savefig('./Hw1/Figures/Figure_7.pdf',bbox='tight')
plt.close()

image_shape = next(iter(train_loader)).permute(0, 3, 1, 2).shape
net = PixelCNN(image_shape=image_shape).to(device)
optimizer = optim.Adam(net.parameters(),lr=lr)

def calc_loss(logits, batch):
    # Divide by 2 for NLL per bit
    #print(logits.shape,batch.shape)
    loss = F.cross_entropy(logits, batch.long(), reduction='sum') / batch_size / np.log(2.0) / (28 * 28 * 3)
    return loss


# Training loop
for epoch in range(n_epochs):
    for mini_batch in train_loader:
        # https://jdhao.github.io/2019/07/10/pytorch_view_reshape_transpose_permute/
        mini_batch = mini_batch.permute(0,3,1,2).to(device) #To save memory we send one batch to cuda at the time
        logits, dist = net(mini_batch)
        loss = calc_loss(logits,mini_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(loss.item())
        k += 1

        mini_batch = mini_batch.cpu() #need to free GPU mem

    val_batch_log = []
    for mini_batch in val_loader:
        with torch.no_grad():
            mini_batch = mini_batch.permute(0, 3, 1, 2).to(device)  # To save memory we send one batch to cuda at the time
            logits, dist = net(mini_batch)
            loss = calc_loss(logits, mini_batch)
            val_batch_log.append(loss.item())

    avg_val_loss = np.mean(val_batch_log)
    val_log[k] = avg_val_loss

    if avg_val_loss < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch,
                         'state_dict': net.state_dict()}, save_dir)

    now = time.strftime("%H:%M", time.localtime(time.time()))
    print(now, '[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch+1, n_epochs, k, np.mean(train_log[-10:]), val_log[k]))


# Plotting each minibatch step
x_val = list(val_log.keys())
y_val = list(val_log.values())
print(np.min(y_val))

# Plotting
plt.figure(2)
x_plot = np.arange(len(train_log))
plt.plot(x_plot, train_log, label='Training Error')
plt.plot(x_val, y_val, label='Validation Error')
plt.legend(loc='best')
plt.title('PixelCNN Training Curve')
plt.xlabel('Num Steps')
plt.ylabel('NLL in bits/dim')
plt.savefig('./Hw1/Figures/Figure_8.pdf', bbox='tight')


# Load best and generate
load_checkpoint('./checkpoints/best.pth.tar', net)
samples = np.zeros([100, 28, 28,3])
for i in range(100):
    with torch.no_grad():
        image = np.transpose(net.sample_once().squeeze(0).cpu().detach(), [1,2,0])
        samples[i] = image

for i in range(10):
    for j in range(10):
        idx = i * 10 + j + 1
        plt.subplot(10, 10, idx)
        plt.imshow(samples[idx - 1] / 3)
        plt.axis('off')
plt.title("Samples from Best Model")
plt.savefig('./Hw1/Figures/Figure_9.pdf', bbox='tight')
