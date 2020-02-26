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
n_epochs = 1
batch_size = 128
lr = 1e-3
train_log = []
val_log = {}
k = 0
best_nll = np.inf
save_dir = './checkpoints/'

# Loading data
pickle_file = 'C:/Users/Johan/Downloads/mnist-hw1.pkl'
#path = '/content/drive/My Drive/mnist-hw1.pkl'
mnist = pickle.load(open(pickle_file,'rb'))

X_train = mnist['train'][:20000].astype('float32')
train_loader = torch.utils.data.DataLoader(torch.from_numpy(X_train), batch_size=batch_size, shuffle=True)

X_val = mnist['test'][:500].astype('float32')
val_loader = torch.utils.data.DataLoader(torch.from_numpy(X_val), batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots(4,4,figsize=(5,5))
for i in range(4):
    for j in range(4):
        ax[i,j].imshow(X_val[4*i+j]/3)
        ax[i,j].axis('off')

image_shape = next(iter(train_loader)).permute(0, 3, 1, 2).shape
net = PixelCNN(image_shape=image_shape).to(device)
optimizer = optim.Adam(net.parameters(),lr=lr)

def calc_loss(logits, batch):
    # Divide by 2 for NLL per bit
    #print(logits.shape,batch.shape)
    loss = F.cross_entropy(logits, batch.long(), reduction='sum') / batch_size
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

    with torch.no_grad():
        logits, dist = net(torch.from_numpy(X_val).permute(0,3,1,2).to(device))
        loss = calc_loss(logits,torch.from_numpy(X_val).permute(0,3,1,2).to(device))
        val_log[k] = loss.item()

    if loss.item() < best_nll:
        best_nll = loss.item()
        save_checkpoint({'epoch': epoch,
                         'state_dict': net.state_dict()}, save_dir)

    print(time.time(), '[Epoch %d/%d][Step: %d] Train Loss: %s Test Loss: %s' \
          % (epoch+1, n_epochs, k, np.mean(train_log[-10:]), val_log[k]))

x = net.sample_once()
plt.imshow(np.transpose(x.cpu().numpy().squeeze(0), (0, 2, 3, 1)) / 3)