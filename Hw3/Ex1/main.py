import numpy as np
import torch
import matplotlib.pyplot as plt
from Hw3.Ex1.Utils import *
from Hw3.Ex1.Utils import *
import seaborn as sns
import torch.optim as optim
sns.set_style("darkgrid")


ds1 = True
ds2 = False
k = 0
batch_size = 125
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NETWORK
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
    train_loader = torch.utils.data.Dataloader(torch.from_numpy(x1), batch_size=batch_size, shuffle=True)
else:
    train_loader = torch.utils.data.Dataloader(torch.from_numpy(x2), batch_size=batch_size, shuffle=True)




for epoch in range(n_epochs):
    for batch in train_loader:


