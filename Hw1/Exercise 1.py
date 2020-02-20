# Homework 1.1 Deep unsupervised learning
# Spring 2020
# Johan Ye
# CS294-158-SP19 Spring 2019

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils.data
from Hw1 import DataGenerator
from Hw1 import model
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("talk")

############ EXERCISE 1 ############
x = DataGenerator.sample_data()
plt.figure(1)
plt.hist(x, 100, density=True)
plt.show()

# Train validation split
x_train, x_val = train_test_split(x, test_size=0.2)

### The 100 thetas
theta = torch.zeros(100, requires_grad=True)
optimizer = optim.Adam([theta], lr=1e-3)

# Hyperparameters:
num_epochs = 500
batch_size = 200
k = 0

# Variable declaration
train_losses, val_losses, train_accs, val_accs = {}, {}, {}, {}

# Data loaders
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=False)

train_losses, val_losses, best_NLL = model.train(num_epochs, train_loader, val_loader, theta, optimizer, train_losses, val_losses,
                                       k)

# Plotting each minibatch step
x_tst = list(val_losses.keys())
y_tst = list(val_losses.values())

# Plot the graphs
plt.figure(2)
plt.xlabel('Num Steps')
plt.ylabel('Negative Log Likelihood')
train_x_vals = sorted(train_losses.keys())
train_y_vals = [train_losses[k] for k in train_x_vals]
plt.ylim(5.5, 7)

plt.plot(train_x_vals, train_y_vals, label='Train')
plt.plot(x_tst, y_tst, label='Validation')
plt.title('Training Curve')
plt.legend(loc='best')
plt.savefig('Figure_2.pdf',bbox_inches='tight')
plt.show()

# Compute model probabilites
theta_prob = torch.exp(theta) / torch.sum(torch.exp(theta))
theta_prob = theta_prob.detach().numpy()

plt.figure(3)
plt.bar(np.arange(100), theta_prob)
plt.title("Model Probabilities")
plt.xlabel("x")
plt.ylabel("P_theta(x)")
plt.savefig('Figure_3.pdf',bbox_inches='tight')
plt.show()

