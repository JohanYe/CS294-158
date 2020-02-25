import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import math
import seaborn as sns
from Hw2.Utils import *
from Hw2.model import *

x, y = sample_data()
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.savefig('Figure_1.pdf')

k = 10
net = flow(k, 100)
