import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
from Hw2.Exercise2.Utils import *
from Hw2.Exercise2.model import *


with open('./hw2_q2.pkl', 'rb') as f:
    data = pickle.load(f)

x_train = np.transpose(data['train'], [0,3,1,2])
x_test = np.transpose(data['test'],[0,3,1,2])
