import torch
import pickle as pkl
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np


class PrintLayerShape(nn.Module):
    def __init__(self):
        super(PrintLayerShape, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print('Layer shape:', x.shape)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def save_checkpoint(state, save_dir, ckpt_name='best.pth.tar'):
    file_path = os.path.join(save_dir, ckpt_name)
    if not os.path.exists(save_dir):
        print("Save directory dosen't exist! Makind directory {}".format(save_dir))
        os.mkdir(save_dir)

    torch.save(state, file_path)

def load_checkpoint(checkpoint, g_model, d_model):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    saved_dict = checkpoint['d_state_dict']
    d_new_dict = d_model.state_dict()
    d_new_dict.update(saved_dict)
    d_model.load_state_dict(d_new_dict)

    saved_dict = checkpoint['g_state_dict']
    g_new_dict = g_model.state_dict()
    g_new_dict.update(saved_dict)
    g_model.load_state_dict(g_new_dict)