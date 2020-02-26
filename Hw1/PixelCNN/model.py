import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import os


class MaskedConv2d(nn.Conv2d):
    """ Masked Conv2d spatial mask only."""

    # args and kwargs take undefined input args
    def __init__(self, mask_type='A', *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        mask_type = mask_type.upper()
        assert mask_type in {'A', 'B'}

        mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.shape

        # Spatial masking
        mask[:, :, height // 2, width // 2 + (mask_type == 'B'):] = 0  # masking along dim 0
        mask[:, :, height // 2 + 1:] = 0  # masking along dim 1
        # RGB masking (in current pixel only)
        mask[:, : height // 2, width // 2] = mask_channels(mask_type, in_channels, out_channels, data_channels)
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data = self.weight.data.mul(self.mask)  # inplace operation is faster
        return super(MaskedConv2d, self).forward(x)


class ResNetBlock(nn.Module):
    """According to PixelCNN paper article 1 figure 5 and table 1
    https://arxiv.org/pdf/1601.06759.pdf
    """

    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()

        self.h = in_channels // 2
        self.out = nn.Sequential(
            nn.ReLU(),  # Relu first, strange setup
            nn.Conv2d(in_channels, self.h, kernel_size=1),
            nn.ReLU(),
            # padding to conserve shape.
            MaskedConv2d(in_channels=self.h, out_channels=self.h, kernel_size=3, padding=1, mask_type='B'),
            nn.ReLU(),
            nn.Conv2d(self.h, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.out(x) + x


def mask_channels(mask_type, in_channels, out_channels, data_channels=3):
    """
    Autoregressive channel masking, B being the lenient mask
    :param mask_type: "A" or "B" according to pixelcnn paper
    :param in_channels: input channels
    :param out_channels: output channels
    :param data_channels: channels in original data
    :return:
    """
    mask_type = mask_type.upper()
    assert mask_type in {'A', 'B'}

    in_factor = in_channels // data_channels + 1
    out_factor = out_channels // data_channels + 1

    mask = torch.ones([data_channels, data_channels])

    # Mask A masked diagonal, while B doesn't - the diagonal is the ability to see itself.
    # if b 0   0
    #  1  if b 0
    #  1   1  if b
    if mask_type == 'A':
        mask = mask.tril(-1)
    else:
        mask = mask.tril(0)

    # Creating mask pattern in order to match weight matrix
    mask = torch.cat([mask] * in_factor, dim=1)  # duplication along dim 1
    mask = torch.cat([mask] * out_factor, dim=0)  # duplication along dim 0

    mask = mask[0:out_channels, 0:in_channels] # This is due to rounding from in_factor / out_factor

    return mask


class PixelCNN(nn.Module):

    def __init__(self, image_shape, n_layers=12, n_filters=128, n_classes=4, final_channels=3):
        # 4 channels due to two bits per dim.
        super(PixelCNN, self).__init__()

        self.n_classes = n_classes
        _, self.final_channels, self.height, self.width = image_shape[0], image_shape[1], image_shape[2], image_shape[3]

        # padding=3 so it can start in [0,0], otherwise it has to start inside the image.
        layers = [MaskedConv2d(in_channels=3, out_channels=n_filters, kernel_size=7, padding=3, mask_type='A')]

        for i in range(n_layers):
            layers.append(ResNetBlock(n_filters))

        # PixelCNN paper uses 1024 for colored data
        layers += [nn.ReLU()] + \
                  [MaskedConv2d(in_channels=n_filters, out_channels=1024, kernel_size=1, mask_type='B')] + \
                  [nn.ReLU()] + \
                  [MaskedConv2d(in_channels=1024, out_channels=final_channels * n_classes, kernel_size=1,
                                mask_type='B')]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        logits = x.view([x.shape[0], self.n_classes, self.final_channels, self.height, self.width])
        out = F.softmax(logits, dim=1)  # [N, n_classes, C, H, W]
        return logits, out

    def sample_once(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.zeros([1, self.final_channels, self.height, self.width]).float().to(device)

        for i in range(28):
            for j in range(28):
                for k in range(3):
                    _, dist = self(x)
                    x[0, k, i, j] = np.random.choice(4, p=dist[0, :, k, i, j].cpu().data.numpy())

        return x
