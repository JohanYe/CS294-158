import torch


def CheckerBoardMask(x):
    _, c, h, w = x.shape
    mask_A = torch.ones([1, c, h, w]).int()
    mask_B = torch.ones([1, c, h, w]).int()
    odds = [i for i in range(h) if not i % 2 == 0]
    even = [i for i in range(w) if i % 2 == 0]
    mask_A[:, :, odds, :] = 0
    mask_B[:, :, :, even] = 0
    mask = mask_A ^ mask_B
    return mask