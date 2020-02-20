import torch
import matplotlib.pyplot as plt
import os


def one_hot(x, n_classes=200):
    return torch.zeros(x.shape[0], n_classes).scatter_(1, x.long().view(-1, 1), 1)


def one_hot_cat(x, n_classes=200):
    l = []
    for i in range(x.shape[1]):
        l.append(one_hot(x[:, i], n_classes))
    return torch.cat(l, dim=1)


def plot_tensor(x):
    plt.figure()
    plt.imshow(x.cpu().data.numpy())


def get_mean_NLL(prob, observation):
    batch_size = prob.shape[0]
    logits = -torch.log2(torch.gather(prob, 2, observation.view([batch_size, -1, 1])).view(batch_size, 2))
    mean_nll = torch.sum(logits) / batch_size
    return mean_nll


def save_checkpoint(state, save_dir, ckpt_name='best.pth.tar'):
    file_path = os.path.join(save_dir, ckpt_name)
    if not os.path.exists(save_dir):
        print("Save directory dosen't exist! Makind directory {}".format(save_dir))
        os.mkdir(save_dir)

    torch.save(state, file_path)


def load_checkpoint(checkpoint, model):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    saved_dict = checkpoint['state_dict']
    new_dict = model.state_dict()
    new_dict.update(saved_dict)
    model.load_state_dict(new_dict)