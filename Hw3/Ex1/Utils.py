import numpy as np



def sample_data_1():
    count = 100000
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2():
    count = 100000
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


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