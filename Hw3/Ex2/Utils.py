import torch
import pickle as pkl
from torchvision import transforms


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, path='./Hw3/Data/hw3-q2.pkl'):
        self.X = pkl.load(open(path,'rb'))
        self.X = self.X[dataset]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]).to(self.device)
