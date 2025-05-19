import os

import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

class FlipperDataset(Dataset):
    def __init__(self, data_dir):
        self.data = np.load(os.path.join(data_dir, "arrays.npz"))["arr_0"]
        self.labels = np.load(os.path.join(data_dir, "labels.npz"))["arr_0"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_array = self.data[index]
        data_label = self.labels[index]

        return torch.Tensor(data_array), torch.Tensor(data_label)


def get_loader(data_dir):
    flipper_dataset = FlipperDataset(data_dir)

    return DataLoader(flipper_dataset, batch_size=4, shuffle=True)
