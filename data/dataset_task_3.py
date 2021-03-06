import os
import numpy as np
import torch
from torch.utils import data

np.load.__defaults__ = (None, True, True, 'ASCII')


class Dataset_copy(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, path, label, transform=None):
        "Initialization"
        self.path = path
        self.list_data = self.load_file()
        self.label = label
        self.list_label = self.load_label()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_data)

    def load_file(self):
        # np_load_old = np.load
        # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        np_data = np.load(self.path)
        list_data = np_data.tolist()
        return list_data

    def load_label(self):
        # np_load_old = np.load
        # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        np_data = np.load(self.label)
        list_data = np_data.tolist()
        return list_data

    def __getitem__(self, index):
        "Generates one sample of data"
        data_idx = self.list_data[index]
        label_idx = self.list_label[index]
        x = torch.FloatTensor(data_idx)
        y = torch.FloatTensor(label_idx)

        return x, y