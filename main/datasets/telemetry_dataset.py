import numpy as np
import torch
from torch.utils.data import Dataset


class TelemetryDataset(Dataset):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 seq_len: int = 1,
                 out_seq_len: int = 1
                 ):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len
        self.out_seq_len = out_seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len+self.out_seq_len)

    def __getitem__(self, index):
        x_from = index
        x_to = index + self.seq_len

        y_from = index + self.seq_len - self.out_seq_len
        y_to = index + self.seq_len

        return (self.X[x_from:x_to], self.y[y_from:y_to])
