import os
import torch
import numpy as np

from torch.utils.data import Dataset

class FineWeb(Dataset):
    def __init__(self, B, T, split):
        self.B = B                              # Batch size
        self.T = T                              # Sequence length
        self.split = split                      # split name
        assert split in {'train', 'test'}

        print(f"Loading {split} dataset... Please wait...")

        # Load the dataset from a binary file, then convert it to a PyTorch tensor
        self.data = np.memmap(os.path.join(os.path.dirname(__file__),f"{self.split}.bin"), dtype=np.uint32, mode='r')
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        # Calculate the number of samples in the dataset
        return len(self.data) - (self.B * self.T) - 1

    def __getitem__(self, idx):
        # Get the data for the current index
        buf = self.data[idx : idx+self.B*self.T+1]

        # Reshape the data into x-source, y-target pairs
        x = buf[:-1].reshape(self.B, self.T)
        y = buf[1:].reshape(self.B, self.T)

        return x, y