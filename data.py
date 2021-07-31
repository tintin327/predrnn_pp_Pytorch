from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class MnistDataset(Dataset):
    def __init__(self,mnist):
        self.mnist = mnist
        self.len = len(self.mnist)
    def __getitem__(self, idx):
        mnist_input = (torch.from_numpy(self.mnist[idx])).float()
        return mnist_input/255
    
    def __len__(self):
        return self.len

def create_batch(samples):

    input_batch = [s[0] for s in samples]
    output_batch = [s[1] for s in samples]

    return input_batch, output_batch
