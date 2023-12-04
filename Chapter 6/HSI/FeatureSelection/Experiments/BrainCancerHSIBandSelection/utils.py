import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

def split_dataset(dataset, config) -> (Subset, Subset):
    generator = torch.Generator().manual_seed(config['seed'])

    train_size = int(config['train_size'] * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)