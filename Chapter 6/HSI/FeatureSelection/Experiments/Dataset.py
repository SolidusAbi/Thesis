from torch.utils.data import Dataset

import os
import torch
import pandas as pd

class BrainCancerHSIDataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        self.X = torch.tensor( pd.read_csv( os.path.join(dir, 'X.csv')).to_numpy(), dtype=torch.float32 )
        self.y = torch.tensor( pd.read_csv( os.path.join(dir, 'y.csv')).to_numpy().ravel(), dtype=torch.long ) - 1 
            
        self.transform = transform
    
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.X[idx]), self.y[idx]
        
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)