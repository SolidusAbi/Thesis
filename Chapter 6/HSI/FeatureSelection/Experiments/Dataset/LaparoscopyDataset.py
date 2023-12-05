import os, torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

def read_laparoscopy_dataset(path:str, selected_wavelength:list = None) -> (pd.DataFrame, pd.DataFrame):
    '''
        Read Laparoscopy dataset from path

        Parameters
        ----------
        path : str
            Path to the dataset.

        Returns
        -------
        X : pd.DataFrame
            Dataframe with the features.
        y : pd.DataFrame
            Dataframe with the labels.            
    '''
    csv_files = ['fat.csv', 'muscle.csv', 'nerve.csv', 'vessels.csv']
    df = list(map(lambda x: pd.read_csv(os.path.join(path, x))[selected_wavelength] if selected_wavelength is not None else pd.read_csv(os.path.join(path, x)), csv_files))
    X_df = pd.concat(df, axis=0)
    y_df = pd.concat([pd.Series(0, index=df[0].index, dtype=int), 
                      pd.Series(1, index=df[1].index, dtype=int),
                      pd.Series(2, index=df[2].index, dtype=int),
                      pd.Series(3, index=df[3].index, dtype=int)], axis=0)

    return X_df, y_df

class LaparoscopyDataset(Dataset):
    def __init__(self, root_dir:str, selected_wavelength: list = None, transform: nn.Module = None):
        super(LaparoscopyDataset, self).__init__()

        X_df, y_df = read_laparoscopy_dataset(root_dir, selected_wavelength)
                        
        self.X = torch.from_numpy(X_df.to_numpy()).float()
        self.y = torch.from_numpy(y_df.to_numpy()).long()
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(self.X[idx]), self.y[idx]
        
        return self.X[idx], self.y[idx]

    def n_features(self):
        return self.X.size(1)