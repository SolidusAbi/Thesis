import numpy as np
import pandas as pd

def plot_boxplot(df: pd.DataFrame, labels = None, n_ticks =12, figsize=(16,8)):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df)
    ticks = np.linspace(0, len(df.columns)-1, n_ticks, dtype=int)

    ax.set_xticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels[ticks])

    ax.tick_params(axis='x', labelrotation=45, labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')

    ax.set_ylabel('Reflectance', fontsize='xx-large')
    ax.set_xlabel('Wavelength (nm)', fontsize='xx-large')
    return fig 

def outlier_removal(df:pd.DataFrame):
    '''
        Remove outliers using IQR.

        Parameters:
            df: pandas DataFrame
        Returns:
            df: pandas DataFrame without outliers
            idx: boolean array with the same length as the number of rows in df. True if the row is an outlier.
    '''

    # Outlier detection by IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    idx = ((df < lower_limit) | (df > upper_limit)).any(axis=1)    
    df = df[~idx]
    
    return df, idx.to_numpy()

def get_data(file):
    ''' 
        Read the HSI data and the ground truth map from a HDF5 file.

        Parameters
        ----------
        file : str
            Path to the HDF5 file.

        Returns
        -------
        data : 2-D array, shape (n_samples, n_features)
            Input array.

        gtMap : 1-D array, shape (n_samples,)
            Ground truth map.
    '''
    import h5py
    with h5py.File(file, 'r') as f:
        gtMap = np.array(f['gtMap'])
        data = np.array(f.get('preProcessedImage'))

        n_bands = data.shape[0]
        _data = data.copy()
        _data = _data.reshape(n_bands, -1).T
        _gtMap = gtMap.reshape(-1)
        # Remove the non-labeled pixels and those that corresponds to the ring
        idx = np.argwhere((_gtMap != 0) & (_gtMap != 4)).flatten()

        _data = _data[idx, :]
        _gtMap = _gtMap[idx].astype(int)

        return _data, _gtMap