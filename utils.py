import numpy as np

def moving_average(a:np.ndarray, n=3, padding_size=2):
    ''' 
        Moving average filter for 1-D array.

        Parameters
        ----------
        a : 1-D array, shape (batch_size, n_features)
            Input array.
        n : int, optional
            Window size. Default is 3.
        padding_size : int, optional
            Padding size. Default is 2.
    
    '''
    if padding_size > 0:
        a = np.pad(a, ((0,0), (padding_size//2, padding_size//2)), 'edge')

    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n