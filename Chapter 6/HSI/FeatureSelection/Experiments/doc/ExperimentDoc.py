from ..ExperimentBase import ExperimentBase
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from torch.nn.functional import hardtanh, sigmoid

def _check_fs(model):
    from FeatureSelection.StochasticGate import GaussianFeatureSelector, ConcreteFeatureSelector
    return isinstance(model.feature_selector, (ConcreteFeatureSelector, GaussianFeatureSelector))

def _get_random_samples(exp: ExperimentBase):
    '''
        Returns a list of random samples from each class.
    '''

    import torch
    dataset = exp.dataset
    X, y = dataset[:]
    n_classes = len(np.unique(y))
    transform = dataset.transform if hasattr(dataset, 'transform') else None

    selected_samples = np.zeros((n_classes, X.shape[1]))

    for i in np.unique(y):
        idx = np.where(y == i)[0][0]
        selected_samples[i] = X[idx].numpy()

    if transform is not None and hasattr(transform, 'inverse_transform'):
        selected_samples = transform.inverse_transform(torch.tensor(selected_samples))
    
    return selected_samples

class ExperimentDoc:
    '''
        This class is used to generate the experiments images.
    '''
    @staticmethod
    def hist_phi(exp: ExperimentBase) -> Figure:
        ''' 
            Generates the histogram of the $\phi_i$ (dropout rate) values for the
            Deep Learning Feature Selection-based method.
        '''        
        model = exp.model
        if not _check_fs(model):
            raise ValueError('Unknown feature selector')
        
        phi = model.feature_selector.variational_parameter(logit=False).detach()

        fig = plt.figure(figsize=(5, 3))
        with plt.style.context('seaborn-colorblind'):
            plt.hist(phi, bins=10)
            plt.yticks(np.arange(0, model.feature_selector.in_features, 20))
            plt.grid(axis='y')
            plt.xlabel('Dropout-rate ($\phi_i$)')

        return fig
    
    @staticmethod
    def plot_band_selection(exp: ExperimentBase, samples=None, labels=None, wv=None) -> Figure:
        import torch 

        # Generate samples and/or labels if not provided
        samples = _get_random_samples(exp) if samples is None else samples

        phi = exp.model.feature_selector.variational_parameter(logit=False).detach()
        activated_gates = torch.where(phi < 1)[0]
        n_features = exp.model.feature_selector.in_features
    
        from .utils import plot_band_selection
        return plot_band_selection(activated_gates, samples, sample_labels=labels, n_features=n_features, wavelength=wv)
