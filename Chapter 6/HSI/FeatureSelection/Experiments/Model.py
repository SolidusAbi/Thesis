import torch

from torch import nn
from FeatureSelection.StochasticGate import ConcreteFeatureSelector, GaussianFeatureSelector, FeatureSelectorSG

from collections import deque
from itertools import islice

def sliding_window_iter(iterable, size):
    '''
        Iterate through iterable using a sliding window of several elements.
        Important: It is a generator!.
        
        Creates an iterable where each element is a tuple of `size`
        consecutive elements from `iterable`, advancing by 1 element each
        time. For example:
        >>> list(sliding_window_iter([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4)]
        
        source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)


class FeatureSelectionMethod:
    '''
        Enum for indicating the feature selection methods.
    '''
    Concrete = 0
    Gaussian = 1

class Model(nn.Module):
    def __init__(self, dims:list, fs_method:FeatureSelectionMethod, fs_threshold=.9, fs_tau=.3, sigma=.5):
        ''' 
            DL model used for feature selection.

            Parameters:
            -----------
            dims: list
                List of dimensions for each layer. The first element is the input dimension
                and last elements corresponds to the output size.
            
            fs_method: FeatureSelectionMethod
                Method used for feature selection by Dropout. Either Concrete or Gaussian approach.
                
            fs_threshold: float
                Threshold for the feature selection layer. Only used for Concrete approach.
            
            fs_tau: float
                Temperature for the feature selection layer. Only used for Concrete approach.

            sigma: float
                Standard deviation for the Gaussian feature selection layer. Only used for Gaussian approach.
        '''
        super(Model, self).__init__()
        if fs_method == FeatureSelectionMethod.Concrete:
            self.feature_selector = ConcreteFeatureSelector(dims[0], p_threshold=fs_threshold, tau=fs_tau)
        elif fs_method == FeatureSelectionMethod.Gaussian:
            self.feature_selector = GaussianFeatureSelector(dims[0], sigma=sigma)
        else: 
            raise ValueError(f'Invalid feature selection method: {fs_method}')


        layers = []
        for idx, (in_features, out_features) in enumerate(sliding_window_iter(dims, 2)):
            last = (True if idx == ((len(dims) + 1) // 2) else False)
            layers.append(self._hidden_layer(in_features, out_features, last))

        self.model = (nn.Sequential(*layers))

    def _hidden_layer(self, in_features, out_features, output=False):
        if output:
            return nn.Linear(in_features, out_features)
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features, affine=False),
                nn.ReLU(inplace=True),
                nn.Dropout(.5),
            )

    def forward(self, x):
        return self.model(self.feature_selector(x))

    def regularization(self, reg_factor = 1e-3):
        reg = 0.
        for module in self.modules():
            if isinstance(module, FeatureSelectorSG):
                reg = reg  + (reg_factor * module.regularize())
        return reg

    def sparse_rate(self):
        r'''
            The sparseness ratio which identifies the number of deactive gates
        '''
        if isinstance(self.feature_selector, ConcreteFeatureSelector):
            return torch.sum(self.feature_selector.logit_p >= self.feature_selector.logit_threshold) / self.feature_selector.in_features
        elif isinstance(self.feature_selector, GaussianFeatureSelector):
            return torch.sum(self.feature_selector.variational_parameter() <= 0) / self.feature_selector.in_features
        
        return 0.