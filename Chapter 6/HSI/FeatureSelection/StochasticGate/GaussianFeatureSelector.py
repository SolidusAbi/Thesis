from .FeatureSelector import FeatureSelectorSG
from torch.nn import Parameter

import torch.nn.functional as F
import numpy as np
import torch

class GaussianFeatureSelector(FeatureSelectorSG):
    def __init__(self, in_features: int, sigma:float=.5):
        super(GaussianFeatureSelector, self).__init__(in_features)
        self.sigma = sigma
        self.mu = Parameter(torch.ones(in_features)*.5) 

    def forward(self, x):
        eps = torch.normal(0, torch.ones_like(self.mu))
        drop_prob = self.mu + (self.sigma * eps * self.training) 
        z = 1 - drop_prob
        gate = F.hardtanh(z, 0, 1)
        return gate * x

    def _guassian_cdf(self, mu:torch.Tensor, sigma:float) -> torch.Tensor:
        r''' 
            Guassian CDF
            
            Based on: https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution

            Parameters
            ----------
            mu: torch.Tensor, shape (in_features,)
                The mean of the Guassian
            
            sigma: float
                The standard deviation of the Guassian
        '''
        return .5 * (1 + torch.erf(mu / (sigma*np.sqrt(2))))

    def regularize(self):
        r'''
            The expected regularization is the sum of the probabilities 
            that the gates are active
        '''
        # return torch.mean(self._guassian_cdf(self.mu, self.sigma))
        return torch.mean(self._guassian_cdf(1-self.mu, self.sigma))

    def variational_parameter(self, logit=True):
        return self.mu if logit else F.hardtanh(self.mu, 0, 1)
    
    def __repr__(self):
        return f'GaussianFeatureSelector(in_features={self.in_features}, sigma={self.sigma:.2f})'