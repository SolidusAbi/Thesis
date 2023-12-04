from .FeatureSelector import FeatureSelectorSG
from torch.nn import Parameter

import numpy as np
import torch

class ConcreteFeatureSelector(FeatureSelectorSG):
    def __init__(self, in_features: int, p_threshold:float = 0.9, tau=.5) -> None:
        super(ConcreteFeatureSelector, self).__init__(in_features)
        self.p_threshold = p_threshold
        self.logit_threshold = np.log(p_threshold) - np.log(1. - p_threshold)

        # dropout rate $\phi_i$
        self.logit_p = Parameter(torch.ones(in_features)*.5)
        
        # temperature $\tau$
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.concrete_bernoulli(x)

        return x * (self.logit_p < self.logit_threshold).float()

    def concrete_bernoulli(self, x):
        eps = 1e-8
        unif_noise = torch.cuda.FloatTensor(*x.size()).uniform_() if self.logit_p.is_cuda else torch.FloatTensor(*x.size()).uniform_()

        p = torch.sigmoid(self.logit_p)

        drop_prob = (torch.log(p + eps) - torch.log((1-p) + eps) + torch.log(unif_noise + eps)
        - torch.log((1. - unif_noise) + eps))
        drop_prob = torch.sigmoid(drop_prob / self.tau)

        random_tensor = 1 - drop_prob
        return x * random_tensor

    def regularize(self):
        p = torch.sigmoid( self.logit_p )
        return torch.mean(1-p)
    

    def variational_parameter(self, logit=True):
        return self.logit_p if logit else torch.sigmoid(self.logit_p)
    
    def __repr__(self):
        p_threshold = torch.sigmoid(torch.tensor(self.logit_threshold))
        return f'ConcreteFeatureSelector(in_features={self.in_features}, p_threshold={p_threshold:.2f}, tau={self.tau})'
