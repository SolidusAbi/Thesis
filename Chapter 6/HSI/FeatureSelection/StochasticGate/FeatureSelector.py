from torch import nn
from IPDL import MatrixEstimator

class FeatureSelectorSG(nn.Module):
    def __init__(self, in_features:int, ipdl=False):
        super().__init__()
        self.in_features = in_features
        self.entropy_estimator = ipdl
        self.matrix_estimator = MatrixEstimator(.1) if self.entropy_estimator else None

    def forward(self, x):
        raise NotImplementedError

    def regularize(self):
        raise NotImplementedError

    def variational_parameter(self, logit=True):
        raise NotImplementedError
    