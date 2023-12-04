from torch import nn

class FeatureSelectorSG(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        raise NotImplementedError

    def regularize(self):
        raise NotImplementedError

    def variational_parameter(self, logit=True):
        raise NotImplementedError