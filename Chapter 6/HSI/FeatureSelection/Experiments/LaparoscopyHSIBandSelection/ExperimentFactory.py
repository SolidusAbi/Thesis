from .LaparoscopyHSIBandSelectionConcrete import LaparoscopyHSIBandSelectionConcrete
from .LaparoscopyHSIBandSelectionGaussian import LaparoscopyHSIBandSelectionGaussian
from enum import Enum

class ExperimentType(Enum):
    '''
        Enum for indicating the feature selection methods.
    '''
    Concrete_5 = 0
    Concrete_2 = 1
    Concrete_3 = 2
    Gaussian_5 = 3

class ExperimentFactory(object):
    """Factory class for creating experiments."""

    def __init__(self):
        '''Initializes the experiment factory with the given experiment type.'''

    def create_experiment(self, experiment: ExperimentType, reg_factor=1):
        '''Creates an experiment with the given type.'''
        if experiment == ExperimentType.Concrete_5:
            return LaparoscopyHSIBandSelectionConcrete(reg_factor=reg_factor, fs_tau=0.5)
        elif experiment == ExperimentType.Concrete_2:
            return LaparoscopyHSIBandSelectionConcrete(reg_factor=reg_factor, fs_tau=0.2)
        elif experiment == ExperimentType.Concrete_3:
            return LaparoscopyHSIBandSelectionConcrete(reg_factor=reg_factor, fs_tau=0.3)
        elif experiment == ExperimentType.Gaussian_5:
            return LaparoscopyHSIBandSelectionGaussian(reg_factor=reg_factor, sigma=0.5)
        else:
            raise NotImplementedError(f'Experiment {experiment} not implemented yet.')
        