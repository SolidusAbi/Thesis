import os
from ..Model import FeatureSelectionMethod
from .LaparoscopyHSIBandSelectionBase import LaparoscopyHSIBandSelectionBase

class LaparoscopyHSIBandSelectionGaussian(LaparoscopyHSIBandSelectionBase):
    def __init__(self, sigma, reg_factor, **kwargs) -> None:
        '''
            Experiment for the LaparoscopyHSI dataset using the Concrete approach for feature selection.

            Parameters:
            -----------
            fs_threshold: float
                Threshold for the feature selection layer.

            fs_tau: float
                Temperature for the feature selection layer.
        '''
        super().__init__(FeatureSelectionMethod.Gaussian, reg_factor=reg_factor, sigma=sigma, **kwargs)
        self.experiment = 'Gaussian_sg{}_reg{}'.format(sigma, reg_factor)

    def config(self) -> dict:
        base_config = super().config()
        if self.experiment is None:
            return base_config

        base_config['save_result_dir'] = os.path.join(base_config['save_result_dir'], self.experiment)
        if not os.path.exists(base_config['save_result_dir']):
            os.makedirs(base_config['save_result_dir'])
            os.makedirs(os.path.join(base_config['save_result_dir'], 'model'))
            os.makedirs(os.path.join(base_config['save_result_dir'], 'imgs'))
            os.makedirs(os.path.join(base_config['save_result_dir'], 'result'))

        config = {
            'save_model_dir': os.path.join(base_config['save_result_dir'], 'model'),
            'save_model_name': 'gaussian.pt',
        }

        return {**base_config, **config}