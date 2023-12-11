import os
from ..Model import FeatureSelectionMethod
from .LaparoscopyHSIBandSelectionBase import LaparoscopyHSIBandSelectionBase

class LaparoscopyHSIBandSelectionConcrete(LaparoscopyHSIBandSelectionBase):
    def __init__(self, fs_tau:float, reg_factor:float, **kwargs) -> None:
        '''
            Experiment for the LaparoscopyHSI dataset using the Concrete approach for feature selection.

            Parameters:
            -----------
            fs_tau: float
                Temperature ($\tau$) for the feature selection layer.

            reg_factor: float
                Regularization factor ($\lambda$) for the feature selection layer.
        '''
        super().__init__(FeatureSelectionMethod.Concrete, reg_factor=reg_factor, fs_threshold=0.9, fs_tau=fs_tau, **kwargs)
        self.experiment = 'Concrete_tau{}_reg{}'.format(fs_tau, reg_factor)

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
            'save_model_name': 'concrete.pt',
        }

        return {**base_config, **config}