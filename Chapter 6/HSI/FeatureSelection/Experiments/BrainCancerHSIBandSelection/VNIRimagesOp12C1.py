from .BrainCancerHSIBandSelectionBase import BrainCancerHSIBandSelectionBase
from ..Model import FeatureSelectionMethod
from ..Dataset import BrainCancerHSIDataset

from ..transforms import Normalize

import os
import config
import pandas as pd
from .utils import split_dataset


class VNIRimagesOp12C1(BrainCancerHSIBandSelectionBase):
    def __init__(self) -> None:
        super().__init__(fs_method=FeatureSelectionMethod.Gaussian, sigma=0.5)
        self.experiment = 'VNIRimagesOp12C1'
        self.data_dir = os.path.join(config.BRAIN_HSI_DIR, f'preprocessed/data/no_outliers/{self.experiment}/')
        self.tb_writer = None

        mean_std_df = pd.read_csv(os.path.join(self.data_dir, 'mean_std.csv'))
        self.transform = Normalize(mean_std_df['mean'], mean_std_df['std'])
        self.dataset = BrainCancerHSIDataset(self.data_dir, transform=self.transform)

        self.train_dataset, self.test_dataset = split_dataset(self.dataset, self.config())
        
    def config(self):
        base_config = super().config()
        base_config['save_result_dir'] = os.path.join(base_config['save_result_dir'], self.experiment)
        
        config = {
            'experiment_name': self.experiment,
            'data_dir': self.data_dir,
            'dataset': self.dataset,
            'train_dataset': self.train_dataset,
            'test_dataset': self.test_dataset,
            'reg_factor': 2,
            'weighted_sampler': True,
            'save_model_dir': os.path.join(base_config['save_result_dir'], 'model'),
            'save_model_name': '{}.pt'.format(self.experiment),
            'tb_writer': self.tb_writer,
        }

        return {**base_config, **config}
