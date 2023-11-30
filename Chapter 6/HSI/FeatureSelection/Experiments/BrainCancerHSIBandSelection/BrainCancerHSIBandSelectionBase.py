import os
import config

from ..ExperimentBase import ExperimentBase
from ..Model import Model

class BrainCancerHSIBandSelectionBase(ExperimentBase):
    def __init__(self) -> None:
        super().__init__()
        self.model = Model([128, 150, 64, 3], fs_threshold=.9, fs_tau=.2) 

    def config(self):
        return {
            'dataset_dir': config.BRAIN_HSI_DIR,
            'dataset': 'BrainCancerHSI',
            'model': self.model,
            'train_size': 0.8,
            'test_size': 0.2,
            'batch_size': 32,
            # 'batch_size': 64,
            # 'batch_size': 128,
            'epochs': 200,
            'lr': 1e-3,
            'seed': 42,
            'log_interval': 10,
            'save_model': False,
            'save_model_dir': 'Test', #config.MODEL_DIR,
            'save_model_name': 'BrainCancerHSIBandSelection.pt',
            'save_result': True,
            'save_result_dir': os.path.join(config.RESULTS_DIR, 'Chapter6/BrainCancerHSI'),
            'save_result_name': 'BrainCancerHSIBandSelection.pkl'
        }