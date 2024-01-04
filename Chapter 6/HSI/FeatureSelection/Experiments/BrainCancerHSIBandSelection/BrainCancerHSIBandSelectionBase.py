import os
import config
import torch

from ..ExperimentBase import ExperimentBase
from ..Model import Model, FeatureSelectionMethod


class BrainCancerHSIBandSelectionBase(ExperimentBase):
    def __init__(self, fs_method:FeatureSelectionMethod, **kwargs) -> None:
        '''
            Experiment base for the BrainCancerHSI dataset.

            Parameters:
            -----------
            fs_method: FeatureSelectionMethod
                Method used for feature selection by Dropout. Either Concrete or Gaussian approach.

            kwargs: dict
                Additional keyword arguments to pass to the ExperimentBase class.
                Keyword arguments:
                    fs_threshold: for the Concrete approach.
                    fs_tau: for the Concrete approach.
                    sigma: for the Gaussian approach. 
        '''
        super().__init__()
        # self.model = Model([128, 64, 32, 3], fs_method, **kwargs) 
        self.model = Model([128, 100, 32, 3], fs_method, **kwargs) 
        self.experiment, self.dataset, self.train_dataset, self.test_dataset = None, None, None, None


    def config(self) -> dict:
        return {
            'dataset_dir': config.BRAIN_HSI_DIR,
            'model': self.model,
            'train_size': 0.8,
            'test_size': 0.2,
            'batch_size': self._batch_size(),
            'n_epochs': 200,
            'lr': 1e-3,
            'seed': 42,
            'save_result': True,
            'save_result_dir': os.path.join(config.RESULTS_DIR, 'Chapter6/BrainCancerHSIBandSelection/'),
            'save_log': True,
            'log_interval': 20,
        }
    
    def run(self) -> None:
        from ..utils import train

        if self.config()['save_log']:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.config()['save_result_dir'], 'logs')
            self.tb_writer = SummaryWriter(log_dir)
        
        self.model = train(**self.config())
        
        if self.config()['save_log']:
            self.tb_writer.close()

        # save model
        if self.config()['save_result']:
            torch.save(self.model.state_dict(), os.path.join(self.config()['save_model_dir'], self.config()['save_model_name']))
    
    def _batch_size(self) -> int:
        # Forcing the 100 iterations per epoch
        return (len(self.train_dataset) // 100) if self.train_dataset else 32
    
    def save_config(self) -> None:
        config = self.config()
        with open(os.path.join(config['save_result_dir'], 'config.txt'), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}: {v}\n')

    def save_results(self) -> None:
        import pandas as pd
        phi = self.model.feature_selector.variational_parameter(logit=False).detach()
        save_dir = os.path.join(self.config()['save_result_dir'], 'result')

        phi_df = pd.DataFrame(phi.numpy(), columns=[self.model.feature_selector.__repr__()])
        phi_df.to_csv(os.path.join(save_dir, 'phi.csv'))

        fs_df = pd.DataFrame(torch.where(phi<1)[0].numpy(), columns=['selected bands'])
        fs_df.to_csv(os.path.join(save_dir, 'selected_bands.csv'), index=False)

    