import os
import config
import torch
import pandas as pd

from ..ExperimentBase import ExperimentBase
from ..Model import Model, FeatureSelectionMethod

from ..Dataset import LaparoscopyDataset
from ..transforms import Normalize
from ..utils import split_dataset

class LaparoscopyHSIBandSelectionBase(ExperimentBase):
    def __init__(self, fs_method:FeatureSelectionMethod, reg_factor, **kwargs) -> None:
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
        self.reg_factor = reg_factor
        self.experiment = None
        self.seed = 42
        self.train_size = 0.8
        self.model = Model([68, 100, 32, 4], fs_method, **kwargs) 
        self.tb_writer = None

        mean_std_df = pd.read_csv(os.path.join(config.LAPAROSCOPY_HSI_DIR,'mean_std.csv'), index_col=0)
        mean, std = mean_std_df['mean'].to_numpy(), mean_std_df['std'].to_numpy()
        self.transform = Normalize(mean, std)
        self.dataset = LaparoscopyDataset(os.path.join(config.LAPAROSCOPY_HSI_DIR, 'balanced/OCSP'), transform=self.transform)
        
        self.train_dataset, self.test_dataset = split_dataset(self.dataset, self.train_size, self.seed)
        

    def config(self) -> dict:
        return {
            'experiment_name': self.experiment,
            'dataset_dir': config.LAPAROSCOPY_HSI_DIR,
            'model': self.model,
            'dataset': self.dataset,
            'train_size': self.train_size,
            'test_size': round(1-self.train_size, 2),
            'train_dataset': self.train_dataset,
            'test_dataset': self.test_dataset,
            'batch_size': 512,
            'weighted_sampler': False,
            'n_epochs': 200,
            'lr': 1e-3,
            'reg_factor': self.reg_factor,
            'seed': self.seed,
            'save_result': True,
            'save_result_dir': os.path.join(config.RESULTS_DIR, 'Chapter6/LaparoscopyHSIBandSelection/'),
            'save_log': True,
            'log_interval': 20,
            'tb_writer': self.tb_writer,
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
    
    def save_config(self) -> None:
        config = self.config()
        with open(os.path.join(config['save_result_dir'], 'config.txt'), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}: {v}\n')

    def save_results(self) -> None:
        import pandas as pd
        from FeatureSelection.StochasticGate import ConcreteFeatureSelector
        phi = self.model.feature_selector.variational_parameter(logit=False).detach()
        save_dir = os.path.join(self.config()['save_result_dir'], 'result')

        phi_df = pd.DataFrame(phi.numpy(), columns=[self.model.feature_selector.__repr__()])
        phi_df.to_csv(os.path.join(save_dir, 'phi.csv'))

        if isinstance(self.model.feature_selector, ConcreteFeatureSelector):
            threshold = self.model.feature_selector.p_threshold
            fs_df = pd.DataFrame(torch.where(phi<threshold)[0].numpy(), columns=['selected bands'])
        else:
            fs_df = pd.DataFrame(torch.where(phi<1)[0].numpy(), columns=['selected bands'])
        
        fs_df.to_csv(os.path.join(save_dir, 'selected_bands.csv'), index=False)
    