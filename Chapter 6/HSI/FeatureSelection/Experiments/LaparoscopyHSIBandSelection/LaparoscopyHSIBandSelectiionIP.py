import numpy as np
from .LaparoscopyHSIBandSelectionConcrete import LaparoscopyHSIBandSelectionConcrete

class LaparoscopyHSIBandSelectiionIP(LaparoscopyHSIBandSelectionConcrete):
    def __init__(self) -> None:
        '''
            Experiment for the LaparoscopyHSI dataset using the Concrete approach for feature selection.

            Parameters:
            -----------
            fs_tau: float
                Temperature ($\tau$) for the feature selection layer.

            reg_factor: float
                Regularization factor ($\lambda$) for the feature selection layer.
        '''
        super().__init__(fs_tau=0.3, reg_factor=2, ipdl=True)
        self.experiment = 'Concrete_IP'
        self.ipdl_dataset = self._ipdl_dataset()
        self.ip = None

        from IPDL.optim import SilvermanOptimizer
        import torch
        x = torch.randn(128, self.model.feature_selector.in_features)
        with torch.no_grad():
            self.model.eval()
            self.model(x)
        optim =  SilvermanOptimizer(self.model, gamma=2e-1, normalize_dim=True)
        optim.step()

    def _ipdl_dataset(self):
        '''
            Obtain a subset which is used for Information Plane estimation. 
        '''
        from torch.utils.data import Subset
        _, y = self.test_dataset[:]
        idx = np.concatenate([np.where(y==i)[0][:32] for i in (np.unique(y))]).tolist()
        ipdl_test_dataset = Subset(self.test_dataset, idx)
        return ipdl_test_dataset


    def config(self) -> dict:
        from IPDL.functional import matrix_estimator
        base_config = super().config()

        val_inputs, val_targets = self.ipdl_dataset[:]          
        _, Ax = matrix_estimator(val_inputs, sigma=1.5)
        Ky, Ay = matrix_estimator(val_targets.unsqueeze(1), sigma=.1)
        
        config = {
            'ip_estimation': True,
            'ipdl_dataset': self.ipdl_dataset,
            'Ax': Ax,
            'Ay': Ay,
            'Ky': Ky,
        }

        return {**base_config, **config}
    
    def run(self):
        from ..utils import train
        import os
        import torch

        if self.config()['save_log']:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.config()['save_result_dir'], 'logs')
            self.tb_writer = SummaryWriter(log_dir)
        
        self.model, self.ip = train(**self.config())
        
        if self.config()['save_log']:
            self.tb_writer.close()

        # save model
        if self.config()['save_result']:
            torch.save(self.model.state_dict(), os.path.join(self.config()['save_model_dir'], self.config()['save_model_name']))