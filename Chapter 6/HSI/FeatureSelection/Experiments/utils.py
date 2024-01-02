import math
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

def split_dataset(dataset, train_size, seed=None, **kwargs) -> (Subset, Subset):
    generator = torch.Generator().manual_seed(seed)

    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)

def cosine_scheduler(timesteps, s=8e-3):
    r'''
        Cosine scheduler for the regularization factor.

        Reference:
            Nichol, A. Q., & Dhariwal, P. (2021, July). Improved denoising
            diffusion probabilistic models. In International Conference on Machine 
            Learning (pp. 8162-8171). PMLR.
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x/timesteps)+s) / (1+s) * math.pi * .5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return 1 - alphas_cumprod

def _weights(dataset):
    '''
        Compute the weights for each sample in the dataset.
        It is used for the WeightedRandomSampler.

        Returns:
        --------
            weights: np.array
                The weights for each sample in the dataset.
    '''
    _, y = dataset[:]
    count = torch.bincount(y)
    weights = 1. / np.array(count)
    weights /= weights.sum()
    
    return weights[y]

def log(tb_writer, model, loss, reg_factor, epoch):
    tb_writer.add_scalar('Loss/test', loss, epoch)
    tb_writer.add_scalar('Sparse/Reg. Factor', reg_factor, epoch)
    tb_writer.add_scalar('Sparse/Sparse Rate', model.sparse_rate(), epoch)
    phi = model.feature_selector.variational_parameter()
    for i in range(phi.shape[0]):
        tb_writer.add_scalar(f'Phi/{i}', phi[i], epoch)

from IPDL import ClassificationInformationPlane
from IPDL.optim import AligmentOptimizer

def train(model, train_dataset, test_dataset, n_epochs=50, lr=1e-3, batch_size=32, weighted_sampler=False,
           reg_factor=1, tb_writer=None, log_interval=5, seed=42, ip_estimation=False, ipdl_dataset=None,
            Ax=None, Ky=None, Ay=None, **kwargs):
    r'''
        Training FS-DL model.

        Parameters:
        -----------
            model: nn.Module
                The model to be trained.
            train_dataset: torch.utils.data.Dataset
                The training dataset.
            test_dataset: torch.utils.data.Dataset
                The test dataset.
            n_epochs: int
                The number of epochs.
            lr: float
                The learning rate.
            batch_size: int
                The batch size.
            weighted_sampler: bool
                If True, use weighted sampler for the training dataset.
            reg_factor: float
                The regularization factor for FS method.
            tb_writer: torch.utils.tensorboard.SummaryWriter
                The tensorboard writer.
            log_interval: int
                The interval to log the results.
            seed: int
                The seed for reproducibility.
            kwargs: dict
                The additional arguments.
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    def seed_worker(worker_id):
        '''
            Seed the workers for reproducibility.

            Reference:
            ----------
                https://pytorch.org/docs/stable/notes/randomness.html
        '''
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        np.seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if weighted_sampler:
        sampler = WeightedRandomSampler(_weights(train_dataset), len(train_dataset), replacement=True)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*4, worker_init_fn=seed_worker, shuffle=False)

    epoch_iterator = tqdm(
            range(n_epochs),
            leave=True,
            unit="epoch",
            postfix={"tls": "%.4f" % 1, 'sparse_rate': "%.2f" % 0},
        )
      
  
    # IP Estimation
    if ip_estimation:
        matrix_optimizer = AligmentOptimizer(model, beta=0.9, n_sigmas=200)
        ip = ClassificationInformationPlane(model, use_softmax=True)
        with torch.no_grad():
            val_inputs, _= ipdl_dataset[:]
            model.eval()
            model(val_inputs.to(device))
            # matrix_optimizer.step(Ky.to(device)) # Commented because it is initialized in the exp.
            ip.computeMutualInformation(Ax.to(device), Ay.to(device))
            

    reg_factor_sch = cosine_scheduler(n_epochs, 1e-3) * reg_factor

    for epoch in epoch_iterator:
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = model(inputs)

            loss = criterion(pred, targets) + model.regularization(reg_factor_sch[epoch].item())
            
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred = model(inputs)
                loss = criterion(pred, targets)
                test_loss += loss.item()
            
            test_loss /= len(test_loader)
            epoch_iterator.set_postfix(reg="%.3f" % reg_factor_sch[epoch], tls="%.4f" % test_loss, sparse_rate="%.2f" % model.sparse_rate())

            if epoch % log_interval == 0 and tb_writer is not None:
                log(tb_writer, model, test_loss, reg_factor_sch[epoch], epoch) 

        # Information Plane
        if ip_estimation:
            with torch.no_grad():
                model.eval()
                model(val_inputs.to(device))
                matrix_optimizer.step(Ky.to(device))
                ip.computeMutualInformation(Ax.to(device), Ay.to(device))

    # Last epoch results if not logged
    if epoch % log_interval != 0 and tb_writer is not None:
        log(tb_writer, model, test_loss, reg_factor_sch[epoch], epoch) 

    return model.cpu() if not ip_estimation else (model.cpu(), ip)