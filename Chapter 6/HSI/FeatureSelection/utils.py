import math, torch
def cosine_scheduler(timesteps, s=8e-3):
    r'''
        Cosine scheduler for the regularization factor.
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x/timesteps)+s) / (1+s) * math.pi * .5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return 1 - alphas_cumprod
