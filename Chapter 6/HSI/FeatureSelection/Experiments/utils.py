import torch

class Normalize(torch.nn.Module):
    def __init__(self, mean, std) -> None:
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        std_inv = 1 / (self.std + 1e-16)
        mean_inv = -self.means * std_inv

        return (x - mean_inv) / std_inv