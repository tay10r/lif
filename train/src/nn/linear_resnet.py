import torch

from torch.nn import functional as F

class LinearResNet(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.l0 = torch.nn.Linear(channels, channels)
        self.l1 = torch.nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.l0(x))
        x = self.l1(x)
        return F.leaky_relu(x + residual)