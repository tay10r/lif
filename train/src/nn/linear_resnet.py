import torch

from torch.nn import functional as F

class LinearResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(64, 64)
        self.l1 = torch.nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.l0(x))
        x = self.l1(x)
        return F.relu(x + residual)