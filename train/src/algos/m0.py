from src.algos.base import Algo

import torch
from torch import nn
from torch.nn import functional as F

from src.nn.linear_resnet import LinearResNet

class Encoder(torch.nn.Module):
    def __init__(self, block_size: int, channels: int, latent_dim: int, hidden_dim: int, num_res_blocks: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(block_size * block_size * channels, hidden_dim),
            *[LinearResNet(hidden_dim) for _ in range(num_res_blocks)],
            torch.nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, block_size: int, channels: int, latent_dim: int, hidden_dim: int, num_res_blocks: int):
        super().__init__()
        self.__layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            *[LinearResNet(hidden_dim) for _ in range(num_res_blocks)],
            torch.nn.Linear(hidden_dim, block_size * block_size * channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__layers(x)
        return x

class M0Algo(Algo):
    def __init__(self, block_size: int, channels: int, latent_dim: int, hidden_dim: int, num_res_blocks: int):
        self.__encoder = Encoder(block_size=block_size, channels=channels, latent_dim=latent_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks)
        self.__decoder = Decoder(block_size=block_size, channels=channels, latent_dim=latent_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks)

    def get_decoder(self) -> nn.Module:
        return self.__decoder

    def get_encoder(self) -> nn.Module:
        return self.__encoder