from src.algos.base import Algo

from torch import nn, Tensor, concat, randn
from torch.nn import functional as F

from src.nn.linear_resnet import LinearResNet

class _Encoder(nn.Module):
    def __init__(self, channels: int, block_size: int, latent_dim: int):
        super().__init__()
        self.__channels = channels
        self.__block_size = block_size
        self.__layers = nn.Sequential(
            nn.Linear(block_size * block_size, 64),
            LinearResNet(64),
            LinearResNet(64),
            nn.Linear(64, latent_dim, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        channels: list[Tensor] = []
        b = self.__block_size * self.__block_size
        for c in range(self.__channels):
            channel = self.__layers(x[:, b*c:b*(c+1)])
            channels.append(channel)
        x = concat(channels, dim=1)
        return x

class _Decoder(nn.Module):
    def __init__(self, channels: int, block_size: int, latent_dim: int):
        super().__init__()
        self.__channels = channels
        self.__block_size = block_size
        self.__latent_dim = latent_dim
        self.__layers = nn.Sequential(
            nn.Linear(latent_dim // channels, 256),
            LinearResNet(256),
            LinearResNet(256),
            nn.Linear(256, block_size * block_size, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        channels: list[Tensor] = []
        l = self.__latent_dim // self.__channels
        for c in range(self.__channels):
            x0 = x[:, l*c:l*(c+1)]
            z = self.__layers(x0)
            channels.append(z)
        z = concat(channels, dim=1)
        return z

class M1Algo(Algo):

    def __init__(self, block_size: int, channels: int, latent_dim: int):
        self.__encoder = _Encoder(channels=channels, block_size=block_size, latent_dim=latent_dim)
        self.__decoder = _Decoder(channels=channels, block_size=block_size, latent_dim=latent_dim)

    def get_decoder(self) -> nn.Module:
        return self.__decoder

    def get_encoder(self) -> nn.Module:
        return self.__encoder