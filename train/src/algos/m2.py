from src.algos.base import Algo

from torch import nn, Tensor

class _Encoder(nn.Module):
    def __init__(self, channels: int, block_size: int, latent_dim: int):
        super().__init__()
        self.__channels = channels
        self.__block_size = block_size
        self.__layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=0)
        )
        self.__linear = nn.Linear(64, latent_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], self.__channels, self.__block_size, self.__block_size)
        x = self.__layers(x)
        x = x.flatten(start_dim=1)
        x = self.__linear(x)
        return x

class _Decoder(nn.Module):
    def __init__(self, channels: int, block_size: int, latent_dim: int):
        super().__init__()
        self.__channels = channels
        self.__block_size = block_size
        self.__latent_dim = latent_dim
        self.__linear = nn.Linear(latent_dim, 16, bias=False)
        self.__layers = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.__linear(x)
        x = x.reshape(x.shape[0], 1, 4, 4)
        x = self.__layers(x)
        x = x.reshape(x.shape[0], self.__channels * self.__block_size * self.__block_size)
        return x

class M2Algo(Algo):

    def __init__(self, block_size: int, channels: int, latent_dim: int):
        self.__encoder = _Encoder(channels=channels, block_size=block_size, latent_dim=latent_dim)
        self.__decoder = _Decoder(channels=channels, block_size=block_size, latent_dim=latent_dim)

    def get_decoder(self) -> nn.Module:
        return self.__decoder

    def get_encoder(self) -> nn.Module:
        return self.__encoder