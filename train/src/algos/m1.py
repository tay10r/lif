from src.algos.base import Algo

from torch import nn, Tensor, concat, randn
from torch.nn import functional as F

from src.nn.linear_resnet import LinearResNet

class _Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.__l0 = nn.Sequential(
            nn.Linear(64, 32),
            LinearResNet(32),
            nn.Linear(32, 16),
            LinearResNet(16),
            nn.Linear(16, latent_dim, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        r = self.__l0(x[:, :64])
        g = self.__l0(x[:, 64:128])
        b = self.__l0(x[:, 128:192])
        x = concat([r, g, b], dim=1)
        x = F.sigmoid(x)
        return x

class _Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            LinearResNet(64),
            nn.Linear(64, 128),
            LinearResNet(128),
            nn.Linear(128, 192, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class M1Algo(Algo):

    def __init__(self, block_size: int, channels: int, latent_dim: int):
        assert block_size == 8, "M1Algo only supports block_size of 8"
        assert channels == 3, "M1Algo only supports 3 input channels"
        self.__encoder = _Encoder(latent_dim=(latent_dim // channels))
        self.__decoder = _Decoder(latent_dim=latent_dim)

    def get_decoder(self) -> nn.Module:
        return self.__decoder

    def get_encoder(self) -> nn.Module:
        return self.__encoder