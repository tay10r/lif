from torch import nn, Tensor, ones

from src.nn.linear_resnet import LinearResNet

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.__layers = nn.Sequential(
            LinearResNet(),
            LinearResNet(),
            LinearResNet(),
            LinearResNet(),
            nn.Linear(64, latent_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.__layers(x)
        x = nn.functional.softsign(x)
        x = x * 0.5 + 0.5
        return x