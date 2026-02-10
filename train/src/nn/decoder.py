from torch import nn, Tensor

from src.nn.linear_resnet import LinearResNet

class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.__layers = nn.Sequential(
            nn.Linear(latent_dim, 64, bias=False),
            LinearResNet(),
            LinearResNet(),
            LinearResNet(),
            LinearResNet()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__layers(x)