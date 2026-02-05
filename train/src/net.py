import torch
from torch.nn import functional as F

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        q = torch.round(x * 255)
        q = torch.clamp(q, 0, 255)
        return q * (1.0 / 255.0)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

class _LinearResNet(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.l0 = torch.nn.Linear(channels, channels)
        self.l1 = torch.nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.l0(x))
        x = self.l1(x)
        return F.leaky_relu(x + residual)

class Encoder(torch.nn.Module):
    def __init__(self, block_size: int, channels: int, latent_dim: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(block_size * block_size * channels, 64),
            _LinearResNet(64),
            _LinearResNet(64),
            _LinearResNet(64),
            _LinearResNet(64),
            torch.nn.Linear(64, latent_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, block_size: int, channels: int, latent_dim: int):
        super().__init__()
        self.__layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            _LinearResNet(128),
            _LinearResNet(128),
            _LinearResNet(128),
            _LinearResNet(128),
            torch.nn.Linear(128, block_size * block_size * channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__layers(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, block_size: int, channels: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(block_size, channels, latent_dim)
        self.decoder = Decoder(block_size, channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = STEQuantize.apply(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    net = Net(block_size=8, channels=3, latent_dim=32)
    x = torch.randn(4, 3, 8, 8)
    x = torch.flatten(x, start_dim=1)
    y: torch.Tensor = net(x)
    print(y.shape)