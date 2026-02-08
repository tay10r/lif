from abc import ABC, abstractmethod

from torch import Tensor
from torch import nn

from src.nn.quantize import STEQuantizePerVectorAbsMax

class Algo(ABC):

    @abstractmethod
    def get_decoder(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        raise NotImplementedError()

def forward(algo: Algo, x: Tensor) -> Tensor:
    encoder = algo.get_encoder()
    decoder = algo.get_decoder()
    z = encoder(x)
    z, _ = STEQuantizePerVectorAbsMax.apply(z)
    y = decoder(z)
    return y