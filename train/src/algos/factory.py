from src.algos.base import Algo

from src.algos.linear_resnet import LinearResnetAlgo
from src.algos.m1 import M1Algo

def make_algo(name: str, **kwargs) -> Algo:
    match name:
        case 'm0':
            return LinearResnetAlgo(**kwargs)
        case 'm1':
            return M1Algo(**kwargs)
        case _:
            raise ValueError(f'Unknown algorithm: {name}')