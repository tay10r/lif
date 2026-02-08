from src.algos.base import Algo

from src.algos.m0 import M0Algo
from src.algos.m1 import M1Algo
from src.algos.m2 import M2Algo

def make_algo(name: str, **kwargs) -> Algo:
    match name:
        case 'm0':
            return M0Algo(**kwargs)
        case 'm1':
            return M1Algo(**kwargs)
        case 'm2':
            return M2Algo(**kwargs)
        case _:
            raise ValueError(f'Unknown algorithm: {name}')