from pathlib import Path
from random import Random

import torch

import numpy as np

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as FT

class RandomSamplerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, block_size: int, seed: int):
        self.__samples = list([path for path in data_dir.glob("*.jpg")])
        self.__rng = Random(seed)
        self.__block_size = block_size

    def __len__(self) -> int:
        return len(self.__samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        p = self.__samples[index]
        img = read_image(str(self.__samples[index]), mode=ImageReadMode.RGB)
        s0 = img.shape[1]
        s1 = img.shape[2]
        x = self.__rng.randint(0, s1 - self.__block_size)
        y = self.__rng.randint(0, s0 - self.__block_size)
        img = FT.crop(img, top=y, left=x, height=self.__block_size, width=self.__block_size)
        return img

class MemoryMappedDataset(torch.utils.data.Dataset):
    """
    Uses a memory mapped numpy file to load data samples.
    """
    def __init__(self, filename: Path, block_size: int, channels: int):
        size = filename.stat().st_size
        num_samples = size // (block_size * block_size * channels)
        self.__data: np.ndarray = np.memmap(filename, dtype=np.uint8, mode='r', shape=(num_samples, channels, block_size, block_size))

    def __len__(self) -> int:
        return self.__data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.__data[index])