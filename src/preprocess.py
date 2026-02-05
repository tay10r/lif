# This script will take the dataset partitions 'train', 'val', and 'test'
# and preprocess them into a numpy array (each) of 8x8 image blocks, so that
# they can be loaded with memory mapping during training.

from pathlib import Path

import numpy as np

from src.data import RandomSamplerDataset

def preprocess(data_dir: Path, out_dir: Path, block_size: int, iterations: int = 8):
    data = RandomSamplerDataset(data_dir=data_dir, block_size=block_size, seed=0)
    result = np.zeros((iterations * len(data), 3, block_size, block_size), dtype=np.float32)
    for iteration in range(iterations):
        for i in range(len(data)):
            result[iteration * len(data) + i] = data[i].numpy()
    # save to flat binary
    out_path = out_dir / f'{data_dir.name}.bin'
    result.tofile(out_path)
    print(f'preprocessed {len(data)} samples to {out_path}')

def main():
    block_size = 8
    out_dir = Path('out/datasets')
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess(Path('data/train'), out_dir=out_dir, block_size=block_size)
    preprocess(Path('data/val'), out_dir=out_dir, block_size=block_size)

if __name__ == '__main__':
    main()