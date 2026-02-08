from argparse import ArgumentParser
from pathlib import Path
import math

from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid, save_image
from loguru import logger
from pytorch_msssim import SSIM

from src.data import MemoryMappedDataset
from src.algos.base import Algo, forward
from src.algos.factory import make_algo
from src.config import load_config, Config
from src.nn.quantize import STEQuantize

def validate(algo: Algo, val_loader: DataLoader, block_size: int, dev: torch.device) -> float:
    enc = algo.get_encoder()
    dec = algo.get_decoder()
    enc.eval()
    dec.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x: torch.Tensor = batch.to(dev).float() * (1.0 / 255.0)
            x = torch.flatten(x, start_dim=1)
            y = forward(algo, x)
            loss = torch.nn.functional.mse_loss(y, x)
            y = y.reshape(x.shape[0], 3, block_size, block_size)
            val_loss += loss.item()
    val_avg = val_loss / len(val_loader)
    return val_avg

def train(config_path: Path, out_dir: Path, datasets_dir: Path):

    # create output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = out_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)

    # clear previous progress images
    for entry in progress_dir.glob('*.png'):
        entry.unlink()

    config: Config = load_config(config_path)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MemoryMappedDataset(filename=(datasets_dir / 'train.bin'), block_size=config.block_size, channels=3)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_data = MemoryMappedDataset(filename=(datasets_dir / 'val.bin'), block_size=config.block_size, channels=3)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    logger.info(f'training samples: {len(train_data)}')
    logger.info(f'validation samples: {len(val_data)}')

    algo = make_algo(config.algo,
                     block_size=config.block_size,
                     channels=config.input_channels,
                     latent_dim=config.latent_dim,
                      **config.algo_params)

    enc = algo.get_encoder()
    dec = algo.get_decoder()

    enc.to(dev)
    dec.to(dev)

    optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=config.lr)
    epoch = 0
    enc.train()
    dec.train()
    counter = 0
    epoch_size = 1000
    num_epochs = len(train_loader) // epoch_size
    best_loss = math.inf
    for batch in train_loader:
        x: torch.Tensor = batch.to(dev).float() * (1.0 / 255.0)
        x = torch.flatten(x, start_dim=1)
        y = forward(algo, x)
        loss = torch.nn.functional.mse_loss(y, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if counter % epoch_size == 0:
            epoch += 1
            enc.eval()
            dec.eval()
            val_loss = validate(algo, val_loader, config.block_size, dev)
            best_delta = val_loss - best_loss if best_loss != math.inf else 0.0
            logger.info(f'[{epoch:04}/{num_epochs:04}]: val_loss={val_loss:.6f}, best_delta={best_delta:.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                enc.to('cpu')
                dec.to('cpu')
                # save to .pt
                torch.save(enc.state_dict(), out_dir / 'best_encoder.pt')
                torch.save(dec.state_dict(), out_dir / 'best_decoder.pt')
                enc.to(dev)
                dec.to(dev)
            enc.train()
            dec.train()

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--out_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--datasets_dir', type=Path, default='out/datasets', help='Datasets directory')
    args = parser.parse_args()
    train(config_path=args.config, out_dir=args.out_dir, datasets_dir=args.datasets_dir)

if __name__ == '__main__':
    main()