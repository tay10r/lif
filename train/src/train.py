from pathlib import Path
from collections import deque
import math

from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid, save_image
from loguru import logger

from src.data import MemoryMappedDataset
from src.net import Net

def export_onnx(net: Net, block_size: int, latent_dim: int, out_path: Path):

    net.eval()

    x = torch.randn(1, 3 * block_size * block_size)
    torch.onnx.export(net.encoder,
                      (x,),
                      str(out_path / 'encoder.onnx'), 
                      external_data=False,
                      verbose=False,
                      input_names=['input'], 
                      output_names=['output'])

    z = torch.randn(1, latent_dim)
    torch.onnx.export(net.decoder,
                      (z,),
                      str(out_path / 'decoder.onnx'), 
                      external_data=False,
                      verbose=False,
                      input_names=['input'], 
                      output_names=['output'])

def validate(net: Net, val_loader: DataLoader, block_size: int, dev: torch.device) -> float:
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x: torch.Tensor = batch.to(dev).float() * (1.0 / 255.0)
            x = torch.flatten(x, start_dim=1)
            z = net.encoder(x)
            y: torch.Tensor = net.decoder(z)
            loss = torch.nn.functional.mse_loss(y, x)
            y = y.reshape(x.shape[0], 3, block_size, block_size)
            val_loss += loss.item()
    val_avg = val_loss / len(val_loader)
    return val_avg

def train(batch_size: int, lr: float, block_size: int, latent_dim: int, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    progress_dir = out_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    # clear previous progress images
    for entry in progress_dir.glob('*.png'):
        entry.unlink()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = MemoryMappedDataset(filename=(out_dir / 'datasets' / 'train.bin'), block_size=block_size, channels=3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = MemoryMappedDataset(filename=(out_dir / 'datasets' / 'val.bin'), block_size=block_size, channels=3)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    logger.info(f'training samples: {len(train_data)}')
    logger.info(f'validation samples: {len(val_data)}')
    net = Net(block_size=block_size, channels=3, latent_dim=latent_dim)
    net.to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epoch = 0
    net.train()
    counter = 0
    epoch_size = 1000
    num_epochs = len(train_loader) // epoch_size
    best_loss = math.inf
    for batch in train_loader:
        x: torch.Tensor = batch.to(dev).float() * (1.0 / 255.0)
        x = torch.flatten(x, start_dim=1)
        y = net(x)
        loss = torch.nn.functional.mse_loss(y, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if counter % epoch_size == 0:
            epoch += 1
            net.eval()
            val_loss = validate(net, val_loader, block_size, dev)
            logger.info(f'[{epoch:04}/{num_epochs:04}]: val_loss={val_loss:.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                net.to('cpu')
                export_onnx(net, block_size, latent_dim=latent_dim, out_path=out_dir)
                net.to(dev)
            net.train()

def main():
    train(batch_size=16, lr=1.0e-4, block_size=8, latent_dim=8, out_dir=Path('out'))

if __name__ == '__main__':
    main()