from argparse import ArgumentParser
from pathlib import Path
import math
import struct

import kagglehub
from torch.utils.data import DataLoader
import torch
from loguru import logger

from src.data import MemoryMappedDataset
from src.config import load_config, Config
from src.nn.encoder import Encoder
from src.nn.decoder import Decoder

def forward(enc: Encoder, dec: Decoder, x: torch.Tensor) -> torch.Tensor:
    x = x.float() * (1.0 / 255.0)
    r: torch.Tensor = dec(enc(x[:, 0:1].flatten(start_dim=1)))
    g: torch.Tensor = dec(enc(x[:, 1:2].flatten(start_dim=1)))
    b: torch.Tensor = dec(enc(x[:, 2:3].flatten(start_dim=1)))
    y = torch.cat([r, g, b], dim=1)
    return torch.nn.functional.mse_loss(y, x.flatten(start_dim=1))

def validate(enc: Encoder, dec: Decoder, val_loader: DataLoader, dev: torch.device) -> float:
    enc.eval()
    dec.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x: torch.Tensor = batch.to(dev)
            loss: torch.Tensor = forward(enc, dec, x)
            val_loss += loss.item()
    val_avg = val_loss / len(val_loader)
    return val_avg

def save_bin(model: torch.nn.Module, path: Path):
    s = model.state_dict()
    p: list[float] = []
    for value in s.values():
        values: torch.Tensor = value.detach().cpu().reshape(-1)
        p += values.tolist()
    with open(path, 'wb') as f:
        f.write(struct.pack(f'<{len(p)}f', *p))

def save_to_c(encoder: torch.nn.Module, decoder: torch.nn.Module, path: Path):
    enc_p: list[float] = []
    for value in encoder.state_dict().values():
        values: torch.Tensor = value.detach().cpu().reshape(-1)
        enc_p += values.tolist()

    dec_p: list[float] = []
    for value in decoder.state_dict().values():
        values: torch.Tensor = value.detach().cpu().reshape(-1)
        dec_p += values.tolist()

    with open(path, 'w') as f:
        f.write(f'const float NICE_Encoder[{len(enc_p)}] = {{')
        f.write(', '.join(f'{v:.6e}f' for v in enc_p))
        f.write('};\n')
        f.write('\n')
        # write decoder
        f.write(f'const float NICE_Decoder[{len(dec_p)}] = {{')
        f.write(', '.join(f'{v:.6e}f' for v in dec_p))
        f.write('};\n')

def save_best(enc: Encoder, dec: Decoder, out_dir: Path):
    save_bin(enc, out_dir / 'best_encoder.bin')
    save_bin(dec, out_dir / 'best_decoder.bin')
    save_to_c(enc, dec, out_dir / 'NICE_Weights.c')
    torch.save(enc.state_dict(), out_dir / 'best_encoder.pt')
    torch.save(dec.state_dict(), out_dir / 'best_decoder.pt')

def train(config_path: Path, out_dir: Path):

    # create output directories
    out_dir.mkdir(parents=True, exist_ok=True)

    config: Config = load_config(config_path)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_dir = Path(kagglehub.dataset_download('tay10r/image-block-compression'))

    train_data = MemoryMappedDataset(filename=(dataset_dir / 'train.bin'), block_size=8, channels=3)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_data = MemoryMappedDataset(filename=(dataset_dir / 'val.bin'), block_size=8, channels=3)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    logger.info(f'training samples: {len(train_data)}')
    logger.info(f'validation samples: {len(val_data)}')

    enc = Encoder(latent_dim=config.latent_dim)
    dec = Decoder(latent_dim=config.latent_dim)

    enc.to(dev)
    dec.to(dev)

    optimizer = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=config.lr)
    epoch = 0
    enc.train()
    dec.train()
    counter = 0
    epoch_size = 1000
    repeat = 16
    num_epochs = (len(train_loader) // epoch_size) * repeat
    best_loss = math.inf
    for batch in train_loader:
        x: torch.Tensor = batch.to(dev)
        loss: torch.Tensor = forward(enc, dec, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if counter % epoch_size == 0:
            epoch += 1
            enc.eval()
            dec.eval()
            val_loss = validate(enc, dec, val_loader, dev=dev)
            best_delta = val_loss - best_loss if best_loss != math.inf else 0.0
            logger.info(f'[{epoch:04}/{num_epochs:04}]: val_loss={val_loss:.6f}, best_delta={best_delta:.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                enc.to('cpu')
                dec.to('cpu')
                save_best(enc, dec, out_dir)
                enc.to(dev)
                dec.to(dev)
            enc.train()
            dec.train()

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--out_dir', type=Path, required=True, help='Output directory')
    args = parser.parse_args()
    train(config_path=args.config, out_dir=args.out_dir)

if __name__ == '__main__':
    main()