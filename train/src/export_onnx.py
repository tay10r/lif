from argparse import ArgumentParser
from pathlib import Path

import torch

from loguru import logger

from src.algos.base import Algo
from src.algos.factory import make_algo
from src.config import load_config, Config

def export_onnx(algo: Algo, block_size: int, latent_dim: int, out_path: Path):

    encoder: torch.nn.Module = algo.get_encoder()
    decoder: torch.nn.Module = algo.get_decoder()

    encoder.eval()
    decoder.eval()

    encoder.load_state_dict(torch.load(out_path / 'best_encoder.pt', map_location='cpu'))
    decoder.load_state_dict(torch.load(out_path / 'best_decoder.pt', map_location='cpu'))

    x = torch.randn(1, 3 * block_size * block_size)
    torch.onnx.export(encoder,
                      (x,),
                      str(out_path / 'encoder.onnx'), 
                      external_data=False,
                      verbose=False,
                      input_names=['input'], 
                      output_names=['output'])

    z = torch.randn(1, latent_dim)
    torch.onnx.export(decoder,
                      (z,),
                      str(out_path / 'decoder.onnx'), 
                      external_data=False,
                      verbose=False,
                      input_names=['input'], 
                      output_names=['output'])

def main():
    out_dir = Path('out/models')
    for model_dir in out_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        logger.info(f'exporting {model_name}')
        config_path = Path('configs')  / f'{model_name}.json'
        config: Config = load_config(config_path)
        algo: Algo = make_algo(config.algo,
                               block_size=config.block_size,
                               channels=config.input_channels,
                               latent_dim=config.latent_dim,
                               **config.algo_params)
        export_onnx(algo, config.block_size, config.latent_dim, model_dir)

if __name__ == '__main__':
    main()
