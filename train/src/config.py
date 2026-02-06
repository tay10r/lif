from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    algo: str
    algo_params: dict
    batch_size: int = 16
    lr: float = 1.0e-4
    block_size: int = 8
    input_channels: int = 3

def load_config(path: str | Path) -> Config:
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return Config(**data)