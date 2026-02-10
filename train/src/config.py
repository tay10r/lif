from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 64
    lr: float = 1.0e-3
    latent_dim: int = 4

def load_config(path: str | Path) -> Config:
    import json
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
        return Config(**data)
    else:
        return Config()