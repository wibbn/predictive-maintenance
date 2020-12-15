from dataclasses import dataclass
from pathlib import Path
from torch.nn import MSELoss

@dataclass
class CommonArguments:
    seed: int = 1
    verbose: bool = True
    version: str = '1.0.0'

@dataclass
class DataArguments:
    seq_len: int = 24
    batch_size: int = 2
    telemetry_path: Path = Path('./data/PdM_telemetry.csv')
    num_workers: int = 4
    val_ratio: float = 0.1
    n_features: int = 7
    hidden_size: int = 100
    num_layers: int = 1

@dataclass
class TrainArguments:
    criterion: MSELoss = MSELoss()
    learning_rate: float = 3e-4
    max_epochs: int = 2
    dropout: float = 0