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
    out_seq_len: int = 8
    batch_size: int = 2
    num_workers: int = 8
    val_ratio: float = 0.1
    n_features: int = 1
    hidden_size: int = 100
    num_layers: int = 1
    machine_id: int = 9
    telemetry_path: Path = Path('./data/PdM_telemetry.csv')
    maint_path: Path = Path('./data/PdM_maint.csv')
    machines_path: Path = Path('./data/PdM_machines.csv')
    errors_path: Path = Path('./data/PdM_errors.csv')
    failures_path: Path = Path('./data/PdM_failures.csv')

@dataclass
class TrainArguments:
    criterion: MSELoss = MSELoss()
    learning_rate: float = 3e-4
    max_epochs: int = 2
    n_iterations: int = 5000
    gbm_learning_rate: float = 1e-2
    dropout: float = 0
    checkpoint_path: str = './model_checkpoints'