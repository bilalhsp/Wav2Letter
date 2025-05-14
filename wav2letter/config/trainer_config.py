from dataclasses import dataclass

@dataclass
class TrainerConfig:
    num_nodes: int = 16
    max_epochs: int = 300
    accelerator: str = 'gpu'

    # checkpoint configs
    monitor: str = "val_loss"
    save_top_k: int = 3
    mode: str = "min"
    every_n_epochs: str = 1