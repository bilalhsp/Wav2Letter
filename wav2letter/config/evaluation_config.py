from dataclasses import dataclass





@dataclass
class DatasetConfig:
    samples_rate: int
    hop_length: int



@dataclass
class EvalConfig:
    checkpoint: str = 'Wav2letter-epoch=024-val_loss=0.37.ckpt'
    val_manifest: str = 'val-clean'
    adv_robust: bool = False
    noise_bound: float = 1.e+1
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = False

