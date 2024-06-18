# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    logger: str = 'None'


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 1
    epochs: int = 100
    patience: int = 30

    verbose: int = 1
    device: str = 'cpu'
    debug: bool = False
    experiment: bool = False
    program_test: bool = False
    record: bool = True



@dataclass
class BaseModelConfig:
    model: str = 'LSTM'
    rank: int = 40
    num_windows: int = 12
    num_preds: int = 1


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'Weather'
    train_size: int = 100
    density: float = 0.80


@dataclass
class TrainingConfig:
    bs: int = 64
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'
    optim: str = 'AdamW'


@dataclass
class OtherConfig:
    classification: bool = False
    visualize: bool = True
