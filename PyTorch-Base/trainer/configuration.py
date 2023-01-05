# # <font style="color:blue">Configurations</font>

from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor


# ## <font style="color:green">System Configuration</font>

@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


# ## <font style="color:green">Data Configuration</font>

@dataclass
class DatasetConfig:
    root_dir: str = "data"  # dataset directory root
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation


# ## <font style="color:green">Dataloader Configuration</font>

@dataclass
class DataloaderConfig:
    batch_size: int = 32
    num_workers: int = 8


# ## <font style="color:green">Optimizer Configuration</font>

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.05
    momentum: float = 0.95
    weight_decay: float = 0.0001
    lr_step_milestones: Iterable = (100,200,300,400,500)
    lr_gamma: float = 0.3


# ## <font style="color:green">Training Configuration</font>

@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_saving_frequency: int = 100  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 600  # number of times the whole dataset will be passed through the network
    progress_bar: bool = True  # enable progress bar visualization during train process
