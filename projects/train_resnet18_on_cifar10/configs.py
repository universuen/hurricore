import path_setup

from torch.optim import AdamW

from hurricane.common_configs import *


class TrainingConfig(ConfigBase):
    epochs = 10
    lr = 1e-3
    batch_size = 128
    optimizer_type = AdamW


class UpdatedPathConfig(PathConfig):
    cifar10_dataset: Path = PathConfig.data / 'cifar10_dataset'


class AcceleratorConfig(ConfigBase):
    split_batches = True
