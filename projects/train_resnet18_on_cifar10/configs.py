import path_setup

from hurricane.common_configs import *


class TrainingConfig(ConfigBase):
    epochs = 10
    lr = 1e-3
    batch_size_per_device = 1024


class UpdatedPathConfig(PathConfig):
    cifar10_dataset: Path = PathConfig.data / 'cifar10_dataset'


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = 1


class CKPTConfig(ConfigBase):
    folder_path = Path(__file__).resolve().parent / 'checkpoints'
