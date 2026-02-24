import logging
from pathlib import Path

from hurricore.utils import ConfigBase, get_file_name


experiment_name = get_file_name()


class PathConfig(ConfigBase):
    project: Path = Path(__file__).parents[1]
    data: Path = project / 'data'
    logs: Path = data / 'logs'
    datasets: Path = data / 'datasets'
    checkpoints: Path = data / 'checkpoints' / experiment_name
    tensor_boards: Path = data / 'tensorboards' / experiment_name
    histories: Path = data / 'histories' / experiment_name
    results: Path = data / 'results' / experiment_name
    others: Path = data / 'others'

    def __post_init__(self) -> None:
        for path in self.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class ModelConfig(ConfigBase):
    ...


class TrainingDatasetConfig(ConfigBase):
    ...


class ValidationDatasetConfig(ConfigBase):
    ...


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps: int = 1
    mixed_precision: str = 'bf16'


class TrainerConfig(ConfigBase):
    # Basic configs
    num_epochs: int = 3
    per_device_batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_clipping_norm: float = 1.0
    gradient_clipping_value: float = None
    data_folder_path: Path = PathConfig().data
    # Hook specific configs
    experiment_name: str = experiment_name
    log_level: int = logging.INFO
    log_interval: int = 1
    tensorboard_record_param_and_grad: bool = True
    evaluation_interval: int = 100
    ckpt_interval: int = 500


class RuntimeConfig(ConfigBase):
    ...
