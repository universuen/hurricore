import logging
from pathlib import Path

from hurricore.utils import ConfigBase, get_file_name


config_name = get_file_name()


class ModelConfig(ConfigBase):
    ...


class DatasetConfig(ConfigBase):
    ...


class PathConfig(ConfigBase):
    project: Path = Path(__file__).parents[1]
    data: Path = project / 'data'
    logs: Path = data / 'logs'
    checkpoints: Path = data / 'checkpoints' / config_name
    tensor_boards: Path = data / 'tensor_boards' / config_name
    histories: Path = data / 'histories' / config_name

    def __post_init__(self) -> None:
        for path in self.values():
            path.mkdir(parents=True, exist_ok=True)


class AcceleratorConfig(ConfigBase):
    gradient_accumulation_steps = 1
    # mixed_precision = 'fp16'


class TrainerConfig(ConfigBase):
    # basic
    num_epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-3
    # logger hook
    logger_name: str = config_name
    log_level: int = logging.DEBUG
    log_folder_path: Path = PathConfig().logs
    log_interval: int = 1 * AcceleratorConfig().gradient_accumulation_steps
    # tensor board hook
    tensor_board_folder_path: Path = PathConfig().tensor_boards
    tensor_board_record_param_and_grad: bool = True
    # history hook
    history_folder_path: Path = PathConfig().histories
    # checkpoint hook
    ckpt_folder_path: Path = PathConfig().checkpoints
    ckpt_interval: int = 10000 * AcceleratorConfig().gradient_accumulation_steps
    ckpt_seed: int = 42
