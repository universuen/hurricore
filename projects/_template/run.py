from typing import TYPE_CHECKING

from accelerate import Accelerator

from hurricore.utils import load_config
from trainers.my_trainer import MyTrainer

if TYPE_CHECKING:
    from configs import debug as config


# Import config from module path
config = load_config('configs.debug')

""" Optional:
import config from file path
`config = load_config('projects/_project_template/configs/default.py')`
import config from url
`config = load_config('https://raw.githubusercontent.com/universuen/hurricore/main/projects/_project_template/configs/default.py')`
"""


def main():
    # Setup accelerator
    accelerator = Accelerator(**config.AcceleratorConfig())
    # Setup model and dataset
    with accelerator.main_process_first():
        model = ...
        training_dataset = ...
        validation_dataset = ...
    # Setup trainer and run
    MyTrainer(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        **config.TrainerConfig(),
    ).run()


if __name__ == '__main__':
    main()
