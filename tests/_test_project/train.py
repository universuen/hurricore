from accelerate import Accelerator

from hurricore.utils import load_config

from models.model import Model
from datasets.dataset import Dataset 
from trainer import TestTrainer

# import config from module path
config = load_config('configs.default')

""" Optional:
import config from file path
`config = load_config('projects/_project_template/configs/default.py')`
import config from url
`config = load_config('https://raw.githubusercontent.com/universuen/hurricore/main/projects/_project_template/configs/default.py')`
"""


def main():
    # setup accelerator
    accelerator = Accelerator(**config.AcceleratorConfig())
    # setup model and dataset
    with accelerator.main_process_first():
        model = Model()
        dataset = Dataset()
    # setup trainer and run
    TestTrainer(
        model=model,
        dataset=dataset,
        accelerator=accelerator,
        **config.TrainerConfig(),
    ).run()


if __name__ == '__main__':
    main()
