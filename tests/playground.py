import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from accelerate import notebook_launcher


def main():
    set_seed(42)
    
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)
    )
    dataset = range(10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = accelerator.prepare(dataloader)
    print(type(dataloader.batch_sampler.batch_sampler.sampler))

notebook_launcher(main, num_processes=2, use_port='8000')

