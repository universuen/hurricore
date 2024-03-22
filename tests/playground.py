import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

CKPT = True


accelerator = Accelerator()
dataset = range(10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
dataloader = accelerator.prepare(dataloader)


accelerator.dataloader_config.use_seedable_sampler = True
set_seed(42)
dataloader._is_accelerate_prepared = False
dataloader = accelerator.prepare(dataloader)
if CKPT:
    print('Resumed from chechpoint')
    accelerator.load_state('ckpt')
    dataloader = accelerator.prepare(dataloader)
    dataloader = accelerator.skip_first_batches(dataloader, 3)
    for idx, batch in enumerate(dataloader, 3):
        print(f'Batch {idx}: {batch}')
else:
    for idx, batch in enumerate(dataloader):
        print(f'Batch {idx}: {batch}')
        if idx == 2:
            accelerator.save_state('ckpt')
            print('Checkpoint saved.')
    
