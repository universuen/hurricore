import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from accelerate import notebook_launcher


def main():
    for use_ckpt in [False, True]:
        set_seed(42)
        
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True)
        )
        dataset = range(10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader = accelerator.prepare(dataloader)
        

            
        if use_ckpt:
            print(f'{accelerator.process_index}: Resumed from checkpoint')
            accelerator.load_state('ckpt')
            dataloader.skip_batches = 3
            for idx, batch in enumerate(dataloader, 3):
                print(f'{accelerator.process_index}: Batch {idx}: {batch}')
    
        else:
            for idx, batch in enumerate(dataloader):
                if idx == 2:
                    accelerator.save_state('ckpt')
                    print(f'{accelerator.process_index}: Checkpoint saved.')
                elif idx > 2:
                    print(f'{accelerator.process_index}: Batch {idx}: {batch}')


notebook_launcher(main, num_processes=2)

