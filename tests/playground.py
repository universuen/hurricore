import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from accelerate import notebook_launcher

from torch.random import get_rng_state


import torch
import numpy as np
import matplotlib.pyplot as plt


def compare_rng_states(*rng_states, name):
    """
    Compares multiple RNG states for equality and visualizes the results in a matrix.

    Parameters:
    - rng_states: Varargs parameter to pass multiple RNG state tensors.
    """
    n = len(rng_states)
    equality_matrix = np.zeros((n, n))

    # Compare each pair of RNG states for equality
    for i in range(n):
        for j in range(n):
            equality_matrix[i, j] = torch.equal(rng_states[i], rng_states[j])

    # Visualize the equality matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(equality_matrix, cmap='hot', interpolation='nearest')
    plt.title('RNG State Equality Matrix')
    plt.xticks(range(n), labels=[f'State {i}' for i in range(n)])
    plt.yticks(range(n), labels=[f'State {i}' for i in range(n)])
    plt.colorbar(label='Equality (1: equal, 0: different)')
    plt.savefig(f'/storage_fast/ysun/hurricane/tests/results_{name}.png')


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
                    

    
notebook_launcher(main, num_processes=1)







from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler