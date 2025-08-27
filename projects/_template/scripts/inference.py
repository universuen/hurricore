import _path_setup  # noqa: F401

import torch

from hurricore.utils import find_latest_checkpoint_folder

from configs import debug as config


@torch.no_grad()
def main():
    model_ckpt_folder_path = find_latest_checkpoint_folder(config.PathConfig().checkpoints) # noqa
    ...


if __name__ == '__main__':
    main()
