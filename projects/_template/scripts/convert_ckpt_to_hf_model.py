import _path_setup  # noqa: F401

import os
from typing import TYPE_CHECKING

import torch 
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from hurricore.utils import load_config
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


if TYPE_CHECKING:
    from configs import debug as config

config = load_config("configs.exp_20250928")
STEP = None


def main() -> None:
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**config.ModelConfig())
    model = model.cpu()
    if STEP is None:
        checkpoint_path = config.PathConfig().checkpoints / "ckpt_lowest_val_loss"
    else:
        checkpoint_path = config.PathConfig().checkpoints / f"ckpt_step_{STEP}"
    
    state_dict = None
    try:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
    except ValueError:
        state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    save_path = config.PathConfig().results / f"{checkpoint_path.name}_hf_model"
    model.save_pretrained(save_directory=save_path, save_peft_format=False)
    processor = AutoProcessor.from_pretrained(**config.ModelConfig()) 
    processor.save_pretrained(save_path)

    print(f"Model and processor saved at: {save_path}")


if __name__ == "__main__":
    main()
