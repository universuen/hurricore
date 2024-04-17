import torch
from diffusers import UNet2DModel


class UNet(UNet2DModel):
    def __init__(
        self, 
        image_size=128,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256, 512, 512),
    ):  
        super().__init__(
            sample_size=image_size,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D", 
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ) 
        self.image_size = image_size
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return super().forward(x, t).sample
