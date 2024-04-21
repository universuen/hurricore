# Download dataset from: https://www.kaggle.com/datasets/andrewmvd/animal-faces/data

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip
from PIL import Image


class NoiseCatDataset(Dataset):
    def __init__(
        self, 
        path: Path,
        seed: int = 0,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.path = Path(path)
        self.cat_img_paths = list(path.rglob('*cat*.jpg'))
        self.noises = torch.randn(len(self.cat_img_paths), 3, image_size, image_size)
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self) -> int:
        return len(self.cat_img_paths)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cat_img_path = self.cat_img_paths[index]
        cat_image = Image.open(cat_img_path)
        cat_image = self.transform(cat_image)
        
        noise = self.noises[index]
        
        return noise, cat_image
