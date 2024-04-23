# Download dataset from: https://www.kaggle.com/datasets/andrewmvd/animal-faces/data

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(
        self, 
        path: Path,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.cat_img_paths = list(path.rglob('*cat*.jpg'))
        self.dog_img_paths = list(path.rglob('*dog*.jpg'))
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self) -> int:
        return len(self.cat_img_paths)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cat_img_path = self.cat_img_paths[index]
        dog_img_path = self.dog_img_paths[torch.randint(0, len(self.dog_img_paths), (1,)).item()]
        cat_image = Image.open(cat_img_path)
        dog_image = Image.open(dog_img_path)
        cat_image = self.transform(cat_image)
        dog_image = self.transform(dog_image)
        return cat_image, dog_image
