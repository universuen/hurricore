# Download dataset from: https://www.kaggle.com/datasets/andrewmvd/animal-faces/data

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image


class CatDataset(Dataset):
    def __init__(
        self, 
        path: Path,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.img_paths = list(path.rglob('*cat*.jpg'))
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.img_paths[index])
        image = self.transform(image)
        return image
