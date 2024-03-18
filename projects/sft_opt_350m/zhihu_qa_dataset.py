from torch.utils.data import Dataset
from datasets import load_dataset


class ZhihuQADataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = load_dataset("wangrui6/Zhihu-KOL", split='train')
    
    def __len__(self) -> None:
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple[str]:
        question, answer, *_ = self.data[index].values()
        return question, answer
