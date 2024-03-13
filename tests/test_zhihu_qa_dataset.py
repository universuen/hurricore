import context

from src.datasets.zhihu_qa_dataset import ZhihuQADataset


dataset = ZhihuQADataset()

for q, a in dataset:
    print(
        f'Question:\n{q}\n'
        f'Answer:\n{a}\n'
    )

print(len(dataset))
