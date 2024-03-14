import context

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.runners.sft_runner import SFTRunner
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.logger import Logger
from src.utils import launch_for_parallel_training


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    dataset = ZhihuQADataset()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    runner = SFTRunner(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=None,
    )

    runner._print('Hello from print')
    runner.logger = Logger('test')
    runner._print('Hello from logger.info', 'info')

    runner.train(epochs=1000, batch_size=32)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
