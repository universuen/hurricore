import path_setup

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from src.trainers.accelerated_trainer_for_hugging_face_llm import AcceleratedTrainerForHFLLM
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.collators.hugging_face_instruction_tuning_collator import HFITCollator
from src.logger import Logger
from src.utils import launch_for_parallel_training
from configs import LoggerConfig


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))

    dataset = ZhihuQADataset()
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        collate_fn=HFITCollator(
            tokenizer=tokenizer, 
            max_len=512,
        ).collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    logger = Logger('test', **LoggerConfig().to_dict())
    accelerator = Accelerator()
    trainer = AcceleratedTrainerForHFLLM(model, data_loader, optimizer, logger, accelerator)
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
