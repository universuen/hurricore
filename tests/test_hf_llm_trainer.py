import path_setup

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from src.trainers.hf_llm_trainer import HFLLMTrainer
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.collators.hf_instruction_tuning_collator import HFITCollator
from src.logger import Logger
from src.utils import launch_for_parallel_training
from configs import LoggerConfig


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        dataset = ZhihuQADataset()
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        collate_fn=HFITCollator(
            tokenizer=tokenizer, 
            max_len=512,
        ).collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    logger = Logger('test', **LoggerConfig().to_dict())

    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        
    )
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=1, use_port='8000')
