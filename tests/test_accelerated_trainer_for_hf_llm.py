import path_setup

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from src.trainers.accelerated_trainer_for_hf_llm import AcceleratedTrainerForHFLLM
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.collators.hf_instruction_tuning_collator import HFITCollator
from src.hooks.hf_llm_peek_hook import HFLLMValidationHooK
from src.logger import Logger
from src.utils import launch_for_parallel_training
from configs import LoggerConfig


def main():
    accelerator = Accelerator(
        # mixed_precision='fp16', 
        gradient_accumulation_steps=4,
        # split_batches=True,
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        dataset = ZhihuQADataset()
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

    trainer = AcceleratedTrainerForHFLLM(model, data_loader, optimizer, logger, accelerator)
    peek_hook = HFLLMValidationHooK(
        [
            "为什么太阳从东边升起？",
            "如何看待大语言模型？",
            "我们对人工智能的理解是否足够深刻？",
        ],
        tokenizer
    )
    trainer.hooks.insert(0, peek_hook)
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
