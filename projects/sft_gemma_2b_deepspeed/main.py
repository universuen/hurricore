import path_setup

import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from hurricane.trainers.hf_llm_trainer import HFLLMTrainer
from hurricane.collators.hf_llm_instruction_tuning_collator import HFLLMITCollator
from hurricane.logger import Logger
from hurricane.utils import launch

from zhihu_qa_dataset import ZhihuQADataset
from configs import *


def main():
    accelerator_config = AcceleratorConfig()
    logger_config = LoggerConfig()
    training_config = TrainingConfig()
    peek_config = PeekConfig()
    
    accelerator = Accelerator(**accelerator_config)
    logger = Logger('sft_gemma_2b_deepspeed', **logger_config)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if accelerator.is_main_process:
        logger.info(accelerator_config)
        logger.info(logger_config)
        logger.info(training_config)
        logger.info(peek_config)
        logger.info('Set TOKENIZERS_PARALLELISM=false to prevent dead lock.')
    
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        dataset = ZhihuQADataset()

    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            max_len=training_config.max_len,
        ).collate_fn,
    )
    optimizer = training_config.optimizer_type(
        params=model.parameters(),
        lr=training_config.lr,
    )

    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        peek_prompts=peek_config.prompts,
        tokenizer=tokenizer,
        interval=peek_config.interval,
    )
    trainer.run(epochs=training_config.epochs)

launch(main, num_processes=4, use_port='8000')
