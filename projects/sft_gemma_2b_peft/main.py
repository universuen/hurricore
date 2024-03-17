import path_setup

import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType

from hurricane.trainers.hf_llm_trainer import HFLLMTrainer
from hurricane.collators.hf_llm_instruction_tuning_collator import HFLLMITCollator
from hurricane.logger import Logger
from hurricane.utils import launch

from zhihu_qa_dataset import ZhihuQADataset
from configs import *


def main():
    logger_config = LoggerConfig()
    logger = Logger('sft_gemma_2b_peft', **logger_config)
    logger.info(logger_config)
    
    training_config = TrainingConfig()
    logger.info(training_config)
    
    accelerator_config = AcceleratorConfig()
    logger.info(accelerator_config)
    
    peek_config = PeekConfig()
    logger.info(peek_config)
    
    peft_config = PEFTConfig()
    logger.info(peft_config)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info('Set TOKENIZERS_PARALLELISM=false to prevent dead lock.')

    accelerator = Accelerator(**accelerator_config)
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        dataset = ZhihuQADataset()
        
    model = get_peft_model(model, peft_config.lora_config)
    model.print_trainable_parameters()
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
