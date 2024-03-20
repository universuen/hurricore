import path_setup

import os

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from hurricane.trainers.hf_llm_trainer import HFLLMTrainer
from hurricane.collators.hf_llm_instruction_tuning_collator import HFLLMITCollator
from hurricane.logger import Logger
from hurricane.utils import launch, log_all_configs

from zhihu_qa_dataset import ZhihuQADataset
from configs.opt_350m import *


def main():
    accelerator_config = AcceleratorConfig()
    accelerator = Accelerator(**accelerator_config)

    logger_config = LoggerConfig()
    logger = Logger(**logger_config)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if accelerator.is_main_process:
        log_all_configs(logger)
        logger.info('Set TOKENIZERS_PARALLELISM=false to prevent dead lock.')
    
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        dataset = ZhihuQADataset()

    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))

    data_loader_config = DataLoaderConfig()
    collator_config = CollatorConfig()
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            **collator_config,
        ).collate_fn,
        **data_loader_config,
    )

    optimizer_config = OptimizerConfig()
    optimizer = AdamW(
        params=model.parameters(),
        **optimizer_config,
    )

    trainer_config = TrainerConfig()
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(len(data_loader) // accelerator_config.gradient_accumulation_steps) * trainer_config.epochs,
    )
    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        tokenizer=tokenizer,
        lr_scheduler=scheduler,
        lr_scheduler_mode='per_step',
        **trainer_config,
    )
    trainer.run()

launch(main, num_processes=4, use_port='8000')
