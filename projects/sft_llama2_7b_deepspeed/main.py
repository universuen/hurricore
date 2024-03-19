import path_setup

import os

from torch.optim import AdamW
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
    ckpt_config = CKPTConfig()
    
    accelerator = Accelerator(**accelerator_config)
    logger = Logger('sft_llama2_7b_deepspeed', **logger_config)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if accelerator.is_main_process:
        logger.info(accelerator_config)
        logger.info(logger_config)
        logger.info(training_config)
        logger.info(peek_config)
        logger.info(ckpt_config)
        logger.info('Set TOKENIZERS_PARALLELISM=false to prevent dead lock.')

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        dataset = ZhihuQADataset()

    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size_per_device,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            max_len=training_config.max_len,
        ).collate_fn,
    )
    optimizer = AdamW(
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
        peek_interval=peek_config.interval,
        log_interval=training_config.log_interval,
        ckpt_folder_path=ckpt_config.folder_path,
    )
    trainer.run(epochs=training_config.epochs)

launch(main, num_processes=4, use_port='8001')
