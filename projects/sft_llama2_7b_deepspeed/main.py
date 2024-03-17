import path_setup

import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DeepSpeedPlugin

from hurricane.trainers.hf_llm_trainer import HFLLMTrainer
from hurricane.collators.hf_llm_instruction_tuning_collator import HFLLMITCollator
from hurricane.logger import Logger
from hurricane.utils import launch_for_parallel_training
from configs import LoggerConfig

from zhihu_qa_dataset import ZhihuQADataset


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    accelerator = Accelerator(
        gradient_accumulation_steps=32,
        mixed_precision='fp16',
        split_batches=True,
        deepspeed_plugin=DeepSpeedPlugin(
            gradient_accumulation_steps=32, 
            zero_stage=3,
            offload_optimizer_device='cpu',
            # offload_param_device='cpu',
            zero3_init_flag=False,
        )
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        dataset = ZhihuQADataset()

    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            max_len=512,
        ).collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    logger_config = LoggerConfig()
    logger = Logger('sft_llama2_7b_deepspeed', **logger_config.to_dict())

    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        peek_prompts=[
            '如何看待明天下雨？',
            '为什么太阳比地球大？',
            '你如何看待近期的股市？',
        ],
        tokenizer=tokenizer,
        interval=32,
    )
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
