import path_setup

import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType

from src.trainers.hf_llm_trainer import HFLLMTrainer
from src.collators.hf_llm_instruction_tuning_collator import HFLLMITCollator
from src.logger import Logger
from src.utils import launch_for_parallel_training
from configs import LoggerConfig

from zhihu_qa_dataset import ZhihuQADataset


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        # deepspeed_plugin=DeepSpeedPlugin(gradient_accumulation_steps=4, zero_stage=3)
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
        dataset = ZhihuQADataset()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=HFLLMITCollator(
            tokenizer=tokenizer, 
            max_len=512,
        ).collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
    logger_config = LoggerConfig()
    logger = Logger('sft_gemma_2b_peft', **logger_config.to_dict())

    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        peek_prompts=[
            '如何看待明天下雨？',
            '为什么太阳比地球大',
            '你如何看待近期的股市？',
        ],
        tokenizer=tokenizer,
    )
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
