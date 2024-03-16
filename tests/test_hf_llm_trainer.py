import path_setup

import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType

from src.trainers.hf_llm_trainer import HFLLMTrainer
from src.datasets.zhihu_qa_dataset import ZhihuQADataset
from src.collators.hf_instruction_tuning_collator import HFITCollator
from src.logger import Logger
from src.utils import launch_for_parallel_training
from configs import LoggerConfig

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
        collate_fn=HFITCollator(
            tokenizer=tokenizer, 
            max_len=512,
        ).collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
    logger = Logger('test', **LoggerConfig().to_dict())

    trainer = HFLLMTrainer(
        model=model, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        logger=logger, 
        accelerator=accelerator,
        peek_prompts=[
            '如何看待太阳比地球大？',
            '太阳比地球大会带来怎样的影响？',
            '太阳距离地球很远，我们该采取怎样的措施？',
        ],
        tokenizer=tokenizer,
    )
    trainer.run(epochs=1000)

launch_for_parallel_training(main, num_processes=4, use_port='8000')
# main()