from logging import Logger
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.hooks.hf_llm_peek_hook import HFLLMPeekHook
from hurricane.hooks.checkpoint_hook import CheckpointHook
from hurricane.hooks.lr_scheduler_hook import LRSchedulerHook
from hurricane.hooks.tensor_board_hook import TensorBoardHook


class HFLLMTrainer(Trainer):
    def __init__(
        self, 
        
        model: PreTrainedModel, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        accelerator: Accelerator,
        epochs: int = 100,
        seed: int = 42,
        
        logger: Logger = None,
        log_interval: int = 1,
        
        peek_prompts: list[str] = None,
        tokenizer: PreTrainedTokenizerBase = None, 
        peek_interval: int = 1,
        
        lr_scheduler: LRScheduler = None,
        lr_scheduler_mode: str = 'per_epoch',
        
        tensor_board_folder_path: Path = None,
        tensor_board_interval: int = 1,
        
        ckpt_folder_path: Path = None,
        ckpt_interval: int = 1000,
        
    ) -> None:
        super().__init__(
            model=model, 
            data_loader=data_loader, 
            optimizer=optimizer, 
            accelerator=accelerator,
            epochs=epochs,
            seed=seed,
        )
        
        if peek_prompts is None:
            peek_prompts = []
            
        self.hooks = [
            HFLLMPeekHook(
                prompts=peek_prompts, 
                tokenizer=tokenizer, 
                interval=peek_interval,
            ),
            LoggerHook(
                logger=logger, 
                interval=log_interval,
            ),
            LRSchedulerHook(
                lr_scheduler=lr_scheduler,
                mode=lr_scheduler_mode,
            ),
            TensorBoardHook(
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
            ),
            CheckpointHook(
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
            ),
        ]
    
    def compute_loss(self) -> torch.Tensor:
        input_ids, attention_masks, labels = self.ctx.batch
        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            use_cache=False,
        )[0]
        return loss
