from logging import Logger
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hurricore.trainers import Trainer
from hurricore.hooks import(
    HFLLMPeekHook, 
    LoggerHook, 
    LRSchedulerHook, 
    TensorBoardHook, 
    CheckpointHook,
)


class HFLLMTrainer(Trainer):
    def __init__(
        self, 
        
        model: PreTrainedModel, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        accelerator: Accelerator,
        num_epochs: int = 100,
        
        
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
        ckpt_seed: int = 42,
        
    ) -> None:
        super().__init__(
            models=[model], 
            optimizers=[optimizer], 
            data_loaders=[data_loader], 
            accelerator=accelerator,
            num_epochs=num_epochs,
        )
        
        if peek_prompts is None:
            peek_prompts = []
            
        self.hooks = [
            HFLLMPeekHook(
                trainer=self,
                prompts=peek_prompts, 
                tokenizer=tokenizer, 
                interval=peek_interval,
            ),
            LoggerHook(
                trainer=self,
                logger=logger, 
                interval=log_interval,
            ),
            LRSchedulerHook(
                trainer=self,
                lr_schedulers=[lr_scheduler],
                mode=lr_scheduler_mode,
            ),
            TensorBoardHook(
                trainer=self,
                folder_path=tensor_board_folder_path,
                interval=tensor_board_interval,
            ),
            CheckpointHook(
                trainer=self,
                folder_path=ckpt_folder_path,
                interval=ckpt_interval,
                seed=ckpt_seed,
            ),
        ]
    
    def compute_loss(self) -> torch.Tensor:
        input_ids, attention_masks, labels = self.ctx.batches[0]
        model = self.models[0]
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            use_cache=False,
        )[0]
        return loss
