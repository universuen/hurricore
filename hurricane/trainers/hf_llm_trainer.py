from logging import Logger
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.hooks.hf_llm_peek_hook import HFLLMPeekHook
from hurricane.hooks.ckpt_hook import CKPTHook


class HFLLMTrainer(Trainer):
    def __init__(
        self, 
        model: PreTrainedModel, 
        data_loader: DataLoader, 
        optimizer: Optimizer,
        accelerator: Accelerator,
        logger: Logger,
        peek_prompts: list[str] = None,
        tokenizer: PreTrainedTokenizerBase = None, 
        peek_interval: int = 1,
        log_interval: int = 1,
        ckpt_interval: int = 1,
        ckpt_folder_path: Path = None,
    ) -> None:
        super().__init__(model, data_loader, optimizer, accelerator)
        
        if peek_prompts is None:
            peek_prompts = []
            
        self.hooks = [
            CKPTHook(ckpt_interval, ckpt_folder_path),
            HFLLMPeekHook(peek_prompts, tokenizer, peek_interval),
            LoggerHook(logger, log_interval),
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
