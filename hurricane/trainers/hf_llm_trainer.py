from logging import Logger

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook
from hurricane.hooks.hf_llm_peek_hook import HFLLMPeekHooK


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
    ) -> None:
        super().__init__(model, data_loader, optimizer, accelerator)
        
        if peek_prompts is None:
            peek_prompts = []
            
        self.hooks = [
            LoggerHook(logger),
            HFLLMPeekHooK(peek_prompts, tokenizer),
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
