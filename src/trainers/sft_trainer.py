import os
from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


class SFTTrainer:
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        logger: Logger = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.env = dict()
    
    def run(self, **training_configs: dict) -> None:
        epochs = training_configs.get('epochs', 3)
        batch_size = training_configs.get('batch_size', 32)
        max_len = training_configs.get('max_len', 512)
        data_loader = DataLoader(
            dataset=self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )
        
        self.env['max_len'] = max_len

        for epoch in range(1, epochs + 1):
            losses = []
            self._print(f'Epoch {epoch} started', 'info')
            for batch in data_loader:
                loss = self.calculate_loss(batch)
                loss.backward()
                self.optimizer.step()
                self._print(f'Current loss: {loss.item(): .5f}')
                losses.append(loss.item())
            if self.scheduler is not None:
                self.scheduler.step()
            self._print(f'Epoch {epoch} finished with average loss: {sum(losses) / len(losses)}', 'info')
    
    def calculate_loss(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(self.model.device)
        attention_masks = attention_masks.to(self.model.device)
        labels = labels.to(self.model.device)

        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
        )[0]
        return loss
            

    def _print(self, msg: str, level: str = None):
        if self.logger is None:
            print(msg)
        else:
            getattr(self.logger, level)(msg)

    def test(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def collate_fn(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chats_strings = [
            self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": f"{question}"},
                    {"role": "assistant", "content": f"{answer}"},
                ],
                tokenize=False
            )
            for question, answer in batch
        ]
        outputs = self.tokenizer(
            text=chats_strings,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids = outputs.input_ids
        attention_masks = outputs.attention_mask

        formatted_questions_ids = [
            self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": f"{question}"}
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            for question, _ in batch
        ]
        labels = input_ids.clone()
        for idx, ids in enumerate(formatted_questions_ids):
            labels[idx, :len(ids)] = -100
        return input_ids, attention_masks, labels
        
        

