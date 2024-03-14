import os
from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import memory_reserved
from transformers import PreTrainedModel, PreTrainedTokenizer


class SFTRunner:
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
    
    def train(self, **training_configs: dict) -> None:
        epochs = training_configs.get('epochs', 1)
        batch_size = training_configs.get('batch_size', 1)
        max_len = training_configs.get('max_len', 512)
        gradient_accumulation_steps = training_configs.get('gradient_accumulation_steps', 1)
        enable_gradient_checkpointing = training_configs.get('enable_gradient_checkpointing', True)
        test_prompt = training_configs.get('test_prompt', None)
        data_loader = DataLoader(
            dataset=self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )
        
        self.env['max_len'] = max_len

        if hasattr(self.model, 'gradient_checkpointing_enable') \
        and enable_gradient_checkpointing:
            self.env['use_cache'] = False
            self.model.gradient_checkpointing_enable()
            self._print('Gradient Checkpointing enabled')

        self.optimizer.zero_grad()

        for epoch in range(1, epochs + 1):
            losses = []
            self._print(f'Epoch {epoch} started')

            for batch_idx, batch in enumerate(data_loader):
                loss = self.calculate_loss(batch)
                loss = loss / gradient_accumulation_steps 
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 \
                or (batch_idx + 1) == len(data_loader):

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    loss_value = loss.item() * gradient_accumulation_steps
                    self._print(f'Current loss: {loss_value: .5f}')
                    losses.append(loss_value) 

                    if self.model.device != 'cpu':
                        memory_used = memory_reserved(self.model.device) / 1024 ** 3
                        self._print(f'Memory used: {memory_used:.2f}GB')
                    
                    if test_prompt is not None:
                        self._print(f'Test prompt: {test_prompt}. Result: {self.test(test_prompt)}')
                    

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = sum(losses) / len(losses)
            self._print(f'Epoch {epoch} finished with average loss: {avg_loss}')
    
    def calculate_loss(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.model.train()

        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(self.model.device)
        attention_masks = attention_masks.to(self.model.device)
        labels = labels.to(self.model.device)

        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            use_cache=self.env.get('use_cache', True),
        )[0]

        return loss

    def _print(self, msg: str, level: str = 'info'):
        if self.logger is None:
            print(msg)
        else:
            getattr(self.logger, level)(msg)

    @torch.no_grad()
    def test(self, prompt: str) -> str:
        self.model.eval()
        input_ids = self.tokenizer.apply_chat_template(
            conversation=[
                {"role": "user", "content": f"{prompt}"}
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.model.device)
        outputs = self.model.generate(input_ids, max_new_tokens=100)
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
            padding='longest',
            add_special_tokens=False,
            return_tensors='pt',
            max_length=self.env['max_len'],
            truncation=True,
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
                return_tensors='pt',
            )
            for question, _ in batch
        ]
        labels = input_ids.clone()
        for i, (label, question_id) in enumerate(zip(labels, formatted_questions_ids)):
            question_id = question_id.squeeze()
            question_start_idx_in_label = self._find_start_index(label, question_id)
            if question_start_idx_in_label == -1:
                self._print(
                    f'Failed to match a question in a lable!\n'
                    f'question_text:{batch[i][0]}\n'
                    f'question_id:{question_id}\n'
                    f'chat_string:{chats_strings[i]}\n'
                    f'chat_label:{label}\n'
                    f'Skipped!'
                )
                continue
            question_end_idx_in_label = question_start_idx_in_label + len(question_id)
            labels[i][question_start_idx_in_label:question_end_idx_in_label] = -100
        return input_ids, attention_masks, labels
    
    def _find_start_index(self, a: torch.Tensor, b: torch.Tensor) -> int:
        for i in range(len(a) - len(b) + 1):
            if torch.all(a[i:i + len(b)] == b):
                return i
        return -1
        

