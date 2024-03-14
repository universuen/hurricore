import os
import time
from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator
from torch.cuda import memory_cached

from src.utils import get_list_mean


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

        self.accelerator = Accelerator(split_batches=True)
        model, optimizer, scheduler = self.accelerator.prepare(model, optimizer, scheduler)

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
    
    def _should_record(self):
        return self.accelerator.is_main_process \
                and (
                    self._batch_idx % 10 == 0 \
                    or self._batch_idx == self._num_batches
                )

    def _should_test(self):
        return self.accelerator.is_main_process and self._test_prompt is not None

    def _record(self):
        progress = (self._batch_idx / self._num_batches)
        elapsed_time = time.time() - self._start_time
        remaining_time = (elapsed_time / self._batch_idx) * (self._num_batches - self._batch_idx)
        formatted_remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
        self._print(
            f"Epoch: {self._epoch}/{self._epochs} | "
            f"Step: {self._batch_idx}/{self._num_batches} | "
            f"Loss: {self._step_loss:.5f} | "
            f"Progress: {progress:.2%} | "
            f"Time left: {formatted_remaining_time} | "
            f"Memory cached: {memory_cached() / 1024 ** 3:.2f}GB"
        )

    def train(
        self, 
        epochs = 1,
        batch_size = 8,
        max_len = 512,
        test_prompt: str = None,
    ) -> None:
        ### TODO: data_loader should be passed to init by user
        data_loader = DataLoader(
            dataset=self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=os.cpu_count(),
            collate_fn=self.collate_fn,
        )
        data_loader = self.accelerator.prepare(data_loader)

        self._test_prompt = test_prompt
        self._num_batches = len(data_loader)
        self._epochs = epochs
        ### TODO: Decouple max_len and collate_fn
        self._max_len = max_len

        self.optimizer.zero_grad()

        for epoch in range(1, epochs + 1):

            self._epoch = epoch
            self._losses_per_batch = []
            self._print(f'Epoch {epoch} started')
            self._start_time = time.time()

            for batch_idx, batch in enumerate(data_loader, start=1):

                self._batch_idx = batch_idx
                loss = self.compute_loss(batch)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                step_loss = self.accelerator.gather(loss).detach().mean().item()

                if self.accelerator.is_main_process:
                    self._losses_per_batch.append(step_loss)
                    self._step_loss = step_loss
                
                if self._should_record():
                    self._record()
                if self._should_test():
                    self._print(f'Test prompt: {test_prompt}. Result: {self.test(test_prompt)}')
                    
            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = get_list_mean(self._losses_per_batch)
            self._print(f'Epoch {epoch} finished with average loss: {avg_loss}')

    
    def compute_loss(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.model.train()

        input_ids, attention_masks, labels = batch

        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            use_cache=False,
        )[0]

        return loss

    def _print(self, msg: str, level: str = 'info') -> None:
        if not self.accelerator.is_main_process:
            return
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
            max_length=self._max_len,
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
    
    @staticmethod
    def _find_start_index( a: torch.Tensor, b: torch.Tensor) -> int:
        for i in range(len(a) - len(b) + 1):
            if torch.all(a[i:i + len(b)] == b):
                return i
        return -1
        
