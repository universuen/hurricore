from logging import Logger

import torch
from transformers import PreTrainedTokenizer


class HFITCollator:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        max_len=512, 
        logger: Logger=None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.logger = logger

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chats_strings = [
            self.tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": f"{question}"},
                    {"role": "assistant", "content": f"{answer}"}
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
            max_length=self.max_len,
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
            start_idx, end_idx = self._find_start_and_end_index(label, question_id)
            if start_idx == -1:
                print(
                    f'Failed to match a question in a label!\n'
                    f'question_text:{batch[i][0]}\n'
                    f'question_id:{question_id}\n'
                    f'chat_string:{chats_strings[i]}\n'
                    f'chat_label:{label}\n'
                    f'Skipped!'
                )
                continue
            labels[i][start_idx:end_idx] = -100

        return input_ids, attention_masks, labels

    @staticmethod
    def _find_start_and_end_index(a: torch.Tensor, b: torch.Tensor) -> int:
        for i in range(len(a) - len(b) + 1):
            if torch.all(a[i:i + len(b)] == b):
                return i, i + len(b)
        return -1, -1
