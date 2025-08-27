from __future__ import annotations

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    Constrained logits processor to filter out the logits of the tokens that are not allowed.
    """
    def __init__(
            self, 
            valid_answers: list[str], 
            tokenizer: PreTrainedTokenizer,
        ) -> None:
        assert tokenizer.eos_token_id is not None, "tokenizer must have an eos_token_id"
        self.valid_answers = valid_answers
        self.allowed_token_ids_list = [
            tokenizer.encode(word) + [tokenizer.eos_token_id]
            for word in valid_answers
        ]
        self.tokenizer = tokenizer
        self.prompt_lengths = []
        self.rules = self._generate_rules()


    def _generate_rules(self) -> dict:
        rules = {str([]): list(set([i[0] for i in self.allowed_token_ids_list]))} 
        for token_ids in self.allowed_token_ids_list:
            prefix = []
            for token_id, next_token_id in zip(token_ids, token_ids[1:]):
                prefix.append(token_id)
                prefix_str = str(prefix)
                if prefix_str not in rules:
                    rules[prefix_str] = set([next_token_id])
                else:
                    rules[prefix_str].add(next_token_id)
        rules = {k: list(v) for k, v in rules.items()}
        return rules


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(self.prompt_lengths) == 0:
            self.prompt_lengths = [len(i) for i in input_ids]

        scores_mask = torch.full_like(scores, -float("inf"))

        for i, seq in enumerate(input_ids.tolist()):
            prefix = seq[self.prompt_lengths[i]:]
            prefix_str = str(prefix)
            valid_next_token_ids = self.rules.get(prefix_str, [self.tokenizer.eos_token_id])
            scores_mask[i, valid_next_token_ids] = 0

        return scores + scores_mask


    def reset(self):
        self.prompt_lengths = []
