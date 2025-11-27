from __future__ import annotations

from typing import Any

from hurricore.collators.llm_sft_collator import LLMSFTCollator


class LLMGRPOCollator(LLMSFTCollator):

    def __call__(
            self, 
            conversations: list[list[dict]],
            targets: list[Any],
        ) -> dict:
        assert len(conversations) == len(targets), "Conversations and targets must have the same number of elements."
        torch_model_inputs = super().__call__(conversations)
        vllm_conversations = self.format_conversations(conversations)
        return {
            "torch_model_inputs": torch_model_inputs,
            "vllm_conversations": vllm_conversations,
            "targets": targets,
        }
