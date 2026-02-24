from transformers import PreTrainedTokenizer

from hurricore.collators.llm_sft_collator import LLMSFTCollator


class LLMRLCollator(LLMSFTCollator):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            generation_mode=True,
            attach_labels=False,
            **kwargs,
        )


    def __call__(
            self, 
            samples: list[dict],
        ) -> dict:
        """
        Expects a list of samples, each containing a conversation and a target.
        E.g.
        [
            {
                "conversation": [
                    {"role": "user", "content": "What is 1 + 1? Think step by step."},
                ],
                "target": 2,
            },
            {
                "conversation": [
                    {"role": "user", "content": "What is 1 + 2? Think step by step."},
                ],
                "target": 3,
            },
        ]
        """
        assert "conversation" in samples[0], "`conversation` must be provided for each sample."
        assert "target" in samples[0], "`target` must be provided for each sample."
        conversations = [sample["conversation"] for sample in samples]
        targets = [sample["target"] for sample in samples]
        assert len(conversations) == len(targets), "Conversations and targets must have the same number of elements."
        torch_model_inputs = super().__call__(conversations)
        vllm_conversations = self.format_conversations(conversations)
        return {
            "torch_model_inputs": torch_model_inputs,
            "vllm_conversations": vllm_conversations,
            "targets": targets,
        }
