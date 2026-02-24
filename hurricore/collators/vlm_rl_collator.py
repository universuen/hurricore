from transformers import ProcessorMixin

from hurricore.collators.vlm_sft_collator import VLMSFTCollator


class VLMRLCollator(VLMSFTCollator):

    def __init__(
        self,
        processor: ProcessorMixin,
        **kwargs,
    ):
        super().__init__(
            processor=processor,
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
                    {"role": "user", "content": [<A red image>, "What is the color of this image? Think step by step."]}
                ],
                "target": "red",
            },
            {
                "conversation": [
                    {"role": "user", "content": [<A blue image>, "What is the color of this image? Think step by step."]}
                ],
                "target": "blue",
            }
        ]
        """
        assert "conversation" in samples[0], "`conversation` must be provided for each sample."
        assert "target" in samples[0], "`target` must be provided for each sample."
        conversations = [sample["conversation"] for sample in samples]
        targets = [sample["target"] for sample in samples]
        assert len(conversations) == len(targets), "Conversations and targets must have the same number of elements."
        torch_model_inputs = super().__call__(conversations)
        formatted_conversations = self.format_conversations(conversations)
        vllm_conversations = self.convert_conversations_to_vllm_format(formatted_conversations)
        return {
            "torch_model_inputs": torch_model_inputs,
            "vllm_conversations": vllm_conversations,
            "targets": targets,
        }

    @staticmethod
    def convert_conversations_to_vllm_format(formatted_conversations: list) -> list:
        for conversation in formatted_conversations:
            for message in conversation:
                for item in message["content"]:
                    if item["type"] == "image":
                        item["type"] = "image_pil"
                        item["image_pil"] = item["image"]
        return formatted_conversations
