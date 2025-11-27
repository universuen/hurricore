from typing import Any

from PIL import Image

from hurricore.collators.vlm_sft_collator import VLMSFTCollator


class VLMGRPOCollator(VLMSFTCollator):

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


    def format_content(self, content: str) -> str:
        if type(content) is not list:
            content = [content]
        formatted_content = []
        for item in content:
            if type(item) is dict:
                assert "type" in item.keys(), f"Content item must have 'type' key, got {item.keys()}"
                formatted_content.append(item)
            elif type(item) is str:
                formatted_content.append({"type": "text", "text": item})
            elif type(item) is Image.Image:
                formatted_content.append({"type": "image_pil", "image_pil": item})
            else:
                raise ValueError(f"Content item must be a dictionary, string or PIL image, got {type(item)}")
        return formatted_content
