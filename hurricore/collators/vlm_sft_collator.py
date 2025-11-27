from transformers import ProcessorMixin
from PIL import Image

from hurricore.collators.llm_sft_collator import LLMSFTCollator


class VLMSFTCollator(LLMSFTCollator):
    def __init__(
        self,
        processor: ProcessorMixin,
        user_prefix: str = "<|im_start|>user",
        assistant_prefix: str = "<|im_start|>assistant",
        generation_mode: bool = False,
        attach_labels: bool = True,
    ):
        super().__init__(
            tokenizer=processor.tokenizer,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            generation_mode=generation_mode,
            attach_labels=attach_labels,
        )
        self.processor = processor


    def __call__(self, conversations: list[list[dict]]) -> dict:
        formatted_conversations = self.format_conversations(conversations)
        model_inputs = self.processor.apply_chat_template(
            formatted_conversations,
            tokenize=True,
            add_generation_prompt= self.generation_mode,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left" if self.generation_mode else "right",
        )
        if self.attach_labels and not self.generation_mode:
            model_inputs = self._attach_labels(model_inputs)
        return model_inputs


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
                formatted_content.append({"type": "image", "image": item})
            else:
                raise ValueError(f"Content item must be a dictionary, string or PIL image, got {type(item)}")
        return formatted_content
