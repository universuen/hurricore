import torch
from transformers import ProcessorMixin
from PIL import Image

from hurricore.collators.llm_sft_collator import LLMSFTCollator


class VLMSFTCollator(LLMSFTCollator):
    def __init__(
        self,
        processor: ProcessorMixin,
        user_prefix: str = "<|im_start|>user",
        assistant_prefix: str = "<|im_start|>assistant",
        user_prefix_ids: list[int] = None,
        assistant_prefix_ids: list[int] = None,
        generation_mode: bool = False,
        attach_labels: bool = True,
        inject_dummy_image_when_empty: bool = True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=processor.tokenizer,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            user_prefix_ids=user_prefix_ids,
            assistant_prefix_ids=assistant_prefix_ids,
            generation_mode=generation_mode,
            attach_labels=attach_labels,
            **kwargs,
        )
        self.processor = processor
        self.inject_dummy_image_when_empty = inject_dummy_image_when_empty


    def __call__(self, conversations: list[list[dict]]) -> dict:
        """
        Expects a list of conversations.
        E.g.
        [
            [   
                {"role": "user", "content": [<A red image>, "What color is this image?"]},
                {"role": "assistant", "content": "The image is red."},
            ],
            [   
                {"role": "user", "content": [<A blue image>, "What color is this image?"]},
                {"role": "assistant", "content": "The image is blue."},
            ],
            ...
        ]
        """
        formatted_conversations = self.format_conversations(conversations)
        model_inputs = self.processor.apply_chat_template(
            formatted_conversations,
            tokenize=True,
            add_generation_prompt=self.generation_mode,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left" if self.generation_mode else "right",
            **self.kwargs,
        )
        if self.attach_labels and not self.generation_mode:
            model_inputs = self._attach_labels(model_inputs)
        if (
            self.inject_dummy_image_when_empty 
            and "labels" in model_inputs
            and "pixel_values" not in model_inputs
        ):
            model_inputs = self._inject_dummy_data(model_inputs)
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


    def _inject_dummy_data(self, model_inputs: dict) -> dict:
        """
        Injects a dummy image into the batch to prevent distributed training hangs.
        Extends the input matrix columns to accommodate the dummy tokens.
        """
        dummy_image = Image.new("RGB", (16, 16), (255, 255, 255))
        dummy_conv = [[{"role": "user", "content": [dummy_image]}]]
        
        dummy_batch = self.processor.apply_chat_template(
            self.format_conversations(dummy_conv),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        for key, value in dummy_batch.items():
            if key not in model_inputs and key not in ["input_ids", "attention_mask", "labels"]:
                model_inputs[key] = value

        batch_size = model_inputs["input_ids"].shape[0]
        dummy_ids = dummy_batch["input_ids"][0] # Shape: (dummy_len,)
        dummy_len = dummy_ids.shape[0]
        device = model_inputs["input_ids"].device

        # Pad input_ids
        pad_block = torch.full(
            (batch_size, dummy_len), 
            self.tokenizer.pad_token_id, 
            dtype=model_inputs["input_ids"].dtype,
            device=device
        )
        pad_block[0] = dummy_ids.to(device)
        model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], pad_block], dim=1)

        # Pad attention_mask
        mask_block = torch.zeros(
            (batch_size, dummy_len), 
            dtype=model_inputs["attention_mask"].dtype,
            device=device
        )
        model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], mask_block], dim=1)

        # Pad labels
        label_block = torch.full(
            (batch_size, dummy_len), 
            -100, 
            dtype=model_inputs["labels"].dtype,
            device=device
        )
        model_inputs["labels"] = torch.cat([model_inputs["labels"], label_block], dim=1)

        return model_inputs