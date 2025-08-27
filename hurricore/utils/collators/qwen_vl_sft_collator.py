from __future__ import annotations

from warnings import warn

from transformers import Qwen2_5_VLProcessor, BatchFeature
from PIL import Image
from qwen_vl_utils import process_vision_info

from hurricore.utils.collators.llm_sft_collator import LLMSFTCollator


class QwenVLSFTCollator(LLMSFTCollator):
    def __init__(
            self, 
            processor: Qwen2_5_VLProcessor,
            max_length: int = 32 * 1024,
            prompt_prefix: str = "<|im_start|>user",
            response_prefix: str = "<|im_start|>assistant",
            generation_mode: bool = False,
            tokenize: bool = True,
            ignore_warnings: bool = False,
            attach_labels: bool = True,
        ) -> None:
        self.processor = processor
        super().__init__(
            tokenizer=processor.tokenizer,
            max_length=max_length,
            prompt_prefix=prompt_prefix,
            response_prefix=response_prefix,
            generation_mode=generation_mode,
            tokenize=tokenize,
            ignore_warnings=ignore_warnings,
            attach_labels=attach_labels,
        )


    def get_raw_model_inputs(self, conversations: list[list[dict]]) -> dict:
        formatted_conversations = []
        for conversation in conversations:
            formatted_conversation = self._format_conversation(conversation)
            formatted_conversations.append(formatted_conversation)
        text = self.processor.apply_chat_template(
            formatted_conversations,
            tokenize=False, 
            add_generation_prompt=self.generation_mode,
        )
        image_inputs, video_inputs = process_vision_info(formatted_conversations)
        return {
            "text_inputs": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
        }


    def __call__(
            self, 
            conversations: list[list[dict]],
            tokenize: bool = None,
        ) -> BatchFeature | dict:
        """
        Example:
            conversations = [
                [
                    {"role": "user", "content": [Image.open("dog.jpg"), whats this?]},
                    {"role": "assistant", "content": ["A dog."]},
                    {"role": "user", "content": ["What's your name?"]}
                    {"role": "assistant", "content": ["I don't want to tell you."]},
                ],
                [
                    {"role": "user", "content": "What's your name?"},
                    {"role": "assistant", "content": "I don't want to tell you."},
                ],
            ]
            collator = LLMSFTCollator(tokenizer=tokenizer)
            model_inputs = collator(conversations)
        """

        if self.generation_mode and len(conversations) > 1:
            warn(
                "You are using generation mode of an LLM with batch size > 1, which may lead to unexpected results.\n"
                "Ref: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535"
            )

        raw_model_inputs = self.get_raw_model_inputs(conversations)
        tokenize = self.tokenize if tokenize is None else tokenize
        if not tokenize:
            return raw_model_inputs
        model_inputs = self.processor(
            text=raw_model_inputs["text_inputs"], 
            images=raw_model_inputs["image_inputs"],
            videos=raw_model_inputs["video_inputs"],
            padding='longest', 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True,
            padding_side="left" if self.generation_mode else "right",
        )
        if not self.generation_mode and self.attach_labels:
            model_inputs = self._attach_labels(model_inputs)
        return model_inputs


    def _format_conversation(self, conversation: dict) -> list[dict]:
        if self.generation_mode:
            # Remove the last message if it's from the assistant
            if conversation[-1]["role"] == "assistant":
                conversation = conversation[:-1]

        formatted_conversation = []
        for message in conversation:
            formatted_conversation.append({"role": message["role"], "content": self.format_content(message["content"])})
            
        return formatted_conversation
    

    @staticmethod
    def format_content(
            content_sequence: list | str | Image.Image,
        ) -> list[dict]:
        if not isinstance(content_sequence, list):
            content_sequence = [content_sequence]

        formal_content_sequence = []
        for item in content_sequence:
            if isinstance(item, dict):
                formal_content_sequence.append(item)
            elif isinstance(item, str):
                formal_content_sequence.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                formal_content_sequence.append({"type": "image", "image": item})
            else:
                raise TypeError(f"Invalid user part item type: {type(item)}, must be str or Image.Image.")
        return formal_content_sequence
