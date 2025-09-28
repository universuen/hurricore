from __future__ import annotations

from warnings import warn

from transformers import PreTrainedTokenizer, BatchEncoding

from hurricore.utils import find_all_sublist_indices


class LLMSFTCollator:
    def __init__(
            self, 
            tokenizer: PreTrainedTokenizer,
            max_length: int = 32 * 1024,
            prompt_prefix: str = "<|im_start|>user",
            response_prefix: str = "<|im_start|>assistant",
            generation_mode: bool = False,
            ignore_warnings: bool = False,
            attach_labels: bool = True,
        ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix
        self.response_prefix = response_prefix
        self.generation_mode = generation_mode
        self.ignore_warnings = ignore_warnings
        self.attach_labels = attach_labels


    def __call__(self, conversations: list[list[dict]], tokenize: bool = True) -> BatchEncoding | list[str]:
        """
        Example:
            conversations = [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine."},
                ],
                [
                    {"role": "system", "content": "You are a secret keeper."},
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

        formatted_conversations = []
        for conversation in conversations:
            formatted_conversation = self._format_conversation(conversation)
            formatted_conversations.append(formatted_conversation)
        text_inputs = self.tokenizer.apply_chat_template(
            formatted_conversations,
            tokenize=False, 
            add_generation_prompt=self.generation_mode,
        )
        if not tokenize:
            return text_inputs
        
        model_inputs = self.tokenizer(
            text=text_inputs, 
            padding='longest', 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True,
            padding_side="left" if self.generation_mode else "right",
        )
        if not self.generation_mode and self.attach_labels:
            model_inputs = self._attach_labels(model_inputs)
        return model_inputs


    def _attach_labels(self, model_inputs: BatchEncoding) -> BatchEncoding:
        labels = model_inputs["input_ids"].clone()
        prompt_prefix_ids = self.tokenizer.encode(self.prompt_prefix)
        response_prefix_ids = self.tokenizer.encode(self.response_prefix)

        for i, input_ids in enumerate(model_inputs["input_ids"]):
            all_prompt_start_indices = find_all_sublist_indices(input_ids.tolist(), prompt_prefix_ids)
            all_response_start_indices = find_all_sublist_indices(input_ids.tolist(), response_prefix_ids)

            if len(all_prompt_start_indices) == 0:
                if not self.ignore_warnings:
                    warn("No prompt prefix found in input_ids.")
            if len(all_response_start_indices) == 0:
                if not self.ignore_warnings:
                    warn("No response prefix found in input_ids.")
                continue
            
            labels[i, :all_response_start_indices[0] + len(response_prefix_ids)] = -100
            for prompt_start_index, response_start_index in zip(all_prompt_start_indices, all_response_start_indices):
                labels[i, prompt_start_index:response_start_index + len(response_prefix_ids)] = -100
            
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
    

    def _format_conversation(self, conversation: list[dict]) -> list[dict]:
        if self.generation_mode:
            # Remove the last message if it's from the assistant
            if conversation[-1]["role"] == "assistant":
                conversation = conversation[:-1]
        return conversation
