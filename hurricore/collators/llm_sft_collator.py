from warnings import warn

from transformers import PreTrainedTokenizer

from hurricore.utils import find_all_sublist_indices


class LLMSFTCollator:
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        user_prefix: str = "<|im_start|>user",
        assistant_prefix: str = "<|im_start|>assistant",
        user_prefix_ids: list[int] = None,
        assistant_prefix_ids: list[int] = None,
        generation_mode: bool = False,
        attach_labels: bool = True,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.user_prefix_ids = user_prefix_ids
        self.assistant_prefix_ids = assistant_prefix_ids
        self.generation_mode = generation_mode
        self.attach_labels = attach_labels
        self.kwargs = kwargs
    

    def __call__(self, conversations: list[list[dict]]) -> dict:
        """
        Expects a list of conversations.
        E.g.
        [
            [   
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm fine."},
            ],
            [   
                {"role": "user", "content": "What's the weather like today?"},
                {"role": "assistant", "content": "It's sunny."},
            ],
            ...
        ]
        """
        formatted_conversations = self.format_conversations(conversations)
        model_inputs = self.tokenizer.apply_chat_template(
            formatted_conversations,
            tokenize=True,
            add_generation_prompt= self.generation_mode,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            tokenizer_kwargs={
                "padding_side": "left" if self.generation_mode else "right",
            },
            **self.kwargs,
        )
        if self.attach_labels and not self.generation_mode:
            model_inputs = self._attach_labels(model_inputs)
        return model_inputs
    

    def format_conversations(self, conversations: list[list[dict]]) -> list[str]:
        return [self.format_conversation(conversation) for conversation in conversations]

    
    def format_conversation(self, conversation: list[dict]) -> str:
        if self.generation_mode:
            # Remove the last message if it's from the assistant
            if conversation[-1]["role"] == "assistant":
                conversation = conversation[:-1]
        for message in conversation:
            assert type(message) is dict, f"Message must be a dictionary, got {type(message)}"
            assert message.keys() >= {"role", "content"}, f"Message must have 'role' and 'content' keys, got {message.keys()}"
            message["content"] = self.format_content(message["content"])
        return conversation


    def format_content(self, content: str) -> str:
        return content
    

    def _attach_labels(self, model_inputs: dict) -> dict:
        labels = model_inputs["input_ids"].clone()
        user_prefix_ids = self.user_prefix_ids or self.tokenizer.encode(self.user_prefix, add_special_tokens=False)
        assistant_prefix_ids = self.assistant_prefix_ids or self.tokenizer.encode(self.assistant_prefix, add_special_tokens=False)

        for i, input_ids in enumerate(model_inputs["input_ids"]):
            all_user_start_indices = find_all_sublist_indices(input_ids.tolist(), user_prefix_ids)
            all_assistant_start_indices = find_all_sublist_indices(input_ids.tolist(), assistant_prefix_ids)

            if len(all_user_start_indices) == 0:
                raise ValueError(f"No user prefix found in input_ids. Raw text: {self.tokenizer.decode(input_ids)}. User prefix: {self.user_prefix}")
            if len(all_assistant_start_indices) == 0:
                raise ValueError(f"No assistant prefix found in input_ids. Raw text: {self.tokenizer.decode(input_ids)}. Assistant prefix: {self.assistant_prefix}")
            
            labels[i, :all_assistant_start_indices[0] + len(assistant_prefix_ids)] = -100
            for user_start_index, assistant_start_index in zip(all_user_start_indices, all_assistant_start_indices):
                labels[i, user_start_index:assistant_start_index + len(assistant_prefix_ids)] = -100
            
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
