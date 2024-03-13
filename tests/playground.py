import context


from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
chats_strings = [
            tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": f"{question}"},
                    {"role": "assistant", "content": f"{answer}"},
                ],
                tokenize=False
            )
            for question, answer in [
                ('Hello', "Thank you"),
                ('Hello Hello', "Thank you Thank you"),
                ('Wow', 'Fantastic'),
            ]
]
outputs = tokenizer(
            text=chats_strings,
            padding=True
)
pass