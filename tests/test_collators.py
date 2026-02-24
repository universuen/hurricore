import os

from transformers import AutoTokenizer, Qwen2_5_VLProcessor
from PIL import Image

from hurricore.collators import LLMSFTCollator, VLMSFTCollator


def test_llm_sft_collator():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    collator = LLMSFTCollator(tokenizer=tokenizer)
    conversations = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ],
        [
            {"role": "user", "content": "What is the capital of Germany?"},
            {"role": "assistant", "content": "Berlin"},
        ],
        [
            {"role": "user", "content": "窗前明月光，疑是地上霜。举头望明月，低头思故乡。"},
            {"role": "assistant", "content": "李白"},
        ],
    ]
    model_inputs = collator(conversations)
    print("Training text:")
    for i in tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True):
        print(i)
    model_inputs["labels"][model_inputs["labels"] == -100] = tokenizer.pad_token_id
    print("Training label:")
    for i in tokenizer.batch_decode(model_inputs["labels"], skip_special_tokens=True):
        print(i)

    collator.training_mode = False
    model_inputs = collator(conversations)
    print("Evaluation text:")
    for i in tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True):
        print(i)


def test_qwen_vl_sft_collator():
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    collator = VLMSFTCollator(processor=processor)
    conversations = [
        [
            {"role": "user", "content": [Image.new("RGB", (100, 100)), Image.new("RGB", (200, 200)), "What is this?"]},
            {"role": "assistant", "content": "A dog"},
        ],
        [
            {"role": "user", "content": [Image.new("RGB", (300, 300)), "What is this?"]},
            {"role": "assistant", "content": "A cat"},
        ],
    ]
    tokenizer = processor.tokenizer
    model_inputs = collator(conversations)
    print("Training text:")
    for i in tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True):
        print(i)
    model_inputs["labels"][model_inputs["labels"] == -100] = tokenizer.pad_token_id
    print("Training label:")
    for i in tokenizer.batch_decode(model_inputs["labels"], skip_special_tokens=True):
        print(i)

    collator.training_mode = False
    model_inputs = collator(conversations)
    print("Evaluation text:")
    for i in tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True):
        print(i)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_llm_sft_collator()
    test_qwen_vl_sft_collator()
