import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM

from hurricore.trainers import LLMGRPOTrainer
from hurricore.utils import LLMSFTCollator, split_dataset


class RandomMathDataset:
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> dict:
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operation = random.choice(["+", "-", "*", "//"])
        result = eval(f"{a} {operation} {b}")
        return {
            "question": f"What is {a} {operation} {b}?",
            "answer": int(result),
        }


class RandomMathGRPOTrainer(LLMGRPOTrainer):
    def compute_reward(self, completion: str, target: int) -> float:
        try:
            return 200 - abs(int(completion) - target)
        except ValueError:
            return -100


class RandomMathCollator(LLMSFTCollator):
    def __call__(self, batch: list[dict]) -> dict:
        conversations = []
        targets = []
        for item in batch:
            conversation = [
                {
                    "role": "system", 
                    "content": (
                        "Your task is to answer the question directly.\n"
                        "E.g.,\n"
                        "user: What is 1 + 1?\n"
                        "assistant: 2\n"
                    )
                },
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": str(item["answer"])},
            ]
            conversations.append(conversation)
            targets.append(item["answer"])

        return {
            "vllm_prompts": super().__call__(conversations, tokenize=False),
            "torch_model_inputs": super().__call__(conversations),
            "targets": targets,
        }


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        deepspeed_plugin=DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            zero_stage=1,
        )
    )

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        dataset = RandomMathDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = RandomMathGRPOTrainer(
        experiment_name="grpo_llm",
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=3,
        learning_rate=1e-6,
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=RandomMathCollator(
            tokenizer=tokenizer,
            generation_mode=True,
        ),
    )
    trainer.run()


if __name__ == "__main__":
    main()
