import re
import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

from hurricore.trainers import LLMGRPOTrainer
from hurricore.collators import LLMRLCollator
from hurricore.utils import split_dataset


EXPERIMENT_NAME = "rl_llm"
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
LEARNING_RATE = 1e-6
GRADIENT_ACCUMULATION_STEPS = 1


class RandomMathDataset(Dataset):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> dict:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        operation = random.choice(["+", "-", "*", "//"])
        result = eval(f"{a} {operation} {b}")
        question = f"What is {a} {operation} {b}?"
        answer = str(result)
        conversation = [
            {
                "role": "system", 
                "content": (
                    "Your task is to answer the question directly.\n"
                    "E.g.,\n"
                    "user: What is 1 + 1?\n"
                    "assistant: <answer>2</answer>\n"
                )
            },
            {"role": "user", "content": question},
        ]
        return {
            "conversation": conversation,
            "target": answer,
        }


class RandomMathRLTrainer(LLMGRPOTrainer):
    def compute_reward(self, completion: str, target: str) -> float:
        base_reward = 1000
        try:
            answer = re.findall(r"<answer>(.*?)</answer>", completion)[0]
            penalty = abs(int(answer) - int(target))
            return base_reward - penalty
        except Exception as e:
            # print(e)
            return -100


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
        deepspeed_plugin=DeepSpeedPlugin(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            zero_stage=1,
        )
    )

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        dataset = RandomMathDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = RandomMathRLTrainer(
        experiment_name=EXPERIMENT_NAME,
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=NUM_EPOCHS,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=LLMRLCollator(
            tokenizer=tokenizer,
            enable_thinking=True,
        ),
        use_vllm=True,
    )
    trainer.run()


if __name__ == "__main__":
    main()
