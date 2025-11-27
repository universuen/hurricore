import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from hurricore.trainers import LLMGRPOTrainer
from hurricore.collators import VLMGRPOCollator
from hurricore.utils import split_dataset


class RandomColorDataset:
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> dict:
        random_color = random.choice(["red", "green", "blue"])
        return {
            "question": [
                Image.new("RGB", (224, 224), random_color),
                "What color is this image?"
            ],
            "answer": random_color,
        }


class RandomColorGRPOTrainer(LLMGRPOTrainer):
    def compute_reward(self, completion: str, target: str) -> float:
        if completion == target:
            return 1.0
        return 0.0


class RandomColorCollator(VLMGRPOCollator):
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
                        "user: What color is this image?\n"
                        "assistant: red\n"
                    )
                },
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            conversations.append(conversation)
            targets.append(item["answer"])

        return super().__call__(
            conversations=conversations,
            targets=targets,
        )


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
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        dataset = RandomColorDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = RandomColorGRPOTrainer(
        experiment_name="grpo_mllm",
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=3,
        learning_rate=1e-6,
        model=model,
        tokenizer=processor.tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=RandomColorCollator(
            processor=processor,
            generation_mode=True,
        ),
        use_vllm=True,
    )
    trainer.run()


if __name__ == "__main__":
    main()
