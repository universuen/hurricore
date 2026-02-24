import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from torch.utils.data import Dataset

from hurricore.trainers import LLMGRPOTrainer
from hurricore.collators import VLMRLCollator
from hurricore.utils import split_dataset


EXPERIMENT_NAME = "rl_vlm"
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
LEARNING_RATE = 1e-6
GRADIENT_ACCUMULATION_STEPS = 1


class RandomColorDataset(Dataset):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> dict:
        random_color = random.choice(["red", "green", "blue"])
        conversation = [
            {
                "role": "system", 
                "content": (
                    "Your task is to answer the question directly.\n"
                    "E.g.,\n"
                    "user: What color is this image?\n"
                    "assistant: The color is red\n"
                )
            },
            {
                "role": "user", 
                "content": [
                    Image.new("RGB", (224, 224), random_color),
                    "What color is this image?"
                ],
            },
        ]
        return {
            "conversation": conversation,
            "target": random_color,
        }


class RandomColorRLTrainer(LLMGRPOTrainer):
    def compute_reward(self, completion: str, target: str) -> float:
        if target in completion:
            return 1.0 - 1e-2 * len(completion)
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
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        dataset = RandomColorDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
        
    trainer = RandomColorRLTrainer(
        experiment_name=EXPERIMENT_NAME,
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=NUM_EPOCHS,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model=model,
        tokenizer=processor.tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=VLMRLCollator(processor=processor),
        use_vllm=True,
    )
    trainer.run()


if __name__ == "__main__":
    main()
