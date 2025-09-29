import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

from hurricore.trainers import LLMTrainer
from hurricore.utils import QwenVLSFTCollator, split_dataset


class RandomColorConversationDataset:
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> list[dict]:
        random_color = random.choice(["red", "green", "blue"])
        conversation = [
            {
                "role": "user", 
                "content": [
                    Image.new("RGB", (224, 224), random_color),
                    "What color is this image?"
                ],
            },
            {
                "role": "assistant", 
                "content": random_color,
            },
        ]
        return conversation


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
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        dataset = RandomColorConversationDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = LLMTrainer(
        experiment_name="sft_mllm",
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=3,
        learning_rate=1e-5,
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=QwenVLSFTCollator(
            processor=processor,
        ),
    )
    trainer.run()


if __name__ == "__main__":
    main()
