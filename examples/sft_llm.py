import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

from hurricore.trainers import LLMTrainer
from hurricore.collators import LLMSFTCollator
from hurricore.utils import split_dataset


EXPERIMENT_NAME = "sft_llm"
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1


class RandomNameConversationDataset(Dataset):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
  
    def __getitem__(self, idx: int) -> list[dict]:
        random_name = random.choice(["John", "Jane", "Bob", "Alice"])
        conversation = [
            {
                "role": "user", 
                "content": f"Hey {random_name}, how are you?"
            },
            {
                "role": "assistant", 
                "content": f"I'm good, thanks for asking {random_name}!",
            },
        ]
        return conversation


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
        dataset = RandomNameConversationDataset(num_samples=1000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = LLMTrainer(
        experiment_name=EXPERIMENT_NAME,
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=NUM_EPOCHS,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        accelerator=accelerator,
        collate_fn=LLMSFTCollator(
            tokenizer=tokenizer,
        ),
    )
    trainer.run()


if __name__ == "__main__":
    main()
