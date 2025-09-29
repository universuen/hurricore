import random
from pathlib import Path

from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM

from hurricore.trainers import LLMTrainer
from hurricore.utils import LLMSFTCollator, split_dataset


class RandomNameConversationDataset:
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
        dataset = RandomNameConversationDataset(num_samples=10000)
        training_dataset, validation_dataset = split_dataset(
            dataset=dataset,
            training_ratio=0.9,
        )
    
    trainer = LLMTrainer(
        experiment_name="sft_llm",
        data_folder_path=Path(__file__).parent / "data",
        num_epochs=3,
        learning_rate=1e-5,
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
