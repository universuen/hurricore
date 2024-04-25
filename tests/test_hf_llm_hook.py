import _path_setup  # noqa: F401

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator

from hurricore.trainers import Trainer
from hurricore.hooks import HFLLMPeekHook, LoggerHook


class _TestTrainer(Trainer):
    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained("nickypro/tinyllama-15M")
        super().__init__(
            models=[AutoModelForCausalLM.from_pretrained('gpt2')],
            optimizers=[AdamW(model.parameters(), lr=1e-3)],
            data_loaders=[DataLoader(range(1), batch_size=1, shuffle=True)],
            accelerator=Accelerator(),
            num_epochs=1,
        )
        self.hooks = [
            HFLLMPeekHook(
                self,
                prompts=[
                    'Hi?',
                    'How are you?',
                    'What is your name?',
                ],
                tokenizer=AutoTokenizer.from_pretrained("nickypro/tinyllama-15M"),
                interval=1,
            )
        ]
        LoggerHook.msg_queue = []
    
    
    def compute_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, requires_grad=True)


def test_hf_llm_hook():
    trainer = _TestTrainer()
    trainer.run()
    assert len(LoggerHook.msg_queue) == 3, "Not all prompts are logged."
