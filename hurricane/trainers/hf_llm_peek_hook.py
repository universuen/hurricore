import torch
from transformers import PreTrainedTokenizer

from hurricane.hooks import HookBase, LoggerHook
from hurricane.trainers.trainer_base import TrainerBase
from hurricane.utils import is_deepspeed_zero3


class HFLLMPeekHook(HookBase):
    def __init__(
        self, 
        trainer: TrainerBase,
        prompts: list[str] = None, 
        tokenizer: PreTrainedTokenizer = None,
        interval: int = 1,
    ) -> None:
        super().__init__(trainer)
        # check validity
        assert interval > 0, 'Peek interval must be greater than 0.'
        assert prompts is not None and len(prompts) > 0, 'Invalid prompts.'
        assert tokenizer is not None, 'Invalid tokenizer.'
        assert hasattr(trainer, 'accelerator'), 'Trainer must have an accelerator.'
        assert len(trainer.originals.models) == 1, 'Only one model is supported.'
        # setup self
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.interval = interval
    
    def on_training_start(self) -> None:
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        if logger_hook is not None:
            self.logger = logger_hook.logger
            self.log_interval = logger_hook.interval
    
    def on_step_end(self) -> None:
        # peek model results
        conditions = (
            self.trainer.accelerator.is_main_process,
            is_deepspeed_zero3(self.trainer.accelerator),
        )
        if any(conditions):
            idx = self.trainer.ctx.batches_idx
            num_batches = len(self.trainer.data_loader)
            if (self.trainer.ctx.tensor_boards + 1) % self.interval == 0 or idx == num_batches:
                original_model = self.trainer.originals.models[0]
                original_model.eval()
                answers = []
                with torch.no_grad():
                    for prompt in self.prompts:
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            conversation=[
                                {"role": "user", "content": f"{prompt}"}
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        inputs = self.tokenizer(
                            text=formatted_prompt, 
                            add_special_tokens=False,
                            return_tensors="pt",
                        ).to(original_model.device)
                        outputs = original_model.generate(**inputs, max_new_tokens=100)
                        answer_ids = outputs[0][len(inputs.input_ids[0]):]
                        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
                        answers.append(answer)
                peek_results = zip(self.prompts, answers)
                # log the results
                if hasattr(self, 'logger') and self.trainer.accelerator.is_main_process:
                    for q, a in peek_results:
                        self.logger.info(f'Prompt: {q}\nAnswer: {a}')