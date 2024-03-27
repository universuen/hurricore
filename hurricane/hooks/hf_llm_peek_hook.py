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
        self.is_available = (None not in (prompts, tokenizer))
        # setup self
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.interval = interval
    
    def on_training_start(self) -> None:
        # check validity
        if not self.is_available:
            return
        # collect logger
        logger_hook = self.trainer.get_hook(LoggerHook)
        if logger_hook is not None:
            self.logger = logger_hook.logger
            self.log_interval = logger_hook.interval
    
    def on_step_end(self) -> None:
        # check validity
        if not self.is_available:
            return
        # peek model results
        conditions = (
            self.trainer.accelerator.is_main_process,
            is_deepspeed_zero3(self.trainer.accelerator),
        )
        if any(conditions):
            idx = self.trainer.ctx.batch_idx
            num_batches = len(self.trainer.data_loader)
            if self.trainer.ctx.global_step % self.interval == 0 or idx == num_batches:
                original_model = self.trainer.accelerator.unwrap_model(self.trainer.model)
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
