import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import env, AttributeDict, Checkpointer, HuggingfaceModelSaver


class LLMRLEvaluationHook(HurricoreHook):
    def __init__(
            self, 
            trainer: HurricoreTrainer,
            data_loader: DataLoader,
            interval: int = 10,
        ):
        super().__init__(
            trainer=trainer,
            data_loader=data_loader,
            interval=interval,
        )
        self.interval = interval


    def execute_on_trainer_init_end(self) -> None:
        self.data_loader = self.trainer.accelerator.prepare(self.init_context.data_loader)
        self.records = AttributeDict(
            best_reward=float("-inf"),
            best_step=0,
        )
        self.trainer.accelerator.register_for_checkpointing(self.records)


    def execute_on_step_end(self) -> None:
        if (self.trainer.training_context.global_step + 1) % self.interval == 0:
            self.evaluate()

    
    def execute_on_training_end(self) -> None:
        self.evaluate()
    

    @torch.no_grad()
    def evaluate(self) -> None:
        data_loader = self.data_loader
        if self.trainer.accelerator.is_main_process:
            data_loader = tqdm(data_loader, desc='Computing validation rewards')
        all_rewards = []
        all_rsp_lens = []
        for batch in data_loader:
            group_completions_ids, group_rewards = self.trainer.get_group_completions_ids_with_rewards(
                batch=batch,
                group_size=1,
            )
            batch_completions = self.trainer.tokenizer.batch_decode(group_completions_ids[0], skip_special_tokens=True)
            batch_rewards = group_rewards[0]
            all_rewards.extend(batch_rewards)
            all_rsp_lens.extend([self.trainer.get_num_tokens(i) for i in batch_completions])
        all_rewards = torch.tensor(all_rewards, device=self.trainer.accelerator.device, dtype=torch.float32)
        all_rsp_lens = torch.tensor(all_rsp_lens, device=self.trainer.accelerator.device, dtype=torch.float32)
        self.trainer.accelerator.wait_for_everyone()
        avg_reward = self.trainer.accelerator.gather(all_rewards).mean().item()
        avg_rsp_len = self.trainer.accelerator.gather(all_rsp_lens).mean().item()

        if avg_reward > self.records.best_reward:
            self.records.best_reward = avg_reward
            self.records.best_step = self.trainer.training_context.global_step + 1
            if isinstance(env.checkpointer, Checkpointer):
                ckpt_path = env.checkpointer.save_state("ckpt_highest_val_reward")
                env.logger.info(f"Saved best checkpoint to {ckpt_path}")
            if isinstance(env.huggingface_model_saver, HuggingfaceModelSaver):
                model_saving_path = env.huggingface_model_saver.save("highest_val_reward_model")
                env.logger.info(f"Saved best model to {model_saving_path}")

        env.logger.info(
            f"Validation reward: {avg_reward:.2f} "
            f"(best: {self.records.best_reward:.2f} at step {self.records.best_step})"
        )
        env.logger.info(f'Validation response length: {avg_rsp_len: .2f}')

        env.tensor_board_writer.add_scalar(
            tag='Reward/Validation',
            scalar_value=avg_reward,
            global_step=self.trainer.training_context.global_step + 1,
        )
        env.tensor_board_writer.add_scalar(
            tag='Response Length/Validation',
            scalar_value=avg_rsp_len,
            global_step=self.trainer.training_context.global_step + 1,
        )

        env.history_recorder.record(
            tag='validation_reward',
            value=avg_reward,
            global_step=self.trainer.training_context.global_step + 1
        )
        env.history_recorder.record(
            tag='validation_response_length',
            value=avg_rsp_len,
            global_step=self.trainer.training_context.global_step + 1
        )
