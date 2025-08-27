import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import env, assign_args, AttributeDict


class LLMRLEvaluationHook(HurricoreHook):
    def __init__(
            self, 
            trainer: HurricoreTrainer,
            data_loader: DataLoader,
            interval: int = 10,
        ):
        super().__init__(trainer)
        self.interval = interval
        assign_args(self.init_context, locals())
    

    def __post_init__(self) -> None:
        self.data_loader = self.trainer.accelerator.prepare(self.init_context.data_loader)
        self.records = AttributeDict(
            best_reward=float("-inf"),
            best_step=0,
        )
        self.trainer.accelerator.register_for_checkpointing(self.records)


    def execute_on_step_end(self) -> None:
        if (self.trainer.context.global_step + 1) % self.interval == 0:
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
            grouped_completions, grouped_rewards = self.trainer.get_grouped_completions_with_rewards(
                batch=batch,
                group_size=1,
            )
            completions, rewards = grouped_completions[0], grouped_rewards[0]
            all_rewards.extend(rewards)
            all_rsp_lens.extend([len(x) for x in completions])
        all_rewards = torch.tensor(all_rewards, device=self.trainer.accelerator.device, dtype=torch.float32)
        all_rsp_lens = torch.tensor(all_rsp_lens, device=self.trainer.accelerator.device, dtype=torch.float32)
        self.trainer.accelerator.wait_for_everyone()
        avg_reward = self.trainer.accelerator.gather(all_rewards).mean().item()
        avg_rsp_len = self.trainer.accelerator.gather(all_rsp_lens).mean().item()

        if avg_reward > self.records.best_reward:
            self.records.best_reward = avg_reward
            self.records.best_step = self.trainer.context.global_step + 1
            # Save the best checkpoint
            if "CheckpointHook" in self.trainer.hooks:
                ckpt_hook = self.trainer.hooks["CheckpointHook"]
                ckpt_path = ckpt_hook.folder_path / "ckpt_highest_val_reward"
                ckpt_path = env.checkpointer.save_state(ckpt_path)
                env.logger.info(f"Saved best checkpoint to {ckpt_path}")
            else:
                env.logger.warning("No checkpoint hook found")

        env.logger.info(f'Validation reward: {avg_reward}')
        env.logger.info(f'Validation response length: {avg_rsp_len}')

        env.tensor_board_writer.add_scalar(
            tag='Reward/Validation',
            scalar_value=avg_reward,
            global_step=self.trainer.context.global_step + 1,
        )
        env.tensor_board_writer.add_scalar(
            tag='Response Length/Validation',
            scalar_value=avg_rsp_len,
            global_step=self.trainer.context.global_step + 1,
        )

        env.history_recorder.record(
            tag='validation_reward',
            value=avg_reward,
            global_step=self.trainer.context.global_step + 1
        )
        env.history_recorder.record(
            tag='validation_response_length',
            value=avg_rsp_len,
            global_step=self.trainer.context.global_step + 1
        )
