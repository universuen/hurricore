from __future__ import annotations

from typing import Any

import torch

from hurricore.trainers import LLMGRPOTrainer
from hurricore.hooks import LoggingHook
from hurricore.utils import Dummy, env


class LLMDAPOTrainer(LLMGRPOTrainer):
    def __init__(
        self, 
        # New in this class
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        **kwargs,
    ) -> None:
        
        super().__init__(
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            ref_model=Dummy(),
            **kwargs,
        )
        
        self.epsilon_low = self.init_context.epsilon_low
        self.epsilon_high = self.init_context.epsilon_high


    def compute_loss(self, batch: Any) -> torch.Tensor:
        # Get grouped completions and advantages
        grouped_completions, grouped_rewards = self.get_grouped_completions_with_rewards(batch)

        # Compute advantages
        grouped_advantages = self.compute_grouped_advantages(grouped_rewards).to(self.accelerator.device)
        
        # Compute initial policy probabilities
        init_token_log_probs, token_masks = self.get_grouped_token_log_probs_with_masks(
            model=self.model,
            encoded_prompts=batch["encoded_prompts"],
            grouped_completions=grouped_completions,
        )

        # Inner Update Loop
        total_loss = 0.0
        for i in range(self.num_updates_per_batch):
            # Re-compute current policy probabilities for the same completions if not the first step
            token_log_probs = self.get_grouped_token_log_probs_with_masks(
                model=self.model,
                encoded_prompts=batch["encoded_prompts"],
                grouped_completions=grouped_completions,
            )[0] if i > 0 else init_token_log_probs
            
            # Compute token losses using the PPO objective
            ratios = (token_log_probs - init_token_log_probs.detach()).exp()
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon_low, 1 + self.epsilon_high)
            token_losses = - torch.min(
                ratios * grouped_advantages.unsqueeze(-1), 
                clipped_ratios * grouped_advantages.unsqueeze(-1),
            )
            
            # Apply mask and compute average loss
            step_loss = (token_losses * token_masks).sum() / (token_masks.sum() + 1e-9)
            
            # Accumulate the loss
            total_loss += step_loss

        # Average the loss over all update steps
        final_loss = total_loss / self.num_updates_per_batch

        self._update_records(
            grouped_rewards=grouped_rewards,
            grouped_completions=grouped_completions,
        )

        return final_loss


    def _update_records(
            self,
            grouped_rewards: torch.Tensor,
            grouped_completions: list[list[str]], 
        ) -> None:
        # Update reward
        gathered_grouped_rewards = self.accelerator.gather(grouped_rewards.to(self.accelerator.device))
        reward = gathered_grouped_rewards.mean().item()

        # Update response length
        response_length = 0
        for batched_completion in grouped_completions:
            response_length += sum([len(i) for i in batched_completion]) / len(batched_completion)
        response_length /= len(grouped_completions)
        gathered_response_length = self.accelerator.gather(torch.tensor(response_length).to(self.accelerator.device))
        response_length = gathered_response_length.mean().item()

        # Update records
        logging_hook: LoggingHook = self.hooks["LoggingHook"]
        logging_hook.extra_log_bar_data.update(
            {
                "Reward": f"{reward:.2f}",
                "RspLen": f"{response_length:.2f}",
            }
        )
        env.tensor_board_writer.add_scalar('Reward/Training', reward, self.context.global_step)
        env.tensor_board_writer.add_scalar('Response Length/Training', response_length, self.context.global_step)
        env.history_recorder.record('training_reward', reward, self.context.global_step)
        env.history_recorder.record('training_response_length', response_length, self.context.global_step)
