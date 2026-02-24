from __future__ import annotations

import torch

from hurricore.trainers import LLMGRPOTrainer


class LLMDAPOTrainer(LLMGRPOTrainer):
    def __init__(
        self, 
        # New in this class
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        **kwargs,
    ) -> None:
        
        super().__init__(
            beta=0,
            **kwargs,
        )
        
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
    

    def compute_loss(self, batch: dict) -> torch.Tensor:
        assert type(batch) is dict, f"Expected batch to be a dict, got {type(batch)}"
        assert "torch_model_inputs" in batch, f"Expected batch to contain 'torch_model_inputs', but got {batch.keys()}"
        # Get group completions and advantages
        group_completions_ids, group_rewards = self.get_group_completions_ids_with_rewards(batch)

        # Compute advantages
        group_advantages = self.compute_group_advantages(group_rewards).to(self.accelerator.device)
        
        # Compute initial policy probabilities
        init_group_token_log_probs, effective_token_mask = self.get_group_token_log_probs_with_mask(
            model=self.model,
            torch_model_inputs=batch["torch_model_inputs"],
            group_completions_ids=group_completions_ids,
        )

        # Update effective_token_mask to ignore truncated samples if needed
        if self.ignore_truncated_samples:
            with torch.no_grad():
                has_eos = (group_completions_ids == self.tokenizer.eos_token_id).any(dim=-1)
                valid_mask = has_eos.unsqueeze(-1)
                effective_token_mask *= valid_mask.to(effective_token_mask.dtype).to(effective_token_mask.device)


        # Inner Update Loop
        total_loss = 0.0
        for i in range(self.num_updates_per_batch):
            # Re-compute current policy probabilities for the same completions if not the first step
            group_token_log_probs = self.get_group_token_log_probs_with_mask(
                model=self.model,
                torch_model_inputs=batch["torch_model_inputs"],
                group_completions_ids=group_completions_ids,
            )[0] if i > 0 else init_group_token_log_probs
            
            # Compute token losses
            ratios = (group_token_log_probs - init_group_token_log_probs.detach()).exp()
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon_low, 1 + self.epsilon_high)
            
            group_token_losses = -torch.min(
                ratios * group_advantages.unsqueeze(-1), 
                clipped_ratios * group_advantages.unsqueeze(-1),
            )
            
            # Apply mask and compute average loss for this update step
            step_loss = (group_token_losses * effective_token_mask).sum() / (effective_token_mask.sum() + 1e-9)
            
            # Accumulate the loss
            total_loss += step_loss

        # Average the loss over all update steps
        final_loss = total_loss / self.num_updates_per_batch

        # Update records
        self._update_records(
            group_rewards=group_rewards,
            group_completions_ids=group_completions_ids,
        )
        return final_loss
