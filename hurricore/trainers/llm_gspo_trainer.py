from __future__ import annotations

from typing import Any

import torch

from hurricore.trainers import LLMGRPOTrainer



class LLMGSPOTrainer(LLMGRPOTrainer):

    def compute_loss(self, batch: dict) -> torch.Tensor:
        assert type(batch) is dict, f"Expected batch to be a dict, got {type(batch)}"
        assert "torch_model_inputs" in batch, f"Expected batch to contain 'torch_model_inputs', but got {batch.keys()}"
        # Get grouped completions and advantages
        grouped_completions, grouped_rewards = self.get_grouped_completions_with_rewards(batch)

        # Compute advantages
        grouped_advantages = self.compute_grouped_advantages(grouped_rewards).to(self.accelerator.device)
        
        # Compute initial policy probabilities
        init_token_log_probs, token_masks = self.get_grouped_token_log_probs_with_masks(
            model=self.model,
            torch_model_inputs=batch["torch_model_inputs"],
            grouped_completions=grouped_completions,
        )
        
        # Get reference model probabilities (frozen)
        with torch.no_grad():
            ref_token_log_probs = self.get_grouped_token_log_probs_with_masks(
                model=self.ref_model,
                torch_model_inputs=batch["torch_model_inputs"],
                grouped_completions=grouped_completions,
            )[0] if self.ref_model is not None else torch.zeros_like(init_token_log_probs)

        # Inner Update Loop
        total_loss = 0.0
        for i in range(self.num_updates_per_batch):
            # Re-compute current policy probabilities for the same completions if not the first step
            token_log_probs = self.get_grouped_token_log_probs_with_masks(
                model=self.model,
                torch_model_inputs=batch["torch_model_inputs"],
                grouped_completions=grouped_completions,
            )[0] if i > 0 else init_token_log_probs

            # Compute KL divergence against the reference model
            kl_divergence = self.compute_kl_divergence(
                ref_log_probs=ref_token_log_probs,
                log_probs=token_log_probs,
            ) if self.ref_model is not None else torch.zeros_like(token_log_probs)
            
            log_ratios = token_log_probs - init_token_log_probs.detach()
            seq_effective_log_ratios = (log_ratios * token_masks).sum(dim=2) / token_masks.sum(dim=2)  # Shape: (num_groups, batch_size)
            seq_effective_ratios = seq_effective_log_ratios.exp()
            seq_kl_divergence = (kl_divergence * token_masks).sum(dim=2) / token_masks.sum(dim=2)  # Shape: (num_groups, batch_size)
            seq_loss = - torch.min(
                seq_effective_ratios * grouped_advantages,
                seq_effective_ratios.clamp(1 - self.epsilon, 1 + self.epsilon) * grouped_advantages,
            ) + self.beta * seq_kl_divergence
            total_loss += seq_loss.mean()
        
        final_loss = total_loss / self.num_updates_per_batch

        with torch.no_grad():
            avg_kl_divergence = seq_kl_divergence.mean()
            
        self._update_records(
            grouped_rewards=grouped_rewards,
            grouped_completions=grouped_completions,
            avg_kl_divergence=avg_kl_divergence,
        )

        return final_loss
