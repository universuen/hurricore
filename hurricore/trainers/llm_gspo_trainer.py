from __future__ import annotations

import torch

from hurricore.trainers import LLMGRPOTrainer


class LLMGSPOTrainer(LLMGRPOTrainer):

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

        # Get reference model probabilities (frozen)
        with torch.no_grad():
            ref_group_token_log_probs = self.get_group_token_log_probs_with_mask(
                model=self.ref_model,
                torch_model_inputs=batch["torch_model_inputs"],
                group_completions_ids=group_completions_ids,
            )[0] if self.ref_model is not None else torch.zeros_like(init_group_token_log_probs)
        
        # Inner Update Loop
        total_loss = 0.0
        for i in range(self.num_updates_per_batch):
            # Re-compute current policy probabilities for the same completions if not the first step
            group_token_log_probs = self.get_group_token_log_probs_with_mask(
                model=self.model,
                torch_model_inputs=batch["torch_model_inputs"],
                group_completions_ids=group_completions_ids,
            )[0] if i > 0 else init_group_token_log_probs
            
            # Compute KL div against the reference model
            group_token_kl_divs = self.compute_token_kl_div(
                ref_log_probs=ref_group_token_log_probs,
                log_probs=group_token_log_probs,
            ) if self.ref_model is not None else torch.zeros_like(group_token_log_probs)

            log_ratios = group_token_log_probs - init_group_token_log_probs.detach()
            seq_effective_log_ratios = (log_ratios * effective_token_mask).sum(dim=2) / (effective_token_mask.sum(dim=2) + 1e-9)  # Shape: (group_size, batch_size)
            seq_effective_ratios = seq_effective_log_ratios.exp()
            seq_kl_divergence = (group_token_kl_divs * effective_token_mask).sum(dim=2) / (effective_token_mask.sum(dim=2) + 1e-9)  # Shape: (group_size, batch_size)
            seq_loss = - torch.min(
                seq_effective_ratios * group_advantages,
                seq_effective_ratios.clamp(1 - self.epsilon, 1 + self.epsilon) * group_advantages,
            ) + self.beta * seq_kl_divergence
            step_loss = seq_loss.mean()
            
            # Accumulate the loss
            total_loss += step_loss

        # Average the loss over all update steps
        final_loss = total_loss / self.num_updates_per_batch

        # Update records
        self._update_records(
            group_rewards=group_rewards,
            group_completions_ids=group_completions_ids,
            group_token_kl_divs=group_token_kl_divs,
            effective_token_mask=effective_token_mask,
        )

        return final_loss
