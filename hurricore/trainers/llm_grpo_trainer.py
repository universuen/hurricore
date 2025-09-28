from __future__ import annotations

import os
import copy
from typing import Any

import torch
from vllm import SamplingParams
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase, 
    GenerationConfig,
)

from hurricore.trainers import StandardTrainer
from hurricore.hooks import (
    LoggingHook, 
    LLMRLEvaluationHook,
)
from hurricore.utils import env, VLLMRollout


class LLMGRPOTrainer(StandardTrainer):
    def __init__(
        self, 
        model: PreTrainedModel,
        # New in this class
        tokenizer: PreTrainedTokenizerBase,
        ref_model: PreTrainedModel = None,
        beta: float = 0.0,
        epsilon: float = 0.2,
        native_generation_config: dict | None = None,
        group_size: int = 8,
        num_updates_per_batch: int = 1,
        gradient_checkpointing_enable: bool = True,
        use_vllm: bool = False,
        vllm_kwargs: dict | None = None,
        vllm_sampling_params: dict | None = None,
        # Task specific hyper parameters
        num_epochs: int = 3,
        per_device_batch_size: int = 1,
        learning_rate: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            ref_model=ref_model,
            beta=beta,
            epsilon=epsilon,
            native_generation_config=native_generation_config,
            group_size=group_size,
            num_updates_per_batch=num_updates_per_batch,
            gradient_checkpointing_enable=gradient_checkpointing_enable,
            use_vllm=use_vllm,
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        # Set default generation config
        self.native_generation_config = {
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "do_sample": True,
            "use_cache": True,
        }
        self.native_generation_config.update(native_generation_config or dict())
        self.temperature = self.native_generation_config["temperature"]
        # Set tokenizer
        self.tokenizer = tokenizer
        # Set ref model
        self.ref_model = None
        if beta > 0:
            if ref_model is None:
                ref_model = copy.deepcopy(model)
            self.ref_model = ref_model.to(self.accelerator.device)
        # Set hyper parameters
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size
        self.num_updates_per_batch = num_updates_per_batch
        # Enable gradient checkpointing
        if gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
        # Set vLLM parameters
        self.use_vllm = use_vllm
        if use_vllm:
            vllm_kwargs = vllm_kwargs or dict()
            vllm_sampling_params = vllm_sampling_params or dict()
            # This param is frequently used, so we save it as a separate variable
            self.vllm_tensor_parallel_size = vllm_kwargs.get("tensor_parallel_size", 1)  
            self.vllm_kwargs = {
                "max_model_len": 16 * 1024,
                "gpu_memory_utilization": 0.5,
                "max_num_seqs": per_device_batch_size * group_size * self.vllm_tensor_parallel_size,
                "tensor_parallel_size": self.vllm_tensor_parallel_size,
                "seed": self.accelerator.process_index // self.vllm_tensor_parallel_size,
            }
            self.vllm_kwargs.update(vllm_kwargs)
            self.vllm_sampling_params = {
                "max_tokens": 4096,
                "temperature": 1.0,
            }
            self.vllm_sampling_params.update(vllm_sampling_params)
            self.temperature = self.vllm_sampling_params["temperature"]
            self.setup_vllm_rollout()


    def setup_vllm_rollout(self) -> None:
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
        self.vllm_rollout = VLLMRollout(
            model=self.original_model,
            vllm_kwargs=self.vllm_kwargs,
        )
        self.vllm_rollout.sleep()
        self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [
                list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
            ]
        )


    def setup_hooks(self) -> None:
        super().setup_hooks()
        self.init_context.hooks.append(
            LLMRLEvaluationHook(
                trainer=self,
                data_loader=self.init_context.validation_data_loader,
                interval=self.init_context.evaluation_interval,
            )
        )
        del self.init_context.hooks["ValidationLossHook"]


    def compute_reward(self, completion: str, target: Any) -> float:
        raise NotImplementedError


    def get_grouped_completions_with_rewards(
            self, 
            batch: dict,
            group_size: int = None,
        ) -> tuple[list[list[str]], torch.Tensor]:
        group_size = group_size or self.group_size
        # Generate completions for each group
        assert "encoded_prompts" in batch.keys(), "encoded_prompts not found in batch"
        assert "targets" in batch, "targets not found in batch"
        encoded_prompts = batch["encoded_prompts"]
        targets = batch["targets"]

        if self.use_vllm:
            # Move model to CPU for weight update
            self.original_model.to("cpu")
            assert "vllm_prompts" in batch.keys(), "vllm_prompts not found in batch"
            vllm_prompts = batch["vllm_prompts"]
            # Wake up vLLM and update weights
            self.vllm_rollout.wake_up()
            self.vllm_rollout.update_weights(model=self.original_model)
            # Prepare flatten grouped prompts
            flatten_grouped_vllm_prompts = vllm_prompts * group_size
            # Get completions
            flatten_grouped_completions = self.get_vllm_completions(flatten_grouped_vllm_prompts)
            # Put vLLM back to sleep and move model back to GPU
            self.vllm_rollout.sleep()
            self.original_model.to(self.accelerator.device)
        else:
            flatten_grouped_encoded_prompts = {
                k: v.repeat(group_size, 1) 
                for k, v in encoded_prompts.items()
            }
            flatten_grouped_completions = self.get_native_completions(flatten_grouped_encoded_prompts)

        # Reshape completions
        batch_size = len(encoded_prompts["input_ids"])
        grouped_completions = [
            flatten_grouped_completions[i * batch_size:(i + 1) * batch_size] 
            for i in range(len(flatten_grouped_completions) // batch_size)
        ]
        # Compute rewards
        grouped_rewards = [
            [
                self.compute_reward(completion, target)
                for completion, target in zip(completions, targets)
            ]
            for completions in grouped_completions
        ]
        grouped_rewards = torch.tensor(grouped_rewards, dtype=torch.float32)
        return grouped_completions, grouped_rewards


    def compute_loss(self, batch: dict) -> torch.Tensor:
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

        # Get reference model probabilities (frozen)
        with torch.no_grad():
            ref_token_log_probs = self.get_grouped_token_log_probs_with_masks(
                model=self.ref_model,
                encoded_prompts=batch["encoded_prompts"],
                grouped_completions=grouped_completions,
            )[0] if self.ref_model is not None else torch.zeros_like(init_token_log_probs)
        
        # Inner Update Loop
        total_loss = 0.0
        for i in range(self.num_updates_per_batch):
            # Re-compute current policy probabilities for the same completions if not the first step
            token_log_probs = self.get_grouped_token_log_probs_with_masks(
                model=self.model,
                encoded_prompts=batch["encoded_prompts"],
                grouped_completions=grouped_completions,
            )[0] if i > 0 else init_token_log_probs
            
            # Compute KL divergence against the reference model
            kl_divergence = self.compute_kl_divergence(
                ref_log_probs=ref_token_log_probs,
                log_probs=token_log_probs,
            ) if self.ref_model is not None else torch.zeros_like(token_log_probs)

            # Compute token losses
            ratios = (token_log_probs - init_token_log_probs.detach()).exp()
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
            
            token_losses = -torch.min(
                ratios * grouped_advantages.unsqueeze(-1), 
                clipped_ratios * grouped_advantages.unsqueeze(-1),
            ) + self.beta * kl_divergence
            
            # Apply mask and compute average loss for this update step
            masked_losses = token_losses * token_masks
            loss_sums = masked_losses.sum(dim=2)
            mask_sums = token_masks.sum(dim=2) + 1e-9
            step_loss = (loss_sums / mask_sums).mean()
            
            # Accumulate the loss
            total_loss += step_loss

        # Average the loss over all update steps
        final_loss = total_loss / self.num_updates_per_batch

        # Update records for logging using the final state
        with torch.no_grad():
            masked_kl_divergence = kl_divergence * token_masks
            avg_kl_divergence = (masked_kl_divergence.sum(dim=2) / mask_sums).mean()
            
        self._update_records(
            grouped_rewards=grouped_rewards,
            grouped_completions=grouped_completions,
            avg_kl_divergence=avg_kl_divergence,
        )

        return final_loss

    
    @torch.no_grad()
    def get_native_completions(self, encoded_prompts: dict) -> list[str]:
        self.model.gradient_checkpointing_disable()
        generated_ids = self.original_model.generate(
            **encoded_prompts, 
            generation_config=GenerationConfig(**self.native_generation_config),
        )
        completions_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(encoded_prompts["input_ids"], generated_ids)
        ]
        completions = self.tokenizer.batch_decode(
            completions_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False,
        )
        if self.init_context.gradient_checkpointing_enable:
            self.model.gradient_checkpointing_enable()
        return completions
    

    @torch.no_grad()
    def get_vllm_completions(self, vllm_prompts: list) -> list[str]:
        # Gather prompts from all processes
        vllm_batch_size = len(vllm_prompts)
        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, vllm_prompts, group=self.vllm_tp_group)
        all_vllm_prompts = sum(gathered_prompts, [])
        # Generate completions
        all_completions = self.vllm_rollout.generate(
            vllm_prompts=all_vllm_prompts,
            sampling_params=SamplingParams(**self.vllm_sampling_params),
        )

        # Extract completions corresponding to the original prompts
        local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
        tp_slice = slice(local_rank_in_group * vllm_batch_size, (local_rank_in_group + 1) * vllm_batch_size)
        completions = all_completions[tp_slice]

        return completions


    @staticmethod
    def compute_grouped_advantages(grouped_rewards: torch.Tensor) -> torch.Tensor:
        mean_r = grouped_rewards.mean(dim=0, keepdim=True) 
        std_r = grouped_rewards.std(dim=0, unbiased=False, keepdim=True)
        advantage = (grouped_rewards - mean_r) / (std_r + 1e-9)
        return advantage


    def get_token_log_probs_with_masks(
            self, 
            model: PreTrainedModel, 
            encoded_prompts: dict,
            completions: list[str], 
        ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_completions = self.tokenizer(
            completions,
            return_tensors="pt",
            padding="longest",
            padding_side="right",
        ).to(self.accelerator.device)
        completion_length = encoded_completions["input_ids"].shape[1]
        model_inputs = {
            "input_ids": torch.cat(
                (encoded_prompts["input_ids"], encoded_completions["input_ids"]), 
                dim=-1,
            ).to(torch.long),
            "attention_mask": torch.cat(
                (encoded_prompts["attention_mask"], encoded_completions["attention_mask"]), 
                dim=-1,
            ).to(torch.long),
        }
        # Assuming <q1, q2, c1, c2, c3>
        model_outputs = model(
            **model_inputs, 
            use_cache=False,
        )
        # q2 carries the prob dist of c1, so we include it
        # c3 carries prob dist of nothing, so we exclude it
        logits = model_outputs.logits[:, -completion_length - 1: -1, :]
        logits = logits / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        # Compute token log probabilities
        completion_log_probs = torch.gather(log_probs, dim=-1, index=encoded_completions["input_ids"].unsqueeze(-1)).squeeze(-1)
        return completion_log_probs, encoded_completions["attention_mask"]


    def get_grouped_token_log_probs_with_masks(
            self, 
            model, 
            encoded_prompts: dict,
            grouped_completions: list[list[str]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(grouped_completions) == self.group_size, f"Expected {self.group_size} completions, got {len(grouped_completions)}"
        all_log_probs = []
        all_masks = []
        max_completion_length = 0  # Track max sequence length across group

        for completions in grouped_completions:
            completion_log_probs, attention_masks = self.get_token_log_probs_with_masks(
                model=model, 
                encoded_prompts=encoded_prompts, 
                completions=completions,
            )
            completion_length = completion_log_probs.shape[1]
            max_completion_length = max(max_completion_length, completion_length)  # Update max sequence length
            all_log_probs.append(completion_log_probs)
            all_masks.append(attention_masks)

        # Pad all tensors to max_completion_length
        padded_log_probs = [torch.nn.functional.pad(p, (0, max_completion_length - p.shape[1]), value=1.0) for p in all_log_probs]
        padded_masks = [torch.nn.functional.pad(m, (0, max_completion_length - m.shape[1]), value=0) for m in all_masks]

        # Stack along group_size dimension
        stacked_log_probs = torch.stack(padded_log_probs, dim=0)  # Shape: (group_size, batch_size, max_completion_length)
        stacked_masks = torch.stack(padded_masks, dim=0)  # Shape: (group_size, batch_size, max_completion_length)
        return stacked_log_probs, stacked_masks


    @staticmethod
    def compute_kl_divergence(
        ref_log_probs: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = ref_log_probs - log_probs
        token_kl_divergence = log_ratio.exp() - log_ratio - 1
        return token_kl_divergence
    

    def _update_records(
            self,
            grouped_rewards: torch.Tensor,
            grouped_completions: list[list[str]], 
            avg_kl_divergence: torch.Tensor,
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

        # Update KL divergence
        gathered_avg_kl_divergence = self.accelerator.gather(avg_kl_divergence)
        kl_div = gathered_avg_kl_divergence.mean().item()
        
        # Update records
        logging_hook: LoggingHook = self.hooks["LoggingHook"]
        logging_hook.extra_log_bar_data.update(
            {
                "Reward": f"{reward:.2f}",
                "RspLen": f"{response_length:.2f}",
                "KLDiv": f"{kl_div:.5f}",
            }
        )
        env.tensor_board_writer.add_scalar('Reward/Training', reward, self.context.global_step)
        env.tensor_board_writer.add_scalar('Response Length/Training', response_length, self.context.global_step)
        env.tensor_board_writer.add_scalar('KL Divergence/Training', kl_div, self.context.global_step)
        env.history_recorder.record('training_reward', reward, self.context.global_step)
        env.history_recorder.record('training_response_length', response_length, self.context.global_step)
        env.history_recorder.record('training_kl_div', kl_div, self.context.global_step)
