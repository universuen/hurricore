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
    HuggingfaceModelSaveHook,
)
from hurricore.utils import env, VLLMRollout, deepspeed_zero3_enabled


class LLMGRPOTrainer(StandardTrainer):
    def __init__(
        self, 
        model: PreTrainedModel,
        # New in this class
        tokenizer: PreTrainedTokenizerBase,
        ref_model: PreTrainedModel = None,
        beta: float = 0.001,
        epsilon: float = 0.2,
        generation_config: dict | None = None,
        group_size: int = 8,
        num_updates_per_batch: int = 1,
        gradient_checkpointing_enable: bool = True,
        use_vllm: bool = False,
        vllm_kwargs: dict | None = None,
        vllm_sampling_params: dict | None = None,
        ignore_truncated_samples: bool = True,
        # Task specific hyper parameters
        learning_rate: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.ref_model = None
        if beta > 0:
            self.ref_model = ref_model or self.create_ref_model(model)
            if deepspeed_zero3_enabled():
                self.ref_model = self.accelerator.prepare(self.ref_model)
                if self.ref_model in self.accelerator._models:
                    self.accelerator._models.remove(self.ref_model)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

        # Set default generation config
        self.generation_config = {
            "max_new_tokens": 4096,
            "temperature": 1.0,
            "do_sample": True,
            "use_cache": True,
        }
        self.generation_config.update(generation_config or dict())
        self.temperature = self.generation_config["temperature"]
        # Set tokenizer
        self.tokenizer = tokenizer

        # Set hyper parameters
        self.beta = beta
        self.epsilon = epsilon
        self.group_size = group_size
        self.num_updates_per_batch = num_updates_per_batch
        self.gradient_checkpointing_enable = gradient_checkpointing_enable
        # Enable gradient checkpointing
        if gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
        # Setup vLLM parameters
        self.use_vllm = use_vllm
        if use_vllm:
            vllm_kwargs = vllm_kwargs or dict()
            vllm_sampling_params = vllm_sampling_params or dict()
            # This param is frequently used, so we save it as a separate variable
            self.vllm_tensor_parallel_size = vllm_kwargs.get("tensor_parallel_size", 8)  
            self.vllm_kwargs = {
                "gpu_memory_utilization": 0.5,
                "max_num_seqs": self.init_context.per_device_batch_size * group_size * self.vllm_tensor_parallel_size,
                "tensor_parallel_size": self.vllm_tensor_parallel_size,
                "seed": self.accelerator.process_index // self.vllm_tensor_parallel_size,
            }
            self.vllm_kwargs.update(vllm_kwargs)
            self.vllm_sampling_params = {
                "max_tokens": 4096,
                "temperature": 1.0,
                "skip_special_tokens": False,
            }
            self.vllm_sampling_params.update(vllm_sampling_params)
            self.temperature = self.vllm_sampling_params["temperature"]
            self.setup_vllm_rollout()
        self.ignore_truncated_samples = ignore_truncated_samples


    def create_ref_model(self, model: PreTrainedModel) -> PreTrainedModel:
        if deepspeed_zero3_enabled():
            # Reload the model from disk/hub to get a fresh, valid instance
            model_name_or_path = getattr(model.config, "_name_or_path", None)
            ref_model = type(model).from_pretrained(
                model_name_or_path,
                config=model.config,
                dtype=model.dtype,
            )
        else:
            ref_model = copy.deepcopy(model)
        return ref_model


    def setup_vllm_rollout(self) -> None:
        os.environ["RANK"] = str(self.accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
        self.accelerator.wait_for_everyone()
        self.vllm_rollout = VLLMRollout(
            model=self.original_model,
            vllm_kwargs=self.vllm_kwargs,
        )
        self.accelerator.wait_for_everyone()
        self.vllm_rollout.sleep()
        self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [
                list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
            ]
        )


    def setup_hooks(self) -> None:
        super().setup_hooks()
        self.init_context.hooks.extend(
            [
                LLMRLEvaluationHook(
                    trainer=self,
                    data_loader=self.init_context.validation_data_loader,
                    interval=self.init_context.evaluation_interval * self.init_context.accelerator.gradient_accumulation_steps,
                ),
                HuggingfaceModelSaveHook(
                    trainer=self,
                    folder_path=self.init_context.data_folder_path / "results" / self.init_context.experiment_name,
                    interval=self.init_context.ckpt_interval * self.init_context.accelerator.gradient_accumulation_steps,
                ),
            ]
        )
        del self.init_context.hooks["ValidationLossHook"]


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

            # Compute token losses
            ratios = (group_token_log_probs - init_group_token_log_probs.detach()).exp()
            clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
            
            group_token_losses = -torch.min(
                ratios * group_advantages.unsqueeze(-1), 
                clipped_ratios * group_advantages.unsqueeze(-1),
            ) + self.beta * group_token_kl_divs
            
            # Apply mask and compute average loss for this update step
            masked_group_token_losses = group_token_losses * effective_token_mask
            group_sequence_losses = masked_group_token_losses.sum(dim=-1) / (effective_token_mask.sum(dim=-1) + 1e-9)  # Shape: (group_size, batch_size)
            step_loss = group_sequence_losses.mean()  # tensor.mean(0).mean(1) == tensor.mean()

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
    

    def compute_reward(self, completion: str, target: Any) -> float:
        raise NotImplementedError


    def get_group_completions_ids_with_rewards(
            self, 
            batch: dict,
            group_size: int = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        group_size = group_size or self.group_size
        # Generate completions for each group
        assert type(batch) is dict, f"Expected batch to be a dict, got {type(batch)}: {batch}"
        assert "torch_model_inputs" in batch.keys(), f"torch_model_inputs not found in batch keys: {batch.keys()}"
        assert "targets" in batch.keys(), f"targets not found in batch keys: {batch.keys()}"
        torch_model_inputs = batch["torch_model_inputs"]
        targets = batch["targets"]

        if self.use_vllm:
            assert "vllm_conversations" in batch.keys(), "vllm_conversations not found in batch"
            vllm_conversations = batch["vllm_conversations"]
            # Wake up vLLM and update weights
            torch.cuda.empty_cache() 
            self.accelerator.wait_for_everyone()
            self.vllm_rollout.wake_up()
            self.vllm_rollout.update_weights(model=self.original_model)
            self.accelerator.wait_for_everyone()
            # Prepare flatten group prompts
            flatten_group_vllm_conversations = vllm_conversations * group_size
            # Get completions
            flatten_group_completions_ids = self.get_vllm_completions_ids(flatten_group_vllm_conversations)
            # Put vLLM back to sleep and move model back to GPU
            self.vllm_rollout.sleep()
        else:
            flatten_group_torch_model_inputs = {
                k: v.repeat(group_size, 1) 
                for k, v in torch_model_inputs.items()
            }
            flatten_group_completions_ids = self.get_completions_ids(flatten_group_torch_model_inputs)

        # Reshape completions
        batch_size = len(torch_model_inputs["input_ids"])
        group_completions_ids = flatten_group_completions_ids.reshape(group_size, batch_size, -1)
        group_completions = []
        for batch_completions_ids in group_completions_ids:
            group_completions.append(self.tokenizer.batch_decode(batch_completions_ids, skip_special_tokens=True))
        # Compute rewards
        group_rewards = [
            [
                self.compute_reward(completion, target)
                for completion, target in zip(completions, targets)
            ]
            for completions in group_completions
        ]
        group_rewards = torch.tensor(group_rewards, dtype=torch.float32)
        return group_completions_ids, group_rewards


    @staticmethod
    def compute_group_advantages(group_rewards: torch.Tensor) -> torch.Tensor:
        mean_r = group_rewards.mean(dim=0, keepdim=True) 
        std_r = group_rewards.std(dim=0, correction=0, keepdim=True)
        advantages = (group_rewards - mean_r) / (std_r + 1e-9)
        return advantages

    
    @torch.no_grad()
    def get_completions_ids(self, torch_model_inputs: dict) -> torch.Tensor:
        self.model.gradient_checkpointing_disable()
        generated_ids = self.model.generate(
            **torch_model_inputs, 
            generation_config=GenerationConfig(**self.generation_config),
            synced_gpus=True 
        )
        if self.gradient_checkpointing_enable:
            self.model.gradient_checkpointing_enable()
        completions_ids = generated_ids[:, torch_model_inputs["input_ids"].shape[1]:]
        return completions_ids
    

    @torch.no_grad()
    def get_vllm_completions_ids(self, vllm_conversations: list) -> torch.Tensor:
        # Gather prompts from all processes
        vllm_batch_size = len(vllm_conversations)
        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
        torch.distributed.all_gather_object(gathered_prompts, vllm_conversations, group=self.vllm_tp_group)
        all_vllm_conversations = sum(gathered_prompts, [])
        # Generate completions
        all_completions_ids = self.vllm_rollout(
            vllm_conversations=all_vllm_conversations,
            sampling_params=SamplingParams(**self.vllm_sampling_params),
            return_text=False,
        )

        # Extract completions corresponding to the original prompts
        local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
        tp_slice = slice(local_rank_in_group * vllm_batch_size, (local_rank_in_group + 1) * vllm_batch_size)
        completions_ids = all_completions_ids[tp_slice]
        completions_ids = [torch.tensor(completion_ids, dtype=torch.long) for completion_ids in completions_ids]
        completions_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=completions_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )
        return completions_ids


    def get_batch_token_log_probs_with_mask(
            self, 
            model: PreTrainedModel, 
            torch_model_inputs: dict,
            batch_completions_ids: list[str], 
        ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_completions_ids = batch_completions_ids.to(self.accelerator.device)
        batch_effective_token_mask = batch_completions_ids.ne(self.tokenizer.pad_token_id)
        model_inputs = {
            "input_ids": torch.cat(
                (torch_model_inputs["input_ids"], batch_completions_ids), 
                dim=-1,
            ).to(torch.long),
            "attention_mask": torch.cat(
                (torch_model_inputs["attention_mask"], batch_effective_token_mask), 
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
        batch_logits = model_outputs.logits[:, -batch_completions_ids.shape[1] - 1: -1, :]
        batch_logits = batch_logits / self.temperature
        batch_log_probs = torch.log_softmax(batch_logits, dim=-1)
        # Compute token log probabilities
        batch_token_log_probs = torch.gather(batch_log_probs, dim=-1, index=batch_completions_ids.unsqueeze(-1)).squeeze(-1)
        return batch_token_log_probs, batch_effective_token_mask


    def get_group_token_log_probs_with_mask(
            self, 
            model, 
            torch_model_inputs: dict,
            group_completions_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(group_completions_ids) == self.group_size, f"Expected {self.group_size} completions, got {len(group_completions_ids)}"
        all_log_probs = []
        all_masks = []
        max_seq_length = 0  # Track max sequence length across group

        for batch_completions_ids in group_completions_ids:
            batch_token_log_probs, batch_effective_token_mask = self.get_batch_token_log_probs_with_mask(
                model=model, 
                torch_model_inputs=torch_model_inputs, 
                batch_completions_ids=batch_completions_ids,
            )
            max_seq_length = max(max_seq_length, batch_token_log_probs.shape[1])  # Update max sequence length
            all_log_probs.append(batch_token_log_probs)
            all_masks.append(batch_effective_token_mask)

        # Pad all tensors to max_seq_length
        padded_log_probs = [torch.nn.functional.pad(p, (0, max_seq_length - p.shape[1]), value=1.0) for p in all_log_probs]
        padded_masks = [torch.nn.functional.pad(m, (0, max_seq_length - m.shape[1]), value=0) for m in all_masks]

        # Stack along group_size dimension
        group_token_log_probs = torch.stack(padded_log_probs, dim=0)  # Shape: (group_size, batch_size, max_seq_length)
        group_effective_token_mask = torch.stack(padded_masks, dim=0)  # Shape: (group_size, batch_size, max_seq_length)
        return group_token_log_probs, group_effective_token_mask


    @staticmethod
    def compute_token_kl_div(
        ref_log_probs: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        log_ratios = ref_log_probs - log_probs
        kl_divs = log_ratios.exp() - log_ratios - 1
        return kl_divs
    

    @torch.no_grad()
    def _update_records(
            self,
            group_rewards: torch.Tensor,
            group_completions_ids: torch.Tensor, 
            group_token_kl_divs: torch.Tensor = None,
            effective_token_mask: torch.Tensor = None,
        ) -> None:
        
        # Update reward
        gathered_group_rewards = self.accelerator.gather(group_rewards.to(self.accelerator.device))
        mean_reward = gathered_group_rewards.mean().item()

        # Update completion length
        completion_length = 0
        group_completions = [
            self.tokenizer.batch_decode(batch_completion, skip_special_tokens=True)
            for batch_completion in group_completions_ids
        ]
        for batch_completion in group_completions:
            completion_length += sum([self.get_num_tokens(i) for i in batch_completion]) / len(batch_completion)
        completion_length /= len(group_completions)
        gathered_completion_length = self.accelerator.gather(torch.tensor(completion_length).to(self.accelerator.device))
        mean_completion_length = gathered_completion_length.mean().item()

        # Update KL div
        if None not in [group_token_kl_divs, effective_token_mask]:
            masked_token_kl_div = group_token_kl_divs * effective_token_mask
            mean_kl_div = (masked_token_kl_div.sum(dim=2) / (effective_token_mask.sum(dim=2) + 1e-9)).mean()
            gathered_mean_kl_div = self.accelerator.gather(mean_kl_div)
            mean_kl_div = gathered_mean_kl_div.mean().item()
            
        # Update records
        logging_hook: LoggingHook = self.hooks["LoggingHook"]
        logging_hook.extra_log_bar_data.update(
            {
                "Reward": f"{mean_reward:.2f}",
                "RspLen": f"{mean_completion_length:.2f}",
                "KLDiv": f"{mean_kl_div:.5f}",
            }
        )
        env.tensor_board_writer.add_scalar('Reward/Training', mean_reward, self.training_context.global_step)
        env.tensor_board_writer.add_scalar('Completion Length/Training', mean_completion_length, self.training_context.global_step)
        env.tensor_board_writer.add_scalar('KL Div/Training', mean_kl_div, self.training_context.global_step)
        env.history_recorder.record('training_reward', mean_reward, self.training_context.global_step)
        env.history_recorder.record('training_completion_length', mean_completion_length, self.training_context.global_step)
        env.history_recorder.record('training_kl_div', mean_kl_div, self.training_context.global_step)


    def get_num_tokens(self, completion: str) -> int:
        return len(self.tokenizer.encode(completion))
