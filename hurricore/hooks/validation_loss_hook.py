from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import env, AttributeDict, Checkpointer, HuggingfaceModelSaver


class ValidationLossHook(HurricoreHook):
    def __init__(
        self,
        trainer: HurricoreTrainer,
        data_loader: DataLoader,
        interval: int = 10,
    ) -> None:
        super().__init__(
            trainer=trainer,
            data_loader=data_loader,
            interval=interval,
        )
        self.interval = interval
    

    def execute_on_trainer_init_end(self) -> None:
        self.data_loader = self.trainer.accelerator.prepare(self.init_context.data_loader)
        self.records = AttributeDict(
            best_loss=float("inf"),
            best_step=0,
        )
        self.trainer.accelerator.register_for_checkpointing(self.records)


    def execute_on_step_end(self) -> None:
        if (self.trainer.training_context.global_step + 1) % self.interval != 0:
            return
        env.logger.info(f'Evaluating model at step {self.trainer.training_context.global_step + 1}...')
        self.evaluate()


    def execute_on_training_end(self) -> None:
        env.logger.info(f'Evaluating model at step {self.trainer.training_context.global_step + 1}...')
        self.evaluate()


    def evaluate(self) -> None:
        for model in self.trainer.models:
            model.eval()
            
        all_losses = []

        data_loader = self.data_loader
        if self.trainer.accelerator.is_main_process:
            data_loader = tqdm(data_loader, desc="Computing validation loss")
        with torch.no_grad():
            for batch in data_loader:
                loss = self.trainer.compute_loss(batch)
                all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        mean_loss = self.trainer.accelerator.gather(all_losses).mean().item()
        if mean_loss < self.records.best_loss:
            self.records.best_loss = mean_loss
            self.records.best_step = self.trainer.training_context.global_step
            if isinstance(env.checkpointer, Checkpointer):
                ckpt_path = env.checkpointer.save_state("ckpt_lowest_val_loss")
                env.logger.info(f"Saved best checkpoint to {ckpt_path}")
            if isinstance(env.huggingface_model_saver, HuggingfaceModelSaver):
                model_saving_path = env.huggingface_model_saver.save("lowest_val_loss_model")
                env.logger.info(f"Saved best model to {model_saving_path}")

        env.logger.info(
            f"Validation loss: {mean_loss:.4f} "
            f"(best: {self.records.best_loss:.4f} at step {self.records.best_step + 1})"
        )
        env.history_recorder.record(
            tag='validation_loss',
            value=mean_loss,
            global_step=self.trainer.training_context.global_step + 1
        )
        env.tensor_board_writer.add_scalar(
            tag='Loss/Validation',
            scalar_value=mean_loss,
            global_step=self.trainer.training_context.global_step + 1,
        )
