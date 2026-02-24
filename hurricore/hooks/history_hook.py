from __future__ import annotations

from pathlib import Path
import json
import shutil
from time import sleep


from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import Dummy, env, HistoryRecorder



class HistoryHook(HurricoreHook):
    def __init__(
            self, 
            trainer: HurricoreTrainer,
            folder_path: Path,
            interval: int = 1,
        ) -> None:
        super().__init__(
            trainer=trainer,
            folder_path=folder_path,
            interval=interval,
        )     
        # check validity
        assert isinstance(folder_path, Path), f'Invalid History folder path. Expected Path, but got {type(folder_path)}.'
        folder_path.mkdir(parents=True, exist_ok=True)
        self.folder_path = folder_path
        self.interval = interval
    

    def execute_on_trainer_init_end(self) -> None:
        self.recoder = HistoryRecorder(self.folder_path) if self.trainer.accelerator.is_main_process else Dummy()
        env.history_recorder = self.recoder


    def execute_on_step_end(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        if (self.trainer.training_context.global_step + 1) % self.interval != 0:
            return
        self.recoder.record(
            tag='training_loss',
            value=self.trainer.training_context.step_loss.item(),
            global_step=self.trainer.training_context.global_step,
        )


    def execute_on_training_start(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        env.checkpointer.register_action(self._recover_from_checkpoint)


    def execute_on_step_start(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        if self.trainer.training_context.global_step == 0:
            shutil.rmtree(self.folder_path)
            while self.folder_path.exists():
                env.logger.info(f'Waiting for {self.folder_path} to be deleted...')
                sleep(1)
            self.folder_path.mkdir(parents=True)
    

    def _recover_from_checkpoint(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return

        histories = HistoryRecorder.load(self.folder_path)
        
        current_step = self.trainer.training_context.global_step

        for tag, history in histories.items():
            valid_records = [
                record for record in history 
                if record['global_step'] < current_step
            ]
            target_file_path = self.folder_path / f'{tag}.jsonl'
            with target_file_path.open('w', encoding='utf-8') as f:
                for record in valid_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
