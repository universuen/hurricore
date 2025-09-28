import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import sleep

from torch.utils.tensorboard import SummaryWriter

from hurricore.hooks import HurricoreHook
from hurricore.trainers import HurricoreTrainer
from hurricore.utils import (
    Dummy,
    env,
    auto_name,
    launch_tensor_board,
)


class TensorBoardHook(HurricoreHook):
    
    def __init__(
        self, 
        trainer: HurricoreTrainer,
        interval: int = 1,
        folder_path: Path = None,
        record_param_and_grad: bool = False,
        auto_launch: bool = False,
    ) -> None:
        super().__init__(
            trainer=trainer,
            interval=interval,
            folder_path=folder_path,
            record_param_and_grad=record_param_and_grad,
            auto_launch=auto_launch,
        )
        # check validity
        assert isinstance(folder_path, Path), f'Invalid TensorBoard folder path. Expected Path, but got {type(folder_path)}.'
        folder_path.mkdir(parents=True, exist_ok=True)
        # setup self
        self.folder_path = folder_path
        self.record_param_and_grad = record_param_and_grad
        self.auto_launch = auto_launch
        self.interval = interval
    

    def execute_on_trainer_init_end(self) -> None:
        self.writer = SummaryWriter(log_dir=self.folder_path) if self.trainer.accelerator.is_main_process else Dummy()
        self.thread_pool_executor = ThreadPoolExecutor() if self.trainer.accelerator.is_main_process else Dummy()
        env.tensor_board_writer = self.writer
    

    def execute_on_step_end(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        if (self.trainer.context.global_step + 1) % self.interval != 0:
            return
        # Record LR
        for optimizer_name, optimizer in zip(auto_name(self.trainer.optimizers), self.trainer.optimizers):
            self.writer.add_scalar(
                tag=f'LR/{optimizer_name}',
                scalar_value=optimizer.param_groups[0]['lr'],
                global_step=self.trainer.context.global_step,
            )
        # Record loss
        self.writer.add_scalar(
            tag='Loss/Training',
            scalar_value=self.trainer.context.step_loss.item(),
            global_step=self.trainer.context.global_step,
        )
    
    
    def execute_on_training_start(self) -> None:
        env.checkpointer.register_action(self._recover_from_checkpoint)
        if self.auto_launch and self.trainer.accelerator.is_main_process:
            url = launch_tensor_board(self.folder_path.parent)
            env.logger.info(f"Tensorboard is launched at {url}")


    def _collect_param_grad_data(self) -> list:
        param_grad_data = []
        models = self.trainer.originals.models
        for model_name, model in zip(auto_name(models), models):
            for layer_name, param in model.named_parameters():
                if param.grad is not None:
                    param_data = {
                        'model_name': model_name,
                        'layer_name': layer_name,
                        'param': param.detach().float().cpu().clone(),
                        'grad': param.grad.detach().float().cpu().clone()
                    }
                    param_grad_data.append(param_data)
        return param_grad_data  


    def execute_on_epoch_end(self) -> None:
        if not self.record_param_and_grad:
            return
        param_grad_data = self._collect_param_grad_data()
        self.thread_pool_executor.submit(
            self._record_parameters_and_gradients, 
            param_grad_data, 
            self.trainer.context.global_step
        )


    def _record_parameters_and_gradients(self, param_grad_data, step) -> None:
        for data in param_grad_data:
            self.writer.add_histogram(
                f"Parameters/{data['model_name']}-{data['layer_name']}", 
                data['param'],
                step
            )
            self.writer.add_histogram(
                f"Gradients/{data['model_name']}-{data['layer_name']}", 
                data['grad'],
                step
            )
        self.writer.flush()


    def execute_on_step_start(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        if self.trainer.context.global_step == 0:
            shutil.rmtree(self.folder_path)
            while self.folder_path.exists():
                env.logger.info(f'Waiting for {self.folder_path} to be deleted...')
                sleep(1)
            self.folder_path.mkdir()


    def execute_on_training_end(self) -> None:
        self.thread_pool_executor.shutdown(wait=True)
        self._shutdown_writer()


    def _recover_from_checkpoint(self) -> None:
        if not self.trainer.accelerator.is_main_process:
            return
        self._shutdown_writer()
        self.writer = SummaryWriter(
            log_dir=self.folder_path, 
            purge_step=self.trainer.context.global_step
        )
        env.tensor_board_writer = self.writer

    
    def _shutdown_writer(self) -> None:
        self.writer.close()
        sleep(2)
