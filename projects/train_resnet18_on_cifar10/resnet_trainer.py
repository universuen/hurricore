from logging import Logger

from accelerate import Accelerator
from torch.nn.functional import cross_entropy

from torch import Tensor
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hurricane.trainers.trainer import Trainer
from hurricane.hooks.logger_hook import LoggerHook


class ResNetTrainer(Trainer):
    def __init__(
        self, 
        model: Module, 
        data_loader: DataLoader, 
        optimizer: Optimizer, 
        accelerator: Accelerator,
        logger: Logger,
    ) -> None:
        super().__init__(model, data_loader, optimizer, accelerator)
        self.hooks = [
            LoggerHook(logger),
        ]
        
    def compute_loss(self) -> Tensor:
        inputs, labels = self.ctx.batch
        outputs = self.model(inputs)
        loss = cross_entropy(outputs, labels)
        return loss
