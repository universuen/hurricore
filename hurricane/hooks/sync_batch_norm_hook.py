from torch.nn import SyncBatchNorm

from hurricane.hooks import HookBase


class SyncBatchNormHook(HookBase):
    def on_training_start(self) -> None:
        self.trainer.models = tuple(SyncBatchNorm.convert_sync_batchnorm(model) for model in self.trainer.models)
