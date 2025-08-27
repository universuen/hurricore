from torch.nn import SyncBatchNorm

from hurricore.hooks import HurricoreHook


class SyncBatchNormHook(HurricoreHook):
    def execute_on_training_start(self) -> None:
        self.trainer.models = tuple(SyncBatchNorm.convert_sync_batchnorm(model) for model in self.trainer.models)
        # check if all models with name containing BatchNorm have been converted
        for model in self.trainer.models:
            for name, module in model.named_modules():
                type_name = type(module).__name__
                if "BatchNorm" in type_name:
                    assert type(module) is SyncBatchNorm, f"{name} is not converted to SyncBatchNorm."
