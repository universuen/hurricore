from hurricane.trainers.trainer_base import TrainerBase


class HookBase:
    def __init__(self, trainer: TrainerBase) -> None:
        self.trainer = trainer
    
    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_epoch_start(self) -> None:
        pass

    def on_epoch_end(self) -> None:
        pass

    def on_step_start(self) -> None:
        pass

    def on_step_end(self) -> None:
        pass
