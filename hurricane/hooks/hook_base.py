from hurricane.trainers.trainer_base import TrainerBase

class HookBase:
    def __init__(self) -> None:
        pass
    
    def on_training_start(self, trainer: TrainerBase) -> None:
        pass

    def on_training_end(self, trainer: TrainerBase) -> None:
        pass

    def on_epoch_start(self, trainer: TrainerBase) -> None:
        pass

    def on_epoch_end(self, trainer: TrainerBase) -> None:
        pass

    def on_step_start(self, trainer: TrainerBase) -> None:
        pass

    def on_step_end(self, trainer: TrainerBase) -> None:
        pass
