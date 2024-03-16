from hurricane.trainers.trainer_base import TrainerBase

class HookBase:
    def __init__(self) -> None:
        pass
    
    def training_start(self, trainer: TrainerBase) -> None:
        pass

    def training_end(self, trainer: TrainerBase) -> None:
        pass

    def epoch_start(self, trainer: TrainerBase) -> None:
        pass

    def epoch_end(self, trainer: TrainerBase) -> None:
        pass

    def iteration_start(self, trainer: TrainerBase) -> None:
        pass

    def iteration_end(self, trainer: TrainerBase) -> None:
        pass
