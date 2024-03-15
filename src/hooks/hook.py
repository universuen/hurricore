from src.trainers.trainer import Trainer

class Hook:
    def __init__(self) -> None:
        pass
    
    def training_start(self, trainer: Trainer) -> None:
        pass

    def training_end(self, trainer: Trainer) -> None:
        pass

    def epoch_start(self, trainer: Trainer) -> None:
        pass

    def epoch_end(self, trainer: Trainer) -> None:
        pass

    def iteration_start(self, trainer: Trainer) -> None:
        pass

    def iteration_end(self, trainer: Trainer) -> None:
        pass
    
