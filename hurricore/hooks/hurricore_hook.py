import inspect
from typing import TYPE_CHECKING

from hurricore.utils import AttributeDict
if TYPE_CHECKING:
    from hurricore.trainers.hurricore_trainer import HurricoreTrainer


class HurricoreHook:
    """Base hook class for Hurricore trainers.
    This class provides hook points that will be called during different stages of training.
    Subclasses should override the execute_on_* methods to implement custom behavior.
    """
    def __init__(self, trainer, name: str = None) -> None:
        """Initialize the hook with the trainer."""
        self.trainer: HurricoreTrainer = trainer
        self.init_context = AttributeDict()
        self.name = name or self.__class__.__name__

    
    def __post_init__(self) -> None:
        """Called after HurricoreTrainer is initialized."""
        pass


    def execute_on_training_start(self) -> None:
        pass


    def execute_on_training_end(self) -> None:
        pass
    

    def execute_on_epoch_start(self) -> None:
        pass


    def execute_on_epoch_end(self) -> None:
        pass


    def execute_on_step_start(self) -> None:
        pass
    

    def execute_on_step_end(self) -> None:
        pass
    

    def __repr__(self):
        cls = self.__class__
        file_path = inspect.getfile(cls)
        _, line_number = inspect.getsourcelines(cls)
        return f"<{self.__class__.__name__} {file_path}:{line_number}>"
