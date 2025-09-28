from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from hurricore.utils.logger import Logger
from hurricore.utils.checkpointer import Checkpointer
from hurricore.utils.history_recorder import HistoryRecorder
from hurricore.utils.singleton import Singleton
from hurricore.utils.dummy import Dummy


class Environment(metaclass=Singleton):
    config_path: str
    logger: Logger
    process_logger: Logger
    history_recorder: HistoryRecorder
    tensor_board_writer: SummaryWriter
    # lr_status: str
    checkpointer: Checkpointer
    accelerator: Accelerator


    def __getattr__(self, name: str):
        return Dummy()


env = Environment()
