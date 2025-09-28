from __future__ import annotations

import logging
import sys
from pathlib import Path


class Logger(logging.Logger):
    def __init__(
            self,
            name: str,
            level: int | str = logging.INFO,
            log_file_path: str | Path = None,
    ) -> None:
        super().__init__(name, level=level)
        self.log_file_path = log_file_path
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(name)s][%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(formatter)
        s_handler.setLevel(level)
        self.addHandler(s_handler)

        if log_file_path is not None:
            # Create the log file directory if it doesn't exist
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            f_handler = logging.FileHandler(log_file_path, 'a')
            f_handler.setFormatter(formatter)
            f_handler.setLevel(level)
            self.addHandler(f_handler)


    def set_level(self, level: int | str) -> None:
        self.setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)
