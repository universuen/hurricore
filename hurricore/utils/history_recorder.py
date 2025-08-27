from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle


class HistoryRecorder:
    def __init__(self, folder_path: Path) -> None:
        self.folder_path = folder_path
    

    def record(self, tag: str, value: Any, global_step: int) -> None:
        target_file_path = self.folder_path / f'{tag}.pkl'
        if not target_file_path.exists():
            history = [{'global_step': global_step, 'value': value}]
        else:
            with target_file_path.open('rb') as f:
                history: list = pickle.load(f)
            history.append({'global_step': global_step, 'value': value}) 
        with target_file_path.open('wb') as f:
            pickle.dump(history, f)
    

    @staticmethod
    def load(folder_path: Path) -> dict[str, list]:
        folder_path = Path(folder_path)
        histories = {}
        for file_path in folder_path.iterdir():
            with file_path.open('rb') as f:
                try:
                    histories[file_path.stem] = pickle.load(f)
                except EOFError:
                    print(f'History file {file_path} is corrupted. Deleting it.')
                    file_path.unlink()
        return histories
