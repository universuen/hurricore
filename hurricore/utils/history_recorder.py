from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HistoryRecorder:
    def __init__(self, folder_path: Path | str) -> None:
        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(parents=True, exist_ok=True)

    def record(self, tag: str, value: Any, global_step: int) -> None:
        """Appends a new record to the corresponding jsonl file."""
        target_file_path = self.folder_path / f'{tag}.jsonl'
        
        entry = {'global_step': global_step, 'value': value}
        
        with target_file_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    @staticmethod
    def load(folder_path: Path | str) -> dict[str, list[dict[str, Any]]]:
        """Loads all .jsonl files from the folder into a dictionary."""
        folder_path = Path(folder_path)
        histories = {}

        if not folder_path.exists():
            return histories

        for file_path in folder_path.glob('*.jsonl'):
            data = []
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip(): # Skip empty lines
                            data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping corrupt line {line_num} in {file_path}")
            
            histories[file_path.stem] = data
            
        return histories