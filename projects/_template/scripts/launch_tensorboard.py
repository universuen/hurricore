import _path_setup  # noqa: F401

from pathlib import Path
import shutil
import os

from hurricore.utils import launch_tensor_board

TARGET_CONFIG_NAMES = [
    "debug",
]

def main() -> None:
    folder_path = Path(__file__).parents[1] / 'data' / 'tensorboards'
    temp_folder_path = folder_path / '_temp'

    if temp_folder_path.exists():
        shutil.rmtree(temp_folder_path)
    temp_folder_path.mkdir()

    if len(TARGET_CONFIG_NAMES) > 0:
        for config_name in TARGET_CONFIG_NAMES:
            print(f"Linking {config_name}")
            if not (folder_path / config_name).exists():
                print(f"Config {config_name} not found")
                continue
            os.symlink(folder_path / config_name, temp_folder_path / config_name)
    else:
        temp_folder_path = folder_path

    url = launch_tensor_board(temp_folder_path)
    print(f"Tensorboard is launched at {url}")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        shutil.rmtree(temp_folder_path)


if __name__ == "__main__":
    main()