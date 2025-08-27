from pathlib import Path

from hurricore.utils import launch_tensor_board


if __name__ == '__main__':
    folder_path = Path(__file__).parent / 'data' / 'tensor_boards'
    url = launch_tensor_board(folder_path)
    print(f"Tensorboard is launched at {url}")
    while True:
        pass
