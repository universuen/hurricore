from hurricore.hooks import HistoryHook
from hurricore.utils import load_config


config = load_config('configs.debug')


if __name__ == '__main__':
    histories = HistoryHook.load(config.PathConfig().histories)
    print(histories)
