import _path_setup  # noqa: F401

from hurricore.utils import HistoryRecorder

from configs import debug as config


if __name__ == '__main__':
    histories = HistoryRecorder.load(config.PathConfig().histories)
    print(histories)
