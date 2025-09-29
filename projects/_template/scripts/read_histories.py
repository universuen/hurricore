import _path_setup  # noqa: F401

from hurricore.utils import HistoryRecorder

from configs import debug as config


def main() -> None:
    histories = HistoryRecorder.load(config.PathConfig().histories)
    print(histories)


if __name__ == "__main__":
    main()
