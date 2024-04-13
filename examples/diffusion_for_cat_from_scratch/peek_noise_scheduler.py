import _path_setup  # noqa: F401

from hurricane.utils import import_config

if __name__ == '__main__':
    config = import_config('configs.default')
    
    
    