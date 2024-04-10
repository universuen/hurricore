import _path_setup

import shutil
from pathlib import Path

from hurricane.utils import import_config

dummy_config_str = """
from hurricane.utils import ConfigBase

class ConfigA(ConfigBase):
    a = 1
    b = '2'

class ConfigB(ConfigBase):
    c = ConfigA().a + 1
    d = int(ConfigA().b) + 1
"""


def test_import_config():
    # test path import
    parent_folder = Path(__file__).parent
    temp_folder_path = parent_folder / 'temp_configs'
    if temp_folder_path.exists():
        shutil.rmtree(temp_folder_path)
    temp_folder_path.mkdir(parents=True)
    config_path = temp_folder_path / 'dummy_config.py'
    with open(config_path, 'w') as f:
        f.write(dummy_config_str)
    config = import_config(str(config_path.absolute()))
    assert config.ConfigA.a == 1, "path import is not working."
    assert config.ConfigA.b == '2', "path import is not working."
    assert config.ConfigB.c == 2, "path import is not working."
    assert config.ConfigB.d == 3, "path import is not working."
    # test url import
    url = r'https://raw.githubusercontent.com/universuen/hurricane/main/examples/resnet18_on_cifar10/configs/default.py'
    config = import_config(url)
    assert config.TrainerConfig is not None, "url import is not working."
    # test module import
    with open(temp_folder_path / '__init__.py', 'w') as f:
        f.write('')
    config = import_config(f"{temp_folder_path.name}.dummy_config")
    assert config.ConfigA.a == 1, "module import is not working."
    assert config.ConfigA.b == '2', "module import is not working."
    assert config.ConfigB.c == 2, "module import is not working."
    assert config.ConfigB.d == 3, "module import is not working."
    # clean up
    shutil.rmtree(temp_folder_path)
    
    

test_import_config()