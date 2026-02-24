
from hurricore.utils import ConfigBase


class ConfigA(ConfigBase):
    a = 1
    b = '2'


class ConfigB(ConfigBase):
    c = ConfigA().a + 1
    d = int(ConfigA().b) + 1
