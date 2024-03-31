import _path_setup

from hurricane.utils import get_total_parameters
from models import Generator, Discriminator


print(get_total_parameters(Generator()))
print(get_total_parameters(Discriminator()))
