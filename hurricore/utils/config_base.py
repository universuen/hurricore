import re

from hurricore.utils.attribute_dict import AttributeDict


class ConfigBase(AttributeDict):
    """
    Base class for all config classes. It initializes instance attributes from class attributes.
    It is also a subclass of AttributeDict which provides convenient access to instance attributes via dot notation.
    """
    def __init__(self, **kwargs) -> None:
        for name, value in vars(self.__class__).items():
            if re.match(r'^__.*__$', name) or callable(value):
                continue
            self[name] = value
        super().__init__(**kwargs)
        self.__post_init__()


    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        attributes = ',\n\t'.join([f"{name}={value!r}" for name, value in self.items()])
        return f"{class_name}(\n\t{attributes}\n)"

    
    def __post_init__(self) -> None:
        pass
