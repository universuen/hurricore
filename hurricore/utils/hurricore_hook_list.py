from typing import Any, Generic, TypeVar, Iterator


T = TypeVar('T')


class HurricoreHookList(list, Generic[T]):
    def __getitem__(self, index: int | str | slice) -> T:
        if type(index) in [int, slice]:
            return super().__getitem__(index)
        elif type(index) is str:
            for hook in self:
                if hook.name == index:
                    return hook
            raise ValueError(f"Hook {index} not found")
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()

    def __contains__(self, key: str | Any) -> bool:
        if type(key) is not str:
            return super().__contains__(key)
        for hook in self:
            if hook.name == key:
                return True
        return False
    

    def __delitem__(self, index: int | str | slice):
        if type(index) is str:
            for hook in self:
                if hook.name == index:
                    self.remove(hook)
                    return
            raise ValueError(f"Hook {index} not found")
        else:
            super().__delitem__(index)
