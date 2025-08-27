from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hurricore.hooks.hurricore_hook import HurricoreHook


class HurricoreHookList(list):
    def __getitem__(self, index: int | str | slice) -> "HurricoreHook":
        if type(index) in [int, slice]:
            return super().__getitem__(index)
        elif type(index) is str:
            for hook in self:
                if hook.name == index:
                    return hook
            raise ValueError(f"Hook {index} not found")
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    

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
