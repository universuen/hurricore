class Context:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        attributes = ', '.join(f"{key}={value}" for key, value in vars(self).items())
        return f"Context({attributes})"

    def state_dict(self) -> dict:
        return vars(self)

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)
