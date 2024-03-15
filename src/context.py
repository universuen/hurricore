class Context:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        attributes = ', '.join(f"{key}={value}" for key, value in vars(self).items())
        return f"Context({attributes})"
