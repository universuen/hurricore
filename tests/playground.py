from dataclasses import dataclass, fields
from typing import Any

def my_dataclass(cls):
    # Apply the dataclass decorator to the class
    cls = dataclass(cls)
    
    def to_dict(self) -> dict:
        """Converts the dataclass instance to a dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    # Add the to_dict method to the class
    setattr(cls, "to_dict", to_dict)
    
    return cls

# Example usage
@my_dataclass
class MyClass:
    name: str
    value: int

# Testing the decorated class
instance = MyClass(name="example", value=42)
print(instance.to_dict())  # Should print: {'name': 'example', 'value': 42}
