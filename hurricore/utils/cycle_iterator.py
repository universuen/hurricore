from typing import Iterable, Iterator

class CycleIterator(Iterator):
    def __init__(self, iterable: Iterable, max_length: int) -> None:
        """
        Initialize the CycleIterator.

        Args:
            iterable (Iterable): The iterable to cycle through.
            max_length (int): Maximum number of items to iterate through.
        """
        self.iterable = iterable
        self.max_length = max_length
        self.reset()

    def __iter__(self) -> Iterator:
        """Every time this is called, start a new iteration."""
        self.reset()  # Reset iterator every time iteration starts
        return self

    def reset(self) -> None:
        """Reset the internal state of the iterator."""
        self.iter_count = 0
        self.iterator = iter(self.iterable)

    def __next__(self):
        """
        Return the next item from the iterable.
        When max_length is reached, stop the iteration.
        """
        if self.iter_count >= self.max_length:
            raise StopIteration

        try:
            item = next(self.iterator)
        except StopIteration:
            # Restart the iterator if we've exhausted the current one
            self.iterator = iter(self.iterable)
            item = next(self.iterator)

        self.iter_count += 1
        return item