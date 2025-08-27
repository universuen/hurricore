import threading
import multiprocessing
from functools import wraps
from typing import Literal
from hurricore.utils.functions import hash_anything


class Cacher:
    def __init__(self, mode: Literal['multithreading', 'multiprocessing'] = None):
        self.mode = mode
        self.cache = dict()
        self.lock = None
        if self.mode == 'multithreading':
            self.lock = threading.Lock()
        elif self.mode == 'multiprocessing':
            self.cache = multiprocessing.Manager().dict()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key for the function arguments
            key = hash_anything((args, kwargs))
            # Check cache first
            if key in self.cache:
                return self.cache[key]
            # Execute the function and cache the result
            result = func(*args, **kwargs)     
            # Ensure that the cache is updated safely
            if self.mode == 'multithreading':
                with self.lock:
                    self.cache[key] = result
            elif self.mode == 'multiprocessing':
                self.cache[key] = result
            else:
                self.cache[key] = result
            return result
        
        return wrapper

