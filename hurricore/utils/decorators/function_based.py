import time
from functools import wraps
import warnings
from typing import Literal, Callable

from hurricore.utils.decorators.frequency_controller import FrequencyController
from hurricore.utils.decorators.cacher import Cacher


def cache(
    mode: Literal["multithreading", "multiprocessing", None] = None
) -> Callable:
    """
    A decorator factory that creates a caching decorator based on the provided mode.

    :param mode: Specifies the caching mode. Can be 'multithreading', 'multiprocessing', or None.
    :return: A decorator that caches the result of the function based on the specified mode.
    """
    # Return the caching decorator based on the mode
    cacher = Cacher(mode=mode)

    def decorator(func: Callable) -> Callable:
        return cacher(func)
    
    return decorator
    


def control_frequency(interval: float, mode: Literal['multithreading', 'multiprocessing'] = None):
    controller = FrequencyController(
        interval=interval,
        mode=mode,
    )
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return controller(func)(*args, **kwargs)
        return wrapper
    return decorator


def record_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time, result
    return wrapper


def retry_when_failed(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        print(f"Attempt {attempts} failed with exception: {e}. Tearing down.")
                        raise
                    print(f"Attempt {attempts} failed with exception: {e} Retrying in {delay} seconds.")
                    time.sleep(delay)
        return wrapper
    return decorator


def safely_run(fallback_func: callable = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exception:
                if fallback_func is not None:
                    return fallback_func(exception)
                else:
                    print(f"Error: {exception}")
                    return None
        return wrapper
    return decorator


def ignore_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper
