import time
import threading
import multiprocessing
from typing import Literal
from functools import wraps


class _RegularFrequencyController:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_call_time = 0.0
    
    def control(self):
        current_time = time.time()
        if current_time - self.last_call_time < self.interval:
            sleep_time = self.interval - (current_time - self.last_call_time)
            time.sleep(sleep_time)


class _ThreadSafeFrequencyController:
    def __init__(self, interval):
        self.interval = interval
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def control(self):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_call_time < self.interval:
                sleep_time = self.interval - (current_time - self.last_call_time)
                time.sleep(sleep_time)
            self.last_call_time = time.time()


class _ProcessSafeFrequencyController:
    def __init__(self, interval):
        self.interval = interval
        self.last_call_time = multiprocessing.Value('d', 0.0)
        self.lock = multiprocessing.Lock()

    def control(self):
        with self.lock: 
            current_time = time.time()
            if current_time - self.last_call_time.value < self.interval:
                sleep_time = self.interval - (current_time - self.last_call_time.value)
                time.sleep(sleep_time)
            self.last_call_time.value = time.time()


class FrequencyController:
    def __init__(
            self, 
            interval: float, 
            mode: Literal['multithreading', 'multiprocessing'] = None,
        ):
        self.interval = interval
        self.mode = mode
        self.regular_controller = _RegularFrequencyController(interval)
        self.thread_controller = _ThreadSafeFrequencyController(interval)
        self.process_controller = _ProcessSafeFrequencyController(interval)


    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.mode is None:
                self.regular_controller.control()
            elif self.mode == 'multithreading':
                self.thread_controller.control()
            elif self.mode == 'multiprocessing':
                self.process_controller.control()
            else:
                raise ValueError(f'Invalid mode: {self.mode}')

            return func(*args, **kwargs)

        return wrapper
