from contextlib import contextmanager
import multiprocessing

@contextmanager
def set_multiprocessing_start_method(method: str):
    # Store the original multiprocessing start method
    original_method = multiprocessing.get_start_method()
    # Set the new start method
    multiprocessing.set_start_method(method, force=True)
    try:
        yield
    finally:
        # Reset to the original start method
        multiprocessing.set_start_method(original_method, force=True)
