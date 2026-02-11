import time
import functools


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.debug(f"{func.__name__!r} finished in {run_time:.4f} secs")
        return value

    return wrapper_timer


if __name__ == "__main__":
    logging.debug("")
