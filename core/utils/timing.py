import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s took %.3fs", func.__name__, elapsed)
        return result
    return wrapper
