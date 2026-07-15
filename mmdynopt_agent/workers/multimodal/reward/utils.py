import signal
from typing import Any, Callable

import ray


def _timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")


@ray.remote
def reward_func_timeout_ray(func: Callable, timeout_seconds: int, *args: Any, **kwargs: Any):
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func(*args, **kwargs)
    except TimeoutError:
        return {"score": 0.0, "extra_info": {"is_filter": "1"}}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
