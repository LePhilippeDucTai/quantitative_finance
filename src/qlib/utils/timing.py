import functools as ft
import os
import time
from typing import Callable

import psutil
from qlib.utils.logger import logger
from typing_extensions import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def elapsed_since(start):
    elapsed = time.perf_counter() - start
    if elapsed < 1:
        return str(round(elapsed * 1000, 2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed / 60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def format_bytes(_bytes):
    if abs(_bytes) < 1000:
        return str(_bytes) + "B"
    elif abs(_bytes) < 1e6:
        return str(round(_bytes / 1e3, 2)) + "kB"
    elif abs(_bytes) < 1e9:
        return str(round(_bytes / 1e6, 2)) + "MB"
    else:
        return str(round(_bytes / 1e9, 2)) + "GB"


def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss


def time_it(func: Callable[P, T]) -> Callable[P, T]:
    @ft.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        function_name = ".".join([func.__module__, func.__qualname__])
        logger.info(f"### Processing in {function_name} ...", end="\n")
        t1 = time.perf_counter()
        rss_before = get_process_memory()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(t1)
        rss_after = get_process_memory()
        logger.success(
            "\t--> Function ended : {:>8}  RSS: {:>8} | time: {:>8}".format(
                "<" + function_name + ">",
                format_bytes(rss_after - rss_before),
                elapsed_time,
            )
        )
        return result

    return wrapper
