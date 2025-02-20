#!/bin/env python

try:
    import psutil
except ImportError:
    psutil = None


def get_memory_use():
    """
    Report memory use on this compute node
    """

    # Do nothing if psutil is not installed
    if psutil is None:
        return None, None

    GB = 1024 ** 3
    mem = psutil.virtual_memory()

    total_mem_gb = mem.total / GB
    free_mem_gb = mem.available / GB

    return total_mem_gb, free_mem_gb
