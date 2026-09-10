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

    GB = 1024**3
    mem = psutil.virtual_memory()

    total_mem_gb = mem.total / GB
    free_mem_gb = mem.available / GB

    return total_mem_gb, free_mem_gb


def get_peak_rss_gb():
    """
    Peak resident set size of this process, in GB.

    TEMPORARY (issue 64): used to check the effect of sharing and evicting the
    particle arrays on per rank memory. Strip this out after testing.

    Returns None if VmHWM cannot be read.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None
