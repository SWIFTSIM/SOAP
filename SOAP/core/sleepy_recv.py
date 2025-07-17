#!/bin/env python

import time


def sleepy_recv(comm, tag, status=None):
    """
    Wait for a message without keeping a core spinning so that we leave
    the core available to run jobs and release the GIL. Checks for
    incoming messages at exponentially increasing intervals starting
    at min_delay up to a limit of max_delay. Sleeps between checks.
    """
    min_delay = 1.0e-5
    max_delay = 5.0
    request = comm.irecv(tag=tag)
    delay = min_delay
    while True:
        completed, message = request.test(status=status)
        if completed:
            return message
        delay = min(max_delay, delay * 2)
        time.sleep(delay)
