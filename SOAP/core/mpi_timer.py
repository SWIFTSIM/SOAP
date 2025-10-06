#!/bin/env python

import time


class MPITimer:
    def __init__(self, name, comm):
        self.name = name
        self.comm = comm

    def __enter__(self):
        self.comm.barrier()
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            self.comm.barrier()
            self.t1 = time.time()
            elapsed = self.t1 - self.t0
            if self.comm.Get_rank() == 0:
                print(f"{self.name} took {elapsed:.2f} seconds")
