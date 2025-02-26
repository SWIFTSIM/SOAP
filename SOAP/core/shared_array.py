#!/bin/env python

from mpi4py import MPI
import numpy as np
import unyt


class SharedArray:
    def __init__(self, local_shape, dtype, comm, units=None):

        self.comm = comm
        self.dtype = np.dtype(dtype)
        self.win = None

        # Determine units
        units = units if units is not None else unyt.dimensionless

        # Find the full size of the array
        n = comm.allreduce(local_shape[0])
        full_shape = (n,) + tuple(local_shape[1:])

        # Find amount of memory to allocate on this rank
        local_elements = 1
        for s in local_shape:
            local_elements *= s

        # Find amount of memory to allocate in total
        full_elements = 1
        for s in full_shape:
            full_elements *= s

        # Allocate shared memory window
        self.win = MPI.Win.Allocate_shared(
            local_elements * self.dtype.itemsize, self.dtype.itemsize, comm=comm
        )

        # Make a numpy array to access the full array
        buf, itemsize = self.win.Shared_query(0)
        nbytes_all = full_elements * itemsize
        buf = MPI.memory.fromaddress(buf.address, nbytes_all)
        self.full = np.ndarray(buffer=buf, dtype=self.dtype, shape=full_shape)

        # Make a numpy array to access the local part of the array
        buf, itemsize = self.win.Shared_query(comm.Get_rank())
        self.local = np.ndarray(buffer=buf, dtype=self.dtype, shape=local_shape)

        # Wrap the numpy arrays in unyt arrays
        self.full = unyt.unyt_array(self.full, units=units)
        self.local = unyt.unyt_array(self.local, units=units)

    def sync(self):
        self.win.Sync()

    def free(self):
        if self.win is not None:
            self.win.Free()
            self.win = None

    def __del__(self):
        if self.win is not None:
            print("ERROR: should not rely on __del__ to free shared memory windows!")
        self.free()
