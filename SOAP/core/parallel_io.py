#!/bin/env python

import h5py
import virgo.mpi.parallel_hdf5 as phdf5

# True if h5py was built without MPI support, in which case we can't use the
# mpio driver and have to fall back to writing output files on a single rank.
SERIAL_HDF5 = phdf5.SERIAL_HDF5


def open_collective(filename, mode, comm):
    """
    Open a file which all ranks in comm will access.

    With parallel HDF5 this returns a file handle opened in MPI mode on every
    rank. Without it, files opened for reading are opened independently on
    each rank, and files being written are opened on rank 0 only with the
    other ranks getting None. Any code which touches the returned object
    directly (creating groups, writing attributes) must check for None.
    """
    if SERIAL_HDF5:
        if mode == "r":
            return h5py.File(filename, "r")
        if comm.Get_rank() == 0:
            return h5py.File(filename, mode)
        return None
    return h5py.File(filename, mode, driver="mpio", comm=comm)


def close_collective(outfile, comm):
    """
    Close a file opened with open_collective().
    """
    comm.barrier()
    if outfile is not None:
        outfile.close()
    comm.barrier()


def collective_write(group, name, data, comm, **kwargs):
    """
    Write a dataset by concatenating the contributions from all ranks in comm
    along the first axis.

    All ranks in comm must call this, including those with no data to write.
    """
    if SERIAL_HDF5:
        return phdf5.serial_collective_write(group, name, data, comm, **kwargs)
    return phdf5.collective_write(group, name, data, comm, **kwargs)
