#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5


def compute_subhalo_rank(host_id, subhalo_mass, comm):
    """
    Given a subhalo catalogue distributed over MPI communicator
    comm, compute the ranking by mass of subhalos within their
    host halos.

    Returns rank array where 0=most massive subhalo in halo.
    This has the same number of elements and distribution over
    MPI ranks as the input arrays.

    host_id      - id of the host  halo
    subhalo_mass - mass to use for ranking
    comm         - MPI communicator to use
    """

    # Record global array indexes of the subhalos so we can restore the ordering later
    nr_local_subhalos = len(host_id)
    subhalo_index = np.arange(nr_local_subhalos, dtype=int)
    nr_prev_subhalos = comm.scan(nr_local_subhalos) - nr_local_subhalos
    subhalo_index += nr_prev_subhalos

    # Create the sort key
    sort_key_t = np.dtype([("host_id", np.int64), ("mass", np.float32)])
    sort_key = np.ndarray(nr_local_subhalos, dtype=sort_key_t)
    sort_key["host_id"] = host_id
    sort_key["mass"] = -subhalo_mass  # negative for descending order
    del subhalo_mass

    # Obtain sorting order
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key

    # Sort the subhalo indexes and hosts
    subhalo_index = psort.fetch_elements(subhalo_index, order, comm=comm)
    host_id = psort.fetch_elements(host_id, order, comm=comm)
    del order

    # Allocate storage for subhalo rank
    subhalo_rank = -np.ones(nr_local_subhalos, dtype=np.int32)

    # Find ranges of subhalos in the same host and assign ranks by mass within this MPI rank
    unique_host, offset, count = np.unique(
        host_id, return_counts=True, return_index=True
    )
    del host_id
    for i, n in zip(offset, count):
        subhalo_rank[i : i + n] = np.arange(n, dtype=np.int32)
    assert np.all(subhalo_rank >= 0)

    # Find the last host ID on each rank and the number of subhalos it contains
    if nr_local_subhalos > 0:
        last_host_id = unique_host[-1]
        last_host_count = count[-1]
    else:
        last_host_id = -1
        last_host_count = 0
    last_host_id = comm.allgather(last_host_id)
    last_host_count = comm.allgather(last_host_count)

    # Now we need to check if any previous MPI rank's last host id is the same as
    # our first. If so, we'll need to increment the ranking of all subhalos in
    # our first host.
    if nr_local_subhalos > 0:
        for prev_rank in range(comm.Get_rank()):
            if (
                last_host_count[prev_rank] > 0
                and last_host_id[prev_rank] == unique_host[0]
            ):
                # Our first host is split between MPI ranks
                subhalo_rank[: count[0]] += last_host_count[prev_rank]

    # Now restore the original ordering
    order = psort.parallel_sort(subhalo_index, return_index=True, comm=comm)
    subhalo_rank = psort.fetch_elements(subhalo_rank, order, comm=comm)

    return subhalo_rank
