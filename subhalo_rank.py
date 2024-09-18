#!/bin/env python

import numpy as np
import h5py
import pytest

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
    for (i, n) in zip(offset, count):
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


@pytest.mark.mpi
def test_subhalo_rank():

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Read VR halos from a small FLAMINGO run (assumes single file catalogue)
    vr_file = "/cosma8/data/dp004/flamingo/Runs/L0100N0180/HYDRO_FIDUCIAL/VR/halos_0006.properties.0"
    with h5py.File(vr_file, "r", driver="mpio", comm=comm) as vr:
        host_id = phdf5.collective_read(vr["hostHaloID"], comm=comm)
        subhalo_id = phdf5.collective_read(vr["ID"], comm=comm)
        subhalo_mass = phdf5.collective_read(vr["Mass_tot"], comm=comm)
    if comm_rank == 0:
        print("Read subhalos")

    field = host_id < 0
    host_id[field] = subhalo_id[field]

    # Compute ranking of subhalos
    subhalo_rank = compute_subhalo_rank(host_id, subhalo_mass, comm)
    if comm_rank == 0:
        print("Computed ranks")

    # Find fraction of VR 'field' halos with rank=0
    nr_field_halos = comm.allreduce(np.sum(field))
    nr_field_rank_nonzero = comm.allreduce(np.sum((field) & (subhalo_rank > 0)))
    fraction = nr_field_rank_nonzero / nr_field_halos
    if comm_rank == 0:
        print(f"Fraction of field halos (hostHaloID<0) with rank>0 is {fraction:.3e}")

    # Sanity check: there should be one instance of each hostHaloID with rank=0
    all_ranks = comm.gather(subhalo_rank)
    all_host_ids = comm.gather(host_id)
    all_ids = comm.gather(subhalo_id)
    if comm_rank == 0:
        all_ranks = np.concatenate(all_ranks)
        all_host_ids = np.concatenate(all_host_ids)
        all_ids = np.concatenate(all_ids)
        all_host_ids[all_host_ids < 0] = all_ids[all_host_ids < 0]
        rank0 = all_ranks == 0
        rank0_hosts = all_host_ids[rank0]
        assert len(rank0_hosts) == len(np.unique(all_host_ids))


if __name__ == "__main__":
    # Run test with "mpirun -np 8 python3 -m mpi4py ./subhalo_rank.py"
    test_subhalo_rank()
