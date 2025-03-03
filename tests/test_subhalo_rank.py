#!/bin/env python

import numpy as np
import h5py
import pytest
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.property_calculation.subhalo_rank import compute_subhalo_rank

import helpers

@pytest.mark.mpi
@helpers.requires('HBT_output/018/SubSnap_018.0.hdf5')
def test_subhalo_rank(filename):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Read HBT halos from a small DMO run
    with h5py.File(filename, "r", driver="mpio", comm=comm) as file:
        sub = phdf5.collective_read(file['Subhalos'], comm=comm)
    if comm_rank == 0:
        print("Read subhalos")
    host_id = sub['HostHaloId']
    subhalo_id = sub['TrackId']
    subhalo_mass = sub['Mbound']
    depth = sub['Depth']

    field = depth == 0
    host_id[field] = subhalo_id[field]

    # Compute ranking of subhalos
    subhalo_rank = compute_subhalo_rank(host_id, subhalo_mass, comm)
    if comm_rank == 0:
        print("Computed ranks")

    # Find fraction of 'field' halos with rank=0
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
