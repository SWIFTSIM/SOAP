#!/bin/env python

import numpy as np
import h5py
import pytest
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.property_calculation.subhalo_rank import compute_subhalo_rank


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
