#!/bin/env python
import os
import warnings

from mpi4py import MPI
import numpy as np
import pytest
import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.catalogue_readers.read_subfind import (
    read_gadget4_groupnr,
    locate_files,
)

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()


@pytest.mark.mpi
def test_read_gadget4_groupnr():
    """
    Read in Gadget-4 group numbers and compute the number of particles
    in each group. This is then compared with the input catalogue as a
    sanity check on the group membershp files.
    """

    # Test with FLAMINGO data on cosma8
    test_data_dir = "/cosma8/data/dp004/dc-mcgi1/SOAP/TEST_DATA/SubFind"
    run = "L1000N1800/DMO_FIDUCIAL"
    snap_nr = 77
    basename = f"{test_data_dir}/{run}/snapdir_{snap_nr:03d}/snapshot_{snap_nr:03d}"
    # Skip this test if we can't find the data (too large to download)
    skip = False
    if comm_rank == 0:
        print(os.path.dirname(basename))
        if not os.path.exists(os.path.dirname(basename)):
            skip = True
    skip = comm.bcast(skip)
    if skip:
        return

    n_halo, ids, grnr = read_gadget4_groupnr(basename)
    del ids  # Don't need the particle IDs

    # Find maximum group number
    max_grnr = comm.allreduce(np.amax(grnr), op=MPI.MAX)
    nr_groups_from_grnr = max_grnr + 1
    if comm_rank == 0:
        print(f"Number of groups from membership files = {nr_groups_from_grnr}")

    # Discard particles in no group
    keep = grnr >= 0
    grnr = grnr[keep]

    # Compute group sizes
    nbound_from_grnr = psort.parallel_bincount(grnr, comm=comm)

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string = locate_files(basename)
    else:
        snap_format_string = None
        group_format_string = None
    snap_format_string, group_format_string = comm.bcast(
        (snap_format_string, group_format_string)
    )

    # Read group sizes from the group catalogue
    subtab = phdf5.MultiFile(group_format_string, file_nr_attr=("Header", "NumFiles"))
    nbound_from_subtab = subtab.read("Subhalo/SubhaloLen")

    # Find number of groups in the subfind output
    nr_groups_from_subtab = comm.allreduce(len(nbound_from_subtab))
    if comm_rank == 0:
        print(f"Number of groups from fof_subhalo_tab = {nr_groups_from_subtab}")
        if nr_groups_from_subtab != nr_groups_from_grnr:
            print("Number of groups does not agree!")
            comm.Abort(1)

    # Ensure nbound arrays are partitioned the same way
    nr_per_rank = comm.allgather(len(nbound_from_subtab))
    nbound_from_grnr = psort.repartition(
        nbound_from_grnr, ndesired=nr_per_rank, comm=comm
    )

    # Compare
    nr_different = comm.allreduce(np.sum(nbound_from_grnr != nbound_from_subtab))
    if comm_rank == 0:
        print(f"Number of group sizes which differ = {nr_different} (should be 0!)")
