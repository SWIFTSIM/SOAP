#!/bin/env python
import os
import warnings

from mpi4py import MPI
import numpy as np
import pytest
import unyt
import virgo.mpi.parallel_sort as psort

from SOAP.catalogue_readers.read_rockstar import read_rockstar_groupnr, locate_files, read_group_file

import helpers

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

@pytest.mark.mpi
def test_read_rockstar_groupnr():
    """
    Read in rockstar group numbers and compute the number of particles
    in each group. This is then compared with the input catalogue as a
    sanity check on the group membershp files.
    """

    # Test with FLAMINGO data on cosma8, skip this test if we can't find it
    test_data_dir = '/cosma8/data/dp004/dc-mcgi1/SOAP/TEST_DATA/ROCKSTAR'
    run = 'L1000N0900/DMO_FIDUCIAL'
    snap_nr = 77
    basename = f"{test_data_dir}/{run}/snapshot_{snap_nr:04d}/halos_{snap_nr:04d}"
    # Skip this test if we can't find the data (too large to download)
    skip = False
    if comm_rank == 0:
        if not os.path.exists(os.path.dirname(basename)):
            skip = True
    skip = comm.bcast(skip)
    if skip:
        return

    # Catch deprecation warning from VirgoDC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _, ids, grnr = read_rockstar_groupnr(basename)
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

