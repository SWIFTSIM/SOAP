#!/bin/env python

import numpy as np
import virgo.util.peano as peano
import virgo.mpi.parallel_sort as psort
import unyt
from mpi4py import MPI


def peano_decomposition(boxsize, centres, nr_chunks, comm):
    """
    Gadget style domain decomposition using Peano-Hilbert curve.
    Allows an arbitrary number of chunks and tries to put equal
    numbers of halos in each chunk.

    The array of halo centres is assumed to be distributed over
    communicator comm.

    Returns chunk index for each input halo centre.
    
    Will not work well for zoom simulations. Could use a grid
    which just covers the zoom region?
    """

    comm_rank = comm.Get_rank()

    # Find size of grid to use to calculate PH keys
    bits_per_dimension = 10
    cells_per_dimension = 2 ** bits_per_dimension
    grid_size = boxsize / cells_per_dimension
    nr_cells = cells_per_dimension ** 3
    nr_halos = centres.shape[0]  # number of halos on this rank
    total_nr_halos = comm.allreduce(nr_halos)  # number on all ranks

    if comm_rank == 0:
        print(f"Using Peano domain decomposition with bits={bits_per_dimension}")

    # Get PH keys for the local halos
    ipos = np.floor(centres / grid_size).value.astype(int)
    ipos = np.clip(ipos, 0, cells_per_dimension - 1)
    phkey = peano.peano_hilbert_keys(
        ipos[:, 0], ipos[:, 1], ipos[:, 2], bits_per_dimension
    )
    del ipos

    # Get sorting index to put halos in PH key order
    order = psort.parallel_sort(phkey, return_index=True, comm=comm)
    del phkey

    # Find number of halos on previous ranks
    nr_halos_previous = comm.scan(nr_halos) - nr_halos

    # Find global indexes of halos on this rank
    index = np.arange(nr_halos, dtype=int) + nr_halos_previous

    # Reorder indexes so that they're sorted by PH key
    index = psort.fetch_elements(index, order, comm=comm)

    # Assign contiguous ranges of PH sorted halos to chunks
    halos_per_chunk = max(1, total_nr_halos // nr_chunks)
    sorted_index = np.arange(nr_halos, dtype=int) + nr_halos_previous
    task_id = (sorted_index // halos_per_chunk).clip(min=0, max=nr_chunks - 1)

    # Sort task_ids back into original order
    order = psort.parallel_sort(index, return_index=True, comm=comm)
    task_id = psort.fetch_elements(task_id, order, comm=comm)

    assert np.all(task_id >= 0)
    assert np.all(task_id < nr_chunks)

    return task_id
