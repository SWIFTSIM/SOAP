#!/bin/env python

import numpy as np
import virgo.util.peano as peano
import virgo.mpi.parallel_sort as psort


def peano_decomposition(boxsize, local_halo, nr_chunks, comm):
    """
    Gadget style domain decomposition using Peano-Hilbert curve.
    Allows an arbitrary number of chunks and tries to put equal
    numbers of halos in each chunk.

    The array of halo centres is assumed to be distributed over
    communicator comm.

    Sorts halos by chunk index and returns the number of halos
    in each chunk. local_halo is a dict of distributed unyt
    arrays with the halo properties.
    
    Will not work well for zoom simulations. Could use a grid
    which just covers the zoom region?
    """

    comm_rank = comm.Get_rank()

    # Find size of grid to use to calculate PH keys
    centres = local_halo["cofp"]
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

    # Reorder the halos
    for name in local_halo:
        local_halo[name] = psort.fetch_elements(local_halo[name], order, comm=comm)

    # Decide how many halos to put in each chunk
    chunk_size = np.zeros(nr_chunks, dtype=int)
    chunk_size[:] = total_nr_halos // nr_chunks
    chunk_size[: total_nr_halos % nr_chunks] += 1
    assert np.sum(chunk_size) == total_nr_halos

    return chunk_size
