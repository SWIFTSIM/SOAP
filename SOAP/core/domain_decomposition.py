#!/bin/env python

import numpy as np
import virgo.util.peano as peano
import virgo.mpi.parallel_sort as psort
from virgo.mpi.gather_array import gather_array


def peano_decomposition(boxsize, local_halo, nr_chunks, comm, separate_chunk_threshold):
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
    comm_size = comm.Get_size()

    # Handle halos which should be on their own chunk
    # for memory reasons
    nr_large_halo = 0
    total_nr_halos = comm.allreduce(local_halo["index"].shape[0])
    if separate_chunk_threshold != -1:
        nr_local_halos = local_halo["index"].shape[0]
        local_halo_offset = comm.scan(nr_local_halos) - nr_local_halos

        # Identify the large halos
        large_halo_mask = local_halo["nr_bound_part"] > separate_chunk_threshold
        idx_large_halo = np.where(large_halo_mask)[0] + local_halo_offset
        idx_large_halo = gather_array(idx_large_halo, root=comm_size - 1)
        if comm_rank == comm_size - 1:
            nr_large_halo = idx_large_halo.shape[0]
        else:
            idx_large_halo = np.zeros(0, dtype=np.int32)
        nr_large_halo = comm.bcast(nr_large_halo, root=comm_size - 1)

        if nr_large_halo > 0:
            if comm_rank == 0:
                print(f"Placing {nr_large_halo} halos on their own chunk")

            # Move the data for these halos to the final rank
            large_halo = {}
            for name in local_halo:
                large_halo[name] = psort.fetch_elements(
                    local_halo[name], idx_large_halo, comm=comm
                )

            # Remove the data for these halos from local_halo
            for name in local_halo:
                local_halo[name] = local_halo[name][~large_halo_mask]

    # Find size of grid to use to calculate PH keys
    centres = local_halo["cofp"]
    bits_per_dimension = 10
    cells_per_dimension = 2**bits_per_dimension
    grid_size = boxsize / cells_per_dimension
    nr_cells = cells_per_dimension**3
    nr_halos = centres.shape[0]  # number of halos on this rank
    total_nr_small_halos = comm.allreduce(nr_halos)  # number on all ranks

    # Reduce the number of chunks if necessary so that all chunks
    # have at least one halo
    nr_chunks = min(nr_chunks, total_nr_small_halos)

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

    # Place the large halos at the end
    if nr_large_halo > 0:
        if comm_rank == comm_size - 1:
            for name in local_halo:
                local_halo[name] = np.concatenate(
                    [
                        local_halo[name],
                        large_halo[name],
                    ],
                    axis=0,
                )

    # Decide how many halos to put in each chunk
    chunk_size = np.zeros(nr_chunks + nr_large_halo, dtype=int)
    chunk_size[:nr_chunks] = total_nr_small_halos // nr_chunks
    chunk_size[: total_nr_small_halos % nr_chunks] += 1
    chunk_size[nr_chunks:] = 1
    assert np.sum(chunk_size) == total_nr_halos

    return chunk_size
