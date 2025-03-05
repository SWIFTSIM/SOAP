#!/bin/env python

from mpi4py import MPI


def mask_cells(comm, cellgrid, centre, radius, done):
    """
    Flag all cells which need to be read in to ensure we have
    all particles within the specified radii of the halo centres.
    Parallelized over all MPI ranks in communicator comm.
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Divide halos between MPI ranks
    nr_halos = len(radius)
    nr_halos_local = nr_halos // comm_size
    first_halo = comm_rank * nr_halos_local
    if comm_rank == comm_size - 1:
        nr_halos_local += nr_halos % comm_size
    assert comm.allreduce(nr_halos_local) == nr_halos

    # Make an empty mask on each rank
    mask = cellgrid.empty_mask()

    # Loop over local halos
    for halo_nr in range(first_halo, first_halo + nr_halos_local):

        # Flag cells around this halo
        if done[halo_nr] == 0:
            pos_min = centre[halo_nr, :] - radius[halo_nr] - 0.5 * cellgrid.cell_size
            pos_max = centre[halo_nr, :] + radius[halo_nr] + 0.5 * cellgrid.cell_size
            cellgrid.mask_region(mask, pos_min, pos_max)

    # Combine masks
    comm.Allreduce(MPI.IN_PLACE, mask, op=MPI.LOR)
    return mask
