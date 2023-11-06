#!/bin/env python

import time
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import swift_cells
import shared_mesh


def io_test():

    comm.barrier()
    t0 = time.time()

    # Open the snapshot
    fname = "/cosma8/data/dp004/flamingo/Runs/L2800N5040/HYDRO_FIDUCIAL/snapshots/flamingo_0037/flamingo_0037.%(file_nr)d.hdf5"
    cellgrid = swift_cells.SWIFTCellGrid(fname)

    # Quantities to read
    property_names = {
        "PartType0": ("Coordinates", "Velocities", "Masses"),
        "PartType1": ("Coordinates", "Velocities", "Masses"),
    }

    # Specify region to read
    pos_min = np.asarray((0.0, 0.0, 0.0)) * cellgrid.units.length
    pos_max = np.asarray((50.0, 50.0, 50.0)) * cellgrid.units.length

    # Read in the region
    mask = cellgrid.empty_mask()
    cellgrid.mask_region(mask, pos_min, pos_max)
    data = cellgrid.read_masked_cells_to_shared_memory(property_names, mask, comm)

    comm.barrier()
    t1 = time.time()

    # Find read rate
    nbytes = 0
    for ptype in property_names:
        for dataset in property_names[ptype]:
            arr = data[ptype][dataset].full
            nbytes += arr.data.nbytes
    elapsed = t1 - t0

    if comm_rank == 0:
        rate = nbytes / elapsed / (1024 ** 3)
        print("Read at %.2f GB/sec on %d ranks" % (rate, comm_size))

    # Build the shared mesh
    mesh = shared_mesh.SharedMesh(
        comm, pos=data["PartType1"]["Coordinates"], resolution=256
    )
    if comm_rank == 0:
        print("Built mesh")

    comm.barrier()

    if comm_rank == 0:

        # Plot all particles
        pos = data["PartType1"]["Coordinates"].full
        plt.plot(pos[:, 0], pos[:, 1], "k,", alpha=0.05)
        plt.gca().set_aspect("equal")

        # Use the mesh to plot a sub-region
        pos_min = np.asarray((20, 20, 20), dtype=float) * cellgrid.units.length
        pos_max = np.asarray((40, 40, 40), dtype=float) * cellgrid.units.length
        idx = mesh.query(pos_min, pos_max)
        plt.plot(pos[idx, 0], pos[idx, 1], "r,")

        # Try selecting a sphere
        centre = np.asarray((30, 30, 30)) * cellgrid.units.length
        radius = 10 * cellgrid.units.length
        idx = mesh.query_radius(centre, radius, pos)
        plt.plot(pos[idx, 0], pos[idx, 1], "g,")

        plt.show()


if __name__ == "__main__":
    io_test()
