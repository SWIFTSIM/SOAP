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
import pytest

@pytest.mark.mpi
def test_io():

    comm.barrier()
    t0 = time.time()

    # Open the snapshot
    fname = "/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/snapshots/flamingo_0037/flamingo_0037.{file_nr}.hdf5"
    try:
        cellgrid = swift_cells.SWIFTCellGrid(fname)
    except FileNotFoundError:
        if comm_rank == 0:
            print("File not found for running io_test")
        return

    # Quantities to read
    property_names = {
        "PartType0": ("Coordinates", "Velocities", "Masses"),
        "PartType1": ("Coordinates", "Velocities", "Masses"),
    }

    # Specify region to read
    pos_min = np.asarray((0.0, 0.0, 0.0)) * cellgrid.get_unit("snap_length")
    pos_max = np.asarray((50.0, 50.0, 50.0)) * cellgrid.get_unit("snap_length")

    # Read in the region
    mask = cellgrid.empty_mask()
    cellgrid.mask_region(mask, pos_min, pos_max)
    data = cellgrid.read_masked_cells_to_shared_memory(property_names, mask, comm, 8)

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
        pos = data["PartType1"]["Coordinates"]
        plt.plot(pos.full[:, 0], pos.full[:, 1], "k,", alpha=0.05)
        plt.gca().set_aspect("equal")

        # Try selecting a sphere
        centre = np.asarray((30, 30, 30)) * cellgrid.get_unit("snap_length")
        radius = 10 * cellgrid.get_unit("snap_length")
        idx = mesh.query_radius_periodic(centre, radius, pos, cellgrid.boxsize)
        plt.plot(pos.full[idx, 0], pos.full[idx, 1], "g,")

        plt.xlim(0, 150)
        plt.ylim(0, 150)
        plt.savefig(f"io_test.png", dpi=300)
        plt.close()

    # Free the shared particle data
    for ptype in data:
        for name in data[ptype]:
            data[ptype][name].free()
    mesh.free()


if __name__ == "__main__":
    test_io()
