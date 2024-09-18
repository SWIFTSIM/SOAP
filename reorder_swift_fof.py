#!/bin/env python
#
# Take the FOFGroupIDs from the first snapshot and write them out in
# the same order as the second snapshot.
#
# Snapshot format strings use f-string formatting with names snap_nr
# and file_nr for the snapshot and file numbers. E.g.
#
# "snapshot_{snap_nr:04d}.{file_nr}.hdf5"
#
# Example run matching a snapshot to itself:
#
# mpirun -np 8 python3 -m mpi4py ./reorder_swift_fof.py \\
#   "/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5" \\
#   "/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5" \\
#   77 "/snap8/scratch/dp004/jch/test_{snap_nr:04d}.{file_nr}.hdf5"
#
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def message(s):
    if comm_rank == 0:
        print(s)


def reorder_swift_fof(snapshot1, snapshot2, snap_nr, output_name):

    # Substitute the snapshot number (but not the file number) into file paths
    pf = PartialFormatter()
    snapshot1 = pf.format(snapshot1, snap_nr=snap_nr, file_nr=None)
    snapshot2 = pf.format(snapshot2, snap_nr=snap_nr, file_nr=None)
    output_name = pf.format(output_name, snap_nr=snap_nr, file_nr=None)

    message(f"\nSnapshot with FOF info: {snapshot1}")
    message(f"Snapshot with ordering: {snapshot2}")
    message(f"Output: {output_name}\n")

    # Determine SWIFT particle types which have FOF info in the snapshot
    # (assumed to be the same between snapshot1 and snapshot2)
    ptypes = []
    if comm_rank == 0:
        with h5py.File(snapshot1.format(file_nr=0), "r") as infile:
            nr_types = infile["Header"].attrs["NumPartTypes"][0]
            numpart_total = infile["Header"].attrs["NumPart_Total"].astype(np.int64) + (
                infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32
            )
            nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
            for i in range(nr_types):
                group_name = "PartType%d" % i
                if numpart_total[i] > 0 and "FOFGroupIDs" in infile[group_name]:
                    ptypes.append(group_name)
    ptypes = comm.bcast(ptypes)

    # Open the snapshots
    snapfile1 = phdf5.MultiFile(
        snapshot1, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )
    snapfile2 = phdf5.MultiFile(
        snapshot2, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Loop over particle types
    create_file = True
    for ptype in ptypes:

        message(f"Reading data for particle type {ptype}")

        # Read particle IDs and FoF membership from snapshot 1
        snap1_data = snapfile1.read(
            ("ParticleIDs", "FOFGroupIDs"), group=ptype, read_attributes=True
        )

        # Read particle IDs from snapshot 2
        snap2_data = snapfile2.read(("ParticleIDs",), group=ptype, read_attributes=True)

        # For each ID in snap2, find the index in snap1. All IDs should match.
        message(f"  Matching particle IDs")
        ptr = psort.parallel_match(
            snap2_data["ParticleIDs"], snap1_data["ParticleIDs"], comm=comm
        )
        if np.any(ptr < 0):
            raise RuntimeError("Failed to match a particle ID!")

        # For each particle in snap2, fetch the FoF group index from snap1 for the same ID
        message(f"  Reordering FoF group IDs")
        snap2_data["FOFGroupIDs"] = psort.fetch_elements(
            snap1_data["FOFGroupIDs"], ptr, comm=comm
        )
        del ptr

        # Write out the result
        message(f"  Writing output")
        mode = "w" if create_file else "r+"
        elements_per_file = snapfile2.get_elements_per_file("ParticleIDs", group=ptype)
        snapfile2.write(
            snap2_data, elements_per_file, filenames=output_name, mode=mode, group=ptype
        )

        # Tidy up before we do the next particle type
        del snap1_data
        del snap2_data

    comm.barrier()
    message("Done.")


if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm=comm, description="Reorder SWIFT FOF information")
    parser.add_argument(
        "snapshot1", type=str, help="Format string for snapshot with FoF info to use"
    )
    parser.add_argument(
        "snapshot2",
        type=str,
        help="Format string for snapshot with particle ordering to use",
    )
    parser.add_argument("snap_nr", type=int, help="Snapshot number to process")
    parser.add_argument(
        "output_name", type=str, help="Format string for the output files"
    )
    args = parser.parse_args()

    reorder_swift_fof(**vars(args))
