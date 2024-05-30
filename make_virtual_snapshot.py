#!/bin/env python

import os.path
import h5py
import shutil


def make_virtual_snapshot(snapshot, membership, output_file):
    """
    Given a FLAMINGO snapshot and group membership files,
    create a new virtual snapshot with group info.
    """

    # Check which datasets exist in the membership files
    filename = membership % {"file_nr": 0}
    with h5py.File(filename, "r") as infile:
        have_grnr_bound = "GroupNr_bound" in infile["PartType1"]
        have_grnr_all = "GroupNr_all" in infile["PartType1"]
        have_rank_bound = "Rank_bound" in infile["PartType1"]

    # Copy the input virtual snapshot to the output
    shutil.copyfile(snapshot, output_file)

    # Open the output file
    outfile = h5py.File(output_file, "r+")

    # Loop over input membership files to get dataset shapes
    file_nr = 0
    filenames = []
    shapes = []
    dtype = None
    while True:
        filename = membership % {"file_nr": file_nr}
        if os.path.exists(filename):
            filenames.append(filename)
            with h5py.File(filename, "r") as infile:
                shape = {}
                for ptype in range(7):
                    name = f"PartType{ptype}"
                    if name in infile:
                        shape[ptype] = infile[name]["GroupNr_bound"].shape
                        if dtype is None:
                            dtype = infile[name]["GroupNr_bound"].dtype
                shapes.append(shape)
        else:
            break
        file_nr += 1
    if file_nr == 0:
        raise IOError(f"Failed to find files matching: {membership}")

    # Loop over particle types in the output
    for ptype in range(7):
        name = f"PartType{ptype}"
        if name in outfile:
            # Create virtual layout for new datasets
            nr_parts = sum([shape[ptype][0] for shape in shapes])
            full_shape = (nr_parts,)
            if have_grnr_all:
                layout_grnr_all = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            if have_grnr_bound:
                layout_grnr_bound = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            if have_rank_bound:
                layout_rank_bound = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            # Loop over input files
            offset = 0
            for (filename, shape) in zip(filenames, shapes):
                count = shape[ptype][0]
                if have_grnr_all:
                    layout_grnr_all[offset : offset + count] = h5py.VirtualSource(
                        filename, f"PartType{ptype}/GroupNr_all", shape=shape[ptype]
                    )
                if have_grnr_bound:
                    layout_grnr_bound[offset : offset + count] = h5py.VirtualSource(
                        filename, f"PartType{ptype}/GroupNr_bound", shape=shape[ptype]
                    )
                if have_rank_bound:
                    layout_rank_bound[offset : offset + count] = h5py.VirtualSource(
                        filename, f"PartType{ptype}/Rank_bound", shape=shape[ptype]
                    )
                offset += count
            # Create the virtual datasets
            if have_grnr_all:
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/GroupNr_all", layout_grnr_all, fillvalue=-999
                )
            if have_grnr_bound:
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/GroupNr_bound", layout_grnr_bound, fillvalue=-999
                )
            if have_rank_bound:
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/Rank_bound", layout_rank_bound, fillvalue=-999
                )

    # Done
    outfile.close()


if __name__ == "__main__":

    import sys
    from update_vds_paths import update_virtual_snapshot_paths

    snapshot = sys.argv[
        1
    ]  # format string for snapshots, e.g. snapshot_0077.%(file_nr).hdf5
    membership = sys.argv[
        2
    ]  # format string for membership files, e.g. membership_0077.%(file_nr).hdf5
    output_file = sys.argv[3]  # Name of the virtual snapshot to create

    # Find input virtual snap file
    virtual_snapshot = (snapshot % {"file_nr": 0})[:-7] + ".hdf5"

    # Make a new virtual snapshot with group info
    make_virtual_snapshot(virtual_snapshot, membership, output_file)

    # Ensure all paths in the virtual file are absolute to avoid VDS prefix issues
    # (we probably need to pick up datasets from two different directories)
    snapshot_dir = os.path.abspath(os.path.dirname(snapshot % {"file_nr": 0}))
    membership_dir = os.path.abspath(os.path.dirname(membership % {"file_nr": 0}))
    update_virtual_snapshot_paths(output_file, snapshot_dir, membership_dir)
