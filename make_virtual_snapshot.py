#!/bin/env python

import os.path
import h5py
import shutil


def make_virtual_snapshot(snapshot, membership, output_file, snap_nr):
    """
    Given a FLAMINGO snapshot and group membership files,
    create a new virtual snapshot with group info.
    """

    # Check which datasets exist in the membership files
    filename = membership.format(file_nr=0, snap_nr=snap_nr)
    with h5py.File(filename, "r") as infile:
        have_grnr_bound = "GroupNr_bound" in infile["PartType1"]
        have_grnr_all = "GroupNr_all" in infile["PartType1"]
        have_rank_bound = "Rank_bound" in infile["PartType1"]
        have_fof_id = "FOFGroupIDs" in infile["PartType1"]

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
        filename = membership.format(file_nr=file_nr, snap_nr=snap_nr)
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
            # PartType6 (neutrinos) are not assigned a FOF group
            if have_fof_id and (ptype != 6):
                layout_fof_id = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
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
                if have_fof_id and (ptype != 6):
                    layout_fof_id[offset : offset + count] = h5py.VirtualSource(
                        filename, f"PartType{ptype}/FOFGroupIDs", shape=shape[ptype]
                    )
                offset += count
            # Create the virtual datasets
            if have_grnr_all:
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/GroupNr_all", layout_grnr_all, fillvalue=-999
                )
            if have_grnr_bound:
                dset_groupnr_bound = outfile.create_virtual_dataset(
                    f"PartType{ptype}/GroupNr_bound", layout_grnr_bound, fillvalue=-999
                )
                outfile[f"PartType{ptype}/HaloCatalogueIndex"] = dset_groupnr_bound
                for k, v in outfile[f"PartType{ptype}/GroupNr_bound"].attrs.items():
                    outfile[f"PartType{ptype}/HaloCatalogueIndex"].attrs[k] = v
            if have_rank_bound:
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/Rank_bound", layout_rank_bound, fillvalue=-999
                )
            if have_fof_id and (ptype != 6):
                outfile.move(
                    f"PartType{ptype}/FOFGroupIDs", f"PartType{ptype}/FOFGroupIDs_old"
                )
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/FOFGroupIDs", layout_fof_id, fillvalue=-999
                )

    # Done
    outfile.close()

if __name__ == "__main__":

    import argparse
    from update_vds_paths import update_virtual_snapshot_paths

    # For description of parameters run the following: $ python make_virtual_snapshot.py --help
    parser = argparse.ArgumentParser(description="Link SWIFT snapshots with SOAP membership files")
    parser.add_argument("virtual_snapshot", type=str, help="Name of the SWIFT virtual snapshot file, e.g. snapshot_{snap_nr:04}.hdf5")
    parser.add_argument("membership", type=str, help="Format string for membership files, e.g. membership_{snap_nr:04}.{file_nr}.hdf5")
    parser.add_argument("output_file", type=str, help="Name of the virtual snapshot to create, e.g. membership_{snap_nr:04}.hdf5")
    parser.add_argument("snap_nr", type=int, nargs='?', default=-1, help="Snapshot number (default: -1). Not required if snap_nr is present in filenames passed.")
    parser.add_argument("--absolute-paths", action='store_true', help="Use absolute paths in the virtual dataset")
    args = parser.parse_args()

    # Substitute snap number
    virtual_snapshot = args.virtual_snapshot.format(snap_nr=args.snap_nr)
    output_file = args.output_file.format(snap_nr=args.snap_nr)

    # Make a new virtual snapshot with group info
    make_virtual_snapshot(virtual_snapshot, args.membership, output_file, args.snap_nr)

    # Set file paths for datasets
    abs_snapshot_dir = os.path.abspath(os.path.dirname(virtual_snapshot))
    abs_membership_dir = os.path.abspath(os.path.dirname(args.membership.format(snap_nr=args.snap_nr, file_nr=0)))
    if args.absolute_paths:
        # Ensure all paths in the virtual file are absolute to avoid VDS prefix issues
        # (we probably need to pick up datasets from two different directories)
        update_virtual_snapshot_paths(output_file, abs_snapshot_dir, abs_membership_dir)
    else:
        abs_output_dir = os.path.abspath(os.path.dirname(output_file))
        rel_snapshot_dir = os.path.relpath(abs_snapshot_dir, abs_output_dir)
        rel_membership_dir = os.path.relpath(abs_membership_dir, abs_output_dir)
        update_virtual_snapshot_paths(output_file, rel_snapshot_dir, rel_membership_dir)
