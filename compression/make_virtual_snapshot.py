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
    # and store their attributes and datatype
    filename = membership.format(file_nr=0, snap_nr=snap_nr)
    dset_attrs = {}
    dset_dtype = {}
    with h5py.File(filename, "r") as infile:
        for ptype in range(7):
            if not f'PartType{ptype}' in infile:
                continue
            dset_attrs[f'PartType{ptype}'] = {}
            dset_dtype[f'PartType{ptype}'] = {}
            for dset in infile[f'PartType{ptype}'].keys():
                attrs = dict(infile[f'PartType{ptype}/{dset}'].attrs)
                dtype = infile[f'PartType{ptype}/{dset}'].dtype

                # Some membership files are missing these attributes
                if not 'Value stored as physical' in attrs:
                    print(f'Setting comoving attrs for PartType{ptype}/{dset}')
                    attrs['Value stored as physical'] = [1]
                    attrs["Property can be converted to comoving"] = [0]

                # Add a flag that these are stored in the membership files
                attrs["Auxilary file"] = [1]

                # Store the values we need for later
                dset_attrs[f'PartType{ptype}'][dset] = attrs
                dset_dtype[f'PartType{ptype}'][dset] = dtype

    # TODO: Remove
    # Check which datasets already exist in the snapshot
    # dset_in_snap = {}
    # with h5py.File(snapshot, "r") as infile:
    #     for ptype in range(7):
    #         if not f'PartType{ptype}' in dset_attrs:
    #             continue
    #         dset_in_snap[f'PartType{ptype}'] = []
    #         for dset in dset_attrs:
    #             if dset in infile[f'PartType{ptype}']:
    #                 dset_in_snap[f'PartType{ptype}'].append(dset)

    # Copy the input virtual snapshot to the output
    shutil.copyfile(snapshot, output_file)

    # Open the output file
    outfile = h5py.File(output_file, "r+")

    # Loop over input membership files to get dataset shapes
    file_nr = 0
    filenames = []
    shapes = []
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
                shapes.append(shape)
        else:
            break
        file_nr += 1
    if file_nr == 0:
        raise IOError(f"Failed to find files matching: {membership}")

    # Loop over particle types in the output
    for ptype in range(7):
        if f"PartType{ptype}" not in dset_attrs:
            continue

        # Create virtual layout for new datasets
        layouts = {}
        nr_parts = sum([shape[ptype][0] for shape in shapes])
        full_shape = (nr_parts,)
        for dset in dset_attrs[f"PartType{ptype}"]:
            dtype = dset_dtype[f"PartType{ptype}"][dset]
            layouts[dset] = h5py.VirtualLayout(shape=full_shape, dtype=dtype)

        # Loop over input files
        offset = 0
        for filename, shape in zip(filenames, shapes):
            count = shape[ptype][0]
            for dset in dset_attrs[f"PartType{ptype}"]:
                layouts[dset][offset : offset + count] = h5py.VirtualSource(
                    filename, f"PartType{ptype}/{dset}", shape=shape[ptype]
                )
            offset += count

        # Create the virtual datasets, renaming datasets if they
        # already exist in the snapshot
        for dset, attrs in dset_attrs[f"PartType{ptype}"].items():
            if f"PartType{ptype}/{dset}" in outfile:
                outfile.move(
                    f"PartType{ptype}/{dset}", f"PartType{ptype}/{dset}_snap"
                )
            outfile.create_virtual_dataset(
                f"PartType{ptype}/{dset}", layouts[dset], fillvalue=-999
            )
            for k, v in attrs.items():
                outfile[f"PartType{ptype}/{dset}"].attrs[k] = v

            # Copy GroupNr_bound to HaloCatalogueIndex, since that is the name in SOAP
            if dset == 'GroupNr_bound':
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/HaloCatalogueIndex",
                    layouts['GroupNr_bound'],
                    fillvalue=-999,
                )
                for k, v in outfile[f"PartType{ptype}/GroupNr_bound"].attrs.items():
                    outfile[f"PartType{ptype}/HaloCatalogueIndex"].attrs[k] = v

    # Done
    outfile.close()


if __name__ == "__main__":

    import argparse
    from update_vds_paths import update_virtual_snapshot_paths

    # For description of parameters run the following: $ python make_virtual_snapshot.py --help
    parser = argparse.ArgumentParser(
        description=(
            "Link SWIFT snapshots with SWIFT auxilary snapshots (snapshot-like
            files with the same number of particles in the same order as the
            snapshot, but with less metadata), such as the SOAP memberships"
        )
    )
    parser.add_argument(
        "virtual_snapshot",
        type=str,
        help="Name of the SWIFT virtual snapshot file, e.g. snapshot_{snap_nr:04}.hdf5",
    )
    parser.add_argument(
        "membership",
        type=str,
        help="Format string for membership files, e.g. membership_{snap_nr:04}.{file_nr}.hdf5",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Name of the virtual snapshot to create, e.g. membership_{snap_nr:04}.hdf5",
    )
    parser.add_argument(
        "snap_nr",
        type=int,
        nargs="?",
        default=-1,
        help="Snapshot number (default: -1). Not required if snap_nr is present in filenames passed.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Use absolute paths in the virtual dataset",
    )
    args = parser.parse_args()

    # Substitute snap number
    virtual_snapshot = args.virtual_snapshot.format(snap_nr=args.snap_nr)
    output_file = args.output_file.format(snap_nr=args.snap_nr)

    # Make a new virtual snapshot with group info
    make_virtual_snapshot(virtual_snapshot, args.membership, output_file, args.snap_nr)

    # Set file paths for datasets
    abs_snapshot_dir = os.path.abspath(os.path.dirname(virtual_snapshot))
    abs_membership_dir = os.path.abspath(
        os.path.dirname(args.membership.format(snap_nr=args.snap_nr, file_nr=0))
    )
    if args.absolute_paths:
        # Ensure all paths in the virtual file are absolute to avoid VDS prefix issues
        # (we probably need to pick up datasets from two different directories)
        update_virtual_snapshot_paths(output_file, abs_snapshot_dir, abs_membership_dir)
    else:
        abs_output_dir = os.path.abspath(os.path.dirname(output_file))
        rel_snapshot_dir = os.path.relpath(abs_snapshot_dir, abs_output_dir)
        rel_membership_dir = os.path.relpath(abs_membership_dir, abs_output_dir)
        update_virtual_snapshot_paths(output_file, rel_snapshot_dir, rel_membership_dir)
