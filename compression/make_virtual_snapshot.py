#!/bin/env python

import os.path
import h5py
import shutil
import numpy as np


class SafeDict(dict):
    def __missing__(self, key):
        # Return the key back in braces so it remains in the string
        return "{" + key + "}"


def update_vds_paths(dset, modify_function):
    """
    Modify the virtual paths of the specified dataset

    Note that querying the source dataspace and selection does not appear
    to work (invalid pointer error from h5py) so here we assume that we're
    referencing all of the source dataspace, which is correct for SWIFT
    snapshots.

    dset:            a h5py.Dataset object
    modify_function: a function which takes the old path as its argument and
                     returns the new path
    """

    # Choose a temporary path for the new virtual dataset
    path = dset.name
    tmp_path = dset.name + ".__tmp__"

    # Build the creation property list for the new dataset
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    for vs in dset.virtual_sources():
        bounds = vs.vspace.get_select_bounds()
        if bounds is not None:
            lower, upper = bounds
            size = np.asarray(upper, dtype=int) - np.asarray(lower, dtype=int) + 1
            src_space = h5py.h5s.create_simple(tuple(size))
            new_name = modify_function(vs.file_name)
            plist.set_virtual(
                vs.vspace, new_name.encode(), vs.dset_name.encode(), src_space
            )

    # Create the new dataset
    tmp_dset = h5py.h5d.create(
        dset.file["/"].id,
        tmp_path.encode(),
        dset.id.get_type(),
        dset.id.get_space(),
        dcpl=plist,
    )
    tmp_dset = h5py.Dataset(tmp_dset)
    for attr_name in dset.attrs:
        tmp_dset.attrs[attr_name] = dset.attrs[attr_name]

    # Rename the new dataset
    f = dset.file
    del f[path]
    f[path] = f[tmp_path]
    del f[tmp_path]


def make_virtual_snapshot(snapshot, auxilary_snapshots, output_file, absolute_paths=False):
    """
    Given a snapshot and auxilary files, create
    a new virtual snapshot with all datasets combine.
    
    snapshot: Path to the snapshot file
    auxilary_snapshots: List of auxiliary file patterns
    output_file: Path to the output virtual snapshot
    absolute_paths: If True, use absolute paths; if False, use relative paths
    """

    # Copy the input virtual snapshot to the output
    shutil.copyfile(snapshot, output_file)

    # Open the output file
    outfile = h5py.File(output_file, "r+")

    # Calculate directories for path updates
    abs_snapshot_dir = os.path.abspath(os.path.dirname(snapshot))
    abs_auxilary_dirs = [
        os.path.abspath(os.path.dirname(aux.format(file_nr=0)))
        for aux in auxilary_snapshots
    ]
    abs_output_dir = os.path.abspath(os.path.dirname(output_file))
    
    if absolute_paths:
        snapshot_dir = abs_snapshot_dir
        auxilary_dirs = abs_auxilary_dirs
    else:
        snapshot_dir = os.path.relpath(abs_snapshot_dir, abs_output_dir)
        auxilary_dirs = [
            os.path.relpath(aux_dir, abs_output_dir) 
            for aux_dir in abs_auxilary_dirs
        ]

    # Create path replacement functions
    def make_replace_path(target_dir):
        def replace_path(old_path):
            basename = os.path.basename(old_path)
            return os.path.join(target_dir, basename)
        return replace_path

    replace_snapshot_path = make_replace_path(snapshot_dir)
    auxilary_path_replacers = [make_replace_path(d) for d in auxilary_dirs]

    all_auxilary_datasets = {}
    
    for aux_index, auxilary in enumerate(auxilary_snapshots):

        # Check which datasets exist in the auxilary files
        # and store their attributes and datatype
        filename = auxilary.format(file_nr=0)
        dset_attrs = {}
        dset_dtype = {}
        with h5py.File(filename, "r") as infile:
            for ptype in range(7):
                if not f"PartType{ptype}" in infile:
                    continue
                dset_attrs[f"PartType{ptype}"] = {}
                dset_dtype[f"PartType{ptype}"] = {}
                for dset in infile[f"PartType{ptype}"].keys():
                    attrs = dict(infile[f"PartType{ptype}/{dset}"].attrs)
                    dtype = infile[f"PartType{ptype}/{dset}"].dtype

                    # Some auxilary files are missing these attributes
                    if not "Value stored as physical" in attrs:
                        print(f"Setting comoving attrs for PartType{ptype}/{dset}")
                        attrs["Value stored as physical"] = [1]
                        attrs["Property can be converted to comoving"] = [0]

                    # Add a flag that these datasets are stored in the auxilary files
                    attrs["Auxilary file"] = [1]

                    # Store the values we need for later
                    dset_attrs[f"PartType{ptype}"][dset] = attrs
                    dset_dtype[f"PartType{ptype}"][dset] = dtype

                    # Check we don't have this dataset in any of the other auxilary files
                    dset_path = f"PartType{ptype}/{dset}"
                    if dset_path in all_auxilary_datasets:
                        other_file = all_auxilary_datasets[f"PartType{ptype}/{dset}"]
                        raise ValueError(f"{dset_path} is in {auxilary} and {other_file}")
                    all_auxilary_datasets[dset_path] = auxilary

        # Loop over input auxilary files to get dataset shapes
        file_nr = 0
        filenames = []
        shapes = []
        counts = []
        while True:
            filename = auxilary.format(file_nr=file_nr)
            if os.path.exists(filename):
                filenames.append(filename)
                with h5py.File(filename, "r") as infile:
                    shape = {}
                    count = {}
                    for ptype in range(7):
                        if f"PartType{ptype}" not in dset_attrs:
                            continue
                        shape[f"PartType{ptype}"] = {}
                        # Get the shape for each dataset
                        for dset in dset_attrs[f"PartType{ptype}"]:
                            s = infile[f"PartType{ptype}/{dset}"].shape
                            shape[f"PartType{ptype}"][dset] = s
                        # Get the number of particles in this chunk file
                        count[f"PartType{ptype}"] = s[0]
                    shapes.append(shape)
                    counts.append(count)
            else:
                break
            file_nr += 1
        if file_nr == 0:
            raise IOError(f"Failed to find files matching: {auxilary}")

        # Loop over particle types in the output
        for ptype in range(7):
            if f"PartType{ptype}" not in dset_attrs:
                continue

            # Create virtual layout for new datasets
            layouts = {}
            nr_parts = sum([count[f"PartType{ptype}"] for count in counts])
            for dset in dset_attrs[f"PartType{ptype}"]:
                full_shape = list(shapes[0][f"PartType{ptype}"][dset])
                full_shape[0] = nr_parts
                full_shape = tuple(full_shape)
                dtype = dset_dtype[f"PartType{ptype}"][dset]
                layouts[dset] = h5py.VirtualLayout(shape=full_shape, dtype=dtype)

            # Loop over input files
            offset = 0
            for filename, count, shape in zip(filenames, counts, shapes):
                n_part = count[f"PartType{ptype}"]
                for dset in dset_attrs[f"PartType{ptype}"]:
                    layouts[dset][offset : offset + n_part] = h5py.VirtualSource(
                        filename,
                        f"PartType{ptype}/{dset}",
                        shape=shape[f"PartType{ptype}"][dset],
                    )
                offset += n_part

            # Create the virtual datasets, renaming datasets if they
            # already exist in the snapshot
            for dset, attrs in dset_attrs[f"PartType{ptype}"].items():
                if f"PartType{ptype}/{dset}" in outfile:
                    outfile.move(f"PartType{ptype}/{dset}", f"PartType{ptype}/{dset}_snap")
                outfile.create_virtual_dataset(
                    f"PartType{ptype}/{dset}", layouts[dset], fillvalue=-999
                )
                for k, v in attrs.items():
                    outfile[f"PartType{ptype}/{dset}"].attrs[k] = v

                # Update paths for this newly created auxiliary dataset
                update_vds_paths(outfile[f"PartType{ptype}/{dset}"], auxilary_path_replacers[aux_index])

                # Copy GroupNr_bound to HaloCatalogueIndex, since 
                # that is the name in SOAP
                if dset == "GroupNr_bound":
                    outfile.create_virtual_dataset(
                        f"PartType{ptype}/HaloCatalogueIndex",
                        layouts["GroupNr_bound"],
                        fillvalue=-999,
                    )
                    for k, v in outfile[f"PartType{ptype}/GroupNr_bound"].attrs.items():
                        outfile[f"PartType{ptype}/HaloCatalogueIndex"].attrs[k] = v
                    
                    # Update paths for HaloCatalogueIndex too
                    update_vds_paths(outfile[f"PartType{ptype}/HaloCatalogueIndex"], auxilary_path_replacers[aux_index])

    # Update paths for all original snapshot datasets
    for ptype in range(7):
        ptype_name = f"PartType{ptype}"
        if ptype_name in outfile:
            for dset_name in list(outfile[ptype_name].keys()):
                dset = outfile[f"{ptype_name}/{dset_name}"]
                if dset.is_virtual:
                    # Check if this is an auxiliary dataset (skip those, already handled)
                    if dset.attrs.get("Auxilary file", [0])[0] != 1:
                        # This is an original snapshot dataset
                        update_vds_paths(dset, replace_snapshot_path)

    # Done
    outfile.close()


if __name__ == "__main__":

    import argparse

    # For description of parameters run the following: $ python make_virtual_snapshot.py --help
    parser = argparse.ArgumentParser(
        description=(
            "Link SWIFT snapshots with SWIFT auxilary snapshots (snapshot-like"
            "files with the same number of particles in the same order as the"
            "snapshot, but with less metadata), such as the SOAP memberships"
        )
    )
    parser.add_argument(
        "virtual_snapshot",
        type=str,
        help="Name of the SWIFT virtual snapshot file, e.g. snapshot_{snap_nr:04}.hdf5",
    )
    parser.add_argument(
        "auxilary_snapshots",
        type=str,
        nargs="+",
        help="One of more format strings for auxilary files, e.g. membership_{snap_nr:04}.{file_nr}.hdf5",
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

    # We don't want to replace {file_nr} for auxilary snapshots
    auxilary_snapshots = [
        filename.format_map(SafeDict({'snap_nr': args.snap_nr}))
        for filename in args.auxilary_snapshots
    ]

    # Make a new virtual snapshot with group info
    make_virtual_snapshot(
        virtual_snapshot, 
        auxilary_snapshots, 
        output_file, 
        absolute_paths=args.absolute_paths
    )
