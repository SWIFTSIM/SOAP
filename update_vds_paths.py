#!/bin/env python

import sys
import h5py
import numpy as np
import os.path


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


def update_virtual_snapshot_paths(filename, snapshot_dir=None, membership_dir=None):
    """
    Add full paths to virtual datasets in the specified file
    """
    f = h5py.File(filename, "r+")

    # Find all datasets in the file
    all_datasets = []

    def visit_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            all_datasets.append(obj)

    f.visititems(visit_datasets)

    def replace_snapshot_path(old_path):
        basename = os.path.basename(old_path)
        return os.path.join(snapshot_dir, basename)

    def replace_membership_path(old_path):
        basename = os.path.basename(old_path)
        return os.path.join(membership_dir, basename)

    # Loop over datasets and update paths if necessary
    for dset in all_datasets:
        if dset.is_virtual:
            name = dset.name.split("/")[-1]
            if name in ("GroupNr_all", "GroupNr_bound", "Rank_bound", "FOFGroupIDs"):
                # Data comes from the membership files
                if membership_dir is not None:
                    update_vds_paths(dset, replace_membership_path)
            else:
                # Data comes from the snapshot files
                if snapshot_dir is not None:
                    update_vds_paths(dset, replace_snapshot_path)

    f.close()


if __name__ == "__main__":

    filename = sys.argv[1]  # Virtual snapshot file to update
    snapshot_dir = sys.argv[2]  # Directory with the real snapshot files
    membership_dir = sys.argv[3]  # Directory with the real membership files

    update_virtual_snapshot_paths(filename, snapshot_dir, membership_dir)
