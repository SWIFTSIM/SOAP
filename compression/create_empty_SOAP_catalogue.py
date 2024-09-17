#!/usr/bin/env

"""
create_empty_SOAP_catalogue.py

Create a placeholder SOAP catalogue with only empty datasets. Useful for
snapshots that do not contain any halos, since SOAP will fail to run for those.
By creating a structurally complete but empty catalogue, tools that blindly
run on all snapshots should still work, even if the SOAP catalogue does not
technically exist.

Usage:
  python3 create_empty_SOAP_catalogue.py REFERENCE SNAPSHOT OUTPUT
where REFERENCE is another SOAP catalogue for the same simulation. This is
used to figure out which datasets and meta-data should be added to the empty
SOAP catalogue. SNAPSHOT is the snapshot for which we want to create the empty
catalogue (the snapshot for which running SOAP itself fails because there are
no halos in it). OUTPUT is the name of the output file that will be created.
Note that SNAPSHOT is required to add the correct SWIFT meta-data to the SOAP
catalogue. Providing the wrong snapshot will work, but this might upset scripts
that parse the SWIFT meta-data in the empty SOAP catalogue.
"""

import argparse
import os
import h5py
import numpy as np


def get_snapshot_index(snapshot_name):
    """
    Turn a snapshot name into a snapshot index number.

    e.g.
      flamingo_0033.hdf5 --> 33

    Should only be used on virtual files or single file snapshots, not on files
    that are part of a multi-file snapshot (e.g. flamingo_0033.25.hdf5).
    """
    name, _ = os.path.splitext(snapshot_name)
    return int(name[-4:])


class H5copier:
    """
    Functor (class that acts as a function) used to copy over groups and
    datasets from one HDF5 file to another.
    """

    def __init__(self, ifile, snapfile, ofile, snapnum):
        """
        Constructor.

        Requires the input SOAP catalogue from which we copy, the snapshot file
        from which we want to copy SWIFT meta-data, and the output file to
        which we want to copy data. Also needs the snapshot index number to
        update some of the SOAP meta-data.
        """
        self.ifile = ifile
        self.snapfile = snapfile
        self.ofile = ofile
        self.snapnum = snapnum

    def __call__(self, name, h5obj):
        """
        Functor function, i.e. what gets called when you use () on an object
        of this class. Conforms to the h5py.Group.visititems() function
        signature.

        Parameters:
         - name: Full path to a group/dataset in the HDF5 file
                  e.g. SO/200_crit/TotalMass
         - h5obj: HDF5 file object pointed at by this path
                   e.g. SO/200_crit/TotalMass --> h5py.Dataset
                        SO/200_crit --> h5py.Group
        """

        # figure out if we are dealing with a dataset or a group
        type = h5obj.__class__
        if isinstance(h5obj, h5py.Group):
            type = "group"
        elif isinstance(h5obj, h5py.Dataset):
            type = "dataset"
        else:
            raise RuntimeError(f"Unknown HDF5 object type: {name}")

        if type == "group":
            # create the group in the output file
            self.ofile.create_group(name)
            # take care of attributes:
            #  - Cosmology, SWIFT/Header, and SWIFT/Parameters attributes are
            #    read directly from the snapshot file
            #  - Header and Parameters are copied, but some snapshot file specific
            #    info is updated
            #  - For all other groups we simply copy all attributes
            if name in ["Cosmology", "SWIFT/Header", "SWIFT/Parameters"]:
                swift_name = name.replace("SWIFT/", "")
                for attr in self.snapfile[swift_name].attrs:
                    self.ofile[name].attrs[attr] = self.snapfile[swift_name].attrs[attr]
            elif name == "Header":
                for attr in self.ifile[name].attrs:
                    self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
                self.ofile[name].attrs["NumSubhalos_ThisFile"] = np.array(
                    [0], dtype="int32"
                )
                self.ofile[name].attrs["NumSubhalos_Total"] = np.array(
                    [0], dtype="int32"
                )
                self.ofile[name].attrs["Redshift"] = snapfile["Cosmology"].attrs[
                    "Redshift"
                ]
                self.ofile[name].attrs["Scale-factor"] = snapfile["Cosmology"].attrs[
                    "Scale-factor"
                ]
            elif name == "Parameters":
                for attr in self.ifile[name].attrs:
                    self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
                self.ofile[name].attrs["halo_indices"] = np.array([], dtype="int64")
                self.ofile[name].attrs["snapshot_nr"] = self.snapnum
            else:
                for attr in self.ifile[name].attrs:
                    self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
        elif type == "dataset":
            if name in ["Cells/Counts/Subhalos", "Cells/OffsetsInFile/Subhalos"]:
                arr = 0 * self.ifile[name][:]
                self.ofile.create_dataset(name, data=arr)
            else:
                # dataset: get the dtype and shape and create a new dataset with
                # the same name, dtype and shape, but with the length of the array
                # (shape[0]) set to 0
                dtype = h5obj.dtype
                shape = h5obj.shape
                new_shape = None
                if len(shape) == 1:
                    new_shape = (0,)
                else:
                    new_shape = (0, *shape[1:])
                self.ofile.create_dataset(name, new_shape, dtype)
                for attr in self.ifile[name].attrs:
                    self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]


if __name__ == "__main__":
    """
    Main entry point.
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "referenceSOAP", help="Existing SOAP catalogue whose structure is copied."
    )
    argparser.add_argument(
        "snapshot", help="Snapshot file for which we want to create an empty catalogue."
    )
    argparser.add_argument("outputSOAP", help="Output catalogue file name.")
    args = argparser.parse_args()

    snapnum = get_snapshot_index(args.snapshot)
    ofile_snapnum = get_snapshot_index(args.outputSOAP)
    assert snapnum == ofile_snapnum

    assert not os.path.exists(args.outputSOAP)

    with h5py.File(args.referenceSOAP, "r") as ifile, h5py.File(
        args.snapshot, "r"
    ) as snapfile, h5py.File(args.outputSOAP, "w") as ofile:
        h5copy = H5copier(ifile, snapfile, ofile, snapnum)
        ifile.visititems(h5copy)
