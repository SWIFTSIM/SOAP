import h5py
import numpy as np


def test_dataset_values(old_file, new_file, cumulative_path=""):
    """
    Checks if all the datasets common to two SOAP generated HDF5 files
    contain the same values.

    Param
    -----
    old_file: h5py.File
        The original SOAP file.
    new_file: h5py.File
        The new SOAP file.
    cumulative_path: str, opt
        The full path that we are currently looking at. Should be ignored, as it
        is only used for printing information when we find a value mismatch.
    """

    # If we have a dataset, test if values agree between old and new files.
    if not isinstance(old_file, h5py._hl.group.Group):
        if not (old_file[()] == new_file[()]).all():
            print(
                f"Value mismatch ({(old_file[()] != new_file[()]).sum()} entries; {(old_file[()] != new_file[()]).sum()/len(old_file[()]) * 100:.5f}% of all) at dataset {cumulative_path[1:]}"
            )
        return

    # If we still haven't reached a dataset, navigate deeper in the file
    old_keys, new_keys = set(old_file.keys()), set(new_file.keys())

    # Print out groups/datasets that are not common to both files
    if len(old_keys - new_keys) or len(new_keys - old_keys):
        print("Keys only present in old file: ", old_keys - new_keys)
        print("Keys only present in new file: ", new_keys - old_keys)
        print()

    for key in old_keys.intersection(new_keys):  # Iterate over the common keys
        test_dataset_values(old_file[key], new_file[key], f"{cumulative_path}/{key}")


if __name__ == "__main__":
    old_soap_file_path = "/cosma8/data/dp004/dc-foro1/colibre/low_res_test/halo_finding/velociraptor/soap/colibre_SOAP_halo_properties_0127.hdf5"
    new_soap_file_path = "/cosma8/data/dp004/dc-foro1/colibre/low_res_test/halo_finding/velociraptor/soap/SOAP_halo_properties_0127.hdf5"

    with h5py.File(old_soap_file_path) as old_soap_file, h5py.File(
        new_soap_file_path
    ) as new_soap_file:
        test_dataset_values(old_soap_file, new_soap_file)
