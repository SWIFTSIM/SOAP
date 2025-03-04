"""
Contains helper functions for downloading test data
"""

import os
import subprocess

import pytest

webstorage_location = "https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP/"
test_output_location = "test_data/"


def requires(filepaths, comm=None):
    """
    Use this as a decorator around tests that require data.

    Can be passed either a single filepath, or a list of filepaths

    If running with MPI then pass the comm so that only a single
    rank will download the data.
    """

    # First check if the test data directory exists
    if (comm is None) or (comm.Get_rank() == 1):
        if not os.path.exists(test_output_location):
            os.mkdir(test_output_location)

    # Handle case where we are passed a single path instead of a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]
        return_str = True
    else:
        return_str = False

    output_locations = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        output_location = f"{test_output_location}{filename}"
        output_locations.append(output_location)

        if (comm is not None) and (comm.Get_rank() != 0):
            file_available = None
        else:
            if not os.path.exists(output_location):
                # Download the file if it doesn't exist
                ret = subprocess.call(
                    ["wget", f"{webstorage_location}{filepath}", "-O", output_location]
                )

                if ret != 0:
                    Warning(f"Unable to download file at {filepath}")
                    # It wrote an empty file, kill it.
                    subprocess.call(["rm", output_location])
                    file_available = False
                else:
                    file_available = True
            else:
                file_available = True

        if comm is not None:
            file_available = comm.bcast(file_available)

        if not file_available:

            def dont_call_test(func):
                def empty(*args, **kwargs):
                    return pytest.skip()

                return empty

            return dont_call_test

    # Return a single path if that's what we were passed
    if return_str:
        output_locations = output_locations[0]

    # We can do the test!
    def do_call_test(func):
        def final_test():
            return func(output_locations)

        return final_test

    return do_call_test


if __name__ == "__main__":
    # Download the data required for run_small_volume.sh
    # Call @requires by passing a dummy function
    dummy = lambda x: x
    requires("swift_output/fof_output_0018.hdf5")(dummy)()
    requires("swift_output/snap_0018.hdf5")(dummy)()
    requires("HBT_output/018/SubSnap_018.0.hdf5")(dummy)()
