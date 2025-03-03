"""
Contains helper functions for downloading test data
"""
import os
import subprocess

webstorage_location = "https://ftp.strw.leidenuniv.nl/mcgibbon/SOAP/"
test_output_location = "test_data/"

def requires(filepath):
    """
    Use this as a decorator around tests that require data.
    """

    # First check if the test data directory exists
    if not os.path.exists(test_output_location):
        os.mkdir(test_output_location)

    filename = os.path.basename(filepath)
    output_location = f"{test_output_location}{filename}"

    # Download the file if it doesn't exist
    if os.path.exists(output_location):
        ret = 0
    else:
        ret = subprocess.call(
            ["wget", f"{webstorage_location}{filepath}", "-O", output_location]
        )

    if ret != 0:
        Warning(f"Unable to download file at {filepath}")
        # It wrote an empty file, kill it.
        subprocess.call(["rm", output_location])

        def dont_call_test(func):
            def empty(*args, **kwargs):
                return True

            return empty

        return dont_call_test

    else:
        # We can do the test!

        def do_call_test(func):
            def final_test():
                return func(f"{output_location}")

            return final_test

        return do_call_test

    raise Exception("You should never have got here.")

if __name__ == '__main__':
    # Download the data required for run_small_volume.sh
    # Call @requires by passing a dummy function
    dummy = lambda x: x
    requires("swift_output/fof_output_0018.hdf5")(dummy)()
    requires("swift_output/snap_0018.hdf5")(dummy)()
    requires("HBT_output/018/SubSnap_018.0.hdf5")(dummy)()

