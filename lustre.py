#!/bin/env python

import subprocess
import os


def setstripe(dirname, stripe_size, stripe_count):
    """
    Try to set Lustre striping on a directory
    """
    args = [
        "lfs",
        "setstripe",
        "--stripe-count=%d" % stripe_count,
        "--stripe-size=%dM" % stripe_size,
        dirname,
    ]
    try:
        subprocess.run(args)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # if the 'lfs' command is not available, this will generate a
        # FileNotFoundError
        print("WARNING: failed to set lustre striping on %s" % dirname)


def ensure_output_dir(filename):

    # Try to ensure the directory exists
    dirname = os.path.dirname(filename)
    try:
        os.makedirs(dirname)
    except OSError:
        pass

    # Try to set striping
    setstripe(dirname, 32, -1)
