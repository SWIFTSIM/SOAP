#!/bin/env python

import subprocess
import os


def setstripe(filename, stripe_size, stripe_count):
    """
    Try to set Lustre striping on a file
    """
    # Remove file if it already exists (or lfs will error)
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    args = [
        "lfs",
        "setstripe",
        f"--stripe-count={stripe_count}",
        f"--stripe-size={stripe_size}M",
        filename,
    ]
    try:
        subprocess.run(args)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # if the 'lfs' command is not available, this will generate a
        # FileNotFoundError
        print(f"WARNING: failed to set lustre striping on {filename}")

