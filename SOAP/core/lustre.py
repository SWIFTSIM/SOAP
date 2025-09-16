#!/bin/env python

import subprocess
import os


def setstripe(filename, stripe_size=32, stripe_count=32):
    """
    Try to set Lustre striping on a file
    """
    # Remove file if it already exists (or lfs will error)
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    # Only set striping for snap8 on cosma
    if not filename.startswith('/snap8/scratch'):
        print(f'Not setting lustre striping on {filename}')
        return

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
