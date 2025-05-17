#!/bin/env python

"""
test_fof_positions.py

This script tests the output of calculate_fof_positions.py.
Usage:

  python misc/calculate_fof_positions.py \
      --snap-filename=SNAPSHOT \
      --fof-filename=FOF

where SNAPSHOT is the filename of the virtual snapshot file,
and FOF is the filename of the virtual FOF cataloge file.
"""

import argparse

import numpy as np
import swiftsimio as sw
import tqdm

# Parse arguments
parser = argparse.ArgumentParser(
    description="Script to calculate extent of FoF groups."
)
parser.add_argument(
    "--snap-filename",
    type=str,
    required=True,
    help="The filename of the virtual snapshot file",
)
parser.add_argument(
    "--fof-filename",
    type=str,
    required=True,
    help="The filename of the virtual FOF catalogue file",
)
parser.add_argument(
    "--n-test",
    type=int,
    required=False,
    default=-1,
    help="The number of FOFs to check. If -1 all objects will be checked",
)
args = parser.parse_args()

# Load the FOF file
fof = sw.load(args.fof_filename)
min_pos = fof.fof_groups.centres - fof.fof_groups.radii[:, None]
max_pos = fof.fof_groups.centres + fof.fof_groups.radii[:, None]

n_test = args.n_test if args.n_test != -1 else fof.fof_groups.sizes.shape[0]
for i_fof in tqdm.tqdm(range(n_test)):

    # Create mask and load data
    mask = sw.mask(args.snap_filename)
    load_region = [
        [min_pos[i_fof, 0], max_pos[i_fof, 0]],
        [min_pos[i_fof, 1], max_pos[i_fof, 1]],
        [min_pos[i_fof, 2], max_pos[i_fof, 2]],
    ]
    mask.constrain_spatial(load_region)
    snap = sw.load(args.snap_filename, mask=mask)

    # Check we loaded all the particles
    n_total = 0
    for ptype in ["gas", "dark_matter", "stars", "black_holes"]:
        fof_id = fof.fof_groups.group_ids[i_fof].value
        part_fof_ids = getattr(snap, ptype).fofgroup_ids.value
        n_total += np.sum(part_fof_ids == fof_id)
    assert n_total == fof.fof_groups.sizes[i_fof].value, f"Failed for object {i_fof}"
