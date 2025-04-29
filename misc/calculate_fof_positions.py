#!/bin/env python

"""

mpirun -np 8 python -u misc/fof_prop.py \
    --snap-basename "${sim_dir}/snapshots/colibre_${snap_nr}/colibre_${snap_nr}" \
    --fof-basename "${sim_dir}/fof/fof_output_${snap_nr}/fof_output_${snap_nr}" \
    --output-basename "${output_dir}/${sim_name}/fof_extent_${snap_nr}/fof_extent_${snap_nr}"
TODO: WRITE DOCSTRING
convert_eagle.py

This script converts EAGLE GADGET snapshots into SWIFT snapshots.
Usage:

  mpirun -- python convert_eagle.py \
          --snapshot-basename=SNAPSHOT \
          --fof-basename=FOF \
          --output-basename=OUTPUT \
          --membership-basename=MEMBERSHIP

where SNAPSHOT is the EAGLE snapshot (use the particledata_*** files,
since the normal snapshots don't store SubGroupNumber), and OUTPUT &
MEMBERSHIP are the names of the output files. You must run with
the same number of ranks as input files.
"""

import argparse
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np
import unyt

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5
from virgo.mpi.gather_array import gather_array

# Parse arguments
parser = argparse.ArgumentParser(
    description=(
        "Script to calculate extent of FoF groups."
    )
)
parser.add_argument(
    "--snap-basename",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--fof-basename",
    type=str,
    required=True,
    help="The basename for the output files",
)
parser.add_argument(
    "--output-basename",
    type=str,
    required=True,
    help=(
        "The basename for the output files"
    ),
)
parser.add_argument(
    "--null-fof-id",
    type=int,
    required=False,
    default=2147483647,
    help=(
        "The FOFGroupIDs of particles not in a FOF group"
    ),
)
args = parser.parse_args()
snap_filename = args.snap_basename + ".{file_nr}.hdf5"
fof_filename = args.fof_basename + ".{file_nr}.hdf5"
output_filename = args.output_basename + ".{file_nr}.hdf5"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

if comm_rank == 0:
    with h5py.File(snap_filename.format(file_nr=0), "r") as file:
        header = dict(file['Header'].attrs)
        coordinate_unit_attrs = dict(file['PartType1/Coordinates'].attrs)
else:
    header = None
    coordinate_unit_attrs = None
header = comm.bcast(header)
coordinate_unit_attrs = comm.bcast(coordinate_unit_attrs)

boxsize = header['BoxSize']
ptypes = np.where(header['TotalNumberOfParticles'] != 0)[0]
nr_files = header['NumFilesPerSnapshot'][0]

# Assign files to ranks
files_per_rank = np.zeros(comm_size, dtype=int)
files_per_rank[:] = nr_files // comm_size
remainder = nr_files % comm_size
if remainder > 0:
    step = max(nr_files // (remainder+1), 1)
    for i in range(remainder):
        files_per_rank[(i*step) % comm_size] += 1
first_file = np.cumsum(files_per_rank) - files_per_rank
assert sum(files_per_rank) == nr_files

# Create output files
for i_file in range(
    first_file[comm_rank], first_file[comm_rank] + files_per_rank[comm_rank]
):
    src_filename = fof_filename.format(file_nr=i_file)
    dst_filename = output_filename.format(file_nr=i_file)

    def copy_attrs(src_obj, dst_obj):
        for key, val in src_obj.attrs.items():
            dst_obj.attrs[key] = val

    with h5py.File(src_filename, "r") as src_file, h5py.File(dst_filename, "w") as dst_file:

        def create_virtual_datasets(src_group, dst_group, src_filename, prefix=""):
            copy_attrs(src_group, dst_group)

            for name, item in src_group.items():
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
                    vsource = h5py.VirtualSource(src_filename, prefix + name, shape=shape)
                    layout[...] = vsource
                    vds = dst_group.create_virtual_dataset(name, layout)
                    copy_attrs(item, vds)
                elif isinstance(item, h5py.Group):
                    new_group = dst_group.create_group(name)
                    create_virtual_datasets(item, new_group, src_filename, prefix + name + "/")

        create_virtual_datasets(src_file, dst_file, src_filename)

# Load FOF catalogue
fof_file = phdf5.MultiFile(
    fof_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
)
fof_group_ids = fof_file.read(f"Groups/GroupIDs")
fof_sizes = fof_file.read(f"Groups/Sizes")
fof_centres = fof_file.read(f"Groups/Centres")

# Initialise arrays for storing results
fof_min_pos = np.inf * np.ones_like(fof_centres)
fof_max_pos = -np.inf * np.ones_like(fof_centres)
total_part_counts = np.zeros_like(fof_sizes)

# Open snapshot file
snap_file = phdf5.MultiFile(
    snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
)
for ptype in ptypes:
    if comm_rank == 0:
        print(f'Processing PartType{ptype}')

    # Load particle positions and their FOF IDs
    part_pos = snap_file.read(f"PartType{ptype}/Coordinates")
    part_fof_ids = snap_file.read(f"PartType{ptype}/FOFGroupIDs")

    # Ignore particles which aren't part of a FOF group
    mask = part_fof_ids != args.null_fof_id
    part_pos = part_pos[mask]
    part_fof_ids = part_fof_ids[mask]

    # Get the centre of the FOF each particle is part of
    idx = psort.parallel_match(part_fof_ids, fof_group_ids, comm=comm)
    assert np.all(idx != -1), 'FOFs could not be found for some particles'
    part_centre = psort.fetch_elements(fof_centres, idx, comm=comm)

    # Centre the particles
    shift = (boxsize[None, :] / 2) - part_centre
    part_pos = ((part_pos + shift) % boxsize[None, :]) - (boxsize[None, :] / 2)

    # Count the number of particles found for each FOF
    unique_fof_ids, unique_counts = psort.parallel_unique(
        part_fof_ids,
        return_counts=True,
        comm=comm,
    )

    # Keep track of number of particles in each FOF (to compare with "Groups/Sizes")
    idx = psort.parallel_match(fof_group_ids, unique_fof_ids, comm=comm)
    mask = idx != -1
    total_part_counts[mask] += psort.fetch_elements(unique_counts, idx[mask], comm=comm)

    # Sort based on unique_fof_ids
    order = psort.parallel_sort(unique_fof_ids, return_index=True, comm=comm)
    unique_counts = psort.fetch_elements(unique_counts, order, comm=comm)

    # Determine how to partition the particles, so groups will not span ranks
    gathered_counts = gather_array(unique_counts)
    if comm_rank == 0:
        n_part_target = np.sum(gathered_counts) / comm_size
        cumsum = np.cumsum(gathered_counts)
        ranks = np.floor(cumsum / n_part_target).astype(int)
        ranks = np.clip(ranks, 0, comm_size-1)
        n_part_per_rank = np.bincount(ranks, weights=gathered_counts, minlength=comm_size)
        assert np.sum(n_part_per_rank) == np.sum(gathered_counts)
    else:
        n_part_per_rank = None
    del gathered_counts
    n_part_per_rank = comm.bcast(n_part_per_rank)

    # Repartition particle data
    part_fof_ids = psort.repartition(part_fof_ids, n_part_per_rank, comm=comm)
    part_pos = psort.repartition(part_pos, n_part_per_rank, comm=comm)

    # Sort particles by FOF ID
    order = psort.parallel_sort(part_fof_ids, return_index=True, comm=comm)
    part_pos = psort.fetch_elements(part_pos, order, comm=comm)

    # Find boundaries where FOF ID changes
    if part_fof_ids.shape[0] != 0:
        change_idx = np.flatnonzero(np.diff(part_fof_ids)) + 1
        reduce_idx = np.concatenate([np.array([0]), change_idx])
    else:
        reduce_idx = np.array([], dtype=int)

    # Determine link to arrays from original FOF catalogue
    local_fof_ids = part_fof_ids[reduce_idx]
    idx = psort.parallel_match(fof_group_ids, local_fof_ids, comm=comm)
    mask = idx != -1

    for i in range(3):
        # Calculate max and minimum particle positions for each FOF ID
        local_min_pos = np.minimum.reduceat(part_pos[:, i], reduce_idx)
        local_max_pos = np.maximum.reduceat(part_pos[:, i], reduce_idx)

        # Compare with values from other particle types
        min_pos = psort.fetch_elements(local_min_pos, idx[mask], comm=comm)
        max_pos = psort.fetch_elements(local_max_pos, idx[mask], comm=comm)
        fof_min_pos[mask, i] = np.minimum(fof_min_pos[mask, i], min_pos)
        fof_max_pos[mask, i] = np.maximum(fof_max_pos[mask, i], max_pos)

# Carry out some sanity checks
assert np.all(total_part_counts == fof_sizes), 'Not all particles found for some FOFs'
assert np.max(fof_min_pos) < 0
assert np.min(fof_max_pos) > 0

# Write data to file
output_data = {
    "MinParticlePosition": fof_min_pos,
    "MaxParticlePosition": fof_max_pos,
}
elements_per_file = fof_file.get_elements_per_file("Groups/GroupIDs")
fof_file.write(
    output_data,
    elements_per_file,
    filenames=output_filename,
    mode='r+',
    group="Groups",
    attrs=coordinate_unit_attrs,
)

if comm_rank == 0:
    print('Done')

