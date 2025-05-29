#!/bin/env python

"""
# TODO: Rewrite
calculate_fof_positions.py

This script calculates the maximum and minimum particles positions for each FOF.
Usage:

  mpirun -- python misc/calculate_fof_positions.py \
          --snap-basename=SNAPSHOT \
          --fof-basename=FOF \
          --output-basename=OUTPUT

where SNAPSHOT is the basename of the snapshot files (the snapshot
name without the .{file_nr}.hdf5 suffix), FOF is the basename of the
fof catalogues, and OUTPUT is the basename of the output fof catalogues.
"""

import argparse
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5
from virgo.mpi.gather_array import gather_array

# Parse arguments
parser = argparse.ArgumentParser(
    description=("Script to calculate extent of FoF groups.")
)
parser.add_argument(
    "--snap-basename1",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--snap-basename2",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--membership-basename1",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--membership-basename2",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--soap-filename1",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
# TODO:
parser.add_argument(
    "--soap-filename",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
# TODO:
parser.add_argument(
    "--output-filename",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
# TODO:
parser.add_argument(
    "--ptypes",
    type=bool,
    required=False,
    help="Only match central halos",
)
parser.add_argument(
    "--nr_particles",
    type=int,
    required=False,
    default=50,
    help="Number of particles to use when matching. -1 to use all particles",
)
parser.add_argument(
    "--centrals-only",
    type=bool,
    required=False,
    help="Only match central halos",
)
args = parser.parse_args()
snap_filename1 = args.snap_basename1 + ".{file_nr}.hdf5"
snap_filename2 = args.snap_basename2 + ".{file_nr}.hdf5"
membership_filename1 = args.membership_basename1 + ".{file_nr}.hdf5"
membership_filename2 = args.membership_basename2 + ".{file_nr}.hdf5"

# TODO: 
ptypes = [1]

def load_particle_data(snap_basename, membership_basename, ptypes, comm):
    '''
    Load the particle IDs and halo membership for the particle types we will use to match
    Removes particles which are not bound
    '''

    # Load particle IDs
    snap_filename = snap_basename + ".{file_nr}.hdf5"
    snap_file = phdf5.MultiFile(
        snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )
    particle_ids = []
    for ptype in ptypes:
        particle_ids.append(file.read(f"PartType{particle_type}/ParticleIDs"))
    particle_ids = np.concatenate(particle_ids)

    # Load membership information
    membership_filename = membership_basename + ".{file_nr}.hdf5"
    membership_file = phdf5.MultiFile(
        membership_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )
    halo_catalogue_idx, rank_bound = [], []
    for ptype in ptypes:
        halo_catalogue_idx.append(file.read(f"PartType{particle_type}/GroupNr_bound"))
        rank_bound.append(file.read(f"PartType{particle_type}/Rank_bound"))
    halo_catalogue_idx, rank_bound = np.concatenate(halo_catalogue_idx), np.concatenate(rank_bound)

    # Check the two files are partitioned the same way
    assert particle_ids.shape == halo_catalogue_idx.shape

    # Remove any particles which are not bound to a subhalo
    mask = halo_catalogue_idx != -1
    particle_ids = particle_ids[mask]
    halo_catalogue_idx = halo_catalogue_idx[mask]
    rank_bound = rank_bound[mask]

    return {
        'particle_ids': particle_ids,
        'halo_catalogue_idx': halo_catalogue_idx,
        'rank_bound': rank_bound,
    }

data_1 = load_particle_data(args.snap_basename_1, args.membership_basename_1, ptypes, comm)
data_2 = load_particle_data(args.snap_basename_2, args.membership_basename_2, ptypes, comm)
datasets = list(data_1.keys())

# Remove particles which are not bound in both snapshots
match_1 = psort.parallel_match(data_1['particle_ids'], data_2['particle_ids'], comm=comm)
for dset in datasets:
    data_1[dset] = data_1[dst][match_1 != -1]

match_2 = psort.parallel_match(data_2['particle_ids'], data_1['particle_ids'], comm=comm)
for dset in datasets:
    data_2[dset] = data_2[dst][match_2 != -1]

# TODO: To function, be careful about passing copies of numpy arrays

# TODO: Overwrite satellites with host halo index

def match_sim(particle_ids, particle_halo_ids, rank_bound, particle_ids_to_match, particle_halo_ids_to_match):
    '''
    Takes in particle_ids, particle_halo_ids, and rank_bound from simulation 1
    and particle_ids_to_match, and particle_halo_ids_to_match from simulation 2.

    Returns halo_ids, matched_halo_ids, n_match
    '''

    # Sort particles
    sort_hash_dtype = [
        ("halo_ids", particle_halo_ids.dtype), 
        ("rank_bound", rank_bound.dtype),
    ]
    sort_hash = np.zeros(particle_halo_ids.shape[0], dtype=sort_hash_dtype)
    sort_hash["halo_ids"] = particle_halo_ids
    sort_hash["rank_bound"] = rank_bound
    order = psort.parallel_sort(sort_hash, return_index=True, comm=comm)
    # We don't require rank_bound after this point, so don't sort it
    particle_ids = psort.fetch_elements(particle_ids, order, comm=comm)
    particle_halo_ids = psort.fetch_elements(particle_halo_ids, order, comm=comm)

    # Count the number of particles for each subhalo
    unique_halo_ids, unique_counts = psort.parallel_unique(
        particle_halo_ids,
        return_counts=True,
        comm=comm,
    )

    # Determine how to partition the particles, so no subhalo spans rank
    gathered_counts = gather_array(unique_counts)
    gathered_halo_ids = gather_array(unique_halo_ids)
    if comm_rank == 0:
        argsort = np.argsort(gathered_halo_ids)
        gathered_counts = gathered_counts[argsort]

        n_part_target = np.sum(gathered_counts) / comm_size
        cumsum = np.cumsum(gathered_counts)
        ranks = np.floor(cumsum / n_part_target).astype(int)
        ranks = np.clip(ranks, 0, comm_size - 1)
        n_part_per_rank = np.bincount(
            ranks, weights=gathered_counts, minlength=comm_size
        )
        assert np.sum(n_part_per_rank) == np.sum(gathered_counts)
    else:
        n_part_per_rank = None
    n_part_per_rank = comm.bcast(n_part_per_rank)

    # Repartition data
    particle_ids = psort.repartition(particle_ids, n_part_per_rank, comm=comm)
    particle_halo_ids = psort.repartition(particle_halo_ids, n_part_per_rank, comm=comm)

    # Return if we don't have any particles on this rank
    if particle_ids.shape[0] == 0:
        # TODO: What are we returning?
        return

    # Only keep the first {args.nr_particles} particles for each subhalo
    # We can't just do a cut on rank_bound since we might be missing some ptypes
    if args.nr_particles != -1:
        # Count how many particles we have for each subhalo
        unique, counts = np.unique(particle_halo_ids, return_counts=True)
        argsort = np.argsort(unique)
        counts = unique[argsort]

        # Calculate a running sum
        cumsum = np.cumsum(counts)
        n_part_before_group_i = np.concatenate(np.array([0]), cumsum[:-1])

        # Calculate the position of each particle within its group
        group_position = np.arange(particle_ids.shape[0])
        group_position -= np.repeat(n_part_before_group_i, counts)

        # Remove unneeded particles
        mask = group_position < args.nr_particles
        particle_ids = particle_ids[mask]
        particle_halo_ids = particle_halo_ids[mask]

    # Identify which subhalo each particle is bound to within simulation 2
    idx = psort.parallel_match(particle_ids, particle_ids_to_match, comm=comm)
    particle_matched_halo_ids = psort.fetch_elements(particle_halo_ids_to_match, idx, comm=comm)

    # Combine (halo_id, matched_halo_id) into a single ID
    combined_ids = particle_halo_ids.astype(np.int64)
    combined_ids <<= 32
    combined_ids += particle_matched_halo_ids.astype(np.int64)

    # Carry out a count on the combined ID
    combined_ids, combined_counts = np.unique(combined_ids, return_counts=True)

    # Extract original halo ids
    matched_halo_ids = combined_ids.astype(np.int32)
    combined_ids -= matched_halo_ids
    combined_ids >>= 32
    halo_ids = combined_ids

    # Sort first based on halo_ids, then by count, then by matched_halo_ids
    idx = np.lexsort((matched_halo_ids, -combined_counts, halo_ids))
    matched_halo_ids = matched_halo_ids[idx]
    halo_ids = halo_ids[idx]
    combined_counts = combined_counts[idx]

    # Use np.unique to find the first instance of each halo_id
    halo_ids, idx = np.unique(halo_ids, return_index=True)
    matched_halo_ids = matched_halo_ids[idx]
    combined_counts = combined_counts[idx]

    return halo_ids, matched_halo_idx, combined_counts



















    print("Done")
