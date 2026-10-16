#!/bin/env python

"""
resort_BirthHaloCatalogueIndex.py

This script takes the output of compute_BirthHaloCatalogueIndex.py for
one snapshot and reorders it to match the star particles of an earlier
snapshot. This is possible since the values only depend on the snapshots
up to when each star first appeared, and star particles are never removed.

Usage:

  mpirun -- python -u misc/resort_BirthHaloCatalogueIndex.py \
            --snap-basename SNAP_BASENAME \
            --input-basename INPUT_BASENAME \
            --output-basename OUTPUT_BASENAME \
            --input-snap-nr INPUT_SNAP_NR \
            --output-snap-nr OUTPUT_SNAP_NR

Run "python misc/resort_BirthHaloCatalogueIndex.py -h" for a full description
of the arguments.

"""

import argparse
import datetime
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import numpy as np

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

datasets = [
    "BirthHaloCatalogueIndex",
    "FirstSnapshot",
    "PreBirthHaloCatalogueIndex",
]


def mpi_print(string, comm_rank):
    if comm_rank == 0:
        print(string)


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(
        description=(
            "Script to reorder the output of compute_BirthHaloCatalogueIndex.py "
            "to match the star particles of an earlier snapshot"
        ),
    )
    parser.add_argument(
        "--snap-basename",
        type=str,
        required=True,
        help=(
            "The basename of the snapshot files (the snapshot "
            "name without the .{file_nr}.hdf5 suffix. Use "
            "{snap_nr:04d} instead of the snapshot number)"
        ),
    )
    parser.add_argument(
        "--input-basename",
        type=str,
        required=True,
        help=(
            "The basename of the existing BirthHaloCatalogueIndex files. "
            "Use {snap_nr:04d} instead of the snapshot number"
        ),
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        required=True,
        help=(
            "The basename of the output files. "
            "Use {snap_nr:04d} instead of the snapshot number"
        ),
    )
    parser.add_argument(
        "--input-snap-nr",
        type=int,
        required=True,
        help="Snapshot the existing BirthHaloCatalogueIndex files were computed for",
    )
    parser.add_argument(
        "--output-snap-nr",
        type=int,
        required=True,
        help="Snapshot to reorder the BirthHaloCatalogueIndex values to",
    )

    args = parser.parse_args()

    # Log the arguments
    for k, v in vars(args).items():
        mpi_print(f"  {k}: {v}", comm_rank)

    assert args.output_snap_nr <= args.input_snap_nr

    # Load star particle IDs from the input snapshot
    mpi_print("Loading input snapshot", comm_rank)
    input_snap_basename = args.snap_basename.format(snap_nr=args.input_snap_nr)
    file = phdf5.MultiFile(
        input_snap_basename + ".{file_nr}.hdf5",
        file_nr_attr=("Header", "NumFilesPerSnapshot"),
        comm=comm,
    )
    input_particle_ids = file.read("PartType4/ParticleIDs")

    # Load existing values. These files don't have a header, so create
    # a list of filenames
    mpi_print("Loading existing BirthHaloCatalogueIndex", comm_rank)
    n_file = len(file.filenames)
    input_basename = args.input_basename.format(snap_nr=args.input_snap_nr)
    input_filenames = [f"{input_basename}.{i}.hdf5" for i in range(n_file)]
    file = phdf5.MultiFile(input_filenames, comm=comm)
    input_data = file.read(datasets, group="PartType4", read_attributes=True)
    for name in ["BirthHaloCatalogueIndex", "FirstSnapshot"]:
        assert input_data.get(name) is not None, f"{name} missing from input files"
    # PreBirthHaloCatalogueIndex is only present if it was requested when
    # running compute_BirthHaloCatalogueIndex.py
    resort_datasets = [name for name in datasets if input_data.get(name) is not None]
    mpi_print(f"Datasets to resort: {resort_datasets}", comm_rank)
    for name in resort_datasets:
        assert input_data[name].shape == input_particle_ids.shape
    attrs = {name: dict(input_data[name].attrs) for name in resort_datasets}

    # Load star particle IDs from the output snapshot
    mpi_print("Loading output snapshot", comm_rank)
    output_snap_basename = args.snap_basename.format(snap_nr=args.output_snap_nr)
    snap_file = phdf5.MultiFile(
        output_snap_basename + ".{file_nr}.hdf5",
        file_nr_attr=("Header", "NumFilesPerSnapshot"),
        comm=comm,
    )
    output_particle_ids = snap_file.read("PartType4/ParticleIDs")
    # Read returns None if there are no stars in the output snapshot
    if output_particle_ids is None:
        mpi_print("No stars in output snapshot", comm_rank)
        output_particle_ids = np.zeros(0, dtype=input_particle_ids.dtype)

    # Match stars. Stars are never removed, so all should be found
    mpi_print("Matching stars", comm_rank)
    idx = psort.parallel_match(output_particle_ids, input_particle_ids, comm=comm)
    n_missing = comm.allreduce(np.sum(idx == -1))
    assert n_missing == 0, f"{n_missing} stars not found in input snapshot"

    output = {}
    for name in resort_datasets:
        output[name] = psort.fetch_elements(input_data[name], idx, comm=comm)

    # Check the output directory exists
    output_filename = (
        args.output_basename.format(snap_nr=args.output_snap_nr) + ".{file_nr}.hdf5"
    )
    if comm_rank == 0:
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
    comm.barrier()

    # Write the output
    mpi_print("Writing output", comm_rank)
    elements_per_file = snap_file.get_elements_per_file(
        "ParticleIDs", group="PartType4"
    )
    snap_file.write(
        output,
        elements_per_file,
        filenames=output_filename,
        mode="w",
        group="PartType4",
        attrs=attrs,
    )

    # Finished
    comm.barrier()
    mpi_print(f"Runtime: {datetime.datetime.now() - start_time}", comm_rank)
    mpi_print("Done!", comm_rank)
