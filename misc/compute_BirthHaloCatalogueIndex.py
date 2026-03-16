#!/bin/env python

"""
compute_BirthHaloCatalogueIndex.py

This script produces an auxiliary snapshot which contains the subhalo
id each star was part of when it first formed.

Usage:

  mpirun -- python -u misc/compute_BirthHaloCatalogueIndex.py \
            --snap-basename SNAP_BASENAME \
            --membership-basename MEMBERSHIP_BASENAME \
            --output-basename OUTPUT_FILENAME \
            --final-snap-nr FINAL_SNAP_NR

Run "python misc/compute_BirthHaloCatalogueIndex.py -h" for a full description
of the arguments, and a list of optional arguments.

"""

import argparse
import datetime
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


def load_particle_data(snap_basename, membership_basename, load_gas, comm):
    """
    Load the particle IDs and halo membership for the particle types
    we will use to match. Removes unbound particles.
    """

    particle_data = {}

    # Load particle IDs
    snap_filename = snap_basename + ".{file_nr}.hdf5"
    file = phdf5.MultiFile(
        snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )
    particle_data["PartType4/ParticleIDs"] = file.read("PartType4/ParticleIDs")
    if load_gas:
        particle_data["PartType0/ParticleIDs"] = file.read("PartType0/ParticleIDs")

    # Membership files don't have a header, so create a list of filenames
    n_file = len(file.filenames)
    membership_filenames = [f"{membership_basename}.{i}.hdf5" for i in range(n_file)]
    # Load membership information
    file = phdf5.MultiFile(
        membership_filenames, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )
    particle_data["PartType4/GroupNr_bound"] = file.read("PartType4/GroupNr_bound")
    if load_gas:
        particle_data["PartType0/GroupNr_bound"] = file.read("PartType0/GroupNr_bound")

    # Check the two files are partitioned the same way
    assert (
        particle_data["PartType4/GroupNr_bound"].shape
        == particle_data["PartType4/ParticleIDs"].shape
    )
    if load_gas:
        assert (
            particle_data["PartType0/GroupNr_bound"].shape
            == particle_data["PartType0/ParticleIDs"].shape
        )

    return particle_data


# Units for the dimensionless fields we will be saving
unit_attrs = {
    "Conversion factor to CGS (not including cosmological corrections)": [1.0],
    "Conversion factor to physical CGS (including cosmological corrections)": [1.0],
    "U_I exponent": [0.0],
    "U_L exponent": [0.0],
    "U_M exponent": [0.0],
    "U_t exponent": [0.0],
    "U_T exponent": [0.0],
    "a-scale exponent": [0.0],
    "h-scale exponent": [0.0],
    "Property can be converted to comoving": [0],
    "Value stored as physical": [1],
}


def mpi_print(string, comm_rank):
    if comm_rank == 0:
        print(string)


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(
        description=("Script to calculate BirthHaloCatalogueIndex of star particles"),
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
        "--membership-basename",
        type=str,
        required=True,
        help="The basename of the membership files",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        required=True,
        help="The basename of the output files",
    )
    parser.add_argument(
        "--final-snap-nr",
        type=int,
        required=True,
        help=("Snapshot at which to load the particles"),
    )
    parser.add_argument(
        "--calculate-PreBirthHaloCatalogueIndex",
        action="store_true",
        help=(
            "Whether to calculate and output the subhalo halo catalogue "
            "index of the gas particle that formed each star"
        ),
    )

    args = parser.parse_args()

    # Log the arguments
    for k, v in vars(args).items():
        mpi_print(f"  {k}: {v}", comm_rank)

    final_snap_basename = args.snap_basename.format(snap_nr=args.final_snap_nr)
    final_membership_basename = args.membership_basename.format(
        snap_nr=args.final_snap_nr
    )
    mpi_print("Loading stars from final snapshot", comm_rank)
    particle_data = load_particle_data(
        final_snap_basename,
        final_membership_basename,
        False,
        comm,
    )
    star_particle_ids = particle_data["PartType4/ParticleIDs"]
    star_birth_ids = particle_data["PartType4/GroupNr_bound"]
    star_birth_ids[:] = -99
    star_first_snapshot = np.copy(star_birth_ids)

    if args.calculate_PreBirthHaloCatalogueIndex:
        particle_data["PartType0/ParticleIDs"] = np.ones(0)
        particle_data["PartType0/GroupNr_bound"] = np.ones(0)
        star_prebirth_ids = np.copy(star_birth_ids)

    for snap_nr in range(0, args.final_snap_nr + 1):

        mpi_print(f"Loading data from snapshot {snap_nr}", comm_rank)
        if args.calculate_PreBirthHaloCatalogueIndex:
            # We need to keep the gas IDs from snapshot N-1
            gas_particle_ids = particle_data["PartType0/ParticleIDs"]
            gas_group_nr = particle_data["PartType0/GroupNr_bound"]
        snap_basename = args.snap_basename.format(snap_nr=snap_nr)
        membership_basename = args.membership_basename.format(snap_nr=snap_nr)
        particle_data = load_particle_data(
            snap_basename,
            membership_basename,
            args.calculate_PreBirthHaloCatalogueIndex,
            comm,
        )

        mpi_print(f"Matching stars", comm_rank)
        # It would be quicker to make use of the BirthScaleFactors
        # instead of checking all stars
        idx = psort.parallel_match(
            star_particle_ids[star_birth_ids == -99],
            particle_data["PartType4/ParticleIDs"],
            comm=comm,
        )

        new_birth_ids = psort.fetch_elements(
            particle_data["PartType4/GroupNr_bound"],
            idx[idx != -1],
            comm=comm,
        )

        has_new_birth_id = star_birth_ids == -99
        has_new_birth_id[has_new_birth_id] = idx != -1
        star_birth_ids[has_new_birth_id] = new_birth_ids
        star_first_snapshot[has_new_birth_id] = snap_nr

        if args.calculate_PreBirthHaloCatalogueIndex:
            mpi_print(f"Matching gas", comm_rank)
            # Identify the gas progenitor of the newly formed stars
            gas_idx = psort.parallel_match(
                star_particle_ids[has_new_birth_id],
                gas_particle_ids,
                comm=comm,
            )
            # The gas progenitor may not exist for all stars due
            # to particle splitting. Note this information is
            # recoverable if required by using the SplitTrees.
            new_prebirth_ids = -99 * np.ones_like(new_birth_ids)
            new_prebirth_ids[gas_idx != -1] = psort.fetch_elements(
                gas_group_nr,
                gas_idx[gas_idx != -1],
                comm=comm,
            )
            star_prebirth_ids[has_new_birth_id] = new_prebirth_ids

    # Check we found a value for every star
    assert np.sum(star_birth_ids == -99) == 0

    # Set up what we want to output
    output = {
        "BirthHaloCatalogueIndex": star_birth_ids,
        "FirstSnapshot": star_first_snapshot,
    }
    attrs = {
        "BirthHaloCatalogueIndex": {
            "Description": "The HaloCatalogueIndex of this particle at the first snapshot it appeared."
        },
        "FirstSnapshot": {
            "Description": "Index of the first simulation snapshot in which the star particle is present."
        },
    }
    attrs["BirthHaloCatalogueIndex"].update(unit_attrs)
    attrs["FirstSnapshot"].update(unit_attrs)
    if args.calculate_PreBirthHaloCatalogueIndex:
        output["PreBirthHaloCatalogueIndex"] = star_prebirth_ids
        attrs["PreBirthHaloCatalogueIndex"] = {
            "Description": "The HaloCatalogueIndex of gas prognitor at the snapshot before the star formed. -99 if no gas progenitor is found."
        }
        attrs["PreBirthHaloCatalogueIndex"].update(unit_attrs)

    # Check the output directory exists
    output_filename = (
        args.output_basename.format(snap_nr=args.final_snap_nr) + ".{file_nr}.hdf5"
    )
    if comm_rank == 0:
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)
    comm.barrier()

    # Write the output
    mpi_print("Writing output", comm_rank)
    snap_file = phdf5.MultiFile(
        final_snap_basename + ".{file_nr}.hdf5",
        file_nr_attr=("Header", "NumFilesPerSnapshot"),
        comm=comm,
    )
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
