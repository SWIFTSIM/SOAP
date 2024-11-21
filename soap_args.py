#!/bin/env python

import argparse
import os
import subprocess
import sys
from mpi4py import MPI

from virgo.mpi.util import MPIArgumentParser

import combine_args


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        print("Could not determine git hash")
        return ""


def get_soap_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """

    # Define command line arguments
    parser = MPIArgumentParser(
        comm=comm, description="Compute halo properties in SWIFT snapshots."
    )
    parser.add_argument(
        "config_file", type=str, help="Name of the yaml configuration file"
    )
    parser.add_argument(
        "--sim-name", type=str, help="Name of the simulation to process"
    )
    parser.add_argument("--snap-nr", type=int, help="Snapshot number to process")
    parser.add_argument(
        "--chunks",
        metavar="N",
        type=int,
        default=1,
        help="Splits volume into N chunks and each compute node processes one chunk at a time",
    )
    parser.add_argument(
        "--dmo", action="store_true", help="Run in dark matter only mode"
    )
    parser.add_argument(
        "--centrals-only", action="store_true", help="Only process central halos"
    )
    parser.add_argument(
        "--record-times", action="store_true", help="Record time taken to process each halo"
    )
    parser.add_argument(
        "--max-halos",
        metavar="N",
        type=int,
        default=0,
        help="(For debugging) only process the first N halos in the catalogue",
    )
    parser.add_argument(
        "--halo-indices",
        nargs="*",
        type=int,
        help="Only process the specified halo indices",
    )
    parser.add_argument(
        "--reference-snapshot",
        help="Specify reference snapshot number containing all particle types",
        metavar="N",
        type=int,
    )
    parser.add_argument(
        "--profile",
        metavar="LEVEL",
        type=int,
        default=0,
        help="Run with profiling (0=off, 1=first MPI rank only, 2=all ranks)",
    )
    parser.add_argument(
        "--max-ranks-reading",
        type=int,
        default=32,
        help="Number of ranks per node reading snapshot data",
    )
    parser.add_argument(
        "--output-parameters",
        type=str,
        default="",
        help="Where to write the used parameters",
    )
    parser.add_argument("--snipshot", action="store_true", help="Run in snipshot mode")
    parser.add_argument("--snapshot", action="store_true", help="Run in snapshot mode")
    all_args = parser.parse_args()

    # Combine with parameters from configuration file
    if comm.Get_rank() == 0:
        all_args = combine_args.combine_arguments(all_args, all_args.config_file)
        all_args["git_hash"] = get_git_hash()
    else:
        all_args = None
    all_args = comm.bcast(all_args)

    # Extract parameters we need for SOAP
    args = argparse.Namespace()
    args.config_filename = all_args["Parameters"]["config_file"]
    args.swift_filename = all_args["Snapshots"]["filename"]
    args.scratch_dir = all_args["HaloProperties"]["chunk_dir"]
    args.halo_basename = all_args["HaloFinder"]["filename"]
    args.halo_format = all_args["HaloFinder"]["type"]
    args.fof_group_filename = all_args["HaloFinder"].get("fof_filename", "")
    args.output_file = all_args["HaloProperties"]["filename"]
    args.snapshot_nr = all_args["Parameters"]["snap_nr"]
    args.chunks = all_args["Parameters"]["chunks"]
    args.centrals_only = all_args["Parameters"]["centrals_only"]
    args.record_times = all_args["Parameters"]["record_times"]
    args.dmo = all_args["Parameters"]["dmo"]
    args.max_halos = all_args["Parameters"]["max_halos"]
    args.halo_indices = all_args["Parameters"]["halo_indices"]
    args.reference_snapshot = all_args["Parameters"]["reference_snapshot"]
    args.profile = all_args["Parameters"]["profile"]
    args.max_ranks_reading = all_args["Parameters"]["max_ranks_reading"]
    args.output_parameters = all_args["Parameters"]["output_parameters"]
    args.git_hash = all_args["git_hash"]
    args.min_read_radius_cmpc = all_args["calculations"]["min_read_radius_cmpc"]
    args.calculations = all_args["calculations"]

    # Extra-input files which are optionally passed in the parameter file are
    # processed the same way as the membership files
    if "ExtraInput" in all_args:
        args.extra_input = list(all_args["ExtraInput"].values())
    else:
        args.extra_input = []
    args.extra_input.append(all_args["GroupMembership"]["filename"])

    # The default behaviour is to determine whether to run in snipshot mode
    # by looking at the value of "SelectOutut" in the snapshot header.
    # Passing --snipshot or --snapshot will override this
    if all_args["Parameters"]["snipshot"]:
        args.snipshot = True
        assert not all_args["Parameters"][
            "snapshot"
        ], "You cannot pass both --snapshot and --snipshot"
    elif all_args["Parameters"]["snapshot"]:
        args.snipshot = False
    else:
        # We will set the value of arg.snipshot later
        args.snipshot = None

    # Check certain input/output paths are valid, as they won't be used until the end
    if comm.Get_rank() == 0:
        # Check we can write to the halo properties file
        dirname = os.path.dirname(os.path.abspath(args.output_file))
        # Directory may not exist yet, so move up the tree until we find one that does
        while not os.path.exists(dirname):
            dirname = os.path.dirname(dirname)
        if not os.access(dirname, os.W_OK):
            print("Can't write to output directory")
            comm.Abort(1)
        # Check if the FOF files exist
        if args.fof_group_filename != "":
            fof_filename = args.fof_group_filename.format(
                snap_nr=args.snapshot_nr, file_nr=0
            )
            if not os.path.exists(fof_filename):
                print(f"Could not find FOF group catalogue: {fof_filename}")
                comm.Abort(1)

    return args
