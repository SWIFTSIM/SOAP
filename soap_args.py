#!/bin/env python

import argparse
import os
import subprocess
import sys
from mpi4py import MPI

from virgo.mpi.util import MPIArgumentParser

import combine_args


def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

def get_soap_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """

    # Define command line arguments
    parser = MPIArgumentParser(comm=comm, description="Compute halo properties in SWIFT snapshots.")
    parser.add_argument("config_file", type=str, help="Name of the yaml configuration file")
    parser.add_argument("--sim-name", type=str, help="Name of the simulation to process")
    parser.add_argument("--snap-nr", type=int, help="Snapshot number to process")    
    parser.add_argument("--chunks", metavar="N", type=int, default=1, help="Splits volume into N chunks and each compute node processes one chunk at a time")
    parser.add_argument("--dmo", action="store_true", help="Run in dark matter only mode")
    parser.add_argument("--centrals-only", action="store_true", help="Only process central halos")
    parser.add_argument("--max-halos", metavar="N", type=int, default=0, help="(For debugging) only process the first N halos in the catalogue")
    parser.add_argument("--halo-ids", nargs="*", type=int, help="Only process the specified halo IDs")
    parser.add_argument("--calculations", nargs="*", help="Which calculations to do (default is to do all)")
    parser.add_argument("--reference-snapshot", help="Specify reference snapshot number containing all particle types", metavar="N", type=int)
    parser.add_argument("--profile", metavar="LEVEL", type=int, default=0, help="Run with profiling (0=off, 1=first MPI rank only, 2=all ranks)")
    parser.add_argument("--max-ranks-reading", type=int, default=32, help="Number of ranks per node reading snapshot data")
    parser.add_argument("--output-parameters", type=str, default='', help="Where to write the used parameters")
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
    args.halo_sizes_file = all_args["GroupMembership"]["halo_sizes_file"]
    args.output_file = all_args["HaloProperties"]["filename"]
    args.snapshot_nr = all_args["Parameters"]["snap_nr"]
    args.chunks = all_args["Parameters"]["chunks"]
    args.extra_input = all_args["GroupMembership"]["filename"]
    args.centrals_only = all_args["Parameters"]["centrals_only"]
    args.dmo = all_args["Parameters"]["dmo"]
    args.max_halos = all_args["Parameters"]["max_halos"]
    args.halo_ids = all_args["Parameters"]["halo_ids"]
    args.calculations = all_args["Parameters"]["calculations"]
    args.reference_snapshot = all_args["Parameters"]["reference_snapshot"]
    args.profile = all_args["Parameters"]["profile"]
    args.max_ranks_reading = all_args["Parameters"]["max_ranks_reading"]
    args.output_parameters = all_args["Parameters"]["output_parameters"]
    args.git_hash = all_args["git_hash"]

    return args


def get_match_vr_halos_args(comm):
    """
    Process command line arguments for halo matching program.

    Returns a dict with the argument values, or None on failure.
    """

    if comm.Get_rank() == 0:

        os.environ[
            "COLUMNS"
        ] = "80"  # Can't detect terminal width when running under MPI?

        parser = ThrowingArgumentParser(description="Match halos between VR outputs.")
        parser.add_argument(
            "vr_basename1",
            help="Base name of the first VELOCIraptor files, excluding trailing .properties[.N] etc.",
        )
        parser.add_argument(
            "vr_basename2",
            help="Base name of the second VELOCIraptor files, excluding trailing .properties[.N] etc.",
        )
        parser.add_argument(
            "nr_particles",
            metavar="N",
            type=int,
            help="Number of most bound particles to use.",
        )
        parser.add_argument("output_file", help="Output file name")
        parser.add_argument(
            "--use-types",
            nargs="*",
            type=int,
            help="Only use the specified particle types (integer, 0-6)",
        )
        parser.add_argument(
            "--to-field-halos-only",
            action="store_true",
            help="Only match to field halos (with hostHaloID=-1 in VR catalogue)",
        )
        try:
            args = parser.parse_args()
        except ArgumentParserError as e:
            args = None

    else:
        args = None

    args = comm.bcast(args)
    if args is None:
        MPI.Finalize()
        sys.exit(0)

    return args