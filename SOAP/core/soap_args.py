#!/bin/env python

import argparse
import os
import subprocess
import sys
import warnings
from mpi4py import MPI
import numpy as np

from virgo.mpi.util import MPIArgumentParser
from virgo.util.partial_formatter import PartialFormatter

from . import combine_args


def get_halo_indices(parameters):
    """
    Return the indices of the halos to process, or None if all halos should
    be processed. The indices are either passed on the command line, or are
    read from a file which contains one index per line. Blank lines and lines
    starting with "#" are ignored.

    Returns a sorted array of the unique indices which were requested.
    """

    filename = parameters["halo_indices_file"]
    if filename is not None:
        # Substitute the snapshot number into the filename
        pf = PartialFormatter()
        filename = pf.format(filename, snap_nr=parameters["snap_nr"], file_nr=None)
        if not os.path.exists(filename):
            raise ValueError(f"Unable to find halo indices file: {filename}")
        try:
            with warnings.catch_warnings():
                # Empty files generate a warning, but we handle them below
                warnings.filterwarnings(
                    "ignore", message="loadtxt: input contained no data"
                )
                # converters=int prevents non-integer values from being silently
                # truncated, which np.loadtxt would otherwise do
                halo_indices = np.loadtxt(
                    filename, dtype=np.int64, ndmin=1, comments="#", converters=int
                )
        except Exception as e:
            raise ValueError(f"Unable to read halo indices file {filename}: {e}")
        if halo_indices.ndim != 1:
            raise ValueError(
                f"Halo indices file must contain one index per line: {filename}"
            )
        if halo_indices.shape[0] == 0:
            raise ValueError(f"Halo indices file is empty: {filename}")
    elif parameters["halo_indices"] is not None:
        halo_indices = np.asarray(parameters["halo_indices"], dtype=np.int64)
    else:
        return None

    return np.unique(halo_indices)


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
        help="Splits volume into N chunks and each compute node processes one chunk "
        "at a time. Should be at least the number of nodes (default: 1)",
    )
    parser.add_argument(
        "--dmo",
        action="store_true",
        help="Run in dark matter only mode, skipping any hydro-only properties",
    )
    parser.add_argument(
        "--centrals-only",
        action="store_true",
        help="Only process central halos, discarding satellites",
    )
    parser.add_argument(
        "--record-halo-timings",
        action="store_true",
        help="Record time taken to process each halo",
    )
    parser.add_argument(
        "--record-property-timings",
        action="store_true",
        help="Record time taken to process each property. This doubles the size of "
        "the output catalogue",
    )
    parser.add_argument(
        "--max-halos",
        metavar="N",
        type=int,
        default=0,
        help="(For debugging) only process the first N halos in the catalogue",
    )
    halo_index_group = parser.add_mutually_exclusive_group()
    halo_index_group.add_argument(
        "--halo-indices",
        nargs="*",
        type=int,
        help="Only process the specified halo indices",
    )
    halo_index_group.add_argument(
        "--halo-indices-file",
        type=str,
        help="Only process the halo indices listed in the specified file, which "
        "must contain one index per line. The snapshot number is substituted "
        "into the filename, e.g. halo_indices_{snap_nr:04d}.txt",
    )
    parser.add_argument(
        "--reference-snapshot",
        help="Specify reference snapshot number containing all particle types. "
        "Used to determine the datasets and units of any particle types which "
        "are missing from the snapshot being processed",
        metavar="N",
        type=int,
    )
    parser.add_argument(
        "--profile",
        metavar="LEVEL",
        type=int,
        default=0,
        help="Run with profiling (0=off, 1=first MPI rank only, 2=all ranks) "
        "(default: 0)",
    )
    parser.add_argument(
        "--max-ranks-reading",
        type=int,
        default=32,
        help="Number of ranks per node reading snapshot data. Can be reduced to "
        "avoid overloading the file system (default: 32)",
    )
    parser.add_argument(
        "--output-parameters",
        type=str,
        default="",
        help="Where to write the parameters used by this run, in yaml format",
    )
    parser.add_argument(
        "--snipshot",
        action="store_true",
        help="Run in snipshot mode, overriding the value of SelectOutput in the "
        "snapshot header",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Run in snapshot mode, overriding the value of SelectOutput in the "
        "snapshot header",
    )
    all_args = parser.parse_args()

    # Combine with parameters from configuration file
    if comm.Get_rank() == 0:
        try:
            all_args = combine_args.combine_arguments(all_args, all_args.config_file)
            # Halo indices are read on this rank and broadcast as an array,
            # since there can be a large number of them
            all_args["Parameters"]["halo_indices"] = get_halo_indices(
                all_args["Parameters"]
            )
        except ValueError as e:
            print(e, flush=True)
            comm.Abort(1)
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
    args.read_potential_energies = all_args["HaloFinder"].get(
        "read_potential_energies", False
    )
    args.fof_group_filename = all_args["HaloFinder"].get("fof_filename", "")
    args.fof_radius_filename = all_args["HaloFinder"].get("fof_radius_filename", "")
    args.output_file = all_args["HaloProperties"]["filename"]
    args.snapshot_nr = all_args["Parameters"]["snap_nr"]
    args.chunks = all_args["Parameters"]["chunks"]
    args.centrals_only = all_args["Parameters"]["centrals_only"]
    args.record_halo_timings = all_args["Parameters"]["record_halo_timings"]
    args.record_property_timings = all_args["Parameters"]["record_property_timings"]
    args.dmo = all_args["Parameters"]["dmo"]
    args.max_halos = all_args["Parameters"]["max_halos"]
    args.halo_indices = all_args["Parameters"]["halo_indices"]
    args.reference_snapshot = all_args["Parameters"]["reference_snapshot"]
    args.profile = all_args["Parameters"]["profile"]
    args.max_ranks_reading = all_args["Parameters"]["max_ranks_reading"]
    args.output_parameters = all_args["Parameters"]["output_parameters"]
    args.git_hash = all_args["git_hash"]
    args.calculations = all_args.get("calculations", {})
    args.min_read_radius_cmpc = args.calculations.get("min_read_radius_cmpc", 0)

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
            print("Can't write to output directory", flush=True)
            comm.Abort(1)
        # Check if the FOF files exist
        if args.fof_group_filename != "":
            fof_filename = args.fof_group_filename.format(
                snap_nr=args.snapshot_nr, file_nr=0
            )
            if not os.path.exists(fof_filename):
                print(f"Could not find FOF group catalogue: {fof_filename}", flush=True)
                comm.Abort(1)
        if args.fof_radius_filename != "":
            assert args.fof_group_filename != ""
            fof_filename = args.fof_radius_filename.format(
                snap_nr=args.snapshot_nr, file_nr=0
            )
            if not os.path.exists(fof_filename):
                print(
                    f"Could not find FOF radius catalogue: {fof_filename}", flush=True
                )
                comm.Abort(1)

    # This really should be done in parameter_file.py
    args.separate_chunks = args.calculations.get("separate_chunks", [])
    if not isinstance(args.separate_chunks, list):
        print("Invalid form for separate_chunks", flush=True)
        comm.Abort(1)
    for threshold in args.separate_chunks:
        if ("n_bound_threshold" not in threshold) or (
            "n_halo_per_chunk" not in threshold
        ):
            print("Invalid form for separate_chunks", flush=True)
            comm.Abort(1)
    args.separate_chunks = sorted(
        args.separate_chunks,
        key=lambda x: -x["n_bound_threshold"],
    )

    return args
