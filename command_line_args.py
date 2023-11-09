#!/bin/env python

import sys
import os
import argparse
from mpi4py import MPI


class ArgumentParserError(Exception):
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message + "\n")
        raise ArgumentParserError(message)


def get_halo_props_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """

    if comm.Get_rank() == 0:

        os.environ[
            "COLUMNS"
        ] = "80"  # Can't detect terminal width when running under MPI?

        parser = ThrowingArgumentParser(
            description="Compute halo properties in SWIFT snapshots."
        )
        parser.add_argument(
            "swift_filename",
            help="Format string to generate snapshot filenames. Use %%(file_nr)d for the file number and %%(snap_nr)04d for the snapshot.",
        )
        parser.add_argument(
            "scratch_dir",
            help="Directory for temporary output. Can use %%(snap_nr)04d for the snapshot number.",
        )
        parser.add_argument(
            "vr_basename",
            help="Format string to generate base name of the VELOCIraptor files, excluding trailing .properties[.N] etc. Use %%(snap_nr)04d for the snapshot.",
        )
        parser.add_argument(
            "output_file",
            help="Format string to generate name of the output file. Use %%(snap_nr)04d for the snapshot.",
        )
        parser.add_argument("snapshot_nr", help="Snapshot number to process", type=int)
        parser.add_argument(
            "--chunks",
            metavar="N",
            type=int,
            default=1,
            help="Splits volume into N chunks and each compute node processes one chunk at a time",
        )
        parser.add_argument(
            "--extra-input",
            metavar="FORMAT_STRING",
            help="Format string to generate names of files with additional particle datasets (e.g. halo membership). Use %%(file_nr)d for the file number and %%(snap_nr)04d for the snapshot.",
        )
        parser.add_argument(
            "--centrals-only", action="store_true", help="Only process central halos"
        )
        parser.add_argument(
            "--dmo", action="store_true", help="Run in dark matter only mode"
        )
        parser.add_argument(
            "--max-halos",
            metavar="N",
            nargs=1,
            type=int,
            default=(0,),
            help="(For debugging) only process the first N halos in the catalogue",
        )
        parser.add_argument(
            "--calculations",
            nargs="*",
            help="Which calculations to do (default is to do all)",
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
            "--halo-ids",
            nargs="*",
            type=int,
            help="Only process the specified halo IDs",
        )
        parser.add_argument(
            "--max-ranks-reading",
            type=int,
            default=32,
            help="Number of ranks per node reading snapshot data",
        )
        parser.add_argument(
            "--parameters",
            help="Name of a parameter file containing properties and halo types to process. Default is to compute all properties for FLAMINGO-like halo types.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output-parameters",
            metavar="FILENAME",
            help="Write the actually used parameters to FILENAME.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--simulation",
            metavar="SIMULATION",
            help="Specifies if the simulation is FLAMINGO or COLIBRE-based.",
            type=str,
            required=True
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


def get_group_membership_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """

    if comm.Get_rank() == 0:

        os.environ[
            "COLUMNS"
        ] = "80"  # Can't detect terminal width when running under MPI?

        parser = ThrowingArgumentParser(
            description="Compute particle group membership in SWIFT snapshots."
        )
        parser.add_argument(
            "swift_filename",
            help="Format string to generate snapshot filenames. Use %%(file_nr)d for the file number.",
        )
        parser.add_argument(
            "vr_basename",
            help="Base name of the VELOCIraptor files, excluding trailing .properties[.N] etc.",
        )
        parser.add_argument(
            "output_file",
            help="Format string to generate output filenames. Use %%(file_nr)d for the file number.",
        )
        parser.add_argument(
            "--update-virtual-file",
            type=str,
            help="Name of a single file virtual snapshot to write group membership to",
        )
        parser.add_argument(
            "--output-prefix",
            type=str,
            help="Prefix for names of datasets added to virtual file",
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
