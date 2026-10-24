#!/bin/env python

"""
run_soap.py

Run all the steps of SOAP for a single snapshot within one MPI job:

1. Compute group membership
2. Compute halo properties
3. Compress the halo properties catalogue
4. Compress the group membership files
5. Create a virtual snapshot linking the snapshot and membership files
   (only if GroupMembership/virtual_snapshot_filename is set)

The membership files and halo properties catalogue are written to the
locations given in the parameter file, as for the individual scripts. Each
file is then compressed to a temporary file with the suffix ".compressed",
which replaces the original once it has been written.

This is intended for small runs. For large runs it can be more efficient to
run each step separately, since they require different amounts of resources.
"""

# Initialize mpi4py with thread support, as for compute_halo_properties.py
import mpi4py

mpi4py.rc.threads = True
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

import os
import time

from virgo.util.partial_formatter import PartialFormatter

from SOAP.core import soap_args
from SOAP.group_membership import compute_membership
from SOAP.compute_halo_properties import compute_halo_properties
from SOAP.compression.compress_soap_catalogue import compress_catalogue
from SOAP.compression.compress_membership import compress_membership
from SOAP.compression.make_virtual_snapshot import make_virtual_snapshot


def get_filenames(all_args):
    """
    Return a dict with the input/output filenames for each step, with the
    snapshot number substituted in. Raises a ValueError if there is a problem
    with the filenames.
    """

    snap_nr = all_args["Parameters"]["snap_nr"]
    pf = PartialFormatter()

    def sub_snapnum(filename):
        return pf.format(filename, snap_nr=snap_nr, file_nr=None)

    filenames = {}
    filenames["membership"] = sub_snapnum(all_args["GroupMembership"]["filename"])
    filenames["halo_properties"] = sub_snapnum(all_args["HaloProperties"]["filename"])
    filenames["compression_scratch_dir"] = all_args["HaloProperties"]["chunk_dir"]

    # The virtual snapshot is optional. For distributed snapshots it is
    # created from the SWIFT virtual file, which has no file number
    filenames["virtual_snapshot"] = None
    if "virtual_snapshot_filename" in all_args["GroupMembership"]:
        filenames["virtual_snapshot"] = sub_snapnum(
            all_args["GroupMembership"]["virtual_snapshot_filename"]
        )
        snapshot = sub_snapnum(all_args["Snapshots"]["filename"])
        filenames["snapshot"] = snapshot.replace(".{file_nr}", "")
        if not os.path.exists(filenames["snapshot"]):
            raise ValueError(
                f"Could not find snapshot {filenames['snapshot']} to create the "
                "virtual snapshot from"
            )

    return filenames


def replace_with_compressed(filenames):
    """
    Replace each file with its compressed version, which was written to the
    same filename with the suffix ".compressed"
    """
    for filename in filenames:
        os.replace(filename + ".compressed", filename)


def print_stage(message):
    if comm_rank == 0:
        print(f"\n######## {message} ########\n", flush=True)


def run_soap():

    t0 = time.time()

    # Read the command line arguments and parameter file. These are the same
    # as for compute_halo_properties.py
    parser = soap_args.get_soap_parser(
        comm,
        description="Run all the steps of SOAP (group membership, halo properties, "
        "and compression) for a single snapshot",
    )
    all_args = soap_args.combine_soap_args(comm, parser)

    # Check the filenames before starting
    if comm_rank == 0:
        try:
            filenames = get_filenames(all_args)
        except ValueError as e:
            print(e, flush=True)
            comm.Abort(1)
        if filenames["virtual_snapshot"] is None:
            print(
                "Warning: GroupMembership/virtual_snapshot_filename is not set, "
                "so no virtual snapshot will be created",
                flush=True,
            )
    else:
        filenames = None
    filenames = comm.bcast(filenames)

    # Extract the arguments for compute_halo_properties. This also checks the
    # input/output paths, so we do it before starting
    args = soap_args.process_soap_args(comm, all_args)

    print_stage("Computing group membership")
    compute_membership(all_args)
    comm.barrier()

    print_stage("Computing halo properties")
    compute_halo_properties(args)
    comm.barrier()

    print_stage("Compressing halo properties")
    compress_catalogue(
        filenames["halo_properties"],
        filenames["halo_properties"] + ".compressed",
        filenames["compression_scratch_dir"],
        comm,
    )
    if comm_rank == 0:
        try:
            replace_with_compressed([filenames["halo_properties"]])
        except OSError as e:
            print(f"Error replacing halo properties catalogue: {e}", flush=True)
            comm.Abort(1)
    comm.barrier()

    print_stage("Compressing group membership")
    membership_files = compress_membership(
        filenames["membership"], filenames["membership"] + ".compressed", comm
    )
    if comm_rank == 0:
        try:
            replace_with_compressed(membership_files)
        except OSError as e:
            print(f"Error replacing membership files: {e}", flush=True)
            comm.Abort(1)
    comm.barrier()

    if filenames["virtual_snapshot"] is not None:
        print_stage("Creating virtual snapshot")
        if comm_rank == 0:
            try:
                os.makedirs(
                    os.path.dirname(os.path.abspath(filenames["virtual_snapshot"])),
                    exist_ok=True,
                )
                make_virtual_snapshot(
                    filenames["snapshot"],
                    [filenames["membership"]],
                    filenames["virtual_snapshot"],
                )
            except Exception as e:
                print(f"Error creating virtual snapshot: {e}", flush=True)
                comm.Abort(1)
            print(f"Created {filenames['virtual_snapshot']}")
        comm.barrier()

    if comm_rank == 0:
        print(f"\nSOAP finished in {time.time() - t0:.1f} seconds")
        print(f"Halo properties: {filenames['halo_properties']}")
        print(f"Membership: {filenames['membership']}")
        if filenames["virtual_snapshot"] is not None:
            print(f"Virtual snapshot: {filenames['virtual_snapshot']}")


if __name__ == "__main__":

    run_soap()
