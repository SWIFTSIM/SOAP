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
which replaces the original once it has been written. The compression steps
are skipped if --no-compression is passed.

If all the membership files already exist they are reused, and steps 1 and 4
are skipped. If only some of them exist then we abort, since they are likely
from a run which didn't complete. Similarly step 5 is skipped if the virtual
snapshot already exists.

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

import h5py
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

    # There is one membership file for each snapshot file
    snapshot = sub_snapnum(all_args["Snapshots"]["filename"])
    if "{file_nr}" in snapshot:
        try:
            with h5py.File(snapshot.format(file_nr=0), "r") as infile:
                n_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
        except OSError as e:
            raise ValueError(f"Unable to read snapshot {snapshot}: {e}")
        filenames["membership_files"] = [
            filenames["membership"].format(file_nr=i) for i in range(n_files)
        ]
    else:
        filenames["membership_files"] = [filenames["membership"]]

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


def membership_files_exist(membership_files, comm):
    """
    Return True if all the membership files exist, and False if none of them
    do. Raises a ValueError if only some of them exist, or if any of them are
    incomplete. The header is written after all the particle data, so a file
    without one is from a run which didn't complete.
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # We open the files directly rather than checking whether they exist first,
    # since the stat call is often the slowest part
    n_missing = 0
    errors = []
    for filename in membership_files[comm_rank::comm_size]:
        try:
            with h5py.File(filename, "r") as infile:
                if "Header" not in infile:
                    errors.append(f"Membership file {filename} has no Header")
        except FileNotFoundError:
            n_missing += 1
        except OSError as e:
            errors.append(f"Unable to read membership file {filename}: {e}")
    n_missing = comm.allreduce(n_missing)
    errors = [error for rank_errors in comm.allgather(errors) for error in rank_errors]

    n_files = len(membership_files)
    if n_missing == n_files:
        return False
    if n_missing > 0:
        errors.insert(0, f"Found {n_files - n_missing} of {n_files} membership files")
    if len(errors) > 0:
        raise ValueError(
            "\n".join(errors) + "\nDelete the membership files to recompute them"
        )
    return True


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
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Don't compress the halo properties catalogue and membership files",
    )
    all_args = soap_args.combine_soap_args(comm, parser)
    compress = not all_args["Parameters"]["no_compression"]

    # Check the filenames before starting, and whether the membership
    # files and virtual snapshot already exist
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
        elif os.path.exists(filenames["virtual_snapshot"]):
            print(
                f"Virtual snapshot {filenames['virtual_snapshot']} already "
                "exists, so it will not be recreated",
                flush=True,
            )
            filenames["virtual_snapshot"] = None
    else:
        filenames = None
    filenames = comm.bcast(filenames)

    try:
        reuse_membership = membership_files_exist(filenames["membership_files"], comm)
    except ValueError as e:
        if comm_rank == 0:
            print(e, flush=True)
        comm.Abort(1)
    if reuse_membership and comm_rank == 0:
        print(
            f"Membership files {filenames['membership']} already exist, "
            "so they will be reused",
            flush=True,
        )

    # Extract the arguments for compute_halo_properties. This also checks the
    # input/output paths, so we do it before starting
    args = soap_args.process_soap_args(comm, all_args)

    if not reuse_membership:
        print_stage("Computing group membership")
        compute_membership(all_args)
        comm.barrier()

    print_stage("Computing halo properties")
    compute_halo_properties(args)
    comm.barrier()

    if compress:
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

    if compress and not reuse_membership:
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
