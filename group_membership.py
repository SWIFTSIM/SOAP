#!/bin/env python

import time
import socket
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

import lustre
import combine_args
import read_vr
import read_hbtplus
import read_subfind
import read_rockstar
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def process_particle_type(
    ptype,
    snap_file,
    ids_bound,
    grnr_bound,
    rank_bound,
    ids_unbound,
    fof_ptypes,
    fof_file,
    create_file,
):
    """
    Compute group membership for one particle type
    """
    if comm_rank == 0:
        print("Calculating group membership for type", ptype)
    swift_ids = snap_file.read(("ParticleIDs",), ptype)["ParticleIDs"]

    # Report memory load
    nr_parts_local = len(swift_ids)
    max_nr_parts_local = comm.allreduce(nr_parts_local, op=MPI.MAX)
    min_nr_parts_local = comm.allreduce(nr_parts_local, op=MPI.MIN)
    if comm_rank == 0:
        print(
            f"  Number of snapshot particle IDs per rank min={min_nr_parts_local}, max={max_nr_parts_local}"
        )

    # Look up FoF group membership of the particles
    if ptype in fof_ptypes:
        if comm_rank == 0:
            print("  Matching FOF catalogue to snapshot")
        fof_particle_ids = fof_file.read(("ParticleIDs",), ptype)["ParticleIDs"]
        fof_group_ids = fof_file.read(("FOFGroupIDs",), ptype)["FOFGroupIDs"]
        ptr = psort.parallel_match(swift_ids, fof_particle_ids)
        del fof_particle_ids
        swift_fof_group_ids = psort.fetch_elements(fof_group_ids, ptr)
        del fof_group_ids
        del ptr

    if comm_rank == 0:
        print("  Matching SWIFT particle IDs to bound IDs")
    ptr = psort.parallel_match(swift_ids, ids_bound)

    if comm_rank == 0:
        print("  Assigning bound group membership to SWIFT particles")
    matched = ptr >= 0
    swift_grnr_bound = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)
    swift_grnr_bound[matched] = psort.fetch_elements(grnr_bound, ptr[matched])
    swift_grnr_bound[matched == False] = -1

    if rank_bound is not None:
        if comm_rank == 0:
            print("  Assigning rank by binding energy to SWIFT particles")
        swift_rank_bound = np.ndarray(len(swift_ids), dtype=rank_bound.dtype)
        swift_rank_bound[matched] = psort.fetch_elements(rank_bound, ptr[matched])
        swift_rank_bound[matched == False] = -1
    del ptr
    del matched

    if ids_unbound is not None:
        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to unbound IDs")
        ptr = psort.parallel_match(swift_ids, ids_unbound)

        if comm_rank == 0:
            print("  Assigning unbound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_unbound = np.ndarray(len(swift_ids), dtype=grnr_unbound.dtype)
        swift_grnr_unbound[matched] = psort.fetch_elements(grnr_unbound, ptr[matched])
        swift_grnr_unbound[matched == False] = -1
        swift_grnr_all = np.maximum(swift_grnr_bound, swift_grnr_unbound)
        del ptr
        del matched

    # Determine if we need to create a new output file set
    if create_file:
        mode = "w"
    else:
        mode = "r+"

    # Set up dataset attributes
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
    }
    attrs = {
        "GroupNr_bound": {
            "Description": "Index of halo in which this particle is a bound member, or -1 if none"
        },
        "Rank_bound": {
            "Description": "Ranking by binding energy of the bound particles (first in halo=0), or -1 if not bound"
        },
        "GroupNr_all": {
            "Description": "Index of halo in which this particle is a member (bound or unbound), or -1 if none"
        },
        "FOFGroupIDs": {
            "Description": "Friends-Of-Friends ID of the group in which this particle is a member, of -1 if none"
        },
    }
    attrs["GroupNr_bound"].update(unit_attrs)
    attrs["Rank_bound"].update(unit_attrs)
    attrs["GroupNr_all"].update(unit_attrs)
    attrs["FOFGroupIDs"].update(unit_attrs)

    # Write these particles out with the same layout as the input snapshot
    if comm_rank == 0:
        print("  Writing out group membership of SWIFT particles")
    elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
    output = {"GroupNr_bound": swift_grnr_bound}
    if rank_bound is not None:
        output["Rank_bound"] = swift_rank_bound
    if ids_unbound is not None:
        output["GroupNr_all"] = swift_grnr_all
    if ptype in fof_ptypes:
        output["FOFGroupIDs"] = swift_fof_group_ids
    snap_file.write(
        output,
        elements_per_file,
        filenames=output_file,
        mode=mode,
        group=ptype,
        attrs=attrs,
    )

if __name__ == "__main__":

    # Read parameters from command line and config file
    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(
        comm=comm, description="Compute particle group membership in SWIFT snapshots."
    )
    parser.add_argument(
        "config_file", type=str, help="Name of the yaml configuration file"
    )
    parser.add_argument(
        "--sim-name", type=str, help="Name of the simulation to process"
    )
    parser.add_argument("--snap-nr", type=int, help="Snapshot number to process")
    args = parser.parse_args()
    args = combine_args.combine_arguments(args, args.config_file)

    # Extract parameters we need
    snap_nr = args["Parameters"]["snap_nr"]
    fof_filename = args["Snapshots"].get("fof_filename", "")
    swift_filename = args["Snapshots"]["filename"]
    halo_format = args["HaloFinder"]["type"]
    halo_basename = args["HaloFinder"]["filename"]
    output_file = args["GroupMembership"]["filename"]

    if comm_rank == 0:
        print(f'Input snapshot is {swift_filename}')
        print(f'Halo basename is {halo_basename}')
        print(f'Snapshot number is {snap_nr}')

    # Substitute in the snapshot number where necessary
    pf = PartialFormatter()
    swift_filename = pf.format(swift_filename, snap_nr=snap_nr, file_nr=None)
    fof_filename = pf.format(fof_filename, snap_nr=snap_nr, file_nr=None)
    halo_basename = pf.format(halo_basename, snap_nr=snap_nr, file_nr=None)
    output_file = pf.format(output_file, snap_nr=snap_nr, file_nr=None)

    # Check both swift and output filenames are (not) chunk files
    if "file_nr" in swift_filename:
        assert "file_nr" in output_file, "Membership filenames require {file_nr}"
    else:
        assert (
            "file_nr" not in output_file
        ), "Membership filenames shouldn't have {file_nr}"

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(output_file)
    comm.barrier()

    # Find group number for each particle ID in the halo finder output
    if halo_format == "VR":
        # Read VELOCIraptor output
        (
            total_nr_halos,
            ids_bound,
            grnr_bound,
            rank_bound,
            ids_unbound,
            grnr_unbound,
        ) = read_vr.read_vr_groupnr(halo_basename)
    elif halo_format == "HBTplus":
        # Read HBTplus output
        total_nr_halos, ids_bound, grnr_bound, rank_bound = read_hbtplus.read_hbtplus_groupnr(
            halo_basename
        )
        ids_unbound = None  # HBTplus does not output unbound particles
        grnr_unbound = None
    elif halo_format == "Subfind":
        # Read Gadget-4 subfind output
        total_nr_halos, ids_bound, grnr_bound = read_subfind.read_gadget4_groupnr(
            halo_basename
        )
        ids_unbound = None
        grnr_unbound = None
        rank_bound = None
    elif halo_format == "Rockstar":
        # Read Rockstar output
        total_nr_halos, ids_bound, grnr_bound = read_rockstar.read_rockstar_groupnr(
            halo_basename
        )
        ids_unbound = None
        grnr_unbound = None
        rank_bound = None
    else:
        raise RuntimeError(f"Unrecognised halo finder name: {halo_format}")

    # Report memory load
    nr_parts_local = len(ids_bound)
    max_nr_parts_local = comm.allreduce(nr_parts_local, op=MPI.MAX)
    min_nr_parts_local = comm.allreduce(nr_parts_local, op=MPI.MIN)
    if comm_rank == 0:
        print(
            f"Number of group particle IDs per rank min={min_nr_parts_local}, max={max_nr_parts_local}"
        )

        # Read snapshot attributes
        header = {}
        ptypes = []
        with h5py.File(swift_filename.format(file_nr=0), "r") as infile:
            # Determine SWIFT particle types which exist in the snapshot
            nr_types = infile["Header"].attrs["NumPartTypes"][0]
            for i in range(nr_types):
                if f"PartType{i}" in infile:
                    ptypes.append(f"PartType{i}")

            for attr in [
                "BoxSize",
                "Dimension",
                "NumFilesPerSnapshot",
                "NumPartTypes",
                "NumPart_ThisFile",
                "NumPart_Total",
                "NumPart_Total_HighWord",
                "Redshift",
                "RunName",
                "Scale-factor",
            ]:
                header[attr] = infile["Header"].attrs[attr]

        header["Code"] = "SOAP"
        header["OutputType"] = "Membership"
        snapshot_date = time.strftime("%H:%M:%S %Y-%m-%d GMT", time.gmtime())
        header["SnapshotDate"] = snapshot_date
        header["System"] = socket.gethostname()
        header["halo_basename"] = halo_basename
        header["halo_format"] = halo_format
        header["snapshot_nr"] = snap_nr
        header["swift_filename"] = swift_filename
        header["fof_filename"] = fof_filename
    else:
        ptypes = None
        header = None
    ptypes = comm.bcast(ptypes)
    header = comm.bcast(header)

    # Open the snapshot
    snap_file = phdf5.MultiFile(
        swift_filename, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Read in FOF file information if a separate file has been passed
    fof_ptypes = []
    fof_file = None
    if fof_filename != "":
        # Determine particle types which exist in the FOF file
        if comm_rank == 0:
            with h5py.File(fof_filename.format(file_nr=0), "r") as infile:
                nr_types = infile["Header"].attrs["NumPartTypes"][0]
                for i in range(nr_types):
                    if f"PartType{i}" in infile:
                        fof_ptypes.append(f"PartType{i}")
        fof_ptypes = comm.bcast(fof_ptypes)

        # Open the FOF file
        fof_file = phdf5.MultiFile(
            fof_filename, file_nr_attr=("Header", "NumFilesPerSnapshot")
        )

    # Loop over particle types
    create_file = True
    for ptype in ptypes:
        process_particle_type(
            ptype,
            snap_file,
            ids_bound,
            grnr_bound,
            rank_bound,
            ids_unbound,
            fof_ptypes,
            fof_file,
            create_file,
        )
        create_file = False
    comm.barrier()

    # Write header
    n_files = header["NumFilesPerSnapshot"][0]
    files_on_rank = phdf5.assign_files(n_files, comm_size)
    first_file = np.cumsum(files_on_rank) - files_on_rank
    for file_nr in range(
        first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
    ):
        with h5py.File(output_file.format(file_nr=file_nr), "r+") as infile:
            group = infile.create_group('Header')
            for k, v in header.items():
                group.attrs[k] = v

    comm.barrier()
    if comm_rank == 0:
        print("Done.")
