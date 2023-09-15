#!/bin/env python

import sys
import os.path
import re
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
from virgo.util.partial_formatter import PartialFormatter

import lustre
import combine_args
import read_vr
import read_hbtplus
import read_gadget4

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


if __name__ == "__main__":

    # Read parameters from command line and config file
    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Compute particle group membership in SWIFT snapshots.")
    parser.add_argument("config_file", type=str, help="Name of the yaml configuration file")
    parser.add_argument("--sim_name", type=str, help="Name of the simulation to process")
    parser.add_argument("--snap_nr", type=int, help="Snapshot number to process")    
    args = parser.parse_args()
    args = combine_args.combine_arguments(args, args.config_file)

    # Extract parameters we need
    snap_nr = args["Parameters"]["snap_nr"]
    swift_filename = args["Snapshots"]["filename"]
    halo_format = args["HaloFinder"]["type"]
    halo_basename = args["HaloFinder"]["filename"]
    halo_basename = args["HaloFinder"]["filename"]    
    output_file = args["GroupMembership"]["filename"]
    halo_sizes_file = args["GroupMembership"]["halo_sizes_file"]
    
    # Substitute in the snapshot number where necessary
    pf = PartialFormatter()
    swift_filename = pf.format(swift_filename, snap_nr=snap_nr, file_nr=None)
    halo_basename = pf.format(halo_basename, snap_nr=snap_nr, file_nr=None)
    output_file = pf.format(output_file, snap_nr=snap_nr, file_nr=None)
    
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
    elif halo_format == "Gadget4":
        # Read Gadget-4 subfind output
        total_nr_halos, ids_bound, grnr_bound = read_gadget4.read_gadget4_groupnr(
            halo_basename
        )
        ids_unbound = None
        grnr_unbound = None
        rank_bound = None
    else:
        raise RuntimeError(f"Unrecognised halo finder name: {halo_format}")

    # Determine SWIFT particle types which exist in the snapshot
    ptypes = []
    with h5py.File(swift_filename.format(file_nr=0), "r") as infile:
        nr_types = infile["Header"].attrs["NumPartTypes"][0]
        numpart_total = infile["Header"].attrs["NumPart_Total"].astype(np.int64) + (
            infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32
        )
        nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
        for i in range(nr_types):
            if numpart_total[i] > 0:
                ptypes.append("PartType%d" % i)

    # Open the snapshot
    snap_file = phdf5.MultiFile(
        swift_filename, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Loop over particle types
    create_file = True
    for ptype in ptypes:

        if comm_rank == 0:
            print("Calculating group membership for type ", ptype)
        swift_ids = snap_file.read(("ParticleIDs",), ptype)["ParticleIDs"]

        # Allocate array to store SWIFT particle group membership
        swift_grnr_bound = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)
        if rank_bound is not None:
            swift_rank_bound = np.ndarray(len(swift_ids), dtype=rank_bound.dtype)
        if ids_unbound is not None:
            swift_grnr_unbound = np.ndarray(len(swift_ids), dtype=grnr_unbound.dtype)

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to bound IDs")
        ptr = psort.parallel_match(swift_ids, ids_bound)

        if comm_rank == 0:
            print("  Assigning bound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_bound[matched] = psort.fetch_elements(grnr_bound, ptr[matched])
        swift_grnr_bound[matched == False] = -1

        if rank_bound is not None:
            if comm_rank == 0:
                print("  Assigning rank by binding energy to SWIFT particles")
            swift_rank_bound[matched] = psort.fetch_elements(rank_bound, ptr[matched])
            swift_rank_bound[matched == False] = -1

        if ids_unbound is not None:
            if comm_rank == 0:
                print("  Matching SWIFT particle IDs to unbound IDs")
            ptr = psort.parallel_match(swift_ids, ids_unbound)

            if comm_rank == 0:
                print("  Assigning unbound group membership to SWIFT particles")
            matched = ptr >= 0
            swift_grnr_unbound[matched] = psort.fetch_elements(
                grnr_unbound, ptr[matched]
            )
            swift_grnr_unbound[matched == False] = -1
            swift_grnr_all = np.maximum(swift_grnr_bound, swift_grnr_unbound)

        # Compute the number of bound particles of this type in each halo
        if comm_rank == 0:
            print("  Computing number of bound particles in each halo")
        in_halo = swift_grnr_bound >= 0
        npart_bound_type = psort.parallel_bincount(
            swift_grnr_bound[in_halo], minlength=total_nr_halos, comm=comm
        )

        # Compute the number of bound+unbound particles of this type in each halo
        if ids_unbound is not None:
            if comm_rank == 0:
                print("  Computing number of bound+unbound particles in each halo")
            in_halo = swift_grnr_all >= 0
            npart_all_type = psort.parallel_bincount(
                swift_grnr_all[in_halo], minlength=total_nr_halos, comm=comm
            )

        # Determine if we need to create a new output file set
        if create_file:
            mode = "w"
            create_file = False
        else:
            mode = "r+"

        # Write the number of particles per halo to an output file
        with h5py.File(halo_sizes_file, mode, driver="mpio", comm=comm) as hsf:
            grp = hsf.create_group(ptype)
            phdf5.collective_write(
                grp, "nr_particles_bound", npart_bound_type, comm=comm
            )
            if ids_unbound is not None:
                phdf5.collective_write(
                    grp, "nr_particles_all", npart_all_type, comm=comm
                )

        # Set up dataset attributes
        unit_attrs = {
            "Conversion factor to CGS (not including cosmological corrections)": [1.0],
            "Conversion factor to CGS (including cosmological corrections)": [1.0],
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
        }
        attrs["GroupNr_bound"].update(unit_attrs)
        attrs["Rank_bound"].update(unit_attrs)
        attrs["GroupNr_all"].update(unit_attrs)

        # Write these particles out with the same layout as the input snapshot
        if comm_rank == 0:
            print("  Writing out group membership of SWIFT particles")
        elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
        output = {"GroupNr_bound": swift_grnr_bound}
        if rank_bound is not None:
            output["Rank_bound"] = swift_rank_bound
        if ids_unbound is not None:
            output["GroupNr_all"] = swift_grnr_all
        snap_file.write(
            output,
            elements_per_file,
            filenames=output_file,
            mode=mode,
            group=ptype,
            attrs=attrs,
        )

    comm.barrier()
    if comm_rank == 0:
        print("Done.")
