#!/bin/env python

import os

import numpy as np
import pandas as pd
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
from virgo.formats.rockstar import HalosFile


def locate_files(basename):
    """
    Generate format strings for rockstar binary and merger tree files,
    given the path to a binary file without the trailing .N.bin.

    Assumes we have a filenames along the lines of

    {rockstar_dir}/snapshot_XXXX/halos_XXXX.0.bin
    ...
    {rockstar_dir}/merger_tree/snapshot_XXXX/parents_XXXX.list
    ...
    """
    snap_format_string = basename + ".%(file_nr)d.bin"

    # Check that the first snapshot file exists
    snap_file = snap_format_string % {"file_nr": 0}
    if not (os.path.exists(snap_file)):
        raise IOError("Snapshot file does not exist: " + snap_file)

    # Get the number of binary files
    n_bin_file = 1
    while os.path.exists(snap_format_string % {"file_nr": n_bin_file}):
        n_bin_file += 1

    # Find the base directory
    top_dir = os.path.dirname(os.path.dirname(snap_file))

    # Get the snapshot number from the filename
    snap_nr = int(basename[-4:])

    group_dir = f"{top_dir}/merger_tree/snapshot_{snap_nr:04d}/"
    group_format_string = f"{group_dir}parents_{snap_nr:04d}.%(file_nr)04d.list"

    # Check group file exists
    if not (os.path.exists(group_format_string % {"file_nr": 0})):
        raise IOError("Group file does not exist: " + group_format_string)

    # Get the number of group files
    n_group_file = 1
    while os.path.exists(group_format_string % {"file_nr": n_group_file}):
        n_group_file += 1

    return snap_format_string, group_format_string, n_bin_file, n_group_file


def read_rockstar_groupnr(basename):
    """
    Read particle IDs and group numbers from rockstar binary output.

    basename should be the name of the rockstar binary files without the
    trailing .N.bin
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, _, n_bin_file, _ = locate_files(basename)
    else:
        snap_format_string = None
        n_bin_file = None
    snap_format_string, n_bin_file = comm.bcast((snap_format_string, n_bin_file))

    # Assign files to ranks
    files_on_rank = phdf5.assign_files(n_bin_file, comm_size)
    first_file = np.cumsum(files_on_rank) - files_on_rank

    # Determine total number of halos and particles processed by this rank
    local_nr_halos = 0
    n_part = 0
    for file_nr in range(
        first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
    ):
        filename = snap_format_string % {"file_nr": file_nr}
        halo_file = HalosFile(filename, hydro=True)
        halo_file.sanity_check()
        local_nr_halos += halo_file["Header"].attrs["num_halos"]
        n_part += halo_file["IDs"].shape[0]

    local_ids = np.zeros(n_part, dtype="int64")
    local_grnr = np.zeros(n_part, dtype="int64")
    offset = 0
    for file_nr in range(
        first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
    ):
        filename = snap_format_string % {"file_nr": file_nr}
        halo_file = HalosFile(filename, hydro=True)
        n_part_file = halo_file["IDs"].shape[0]

        # Store particle ids associated with all halos in this file
        local_ids[offset : offset + n_part_file] = halo_file["IDs"]

        # Store halo id of each halo in this file
        file_offset = 0
        for halo_id, halo_offset in zip(
            halo_file["Halo"]["id"], halo_file["Halo"]["num_p"]
        ):
            local_grnr[offset + file_offset : offset + file_offset + halo_offset] = (
                halo_id
            )
            file_offset += halo_offset

        offset += n_part_file

    total_nr_halos = comm.allreduce(local_nr_halos)

    return total_nr_halos, local_ids, local_grnr


def read_rockstar_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the Rockstar halo catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - Rockstar binary filename without the .N suffix
    a_unit   - unyt a factor
    registry - unyt unit registry
    boxsize  - box size as a unyt quantity

    Returns a dict of unyt arrays with the halo properies.
    Arrays which must always be returned:

    index - index of each halo in the input catalogue
    cofp  - (N,3) array with centre to use for SO calculations
    search_radius - initial search radius which includes all member particles
    is_central - integer 1 for centrals, 0 for satellites
    nr_bound_part - number of bound particles in each halo

    Any other arrays will be passed through to the output ONLY IF they are
    documented in property_table.py.

    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string, _, n_group_file = locate_files(
            basename
        )
    else:
        snap_format_string = None
        group_format_string = None
        n_group_file = None
    snap_format_string, group_format_string, n_group_file = comm.bcast(
        (snap_format_string, group_format_string, n_group_file)
    )

    # Assign files to ranks
    files_on_rank = phdf5.assign_files(n_group_file, comm_size)
    first_file = np.cumsum(files_on_rank) - files_on_rank

    # Extract properties from group catalogue files
    local_halo = {
        "index": [],
        "cofp": [],
        "is_central": [],
        "PID": [],
        "DescID": [],
        "nr_bound_part": [],
        "search_radius": [],
    }
    for file_nr in range(
        first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
    ):
        filename = group_format_string % {"file_nr": file_nr}

        with open(filename, "r") as file:
            cols = file.readline()[1:]
        cols = cols.split()
        data = pd.read_csv(filename, names=cols, comment="#", delim_whitespace=True)

        local_halo["index"].append(np.array(data["ID"]))

        # Note this is not the most bound particle
        x = np.array(data["X"]).reshape(-1, 1)
        y = np.array(data["Y"]).reshape(-1, 1)
        z = np.array(data["Z"]).reshape(-1, 1)
        local_halo["cofp"].append(np.concatenate([x, y, z], axis=1))

        parent_id = np.array(data["PID"])
        local_halo["is_central"].append(parent_id == -1)
        local_halo["PID"].append(parent_id)
        local_halo["DescID"].append(np.array(data["DescID"]))

        local_halo["nr_bound_part"].append(np.array(data["Np"]))

        local_halo["search_radius"].append(np.array(data["Rvir"]))

    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
    swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
    a = a_unit.base_value

    # Get hubble param
    snap_file = snap_format_string % {"file_nr": 0}
    halo_file = HalosFile(snap_file, hydro=True)
    hubble_param = halo_file["Header"].attrs["h0"]

    # Unit conversion and creation of unyt arrays
    for name in local_halo:
        if local_halo[name]:
            local_halo[name] = np.concatenate(local_halo[name], axis=0)
        # Handling cases where rank didn't process any files
        elif name == "cofp":
            local_halo[name] = np.array([]).reshape(-1, 3)
        else:
            local_halo[name] = np.array([])

        if name == "cofp":
            # Rockstar units are Mpc/h (comoving)
            local_halo[name] = unyt.unyt_array(
                (local_halo[name] / hubble_param) * swift_cmpc, registry=registry
            )
        elif name == "search_radius":
            # Rockstar units are kpc/h (comoving)
            local_halo[name] *= a / (hubble_param * 1000)
            local_halo[name] = unyt.unyt_array(
                local_halo[name] * swift_pmpc, registry=registry
            )
        else:
            local_halo[name] = unyt.unyt_array(
                local_halo[name], dtype=int, units=unyt.dimensionless, registry=registry
            )

    return local_halo


def test_read_rockstar_groupnr(basename):
    """
    Read in rockstar group numbers and compute the number of particles
    in each group. This is then compared with the input catalogue as a
    sanity check on the group membershp files.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    _, ids, grnr = read_rockstar_groupnr(basename)
    del ids  # Don't need the particle IDs

    # Find maximum group number
    max_grnr = comm.allreduce(np.amax(grnr), op=MPI.MAX)
    nr_groups_from_grnr = max_grnr + 1
    if comm_rank == 0:
        print(f"Number of groups from membership files = {nr_groups_from_grnr}")

    # Discard particles in no group
    keep = grnr >= 0
    grnr = grnr[keep]

    # Compute group sizes
    import virgo.mpi.parallel_sort as psort

    nbound_from_grnr = psort.parallel_bincount(grnr, comm=comm)

    # Rockstar outputs are csv, so can't use phdf5 to read
    if comm_rank == 0:
        snap_format_string, group_format_string, _, n_group_file = locate_files(
            basename
        )
        for file_nr in range(n_group_file):

            filename = group_format_string % {"file_nr": file_nr}

            with open(filename, "r") as file:
                cols = file.readline()[1:]
            cols = cols.split()
            data = pd.read_csv(filename, names=cols, comment="#", delim_whitespace=True)

            # Extract halo ids and number of particles from group files
            halo_ids = np.array(data["ID"], dtype=int)
            num_p = np.array(data["Np"], dtype=int)

            # Compare
            if not np.all(nbound_from_grnr[halo_ids] == num_p):
                different = nbound_from_grnr[halo_ids] != num_p
                print("The following halo ids differ:", halo_ids[different])


if __name__ == "__main__":

    import sys

    basename = sys.argv[1]
    test_read_rockstar_groupnr(basename)
