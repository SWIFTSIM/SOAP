#!/bin/env python

import os

import numpy as np
import h5py
import unyt

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5


def locate_files(basename):
    """
    Generate format strings for Gadget-4 snapshot and fof_subhalo_tab files,
    given the path to a snapshot file without the trailing .N.hdf5.

    Assumes we have a filenames along the lines of

    snapdir_XXX/snap_basename_XXX.0.hdf5
    ...
    groups_XXX/fof_subhalo_tab_XXX.0.hdf5
    ...
    """
    snap_format_string = basename + ".%(file_nr)d.hdf5"

    # Check that the first snapshot file exists
    snap_file = snap_format_string % {"file_nr": 0}
    if not (os.path.exists(snap_file)):
        raise IOError("Snapshot file does not exist: " + snap_file)

    # Find the base directory
    topdir = os.path.dirname(os.path.dirname(snap_file))

    # Get the snapshot number from the filename
    snap_nr = int(basename[-3:])

    # Make format string for halo filenames
    group_format_string = (
        f"{topdir}/groups_{snap_nr:03d}/fof_subhalo_tab_{snap_nr:03d}"
        + ".%(file_nr)d.hdf5"
    )

    # Check group file exists
    group_file = group_format_string % {"file_nr": 0}
    if not (os.path.exists(group_file)):
        raise IOError("Group file does not exist: " + group_file)

    return snap_format_string, group_format_string


def read_gadget4_groupnr(basename):
    """
    Read particle IDs and group numbers from Gadget-4 output.

    basename should be the name of a group sorted snapshot file without the
    trailing .N.hdf5.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string = locate_files(basename)
    else:
        snap_format_string = None
        group_format_string = None
    snap_format_string, group_format_string = comm.bcast(
        (snap_format_string, group_format_string)
    )

    # Check what particle types we have
    if comm_rank == 0:
        snap_file = snap_format_string % {"file_nr": 0}
        with h5py.File(snap_file, "r") as infile:
            numpart_total = infile["Header"].attrs["NumPart_Total"]
        type_nrs = np.arange(len(numpart_total), dtype=int)
        type_nrs = type_nrs[numpart_total > 0]
    else:
        type_nrs = None
    type_nrs = comm.bcast(type_nrs)

    # Read in the sorted particle IDs from the snapshot
    particle_ids = {}
    snap = phdf5.MultiFile(
        snap_format_string, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )
    for type_nr in type_nrs:
        particle_ids[type_nr] = snap.read(f"PartType{type_nr}/ParticleIDs")

    # Read in the group lengths and offsets
    subtab = phdf5.MultiFile(group_format_string, file_nr_attr=("Header", "NumFiles"))
    suboffset_type, sublen_type = subtab.read(
        ("Subhalo/SubhaloOffsetType", "Subhalo/SubhaloLenType"), unpack=True
    )

    # Compute group index for each particle ID
    particle_grnr = {}
    for type_nr in type_nrs:
        particle_grnr[type_nr] = virgo.mpi.util.group_index_from_length_and_offset(
            np.ascontiguousarray(sublen_type[:, type_nr]),
            np.ascontiguousarray(suboffset_type[:, type_nr]),
            len(particle_ids[type_nr]),
            return_rank=False,
            comm=comm,
        )
    # Concatenate and return arrays
    all_grnr = []
    all_ids = []
    for type_nr in type_nrs:
        all_grnr.append(particle_grnr[type_nr])
        all_ids.append(particle_ids[type_nr])

    local_nr_halos = suboffset_type.shape[0]
    total_nr_halos = comm.allreduce(local_nr_halos)

    return total_nr_halos, np.concatenate(all_ids), np.concatenate(all_grnr)


def read_gadget4_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the Gadget-4 halo catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - HBTPlus SubSnap filename without the .N suffix
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

    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
    swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
    swift_msun = unyt.Unit("swift_msun", registry=registry)

    # Get expansion factor as a float
    a = a_unit.base_value

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string = locate_files(basename)
    else:
        snap_format_string = None
        group_format_string = None
    snap_format_string, group_format_string = comm.bcast(
        (snap_format_string, group_format_string)
    )

    # Get Gadget-4 unit information (only need lengths here)
    with h5py.File(group_format_string % {"file_nr": 0}, "r") as subtab:
        length_cgs = float(subtab["Parameters"].attrs["UnitLength_in_cm"])
        hubble = float(subtab["Parameters"].attrs["Hubble"])
        hubbleparam = float(subtab["Parameters"].attrs["HubbleParam"])

    # Gadget-4 can be set up to use no-h units. Wont try to support that here.
    # Hubble=100 means Gadget is using 1/h units with h given by HubbleParam.
    if hubble != 100.0:
        raise ValueError("Runs with Hubble != 100.0 not supported")

    # Read halo properties we need
    datasets = (
        "Subhalo/SubhaloPos",
        "Subhalo/SubhaloHalfmassRad",
        "Subhalo/SubhaloRankInGr",
        "Subhalo/SubhaloLen",
        "Subhalo/SubhaloGroupNr",
    )
    subtab = phdf5.MultiFile(group_format_string, file_nr_attr=("Header", "NumFiles"))
    data = subtab.read(datasets)

    # Assign indexes to the subhalos
    nr_local_halos = len(data["Subhalo/SubhaloLen"])
    local_offset = comm.scan(nr_local_halos) - nr_local_halos
    index = np.arange(nr_local_halos, dtype=int) + local_offset
    index = unyt.unyt_array(
        index, dtype=int, units=unyt.dimensionless, registry=registry
    )

    # Get length unit conversion (ignoring any a factors and assuming Gadget uses 1/h units)
    gadget_length_unit = length_cgs * unyt.cm / hubbleparam
    length_conversion = (gadget_length_unit / swift_pmpc).to(unyt.dimensionless)

    # Get position in comoving Mpc, assuming input position from Gadget is comoving
    cofp = data["Subhalo/SubhaloPos"] * length_conversion * swift_cmpc

    # Store central halo flag
    is_central = np.where(data["Subhalo/SubhaloRankInGr"] == 0, 1, 0)
    is_central = unyt.unyt_array(
        is_central, dtype=int, units=unyt.dimensionless, registry=registry
    )

    # Store number of bound particles in each halo
    nr_bound_part = unyt.unyt_array(
        data["Subhalo/SubhaloLen"],
        dtype=int,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Store FOF group ID of this halo
    group_nr = unyt.unyt_array(
        data["Subhalo/SubhaloGroupNr"],
        dtype=int,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Store initial search radius (TODO: check this is still in physical units, unlike the position)
    search_radius = (
        data["Subhalo/SubhaloHalfmassRad"] * length_conversion * swift_pmpc
    )  # different units from cofm, not a typo!

    local_halo = {
        "cofp": cofp,
        "index": index,
        "search_radius": search_radius,
        "is_central": is_central,
        "nr_bound_part": nr_bound_part,
        "GroupNr": group_nr,
    }

    for name in local_halo:
        local_halo[name] = unyt.unyt_array(local_halo[name], registry=registry)

    return local_halo


def test_read_gadget4_groupnr(basename):
    """
    Read in Gadget-4 group numbers and compute the number of particles
    in each group. This is then compared with the input catalogue as a
    sanity check on the group membershp files.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    n_halo, ids, grnr = read_gadget4_groupnr(basename)
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

    # Locate the snapshot and fof_subhalo_tab files
    if comm_rank == 0:
        snap_format_string, group_format_string = locate_files(basename)
    else:
        snap_format_string = None
        group_format_string = None
    snap_format_string, group_format_string = comm.bcast(
        (snap_format_string, group_format_string)
    )

    # Read group sizes from the group catalogue
    subtab = phdf5.MultiFile(group_format_string, file_nr_attr=("Header", "NumFiles"))
    nbound_from_subtab = subtab.read("Subhalo/SubhaloLen")

    # Find number of groups in the subfind output
    nr_groups_from_subtab = comm.allreduce(len(nbound_from_subtab))
    if comm_rank == 0:
        print(f"Number of groups from fof_subhalo_tab = {nr_groups_from_subtab}")
        if nr_groups_from_subtab != nr_groups_from_grnr:
            print("Number of groups does not agree!")
            comm.Abort(1)

    # Ensure nbound arrays are partitioned the same way
    nr_per_rank = comm.allgather(len(nbound_from_subtab))
    nbound_from_grnr = psort.repartition(
        nbound_from_grnr, ndesired=nr_per_rank, comm=comm
    )

    # Compare
    nr_different = comm.allreduce(np.sum(nbound_from_grnr != nbound_from_subtab))
    if comm_rank == 0:
        print(f"Number of group sizes which differ = {nr_different} (should be 0!)")


if __name__ == "__main__":

    import sys

    basename = sys.argv[1]
    test_read_gadget4_groupnr(basename)
