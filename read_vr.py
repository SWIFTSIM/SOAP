#!/bin/env python

import os

import numpy as np
import h5py
import unyt

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as ga
import virgo.mpi.parallel_sort as ps


from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from read_vr_group_sizes import read_vr_group_sizes


def read_vr_datasets(vr_basename, file_type, datasets, return_file_nr=None):
    """
    Parallel read of datasets from VR split over any number of files
    """
    # Make a format string for the filename
    filename_format = vr_basename + "." + file_type + ".%(file_nr)d"

    # Open the file
    vr_file = phdf5.MultiFile(filename_format, file_nr_dataset="Num_of_files")

    # Read the data
    return vr_file.read(datasets, return_file_nr=return_file_nr)


def compute_lengths(offsets, total_nr_ids):
    """
    Compute group lengths given the offsets and the total number
    of particle IDs. 
    """

    # Only include ranks with >0 groups
    if len(offsets) > 0:
        color = 1
    else:
        color = 0
    local_comm = comm.Split(color)

    if color == 1:
        # Find first offset on each rank
        first_offset = local_comm.allgather(offsets[0])
        # Allocate lengths array
        lengths = np.ndarray(len(offsets), dtype=int)
        # Compute lengths of all but last group
        lengths[:-1] = offsets[1:] - offsets[:-1]
        # Compute length of last group
        if local_comm.Get_rank() == local_comm.Get_size() - 1:
            lengths[-1] = total_nr_ids - offsets[-1]
        else:
            lengths[-1] = first_offset[local_comm.Get_rank() + 1] - offsets[-1]
    else:
        # Have zero groups on this rank
        lengths = np.ndarray(0, dtype=int)

    local_comm.Free()
    return lengths


def read_vr_lengths_and_offsets(vr_basename):
    """
    Read the bound and unbound halo lengths, offsets and IDs from VR.
    Offsets are modified to be relative to the start of the first file.

    All output arrays are distributed over ranks in MPI_COMM_WORLD.
    """

    # Find number of VR output files
    fname = vr_basename + ".catalog_particles.0"
    if comm_rank == 0:
        infile = h5py.File(fname, "r")
        nr_files = infile["Num_of_files"][0]
        infile.close()
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Assign files to ranks
    files_on_rank = phdf5.assign_files(nr_files, comm_size)
    first_file = np.cumsum(files_on_rank) - files_on_rank

    # Loop over files on this rank and read numbers of bound, unbound IDs
    nr_ids_bound = []
    nr_ids_unbound = []
    for file_nr in range(
        first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
    ):
        fname = vr_basename + (".catalog_particles.%d" % file_nr)
        with h5py.File(fname, "r") as infile:
            nr_ids_bound.append(infile["Particle_IDs"].shape[0])
        fname = vr_basename + (".catalog_particles.unbound.%d" % file_nr)
        with h5py.File(fname, "r") as infile:
            nr_ids_unbound.append(infile["Particle_IDs"].shape[0])

    # Combine results from all ranks
    nr_ids_bound = np.concatenate(
        comm.allgather(np.asarray(nr_ids_bound, dtype=int)), dtype=int
    )
    nr_ids_unbound = np.concatenate(
        comm.allgather(np.asarray(nr_ids_unbound, dtype=int)), dtype=int
    )

    # Find offsets which need to be added to particle offsets read from each file
    cumulative_nr_bound_ids = np.cumsum(nr_ids_bound) - nr_ids_bound
    cumulative_nr_unbound_ids = np.cumsum(nr_ids_unbound) - nr_ids_unbound

    # Read in the VR particle offsets
    names = ("Offset", "Offset_unbound")
    data, file_nr = read_vr_datasets(
        vr_basename, "catalog_groups", names, return_file_nr=names
    )

    # Make all offsets relative to the start of file zero
    offset_bound = data["Offset"] + cumulative_nr_bound_ids[file_nr["Offset"]]
    offset_unbound = (
        data["Offset_unbound"] + cumulative_nr_unbound_ids[file_nr["Offset_unbound"]]
    )

    # Report number of groups read in
    nr_bound_offsets = comm.reduce(len(offset_bound))
    nr_unbound_offsets = comm.reduce(len(offset_unbound))
    if comm_rank == 0:
        print(
            "Read in %d bound offsets and %d unbound offsets"
            % (nr_bound_offsets, nr_unbound_offsets)
        )

    # Read in the particle IDs
    data = read_vr_datasets(vr_basename, "catalog_particles", ("Particle_IDs",))
    ids_bound = data["Particle_IDs"]
    data = read_vr_datasets(vr_basename, "catalog_particles.unbound", ("Particle_IDs",))
    ids_unbound = data["Particle_IDs"]
    nr_bound_ids = comm.reduce(len(ids_bound))
    nr_unbound_ids = comm.reduce(len(ids_unbound))
    if comm_rank == 0:
        print(
            "Read in %d bound ids and %d unbound ids" % (nr_bound_ids, nr_unbound_ids)
        )

    # Find the bound and unbound length of each halo
    total_nr_bound = comm.allreduce(len(ids_bound))
    length_bound = compute_lengths(offset_bound, total_nr_bound)
    total_nr_unbound = comm.allreduce(len(ids_unbound))
    length_unbound = compute_lengths(offset_unbound, total_nr_unbound)
    if comm_rank == 0:
        print("Calculated halo lengths ")

    return (
        length_bound,
        offset_bound,
        ids_bound,
        length_unbound,
        offset_unbound,
        ids_unbound,
    )


def vr_group_membership_from_ids(
    length, offset, ids, max_nr_particles=None, return_rank=False
):
    """
    Return VR group membership for the supplied IDs. Only the first
    max_nr_particles in each group are assigned group numbers if
    max_nr_particles is not None.

    Returns -1 for particles in no group.
    """

    # Find group lengths to use
    if max_nr_particles is None:
        lengths_to_use = length
    else:
        lengths_to_use = np.clip(length, None, max_nr_particles)

    # Associate a group index to each particle ID
    return virgo.mpi.util.group_index_from_length_and_offset(
        lengths_to_use, offset, len(ids), return_rank=return_rank
    )


def read_vr_groupnr(basename):
    """
    Read VR output and return group number for each particle ID 
    """
    (
        length_bound,
        offset_bound,
        ids_bound,
        length_unbound,
        offset_unbound,
        ids_unbound,
    ) = read_vr_lengths_and_offsets(basename)
    grnr_bound, rank_bound = vr_group_membership_from_ids(
        length_bound, offset_bound, ids_bound, return_rank=True
    )
    grnr_unbound = vr_group_membership_from_ids(
        length_unbound, offset_unbound, ids_unbound
    )

    local_nr_halos = len(offset_bound)
    total_nr_halos = comm.allreduce(local_nr_halos)

    return total_nr_halos, ids_bound, grnr_bound, rank_bound, ids_unbound, grnr_unbound


def read_vr_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the VR halo catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - VR filename without the .properties.* suffix
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
    nr_unbound_part - number of unbound particles in each halo
    
    Any other arrays will be passed through to the output ONLY IF they are
    documented in property_table.py.
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
    swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
    swift_msun = unyt.Unit("swift_msun", registry=registry)

    # Get expansion factor as a float
    a = a_unit.base_value

    # Check for single file VR output
    if comm_rank == 0:
        if os.path.exists(f"{basename}.properties"):
            suffix = ""
        else:
            suffix = ".%(file_nr)d"
    else:
        suffix = None
    suffix = comm.bcast(suffix)

    def vr_filename(file_type, file_nr=None):
        filebase = f"{basename}.{file_type}"
        if file_nr is not None:
            return filebase + (suffix % {"file_nr": file_nr})
        else:
            return filebase + suffix

    # Datasets we need to read from the .properties files
    datasets = (
        "Xcminpot",
        "Ycminpot",
        "Zcminpot",
        "Xc",
        "Yc",
        "Zc",
        "R_size",
        "Structuretype",
        "ID",
        "npart",
        "hostHaloID",
        "numSubStruct",
    )

    # Read in positions and radius of each halo, distributed over all MPI ranks
    mf = phdf5.MultiFile(vr_filename("properties"), file_nr_dataset="Num_of_files")
    local_halo = mf.read(datasets)

    # Read numbers of bound and unbound particles in each halo
    local_halo["nr_bound_part"], local_halo["nr_unbound_part"] = read_vr_group_sizes(
        basename, suffix, comm
    )

    # Read parent halo ID from the .catalog_groups files
    mf = phdf5.MultiFile(vr_filename("catalog_groups"), file_nr_dataset="Num_of_files")
    local_halo.update(mf.read(["Parent_halo_ID"]))

    # Compute array index of each halo
    nr_local = local_halo["ID"].shape[0]
    offset = comm.scan(nr_local) - nr_local
    local_halo["index"] = np.arange(offset, offset + nr_local, dtype=int)

    # Combine positions into one array each
    local_halo["cofm"] = np.column_stack(
        (local_halo["Xc"], local_halo["Yc"], local_halo["Zc"])
    )
    del local_halo["Xc"]
    del local_halo["Yc"]
    del local_halo["Zc"]
    local_halo["cofp"] = np.column_stack(
        (local_halo["Xcminpot"], local_halo["Ycminpot"], local_halo["Zcminpot"])
    )
    del local_halo["Xcminpot"]
    del local_halo["Ycminpot"]
    del local_halo["Zcminpot"]

    # Extract unit information from the first file
    if comm_rank == 0:
        filename = vr_filename("properties", 0)
        with h5py.File(filename, "r") as infile:
            units = dict(infile["UnitInfo"].attrs)
            siminfo = dict(infile["SimulationInfo"].attrs)
    else:
        units = None
        siminfo = None
    units, siminfo = comm.bcast((units, siminfo))

    # Make central halo flag
    local_halo["is_central"] = np.zeros(len(local_halo["index"]), dtype=np.int32)
    local_halo["is_central"][local_halo["Structuretype"] == 10] = 1

    # Compute conversion factors to comoving Mpc (no h)
    comoving_or_physical = int(units["Comoving_or_Physical"])
    length_unit_to_kpc = float(units["Length_unit_to_kpc"])
    h = float(siminfo["h_val"])
    if comoving_or_physical == 0:
        # File contains physical units with no h factor
        length_conversion = (1.0 / a) * length_unit_to_kpc / 1000.0  # to comoving Mpc
    else:
        # File contains comoving 1/h units
        length_conversion = h * length_unit_to_kpc / 1000.0  # to comoving Mpc

    # Convert units and wrap in unyt_arrays
    for name in local_halo:
        dtype = local_halo[name].dtype
        if name in ("cofm", "cofp", "R_size"):
            conv_fac = length_conversion
            units = swift_cmpc
        elif name in (
            "Structuretype",
            "ID",
            "index",
            "npart",
            "hostHaloID",
            "numSubStruct",
            "Parent_halo_ID",
            "is_central",
            "nr_bound_part",
            "nr_unbound_part",
        ):
            conv_fac = None
            units = unyt.dimensionless
        else:
            raise Exception("Unrecognized property name: " + name)
        if conv_fac is not None:
            local_halo[name] = unyt.unyt_array(
                local_halo[name] * conv_fac, units=units, dtype=dtype, registry=registry
            )
        else:
            local_halo[name] = unyt.unyt_array(
                local_halo[name], units=units, dtype=dtype, registry=registry
            )

    # Compute initial search radius for each halo:
    #
    # Need to ensure that our radius about the potential minimum
    # includes all particles within r_size of the centre of mass.
    #
    # Find distance from centre of mass to centre of potential,
    # taking the periodic box into account
    dist = np.abs(local_halo["cofp"] - local_halo["cofm"])
    for dim in range(3):
        need_wrap = dist[:, dim] > 0.5 * boxsize
        dist[need_wrap, dim] = boxsize - dist[need_wrap, dim]
    dist = np.sqrt(np.sum(dist ** 2, axis=1))

    # Store the initial search radius
    local_halo["search_radius"] = local_halo["R_size"] * 1.01 + dist

    return local_halo
