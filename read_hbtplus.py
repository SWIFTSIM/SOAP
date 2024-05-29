#!/bin/env python

import os.path
import numpy as np
import h5py
import unyt

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.util


def hbt_filename(hbt_basename, file_nr):
    return f"{hbt_basename}.{file_nr}.hdf5"


def read_hbtplus_groupnr(basename):
    """
    Read HBTplus output and return group number for each particle ID
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Find number of HBT output files
    if comm_rank == 0:
        with h5py.File(hbt_filename(basename, 0), "r") as infile:
            nr_files = int(infile["NumberOfFiles"][...])
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Assign files to MPI ranks
    files_per_rank = np.zeros(comm_size, dtype=int)
    files_per_rank[:] = nr_files // comm_size
    files_per_rank[: nr_files % comm_size] += 1
    assert np.sum(files_per_rank) == nr_files
    first_file_on_rank = np.cumsum(files_per_rank) - files_per_rank

    # Read in the halos from the HBT output:
    # 'halos' will be an array of structs with the halo catalogue
    # 'ids_bound' will be an array of particle IDs in halos, sorted by halo
    halos = []
    ids_bound = []
    for file_nr in range(
        first_file_on_rank[comm_rank],
        first_file_on_rank[comm_rank] + files_per_rank[comm_rank],
    ):
        with h5py.File(hbt_filename(basename, file_nr), "r") as infile:
            halos.append(infile["Subhalos"][...])
            ids_bound.append(infile["SubhaloParticles"][...])

    # Get the dtype for particle IDs
    if len(ids_bound) > 0:
        id_dtype = h5py.check_vlen_dtype(ids_bound[0].dtype)
    else:
        id_dtype = None

    # Concatenate arrays of halos from different files
    if len(halos) > 0:
        halos = np.concatenate(halos)
    else:
        # This rank was assigned no files
        halos = None
    halos = virgo.mpi.util.replace_none_with_zero_size(halos, comm=comm)

    # Combine arrays of particles in halos
    if len(ids_bound) > 0:
        ids_bound = np.concatenate(ids_bound)  # Combine arrays of halos from different files
        if len(ids_bound) > 0:
            ids_bound = np.concatenate(ids_bound)  # Combine arrays of particles from different halos
        else:
            # The files assigned to this rank contain zero halos
            ids_bound = np.zeros(0, dtype=id_dtype)
    else:
        # This rank was assigned no files
        ids_bound = None
    ids_bound = virgo.mpi.util.replace_none_with_zero_size(ids_bound, comm=comm)

    # Assign halo indexes to the particles
    nr_local_halos = len(halos)
    total_nr_halos = comm.allreduce(nr_local_halos)
    halo_offset = comm.scan(len(halos), op=MPI.SUM) - len(halos)
    halo_index = np.arange(nr_local_halos, dtype=int) + halo_offset
    halo_size = halos["Nbound"]
    del halos
    grnr_bound = np.repeat(halo_index, halo_size)

    # Assign ranking by binding energy to the particles
    rank_bound = -np.ones(grnr_bound.shape[0], dtype=int)
    offset = 0
    for halo_nr in range(nr_local_halos):
        rank_bound[offset : offset + halo_size[halo_nr]] = np.arange(
            halo_size[halo_nr], dtype=int
        )
        offset += halo_size[halo_nr]
    assert np.all(rank_bound >= 0)  # HBT only outputs bound particles
    del halo_size
    del halo_offset
    del halo_index

    # Now check for duplicates. HBTplus can assign the same particle to multiple
    # subhalos in cases where a subhalo escapes its host and enters another halo.
    # Here we assign such particles to the subhalo in which their rank_bound is lowest.
    # Make a key to sort the particles by ID and then by bound rank where ID is equal.
    sort_key_t = np.dtype([("id", ids_bound.dtype), ("rank", rank_bound.dtype)])
    sort_key = np.ndarray(len(ids_bound), dtype=sort_key_t)
    sort_key["id"] = ids_bound
    sort_key["rank"] = rank_bound
    assert np.all(sort_key["rank"] >= 0)  # HBTplus does not output unbound particles

    # Sort the particles by ID and then by bound rank where the ID is the same.
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    ids_bound = psort.fetch_elements(ids_bound, order, comm=comm)
    grnr_bound = psort.fetch_elements(grnr_bound, order, comm=comm)
    rank_bound = psort.fetch_elements(rank_bound, order, comm=comm)
    del order

    # Then find the unique particle IDs
    unique_ids_bound, unique_counts = psort.parallel_unique(
        ids_bound, comm=comm, arr_sorted=True, return_counts=True
    )

    # Compute sum of the counts of unique IDs on this MPI rank. This is
    # not necessarily the same as len(ids_bound).
    nr_ids_local = np.sum(unique_counts, dtype=int)

    # Compute sum of the counts of unique IDs on all previous MPI ranks
    nr_ids_prev_rank = comm.scan(nr_ids_local) - nr_ids_local

    # Find the global offset of the first instance of each unique ID in the
    # full array of IDs
    unique_offsets = np.cumsum(unique_counts) - unique_counts + nr_ids_prev_rank

    # Fetch the ID, grnr_bound and rank_bound of the first instance of each particle ID
    ids_bound = psort.fetch_elements(ids_bound, unique_offsets, comm=comm)
    assert np.all(
        ids_bound == unique_ids_bound
    )  # Check we computed unique_offsets correctly
    rank_bound = psort.fetch_elements(rank_bound, unique_offsets, comm=comm)
    grnr_bound = psort.fetch_elements(grnr_bound, unique_offsets, comm=comm)

    return total_nr_halos, ids_bound, grnr_bound, rank_bound


def read_hbtplus_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the HBTplus halo catalogue, distributed over communicator comm.

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

    Note that in case of HBT we only want to compute properties of resolved
    halos, so we discard those with 0-1 bound particles.
    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
    swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
    swift_msun = unyt.Unit("swift_msun", registry=registry)

    # Get km/s
    kms = unyt.Unit("km/s", registry=registry)

    # Get expansion factor as a float
    a = a_unit.base_value

    # Get h as a float
    h_unit = unyt.Unit("h", registry=registry)
    h = h_unit.base_value

    # Get HBTplus unit information
    if comm_rank == 0:
        # Try to get units from the HDF5 output
        have_units = False
        filename = hbt_filename(basename, 0)
        with h5py.File(filename, "r") as infile:
            if "Units" in infile:
                LengthInMpch = float(infile["Units/LengthInMpch"][...])
                MassInMsunh = float(infile["Units/MassInMsunh"][...])
                VelInKmS = float(infile["Units/VelInKmS"][...])
                have_units = True
        # Otherwise, will have to read the Parameters.log file
        if not(have_units):
            dirname = os.path.dirname(os.path.dirname(filename))
            with open(dirname+"/Parameters.log", "r") as infile:
                for line in infile:
                    fields = line.split()
                    if len(fields) == 2:
                        name, value = fields
                        if name == "MassInMsunh":
                            MassInMsunh = float(value)
                        elif name == "LengthInMpch":
                            LengthInMpch = float(value)
                        elif name == "VelInKmS":
                            VelInKmS = float(value)
    else:
        LengthInMpch = None
        MassInMsunh = None
        VelInKmS = None
    (LengthInMpch, MassInMsunh, VelInKmS) = comm.bcast(
        (LengthInMpch, MassInMsunh, VelInKmS)
    )

    # Read the subhalos for this snapshot
    filename = f"{basename}.%(file_nr)d.hdf5"
    mf = phdf5.MultiFile(filename, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalo = mf.read("Subhalos")

    # Load the number of bound particles
    nr_bound_part = unyt.unyt_array(
        subhalo['Nbound'], units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Only process resolved subhalos (HBTplus also outputs unresolved "orphan" subhalos)
    keep = nr_bound_part > 1

    # Assign indexes to halos: for each halo we're going to process we store the
    # position in the input catalogue.
    nr_local_halos = len(keep)
    local_offset = comm.scan(nr_local_halos) - nr_local_halos
    index = np.arange(nr_local_halos, dtype=int) + local_offset
    index = index[keep]
    index = unyt.unyt_array(
        index, units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Find centre of potential
    cofp = (
        subhalo["ComovingMostBoundPosition"][keep, :] * LengthInMpch / h
    ) * swift_cmpc

    # Initial guess at search radius for each halos.
    # Search radius will be expanded if we don't find all of the bound particles.
    search_radius = (
        1.01 * (subhalo["REncloseComoving"][keep] * LengthInMpch / h) * swift_cmpc
    )

    # Central halo flag
    is_central = np.where(subhalo["Rank"] == 0, 1, 0)[keep]
    is_central = unyt.unyt_array(
        is_central, units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Subhalo tracking information
    track_id = unyt.unyt_array(
        subhalo["TrackId"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    host_halo_id = unyt.unyt_array(
        subhalo["HostHaloId"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    depth = unyt.unyt_array(
        subhalo["Depth"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    snapshot_birth = unyt.unyt_array(
        subhalo["SnapshotIndexOfBirth"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    parent_id = unyt.unyt_array(
        subhalo["NestedParentTrackId"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    descendant_id = unyt.unyt_array(
        subhalo["DescendantTrackId"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Peak mass
    max_mass = (subhalo["LastMaxMass"][keep] * MassInMsunh / h ) * swift_msun
    snapshot_max_mass = subhalo["SnapshotIndexOfLastMaxMass"][keep]
    snapshot_max_mass = unyt.unyt_array(
        snapshot_max_mass, units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Peak vmax
    max_vmax = (subhalo["LastMaxVmaxPhysical"][keep] * VelInKmS ) * kms
    snapshot_max_vmax = subhalo["SnapshotIndexOfLastMaxVmax"][keep]
    snapshot_max_vmax = unyt.unyt_array(
        snapshot_max_vmax, units=unyt.dimensionless, dtype=int, registry=registry
    )


    # Number of bound particles
    nr_bound_part = nr_bound_part[keep]

    local_halo = {
        "cofp": cofp,
        "index": index,
        "search_radius": search_radius,
        "is_central": is_central,
        "nr_bound_part": nr_bound_part,
        "HostHaloId": host_halo_id,
        "Depth": depth,
        "TrackId": track_id,
        "SnapshotIndexOfBirth": snapshot_birth,
        "NestedParentTrackId": parent_id,
        "DescendantTrackId": descendant_id,
        "LastMaxMass": max_mass,
        "SnapshotIndexOfLastMaxMass": snapshot_max_mass,
        "LastMaxVmaxPhysical": max_vmax,
        "SnapshotIndexOfLastMaxVmax": snapshot_max_vmax,
    }
    return local_halo
