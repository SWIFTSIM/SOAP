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


def read_hbtplus_groupnr(basename, read_potential_energies=False, registry=None):
    """
    Read HBTplus output and return group number for each particle ID

    Potential energies will not be returned by default. To return the potential
    energies a unit registry must be passed.

    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Find number of HBT output files, and if we're dealing with sorted output
    if comm_rank == 0:
        if os.path.exists(hbt_filename(basename, 0)):
            with h5py.File(hbt_filename(basename, 0), "r") as infile:
                nr_files = int(infile["NumberOfFiles"][...])
            sorted_file = False
        elif os.path.exists(basename):
            with h5py.File(basename, "r") as infile:
                assert "Particles" in infile
            nr_files = 1
            sorted_file = True
        else:
            print(f"No HBT files found for basename {basename}")
            comm.Abort()
    else:
        nr_files = None
        sorted_file = None
    nr_files = comm.bcast(nr_files)
    sorted_file = comm.bcast(sorted_file)

    # There are two different read routines. One is for the original HBTplus
    # output. The second is for the "nice" output, where the subhalos are
    # sorted by TrackId.
    if not sorted_file:
        # Assign files to MPI ranks
        files_per_rank = np.zeros(comm_size, dtype=int)
        files_per_rank[:] = nr_files // comm_size
        remainder = nr_files % comm_size
        if remainder == 1:
            files_per_rank[0] += 1
        elif remainder > 1:
            for i in range(remainder):
                files_per_rank[int((comm_size - 1) * i / (remainder - 1))] += 1
        assert np.sum(files_per_rank) == nr_files, f"{nr_files=}, {comm_size=}"
        first_file_on_rank = np.cumsum(files_per_rank) - files_per_rank

        halos = []
        ids_bound = []
        if read_potential_energies:
            potential_energies = []

        for file_nr in range(
            first_file_on_rank[comm_rank],
            first_file_on_rank[comm_rank] + files_per_rank[comm_rank],
        ):
            with h5py.File(hbt_filename(basename, file_nr), "r") as infile:
                halos.append(infile["Subhalos"][...])
                ids_bound.append(infile["SubhaloParticles"][...])
                if read_potential_energies:
                    potential_energies.append(infile["PotentialEnergies"][...])

        # Concatenate arrays of halos from different files
        if len(halos) > 0:
            halos = np.concatenate(halos)
        else:
            # This rank was assigned no files
            halos = None
        halos = virgo.mpi.util.replace_none_with_zero_size(halos, comm=comm)

        if len(ids_bound) > 0:
            # Get the dtype for particle IDs
            id_dtype = h5py.check_vlen_dtype(ids_bound[0].dtype)
            # Combine arrays of halos from different files
            ids_bound = np.concatenate(ids_bound)
            if len(ids_bound) > 0:
                # Combine arrays of particles from different halos
                ids_bound = np.concatenate(ids_bound)
            else:
                # The files assigned to this rank contain zero halos
                ids_bound = np.zeros(0, dtype=id_dtype)
        else:
            # This rank was assigned no files
            ids_bound = None
        ids_bound = virgo.mpi.util.replace_none_with_zero_size(ids_bound, comm=comm)

        # Number of particles in each subhalo
        halo_size = halos["Nbound"]
        del halos

        # Apply same combination process to potential energies
        if read_potential_energies:
            if len(potential_energies) > 0:
                potential_dtype = h5py.check_vlen_dtype(potential_energies[0].dtype)
                potential_energies = np.concatenate(potential_energies)
                if len(potential_energies) > 0:
                    potential_energies = np.concatenate(potential_energies)
                else:
                    potential_energies = np.zeros(0, dtype=potential_dtype)
            else:
                # This rank was assigned no files
                potential_energies = None
            potential_energies = virgo.mpi.util.replace_none_with_zero_size(
                potential_energies, comm=comm
            )
    else:
        # Read the fields we require from the sorted catalogues
        hbt_file = phdf5.MultiFile([basename], comm=comm)
        ids_bound = hbt_file.read("Particles/ParticleIDs")
        halo_size = hbt_file.read("Subhalos/Nbound")
        if read_potential_energies:
            potential_energies = hbt_file.read("Particles/PotentialEnergies")

    if read_potential_energies:
        # Get HBTplus unit information
        if comm_rank == 0:
            filename = basename if sorted_file else hbt_filename(basename, 0)
            with h5py.File(filename, "r") as infile:
                if "Units" in infile:
                    VelInKmS = float(infile["Units/VelInKmS"][...])
        else:
            VelInKmS = None
        VelInKmS = comm.bcast(VelInKmS)

        # Add units to potential energies
        potential_energies = (potential_energies * (VelInKmS**2)) * unyt.Unit(
            "km/s", registry=registry
        ) ** 2

    # Assign halo indexes to the particles
    nr_local_halos = len(halo_size)
    total_nr_halos = comm.allreduce(nr_local_halos)
    halo_offset = comm.scan(len(halo_size), op=MPI.SUM) - len(halo_size)
    halo_index = np.arange(nr_local_halos, dtype=int) + halo_offset
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

    # HBTplus originally output duplicate particles, so this script previously
    # assigned duplicate particles to a single subhalo based on their bound rank.
    # HBT should no longer have duplicates, which is tested by this assert
    unique_ids_bound, unique_counts = psort.parallel_unique(
        ids_bound, comm=comm, arr_sorted=False, return_counts=True
    )
    assert len(unique_counts) == 0 or np.max(unique_counts) == 1

    if read_potential_energies:
        return total_nr_halos, ids_bound, grnr_bound, rank_bound, potential_energies

    return total_nr_halos, ids_bound, grnr_bound, rank_bound


def read_hbtplus_catalogue(
    comm, basename, a_unit, registry, boxsize, keep_orphans=False
):
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
        # Check if this is a sorted HBT file
        if os.path.exists(hbt_filename(basename, 0)):
            filename = hbt_filename(basename, 0)
            sorted_file = False
        else:
            filename = basename
            sorted_file = True

        # Try to get units from the HDF5 output
        have_units = False
        with h5py.File(filename, "r") as infile:
            if "Units" in infile:
                LengthInMpch = float(infile["Units/LengthInMpch"][...])
                MassInMsunh = float(infile["Units/MassInMsunh"][...])
                VelInKmS = float(infile["Units/VelInKmS"][...])
                have_units = True
        # Otherwise, will have to read the Parameters.log file
        if not (have_units):
            dirname = os.path.dirname(os.path.dirname(filename))
            with open(dirname + "/Parameters.log", "r") as infile:
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
        sorted_file = None
    (LengthInMpch, MassInMsunh, VelInKmS) = comm.bcast(
        (LengthInMpch, MassInMsunh, VelInKmS)
    )
    sorted_file = comm.bcast(sorted_file)

    # Read the subhalos for this snapshot
    if not sorted_file:
        filename = f"{basename}.%(file_nr)d.hdf5"
        mf = phdf5.MultiFile(filename, file_nr_dataset="NumberOfFiles", comm=comm)
        subhalo = mf.read("Subhalos")
        subhalo_props = list(subhalo.dtype.names)
    else:
        # Just recreate an object the same as the unsorted file
        if comm_rank == 0:
            with h5py.File(filename, "r") as infile:
                subhalo_props = list(infile["Subhalos"].keys())
        else:
            subhalo_props = None
        subhalo_props = comm.bcast(subhalo_props)
        mf = phdf5.MultiFile([basename], comm=comm)
        subhalo = {}
        for prop in subhalo_props:
            subhalo[prop] = mf.read(f"Subhalos/{prop}")

    # Load the number of bound particles
    nr_bound_part = unyt.unyt_array(
        subhalo["Nbound"], units=unyt.dimensionless, dtype=int, registry=registry
    )

    # Do we only process resolved subhalos? (HBT also outputs unresolved "orphan" subhalos)
    if not keep_orphans:
        keep = nr_bound_part > 0
    else:
        keep = np.ones_like(nr_bound_part, dtype=bool)

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
        subhalo["HostHaloId"][keep],
        units=unyt.dimensionless,
        dtype=int,
        registry=registry,
    )
    depth = unyt.unyt_array(
        subhalo["Depth"][keep], units=unyt.dimensionless, dtype=int, registry=registry
    )
    parent_id = unyt.unyt_array(
        subhalo["NestedParentTrackId"][keep],
        units=unyt.dimensionless,
        dtype=int,
        registry=registry,
    )
    descendant_id = unyt.unyt_array(
        subhalo["DescendantTrackId"][keep],
        units=unyt.dimensionless,
        dtype=int,
        registry=registry,
    )

    # Peak mass
    max_mass = (subhalo["LastMaxMass"][keep] * MassInMsunh / h) * swift_msun

    # Peak vmax
    max_vmax = (subhalo["LastMaxVmaxPhysical"][keep] * VelInKmS) * kms

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
        "NestedParentTrackId": parent_id,
        "DescendantTrackId": descendant_id,
        "LastMaxMass": max_mass,
        "LastMaxVmaxPhysical": max_vmax,
    }

    if "SnapshotIndexOfBirth" in subhalo_props:
        snapshot_birth = subhalo["SnapshotIndexOfBirth"][keep]
        snapshot_max_mass = subhalo["SnapshotIndexOfLastMaxMass"][keep]
        snapshot_max_vmax = subhalo["SnapshotIndexOfLastMaxVmax"][keep]
        snapshot_isolation = subhalo["SnapshotIndexOfLastIsolation"][keep]
    else:
        snapshot_birth = subhalo["SnapshotOfBirth"][keep]
        snapshot_max_mass = subhalo["SnapshotOfLastMaxMass"][keep]
        snapshot_max_vmax = subhalo["SnapshotOfLastMaxVmax"][keep]
        snapshot_isolation = subhalo["SnapshotOfLastIsolation"][keep]

    local_halo["SnapshotOfBirth"] = unyt.unyt_array(
        snapshot_birth, units=unyt.dimensionless, dtype=int, registry=registry
    )
    local_halo["SnapshotOfLastMaxMass"] = unyt.unyt_array(
        snapshot_max_mass, units=unyt.dimensionless, dtype=int, registry=registry
    )
    local_halo["SnapshotOfLastMaxVmax"] = unyt.unyt_array(
        snapshot_max_vmax, units=unyt.dimensionless, dtype=int, registry=registry
    )
    local_halo["SnapshotOfLastIsolation"] = unyt.unyt_array(
        snapshot_isolation, units=unyt.dimensionless, dtype=int, registry=registry
    )

    return local_halo
