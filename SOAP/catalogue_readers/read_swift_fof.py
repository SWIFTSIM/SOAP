#!/bin/env python

import h5py
import numpy as np
import unyt

import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.core import swift_units

# Value of FOFGroupIDs for particles which are not in a FOF group, used if
# the snapshot does not record FOF:group_id_default
DEFAULT_NULL_FOF_ID = 2147483647


def read_null_fof_id(filename):
    """
    Return the FOFGroupIDs value used for particles not in a FOF group,
    reading it from the snapshot parameters if possible.

    filename - name of a single SWIFT snapshot file
    """
    with h5py.File(filename, "r") as infile:
        params = infile["Parameters"].attrs if "Parameters" in infile else {}
        value = params.get("FOF:group_id_default", DEFAULT_NULL_FOF_ID)
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    return int(value)


def read_swift_fof_groupnr(snap_filename):
    """
    Read particle IDs and FOF group IDs from a SWIFT snapshot.

    snap_filename - format string for the snapshot (or another file set with
                    ParticleIDs and FOFGroupIDs datasets), with {file_nr}
                    left unsubstituted for multi-file outputs

    Returns the total number of FOF groups found, and the particle IDs and
    group IDs of the particles which are in a FOF group. The group IDs
    are used as the halo index, matching read_swift_fof_catalogue.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Determine which particle types have FOF information
    if comm_rank == 0:
        filename = phdf5.substitute_file_nr(snap_filename, 0)
        null_fof_id = read_null_fof_id(filename)
        ptypes = []
        with h5py.File(filename, "r") as infile:
            nr_types = infile["Header"].attrs["NumPartTypes"][0]
            for i in range(nr_types):
                name = f"PartType{i}"
                if (name in infile) and ("FOFGroupIDs" in infile[name]):
                    ptypes.append(name)
        if len(ptypes) == 0:
            raise RuntimeError(f"No FOFGroupIDs datasets found in {filename}")
    else:
        null_fof_id = None
        ptypes = None
    null_fof_id, ptypes = comm.bcast((null_fof_id, ptypes))

    snap_file = phdf5.MultiFile(
        snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )

    # Read IDs of particles in FOF groups for all types
    local_ids = []
    local_grnr = []
    for ptype in ptypes:
        data = snap_file.read(("ParticleIDs", "FOFGroupIDs"), group=ptype)
        keep = data["FOFGroupIDs"] != null_fof_id
        local_ids.append(data["ParticleIDs"][keep])
        local_grnr.append(data["FOFGroupIDs"][keep].astype(np.int64))
        del data, keep
    local_ids = np.concatenate(local_ids)
    local_grnr = np.concatenate(local_grnr)

    # Count number of groups
    local_max = np.amax(local_grnr) if local_grnr.shape[0] > 0 else -1
    max_grnr = comm.allreduce(local_max, op=MPI.MAX)
    total_nr_halos = max_grnr + 1

    return total_nr_halos, local_ids, local_grnr


def read_swift_fof_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in a SWIFT FOF catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - FOF catalogue filename, with {file_nr} left unsubstituted
               for multi-file catalogues
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

    FOF groups have no substructure, so every group is treated as a central.
    nr_bound_part is the number of particles in the group (which are not
    necessarily bound).
    """

    comm_rank = comm.Get_rank()

    # Read the units of the datasets from the first file
    if comm_rank == 0:
        filename = phdf5.substitute_file_nr(basename, 0)
        with h5py.File(filename, "r") as infile:
            if "Groups/Radii" not in infile:
                raise RuntimeError(
                    f"FOF catalogue {filename} has no Groups/Radii dataset. "
                    "This can be created with misc/calculate_fof_radii.py"
                )
            attrs = {
                name: dict(infile[f"Groups/{name}"].attrs)
                for name in ("Centres", "Masses", "Radii")
            }
    else:
        attrs = None
    attrs = comm.bcast(attrs)
    units = {
        name: swift_units.units_from_attributes(attrs[name], registry) for name in attrs
    }

    # Read the catalogue
    fof_file = phdf5.MultiFile(
        basename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
    )
    data = fof_file.read(
        ("GroupIDs", "Centres", "Masses", "Radii", "Sizes"), group="Groups"
    )

    local_halo = {}
    local_halo["index"] = unyt.unyt_array(
        data["GroupIDs"].astype(np.int64),
        dtype=np.int64,
        units=unyt.dimensionless,
        registry=registry,
    )
    # Centres of mass can lie slightly outside the box, so wrap them
    local_halo["cofp"] = (
        unyt.unyt_array(data["Centres"], units=units["Centres"], registry=registry)
        % boxsize
    )
    local_halo["search_radius"] = 1.001 * unyt.unyt_array(
        data["Radii"], units=units["Radii"], registry=registry
    )
    nr_halos = local_halo["index"].shape[0]
    local_halo["is_central"] = unyt.unyt_array(
        np.ones(nr_halos, dtype=np.int64),
        dtype=np.int64,
        units=unyt.dimensionless,
        registry=registry,
    )
    local_halo["nr_bound_part"] = unyt.unyt_array(
        data["Sizes"].astype(np.int64),
        dtype=np.int64,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Quantities passed through to the output as InputHalos/FOF/*
    local_halo["Centres"] = local_halo["cofp"].copy()
    local_halo["Radii"] = unyt.unyt_array(
        data["Radii"], units=units["Radii"], registry=registry
    )
    local_halo["Masses"] = unyt.unyt_array(
        data["Masses"], units=units["Masses"], registry=registry
    )
    local_halo["Sizes"] = unyt.unyt_array(
        data["Sizes"].astype(np.uint64),
        dtype=np.uint64,
        units=unyt.dimensionless,
        registry=registry,
    )

    return local_halo
