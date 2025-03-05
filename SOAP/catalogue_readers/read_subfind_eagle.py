#!/bin/env python

import os

import numpy as np
import h5py
import unyt

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5


def read_subfind_catalogue(comm, basename, a_unit, registry, boxsize):
    """
    Read in the EAGLE subhalo catalogue, distributed over communicator comm.

    comm     - communicator to distribute catalogue over
    basename - SubFind catalogue filename without the .N.hdf5 suffix
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

    """

    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    sub_format_string = basename + ".{file_nr}.hdf5"
    sub_file = phdf5.MultiFile(
        sub_format_string, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Get SWIFT's definition of physical and comoving Mpc units
    swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
    swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)

    if comm_rank == 0:
        with h5py.File(sub_format_string.format(file_nr=0), "r") as file:
            h = file["Header"].attrs["HubbleParam"]
            # Check units are indeed what we are assuming below
            units_header = file["Units"].attrs
            mpc_in_cm = (1 * unyt.Mpc).to("cm").value
            assert np.isclose(units_header["UnitLength_in_cm"], mpc_in_cm)
            M_in_g = (10**10 * unyt.Msun).to("g").value
            assert np.isclose(units_header["UnitMass_in_g"], M_in_g, rtol=1e-2)
            assert file["Subhalo/CentreOfPotential"].attrs["h-scale-exponent"] == -1
            assert file["Subhalo/CentreOfPotential"].attrs["aexp-scale-exponent"] == 1
            assert file["Subhalo/VmaxRadius"].attrs["h-scale-exponent"] == -1
            assert file["Subhalo/VmaxRadius"].attrs["aexp-scale-exponent"] == 1
    else:
        h = None
    h = comm.bcast(h)

    sub_format_string = basename + ".{file_nr}.hdf5"
    sub_file = phdf5.MultiFile(
        sub_format_string, file_nr_attr=("Header", "NumFilesPerSnapshot")
    )

    # Read halo properties we need
    datasets = (
        "Subhalo/GroupNumber",
        "Subhalo/SubGroupNumber",
        "Subhalo/CentreOfPotential",
        "Subhalo/SubLength",
        "Subhalo/VmaxRadius",
    )
    data = sub_file.read(datasets)

    # Create a halo index by combining the GroupNumber and SubGroupNumber
    group = data["Subhalo/GroupNumber"]
    sub_group = data["Subhalo/SubGroupNumber"]
    index = group.astype(np.int64)
    index <<= 32
    index += sub_group
    index = unyt.unyt_array(
        index,
        dtype=int,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Get position in comoving Mpc
    cofp = (data["Subhalo/CentreOfPotential"] / h) * swift_cmpc

    # Store central halo flag
    is_central = unyt.unyt_array(
        data["Subhalo/SubGroupNumber"] == 0,
        dtype=int,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Store number of bound particles in each halo
    nr_bound_part = unyt.unyt_array(
        data["Subhalo/SubLength"],
        dtype=int,
        units=unyt.dimensionless,
        registry=registry,
    )

    # Store initial search radius
    search_radius = (5 * data["Subhalo/VmaxRadius"] / h) * swift_cmpc

    local_halo = {
        "cofp": cofp,
        "index": index,
        "search_radius": search_radius,
        "is_central": is_central,
        "nr_bound_part": nr_bound_part,
    }

    for name in local_halo:
        local_halo[name] = unyt.unyt_array(local_halo[name], registry=registry)

    return local_halo
