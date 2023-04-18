#!/bin/env python

import os.path
import h5py
import numpy as np
import unyt
import virgo.util.match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g

import domain_decomposition


def gather_to_rank_zero(arr):
    """Gather the specified array on rank 0, preserving units"""
    units = arr.units
    arr = g.gather_array(arr.value)
    return unyt.unyt_array(arr, units=units)


class SOCatalogue:
    def __init__(
        self,
        comm,
        vr_basename,
        a_unit,
        registry,
        boxsize,
        max_halos,
        centrals_only,
        halo_ids,
        halo_prop_list,
        nr_chunks,
    ):
        """
        This reads in the VR catalogues and stores the halo properties in a
        dict of unyt_arrays, self.halo_arrays, on rank 0 of communicator comm.
        It also calculates the radii to read in around each halo.

        self.halo_arrays["read_radius"] contains the radius to read in about
        the potential minimum of each halo.

        self.halo_arrays["search_radius"] contains an initial guess for the
        radius we need to search to reach the required overdensity. This will
        be increased up to read_radius if necessary.

        Both read_radius and search_radius will be set to be at least as large
        as the largest physical_radius_mpc specified by the halo property
        calculations.
        """

        comm_rank = comm.Get_rank()

        # Get SWIFT's definition of physical and comoving Mpc units
        swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
        swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
        swift_msun = unyt.Unit("swift_msun", registry=registry)

        # Get expansion factor as a float
        a = a_unit.base_value

        # Find minimum physical radius to read in
        physical_radius_mpc = 0.0
        for halo_prop in halo_prop_list:
            physical_radius_mpc = max(
                physical_radius_mpc, halo_prop.physical_radius_mpc
            )
        physical_radius_mpc = unyt.unyt_quantity(physical_radius_mpc, units=swift_pmpc)

        # Here we need to read the centre of mass AND potential minimum:
        # The radius R_size about (Xc, Yc, Zc) contains all particles which
        # belong to the group. But we want to compute spherical overdensity
        # quantities about the potential minimum.
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

        # Check for single file VR output - will prefer filename without
        # extension if both are present
        vr_basename_props = f"{vr_basename}.properties"
        if comm_rank == 0:
            if os.path.exists(vr_basename_props):
                filenames = vr_basename_props
            else:
                filenames = vr_basename_props + ".%(file_nr)d"
        else:
            filenames = None
        filenames = comm.bcast(filenames)

        # Read in positions and radius of each halo, distributed over all MPI ranks
        mf = phdf5.MultiFile(filenames, file_nr_dataset="Num_of_files")
        local_halo = mf.read(datasets)

        vr_basename_groups = f"{vr_basename}.catalog_groups"
        if comm_rank == 0:
            if os.path.exists(vr_basename_groups):
                group_filenames = vr_basename_groups
            else:
                group_filenames = vr_basename_groups + ".%(file_nr)d"
        else:
            group_filenames = None
        group_filenames = comm.bcast(group_filenames)
        mf = phdf5.MultiFile(group_filenames, file_nr_dataset="Num_of_files")
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
            filename = filenames % {"file_nr": 0}
            with h5py.File(filename, "r") as infile:
                units = dict(infile["UnitInfo"].attrs)
                siminfo = dict(infile["SimulationInfo"].attrs)
        else:
            units = None
            siminfo = None
        units, siminfo = comm.bcast((units, siminfo))

        # Compute conversion factors to comoving Mpc (no h)
        comoving_or_physical = int(units["Comoving_or_Physical"])
        length_unit_to_kpc = float(units["Length_unit_to_kpc"])
        h = float(siminfo["h_val"])
        if comoving_or_physical == 0:
            # File contains physical units with no h factor
            length_conversion = (
                (1.0 / a) * length_unit_to_kpc / 1000.0
            )  # to comoving Mpc
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
            ):
                conv_fac = None
                units = unyt.dimensionless
            else:
                raise Exception("Unrecognized property name: " + name)
            if conv_fac is not None:
                local_halo[name] = unyt.unyt_array(
                    local_halo[name] * conv_fac,
                    units=units,
                    dtype=dtype,
                    registry=registry,
                )
            else:
                local_halo[name] = unyt.unyt_array(
                    local_halo[name], units=units, dtype=dtype, registry=registry
                )

        # Assign halos to chunk tasks
        task_id = domain_decomposition.peano_decomposition(
            boxsize, local_halo["cofp"], nr_chunks, comm
        )
        local_halo["task_id"] = unyt.unyt_array(
            task_id, units=unyt.dimensionless, registry=registry, dtype=task_id.dtype
        )

        #
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

        # Compute radius to read in about each halo:
        # this is the maximum radius we'll search to reach the required overdensity
        local_halo["read_radius"] = local_halo["search_radius"].copy()
        min_radius = 5.0 * swift_cmpc
        local_halo["read_radius"] = local_halo["read_radius"].clip(min=min_radius)

        # Ensure that both the initial search radius and the radius to read in
        # are >= the minimum physical radius required by property calculations
        local_halo["read_radius"] = local_halo["read_radius"].clip(
            min=physical_radius_mpc
        )
        local_halo["search_radius"] = local_halo["search_radius"].clip(
            min=physical_radius_mpc
        )

        # Discard satellites, if necessary
        if centrals_only:
            keep = local_halo["Structuretype"] == 10
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # Only keep halos in the supplied list of halo IDs.
        if halo_ids is not None:
            halo_ids = np.asarray(halo_ids, dtype=np.int64)
            keep = np.zeros_like(local_halo["ID"], dtype=bool)
            matching_index = virgo.util.match.match(halo_ids, local_halo["ID"])
            have_match = matching_index >= 0
            keep[matching_index[have_match]] = True
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # Gather subhalo arrays on rank zero.
        halo = {}
        for name in local_halo:
            halo[name] = gather_to_rank_zero(local_halo[name])
        del local_halo

        # For testing: limit number of halos
        if comm_rank == 0 and max_halos > 0:
            for name in halo:
                halo[name] = halo[name][:max_halos, ...]

        # Rank 0 stores the subhalo catalogue
        if comm_rank == 0:
            self.nr_halos = len(halo["search_radius"])
            self.halo_arrays = halo
