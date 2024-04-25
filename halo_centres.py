#!/bin/env python

import os.path
import h5py
import numpy as np
import unyt
import virgo.util.match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g

import domain_decomposition
import read_vr
import read_hbtplus
import read_subfind
import read_rockstar


def gather_to_rank_zero(arr):
    """Gather the specified array on rank 0, preserving units"""
    units = arr.units
    arr = g.gather_array(arr.value)
    return unyt.unyt_array(arr, units=units)


class SOCatalogue:
    def __init__(
        self,
        comm,
        halo_basename,
        halo_format,
        a_unit,
        registry,
        boxsize,
        max_halos,
        centrals_only,
        halo_indices,
        halo_prop_list,
        nr_chunks,
        halo_size_file,
        min_read_radius_cmpc,
    ):
        """
        This reads in the halo catalogue and stores the halo properties in a
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

        # Read the input halo catalogue
        common_props = (
            "index",
            "cofp",
            "search_radius",
            "is_central",
            "nr_bound_part",
            "nr_unbound_part",
        )
        if halo_format == "VR":
            halo_data = read_vr.read_vr_catalogue(
                comm, halo_basename, a_unit, registry, boxsize
            )
        elif halo_format == "HBTplus":
            halo_data = read_hbtplus.read_hbtplus_catalogue(
                comm, halo_basename, a_unit, registry, boxsize, halo_size_file
            )
        elif halo_format == "Subfind":
            halo_data = read_subfind.read_gadget4_catalogue(
                comm, halo_basename, a_unit, registry, boxsize
            )
        elif halo_format == "Rockstar":
            halo_data = read_rockstar.read_rockstar_catalogue(
                comm, halo_basename, a_unit, registry, boxsize
            )
        else:
            raise RuntimeError(f"Halo format {format} not recognised!")

        # Add halo finder prefix to halo finder specific quantities:
        # This in case different finders use the same property names.
        local_halo = {}
        for name in halo_data:
            if name in common_props:
                local_halo[name] = halo_data[name]
            else:
                local_halo[f"{halo_format}/{name}"] = halo_data[name]
        del halo_data

        # Only keep halos in the supplied list of halo IDs.
        if (halo_indices is not None) and (local_halo['index'].shape[0]):
            halo_indices = np.asarray(halo_indices, dtype=np.int64)
            keep = np.zeros_like(local_halo["index"], dtype=bool)
            matching_index = virgo.util.match.match(halo_indices, local_halo["index"])
            have_match = matching_index >= 0
            keep[matching_index[have_match]] = True
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # Discard satellites, if necessary
        if centrals_only:
            keep = local_halo["is_central"] == 1
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # For testing: limit number of halos processed
        if max_halos > 0:
            nr_halos_local = len(local_halo["index"])
            nr_halos_prev = comm.scan(nr_halos_local) - nr_halos_local
            nr_keep_local = max_halos - nr_halos_prev
            if nr_keep_local < 0:
                nr_keep_local = 0
            if nr_keep_local > nr_halos_local:
                nr_keep_local = nr_halos_local
            for name in local_halo:
                local_halo[name] = local_halo[name][:nr_keep_local, ...]

        # Assign halos to chunk tasks
        task_id = domain_decomposition.peano_decomposition(
            boxsize, local_halo["cofp"], nr_chunks, comm
        )
        local_halo["task_id"] = unyt.unyt_array(
            task_id, units=unyt.dimensionless, registry=registry, dtype=task_id.dtype
        )

        # Compute initial radius to read in about each halo
        local_halo["read_radius"] = local_halo["search_radius"].copy()
        min_radius = min_read_radius_cmpc * swift_cmpc
        local_halo["read_radius"] = local_halo["read_radius"].clip(min=min_radius)

        # Find minimum physical radius to read in
        physical_radius_mpc = 0.0
        for halo_prop in halo_prop_list:
            # Skip halo_types with a nonzero mass limit
            # Assume mass limit is large enough to exclude majority of halos
            if hasattr(halo_prop, "mass_limit"):
                if halo_prop.mass_limit > 0 * unyt.Msun:
                    continue
            physical_radius_mpc = max(
                physical_radius_mpc, halo_prop.physical_radius_mpc
            )
        physical_radius_mpc = unyt.unyt_quantity(physical_radius_mpc, units=swift_pmpc)

        # Ensure that both the initial search radius and the radius to read in
        # are >= the minimum physical radius required by property calculations
        local_halo["read_radius"] = local_halo["read_radius"].clip(
            min=physical_radius_mpc
        )
        local_halo["search_radius"] = local_halo["search_radius"].clip(
            min=physical_radius_mpc
        )

        # Gather subhalo arrays on rank zero.
        halo = {}
        for name in local_halo:
            halo[name] = gather_to_rank_zero(local_halo[name])
        del local_halo

        # Rank 0 stores the subhalo catalogue
        if comm_rank == 0:
            self.nr_halos = len(halo["search_radius"])
            self.halo_arrays = halo
