#!/bin/env python

import time

import numpy as np
import unyt

from dataset_names import mass_dataset, ptypes_for_so_masses
from halo_properties import ReadRadiusTooSmallError
import shared_array
import result_set
import halo_properties
from property_table import PropertyTable


# Factor by which to increase search radius when looking for density threshold
SEARCH_RADIUS_FACTOR = 1.2

# Factor by which to increase the region read in around a halo if too small
READ_RADIUS_FACTOR = 1.5

# Radius in Mpc at which we report halos which have a large search radius
REPORT_RADIUS = 20.0


def process_single_halo(
    mesh,
    unit_registry,
    data,
    halo_prop_list,
    critical_density,
    mean_density,
    boxsize,
    input_halo,
    target_density,
    xray_calc,
):
    """
    This computes properties for one halo and runs on a single
    MPI rank. Result is a dict of properties of the form

    halo_result[property_name] = (unyt_array, description)

    where the property_name will be used as the HDF5 dataset name
    in the output and the units of the unyt_array determine the unit
    attributes.

    Two radii are passed in:

    search_radius is an initial guess at the radius we need to fall
    below the specified overdensity threshold. read_radius is the
    radius within which we have all of the particles. If we find that
    the density within read_radius is above the threshold, then we
    didn't read in a large enough region.

    Returns None if we need to try again with a larger region.
    """

    swift_mpc = unyt.Unit("swift_mpc", registry=unit_registry)
    snap_length = unyt.Unit("snap_length", registry=unit_registry)
    snap_mass = unyt.Unit("snap_mass", registry=unit_registry)
    snap_density = snap_mass / (snap_length ** 3)

    # Record which calculations are still to do for this halo
    halo_prop_done = np.zeros(len(halo_prop_list), dtype=bool)

    # Dict to store the results
    halo_result = {}

    # Loop until we fall below the required density
    current_radius = input_halo["search_radius"]
    while True:

        # Sanity checks on the radius
        assert current_radius <= input_halo["read_radius"]
        if current_radius > REPORT_RADIUS * swift_mpc:
            print(
                f"Halo ID={input_halo['ID']} has large search radius {current_radius}"
            )

        # Find the mass within the search radius
        mass_total = unyt.unyt_quantity(0.0, units=snap_mass)
        idx = {}
        for ptype in data:
            pos = data[ptype]["Coordinates"]
            idx[ptype] = mesh[ptype].query_radius_periodic(
                input_halo["cofp"], current_radius, pos, boxsize
            )
            if ptype in ptypes_for_so_masses:
                mass = data[ptype][mass_dataset(ptype)]
                mass_total += np.sum(mass.full[idx[ptype]], dtype=float)

        # Find mean density in the search radius
        density = mass_total / (4.0 / 3.0 * np.pi * current_radius ** 3)

        # If we've reached the target density, we can try to compute halo properties
        max_physical_radius_mpc = (
            0.0  # Will store the largest physical radius requested by any calculation
        )
        if target_density is None or density <= target_density:

            # Extract particles in this halo
            particle_data = {}
            for ptype in data:
                particle_data[ptype] = {}
                for name in data[ptype]:
                    particle_data[ptype][name] = data[ptype][name].full[idx[ptype], ...]

            # Wrap coordinates to copy closest to the halo centre
            for ptype in particle_data:
                pos = particle_data[ptype]["Coordinates"]
                # Shift halo to box centre, wrap all particles into box, shift halo back
                offset = input_halo["cofp"] - 0.5 * boxsize
                pos[:, :] = ((pos - offset) % boxsize) + offset

            ptype = "PartType0"
            idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n = xray_calc.find_indices(
                particle_data[ptype]["Densities"],
                particle_data[ptype]["Temperatures"],
                particle_data[ptype]["SmoothedElementMassFractions"],
                particle_data[ptype]["Masses"],
                fill_value=0,
            )

            xray_bands = ["erosita-low", "erosita-high", "ROSAT"]
            observing_types = [
                "energies_intrinsic",
                "energies_intrinsic",
                "energies_intrinsic",
            ]
            particle_data[ptype]["XrayLuminosities"] = xray_calc.interpolate_X_Ray(
                idx_he,
                idx_T,
                idx_n,
                t_z,
                d_z,
                t_T,
                d_T,
                t_nH,
                d_nH,
                t_He,
                d_He,
                abundance_to_solar,
                joint_mask,
                volumes,
                data_n,
                bands=xray_bands,
                observing_types=observing_types,
                fill_value=0,
            ).to(particle_data[ptype]["XrayLuminosities"].units)

            observing_types = [
                "photons_intrinsic",
                "photons_intrinsic",
                "photons_intrinsic",
            ]
            particle_data[ptype][
                "XrayPhotonLuminosities"
            ] = xray_calc.interpolate_X_Ray(
                idx_he,
                idx_T,
                idx_n,
                t_z,
                d_z,
                t_T,
                d_T,
                t_nH,
                d_nH,
                t_He,
                d_He,
                abundance_to_solar,
                joint_mask,
                volumes,
                data_n,
                bands=xray_bands,
                observing_types=observing_types,
                fill_value=0,
            ).to(
                particle_data[ptype]["XrayPhotonLuminosities"].units
            )

            observing_types = [
                "energies_intrinsic_restframe",
                "energies_intrinsic_restframe",
                "energies_intrinsic_restframe",
            ]
            particle_data[ptype][
                "XrayLuminositiesRestframe"
            ] = xray_calc.interpolate_X_Ray(
                idx_he,
                idx_T,
                idx_n,
                t_z,
                d_z,
                t_T,
                d_T,
                t_nH,
                d_nH,
                t_He,
                d_He,
                abundance_to_solar,
                joint_mask,
                volumes,
                data_n,
                bands=xray_bands,
                observing_types=observing_types,
                fill_value=0,
            ).to(
                particle_data[ptype]["XrayLuminosities"].units
            )

            observing_types = [
                "photons_intrinsic_restframe",
                "photons_intrinsic_restframe",
                "photons_intrinsic_restframe",
            ]
            particle_data[ptype][
                "XrayPhotonLuminositiesRestframe"
            ] = xray_calc.interpolate_X_Ray(
                idx_he,
                idx_T,
                idx_n,
                t_z,
                d_z,
                t_T,
                d_T,
                t_nH,
                d_nH,
                t_He,
                d_He,
                abundance_to_solar,
                joint_mask,
                volumes,
                data_n,
                bands=xray_bands,
                observing_types=observing_types,
                fill_value=0,
            ).to(
                particle_data[ptype]["XrayPhotonLuminosities"].units
            )

            # Try to compute properties of this halo which haven't been done yet
            for prop_nr, halo_prop in enumerate(halo_prop_list):
                if halo_prop_done[prop_nr]:
                    # Already have the result for this one
                    continue
                try:
                    halo_prop.calculate(
                        input_halo, current_radius, particle_data, halo_result
                    )
                except ReadRadiusTooSmallError:
                    # Search radius was too small, so will need to try again with a larger radius.
                    max_physical_radius_mpc = max(
                        max_physical_radius_mpc, halo_prop.physical_radius_mpc
                    )
                    break
                except FloatingPointError as fpe:
                    # Calculation cause a floating point exception.
                    # Output the halo ID so we can debug this.
                    print(
                        f"Halo ID={input_halo['ID']} encountered a floating point error"
                    )
                    raise
                else:
                    # The property calculation worked!
                    halo_prop_done[prop_nr] = True

            # If we computed all of the properties, we're done with this halo
            if np.all(halo_prop_done):
                break

        # Either the density is still too high or the property calculation failed.
        search_radius = input_halo["search_radius"]
        required_radius = (max_physical_radius_mpc * swift_mpc).to(search_radius.units)
        if required_radius > input_halo["read_radius"]:
            # A calculation has set its physical_radius_mpc larger than the region
            # which we read in, so we can't process the halo on this iteration regardless
            # of the current radius.
            input_halo["search_radius"] = max(search_radius, required_radius)
            return None
        elif current_radius >= input_halo["read_radius"]:
            # The current radius has exceeded the region read in. Will need to redo this
            # halo using current_radius as the starting point for the next iteration.
            input_halo["search_radius"] = max(search_radius, current_radius)
            return None
        else:
            # We still have a large enough region in memory that we can try a larger radius
            current_radius = min(
                current_radius * SEARCH_RADIUS_FACTOR, input_halo["read_radius"]
            )
            current_radius = max(current_radius, required_radius)

    # In case we're not doing any calculations with a target density
    if target_density is None:
        target_density = density * 0.0

    # Add the halo index to the result set
    for vrkey in PropertyTable.vr_properties:
        vrprops = PropertyTable.full_property_list[f"VR{vrkey}"]
        vrname = vrprops[0]
        vrdescription = vrprops[4]
        halo_result[f"VR/{vrname}"] = (input_halo[vrkey], vrdescription)

    # Store search radius and density within that radius
    halo_result["SearchRadius/search_radius"] = (
        current_radius,
        "Search radius for property calculation",
    )
    halo_result["SearchRadius/density_in_search_radius"] = (
        density.to(snap_density),
        "Density within the search radius",
    )
    halo_result["SearchRadius/target_density"] = (
        target_density.to(snap_density),
        "Target density for property calculation",
    )

    return halo_result


def process_halos(
    comm,
    unit_registry,
    data,
    mesh,
    halo_prop_list,
    critical_density,
    mean_density,
    boxsize,
    halo_arrays,
    results,
    xray_calculator,
):
    """
    This uses all of the MPI ranks on one compute node to compute halo properties
    for a single "chunk" of the simulation.

    Each rank returns a dict with the properties of the halos it processed.
    The dict keys are the property names. The values are tuples containing
    (unyt_array, description).

    Halos where done=1 are not processed and don't generate an entry in the
    output arrays.

    If a halo can't be processed because we didn't read enough particles,
    its read_radius is doubled but no entry is generated in the output.
    """
    # Compute density threshold at this redshift in comoving units:
    # This determines the size of the sphere we use for all other SO quantities.
    # We need to find the minimum density required for any of the halo property
    # calculations
    target_density = None
    for halo_prop in halo_prop_list:
        # Ensure target density is no greater than mean density multiple
        if halo_prop.mean_density_multiple is not None:
            density = halo_prop.mean_density_multiple * mean_density
            if target_density is None or density < target_density:
                target_density = density
        # Ensure target density is no greater than critical density multiple
        if halo_prop.critical_density_multiple is not None:
            density = halo_prop.critical_density_multiple * critical_density
            if target_density is None or density < target_density:
                target_density = density

    # Allocate shared storage for a single integer and initialize to zero
    if comm.Get_rank() == 0:
        local_shape = (1,)
    else:
        local_shape = (0,)
    next_task = shared_array.SharedArray(local_shape, np.int64, comm)
    if comm.Get_rank() == 0:
        next_task.full[0] = 0
    next_task.sync()

    # Start the clock
    comm.barrier()
    t0_all = time.time()

    # Count halos to do
    nr_halos_left = comm.allreduce(np.sum(halo_arrays["done"].local.value == 0))

    # Loop until all halos are done
    nr_halos = len(halo_arrays["index"].full)
    nr_done_this_rank = 0
    nr_halos_this_rank_guess = int(nr_halos_left / comm.Get_size() * 1.5)
    task_time = 0.0
    while True:

        # Get a task by atomic incrementing the counter. Don't know how to do
        # an atomic fetch and add in python, so will use MPI RMA calls!
        task_to_do = np.ndarray(1, dtype=np.int64)
        one = np.ones(1, dtype=np.int64)
        next_task.win.Lock(0)
        next_task.win.Fetch_and_op(one, task_to_do, 0)
        next_task.win.Unlock(0)
        task_to_do = int(task_to_do)

        # Execute the task, if there's one left
        if task_to_do < nr_halos:
            t0_task = time.time()

            # Skip halos we already did
            if halo_arrays["done"].full[task_to_do].value == 0:

                # Extract this halo's VR information (centre, radius, index etc)
                input_halo = {}
                for name in halo_arrays:
                    input_halo[name] = halo_arrays[name].full[task_to_do, ...].copy()

                # Fetch the results for this particular halo
                halo_result = process_single_halo(
                    mesh,
                    unit_registry,
                    data,
                    halo_prop_list,
                    critical_density,
                    mean_density,
                    boxsize,
                    input_halo,
                    target_density,
                    xray_calculator,
                )
                if halo_result is not None:
                    # Store results and flag this halo as done
                    results.append(halo_result)
                    nr_done_this_rank += 1
                    halo_arrays["done"].full[task_to_do] = 1
                else:
                    # We didn't read in a large enough region. Update the shared radius
                    # arrays so that we read a larger region next time and start the
                    # search from whatever radius we had reached this time.
                    new_read_radius = max(
                        input_halo["read_radius"] * READ_RADIUS_FACTOR,
                        input_halo["search_radius"],
                    )
                    # Set the radius around the halo to read in
                    halo_arrays["read_radius"].full[task_to_do] = new_read_radius
                    # Set the initial guess at the radius we need
                    halo_arrays["search_radius"].full[task_to_do] = input_halo[
                        "search_radius"
                    ]

            t1_task = time.time()
            task_time += t1_task - t0_task
        else:
            # We ran out of halos to do
            break

    # Free the shared task counter
    next_task.free()

    # Count halos left to do
    comm.barrier()
    nr_halos_left = comm.allreduce(np.sum(halo_arrays["done"].local.value == 0))

    # Stop the clock
    comm.barrier()
    t1_all = time.time()

    return t1_all - t0_all, task_time, nr_halos_left, comm.allreduce(nr_done_this_rank)
