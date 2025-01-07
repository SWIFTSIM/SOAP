#!/bin/env python

import os
import time

import h5py
import numpy as np
import unyt

import shared_mesh
import shared_array
from dataset_names import mass_dataset, ptypes_for_so_masses
from halo_tasks import process_halos
from mask_cells import mask_cells
import memory_use
import result_set

# Will label messages with time since run start
time_start = time.time()


def share_array(comm, arr):
    """
    Take the array on rank 0 of communicator comm and copy it into
    shared memory. All ranks in comm must be on the same node.
    """
    unyt_array = isinstance(arr, unyt.unyt_array)
    comm_rank = comm.Get_rank()
    shape = None
    dtype = None
    units = None
    if comm_rank == 0:
        shape = list(arr.shape)
        dtype = arr.dtype
        if unyt_array:
            units = arr.units
    shape, dtype, units = comm.bcast((shape, dtype, units))
    if comm_rank > 0:
        shape[0] = 0
    shared_arr = shared_array.SharedArray(shape, dtype, comm, units)
    if comm_rank == 0:
        shared_arr.full[...] = arr[...]
    shared_arr.sync()
    comm.barrier()
    return shared_arr


def box_wrap(pos, ref_pos, boxsize):
    shift = ref_pos[None, :] - 0.5 * boxsize
    return (pos - shift) % boxsize + shift


class ChunkTask:
    """
    Each ChunkTask is a set of halos in a patch of the simulation volume
    for which we want to evaluate spherical overdensity properties.

    Each ChunkTask is called collectively on all of the MPI ranks in one
    compute node. The task imports the halos to be processed, reads in
    the required patch of the snapshot and computes halo properties.    
    """

    def __init__(self, halo_prop_list=None, chunk_nr=0, nr_chunks=1):

        self.halo_prop_list = halo_prop_list
        self.chunk_nr = chunk_nr
        self.nr_chunks = nr_chunks
        self.shared = False

    def __call__(
        self,
        cellgrid,
        so_cat,
        comm,
        inter_node_rank,
        timings,
        max_ranks_reading,
        scratch_file_format,
    ):

        # Get communicator size and rank within this compute node
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        def message(m):
            if inter_node_rank >= 0:
                print(
                    "[%8.1fs] %d: [%d/%d] %s"
                    % (
                        time.time() - time_start,
                        inter_node_rank,
                        self.chunk_nr,
                        self.nr_chunks,
                        m,
                    )
                )


        # The first rank on this node import the halos to be processed.
        # It also checks if this chunk has already been processed (by
        # a previous SOAP run that crashed).
        comm.barrier()
        t0_halos = time.time()
        result_metadata = None
        if comm_rank == 0:
            # Receive arrays
            self.halo_arrays = so_cat.request_chunk(self.chunk_nr)
            # Add a done flag for each halo
            nr_halos = len(self.halo_arrays["index"])
            self.halo_arrays["done"] = unyt.unyt_array(
                np.zeros(nr_halos, dtype=np.int8), dtype=np.int8
            )
            # Will need to broadcast names of the halo properties
            names = list(self.halo_arrays.keys())

            # Sort halos based on their number of bound particles
            # We do this since larger halos will take longer to process
            # and so we want to do them first
            order = np.argsort(self.halo_arrays['nr_bound_part'])[::-1]
            for name in names:
                self.halo_arrays[name] = self.halo_arrays[name][order]

            chunk_file_already_exists = False
            # Check if the chunk file exists, was fully written, and has the correct objects
            filename = scratch_file_format % {"file_nr": self.chunk_nr}
            if os.path.exists(filename):
                try:
                    with h5py.File(filename, 'r') as outfile:
                        chunk_file_already_exists = outfile.attrs.get('Write complete', False)
                        index = np.sort(outfile['InputHalos/HaloCatalogueIndex'][:])
                        file_calc_names = sorted(outfile.attrs["calc_names"].tolist())
                    # Check we have the correct halo indices
                    if not np.all(index == np.sort(self.halo_arrays['index'].value)):
                        chunk_file_already_exists = False
                    # Check halo properties are the same
                    calc_names = sorted([hp.name for hp in self.halo_prop_list])
                    if len(calc_names) != len(file_calc_names):
                        chunk_file_already_exists = False
                    for name1, name2 in zip(calc_names, file_calc_names):
                        if name1 != name2:
                            chunk_file_already_exists = False
                except Exception as e:
                    # Blanket catch in case there are i/o issues with the chunk file
                    chunk_file_already_exists = False

            # File is valid, let's extracting the metadata from it
            if chunk_file_already_exists:
                result_metadata = result_set.get_metadata_from_chunk_file(filename, self.halo_prop_list, cellgrid.snap_unit_registry)

        else:
            chunk_file_already_exists = None
            names = None
            self.halo_arrays = {}
        names = comm.bcast(names)

        chunk_file_already_exists = comm.bcast(chunk_file_already_exists)
        if chunk_file_already_exists:
            message(f'Using pre-existing file for chunk')
            return result_metadata

        # Then we copy the halo arrays into shared memory
        for name in names:
            if comm_rank == 0:
                arr = self.halo_arrays[name]
            else:
                arr = None
            self.halo_arrays[name] = share_array(comm, arr)
        t1_halos = time.time()
        nr_halos = len(self.halo_arrays["index"].full)
        self.shared = True  # So we know to explicitly free the shared memory regions
        message(
            "receiving %d halos for chunk %d took %.2fs"
            % (nr_halos, self.chunk_nr, t1_halos - t0_halos)
        )

        # Create object to store the results for this chunk
        results = result_set.ResultSet(initial_size=max(1, nr_halos // comm_size))

        # Unpack arrays we need
        centre = self.halo_arrays["cofp"]
        read_radius = self.halo_arrays["read_radius"]
        done = self.halo_arrays["done"]

        # Repeat until all halos have been done
        task_time_all_iterations = 0.0
        while True:

            # Find the region we need to read in, allowing for particles outside their cells
            comm.barrier()
            t0_mask = time.time()
            mask = mask_cells(comm, cellgrid, centre.full, read_radius.full, done.full)
            nr_cells = np.sum(mask == True)
            comm.barrier()
            t1_mask = time.time()
            message(
                "identified %d cells to read in %.2fs" % (nr_cells, t1_mask - t0_mask)
            )
            nr_halos = len(self.halo_arrays["index"].full)

            # Get the cosmology info from the input snapshot
            critical_density = cellgrid.critical_density
            mean_density = cellgrid.mean_density
            a = cellgrid.a
            z = cellgrid.z
            boxsize = cellgrid.boxsize

            # Find reference position for box wrapping:
            # Coordinates will be wrapped in order to minimize the size of the volume we place
            # the mesh over. TODO: use a tree instead so that this isn't necessary.
            pos_min = np.amin(centre.full, axis=0)
            pos_max = np.amax(centre.full, axis=0)
            ref_pos = (pos_min + pos_max) / 2

            # Find all particle properties we need to read in:
            # For each particle type this is the union of the quantities
            # needed for each calculation.
            if comm_rank == 0:
                properties = {}
                # Check if we need to compute spherical overdensity masses
                need_so = False
                for halo_prop in self.halo_prop_list:
                    if (
                        halo_prop.mean_density_multiple is not None
                        or halo_prop.critical_density_multiple is not None
                    ):
                        need_so = True
                # If we're computing SO masses, we need masses and positions of all particle types
                if need_so:
                    for ptype in ptypes_for_so_masses:
                        properties[ptype] = set(["Coordinates", mass_dataset(ptype)])
                # Add particle properties needed for halo property calculations
                for halo_prop in self.halo_prop_list:
                    for ptype in halo_prop.particle_properties:
                        if ptype not in properties:
                            properties[ptype] = set()
                        properties[ptype] = properties[ptype].union(
                            halo_prop.particle_properties[ptype]
                        )
                for ptype in properties:
                    properties[ptype] = list(properties[ptype])
                try:
                    cellgrid.check_datasets_exist(properties, self.halo_prop_list)
                except KeyError as err_msg:
                    print(err_msg)
                    comm.Abort(1)
            else:
                properties = None
            properties = comm.bcast(properties)

            # Read in particles in the required region
            comm.barrier()
            t0_read = time.time()
            data = cellgrid.read_masked_cells_to_shared_memory(
                properties, mask, comm, max_ranks_reading
            )
            comm.barrier()
            t1_read = time.time()

            # Count how many particles we read in
            nr_parts = 0
            for ptype in data:
                name = mass_dataset(ptype)
                nr_parts += data[ptype][name].full.shape[0]
            if nr_parts == 0:
                # Should be impossible: all halos have particles!
                raise Exception("Task has zero particles?!")

            # Compute number of bytes read
            nr_bytes = 0
            for ptype in data:
                for name in data[ptype]:
                    nr_bytes += data[ptype][name].full.nbytes
            nr_mb = nr_bytes / (1024 ** 2)
            rate = nr_mb / (t1_read - t0_read)
            message(
                "read in %d particles in %.1fs = %.1fMB/s (uncompressed)"
                % (nr_parts, t1_read - t0_read, rate)
            )

            # Do periodic shift of particles to copies nearest the reference point
            for ptype in data:
                if "Coordinates" in data[ptype]:
                    data[ptype]["Coordinates"].local[:] = box_wrap(
                        data[ptype]["Coordinates"].local[:], ref_pos, boxsize
                    )

            # Build the mesh for each particle type
            comm.barrier()
            t0_mesh = time.time()
            mesh = {}
            for ptype in data:
                # Find the particle coordinates
                pos = data[ptype]["Coordinates"]
                nr_parts_type = pos.full.shape[0]
                # Compute mesh resolution to give roughly fixed number of particles per cell
                target_nr_per_cell = 1000
                max_resolution = 256
                resolution = int((nr_parts_type / target_nr_per_cell) ** (1.0 / 3.0))
                resolution = min(max(resolution, 1), max_resolution)
                # Build the mesh for this particle type
                mesh[ptype] = shared_mesh.SharedMesh(comm, pos, resolution)
            comm.barrier()
            t1_mesh = time.time()
            message("constructing shared mesh took %.1fs" % (t1_mesh - t0_mesh))

            # Report remaining memory after particles have been read in and mesh has been built
            total_mem_gb, free_mem_gb = memory_use.get_memory_use()
            if total_mem_gb is not None:
                message(
                    "node has %.1fGB of %.1fGB memory free"
                    % (free_mem_gb, total_mem_gb)
                )

            # Calculate the halo properties
            t0_halos = time.time()
            total_time, task_time, nr_left, nr_done = process_halos(
                comm,
                cellgrid.snap_unit_registry,
                data,
                mesh,
                self.halo_prop_list,
                critical_density,
                mean_density,
                boxsize,
                self.halo_arrays,
                results,
            )
            t1_halos = time.time()
            task_time_all_iterations += task_time
            dead_time_fraction = 1.0 - comm.allreduce(task_time) / comm.allreduce(
                total_time
            )
            message(
                "processing %d of %d halos on %d ranks took %.1fs (dead time frac.=%.2f)"
                % (
                    nr_done,
                    nr_halos,
                    comm_size,
                    t1_halos - t0_halos,
                    dead_time_fraction,
                )
            )
            # Free the shared particle data
            for ptype in data:
                for name in data[ptype]:
                    data[ptype][name].free()
            del data

            # Free the shared mesh
            for ptype in mesh:
                mesh[ptype].free()
            del mesh

            # Check if we're done
            if nr_left == 0:
                break
            else:
                message("need to repeat chunk for %d halos" % nr_left)

        # Free shared halo catalogue
        if self.shared:
            for name in sorted(self.halo_arrays):
                self.halo_arrays[name].free()

        # MPI ranks with results write the output file in collective mode
        colour = 0 if len(results) > 0 else 1
        comm_have_results = comm.Split(colour, comm_rank)
        if len(results) > 0:
            filename = scratch_file_format % {"file_nr": self.chunk_nr}
            with h5py.File(
                filename, "w", driver="mpio", comm=comm_have_results
            ) as outfile:
                results.collective_write(outfile, comm_have_results)
        comm_have_results.Free()

        # Store time taken for this task
        timings.append(task_time_all_iterations)

        # Write metadata in case this file is used for restarts
        if comm_rank == 0:
            with h5py.File(filename, 'a') as outfile:
                units = outfile.create_group("Units")
                for name, value in cellgrid.swift_units_group.items():
                    units.attrs[name] = [value]
                calc_names = sorted([hp.name for hp in self.halo_prop_list])
                outfile.attrs["calc_names"] = calc_names
                outfile.attrs['Write complete'] = True

        # Return the names, dimensions and units of the quantities we computed
        # so that we can check they're consistent between chunks
        return results.get_metadata(comm)
