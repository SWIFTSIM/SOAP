#!/bin/env python

import time
import numpy as np
from mpi4py import MPI
import unyt
import h5py

import task_queue
import shared_mesh
import shared_array
import halo_tasks
from dataset_names import mass_dataset, ptypes_for_so_masses
from halo_tasks import process_halos
from mask_cells import mask_cells
import memory_use
import result_set

# Will label messages with time since run start
time_start = time.time()


def send_array(comm, arr, dest, tag):
    """Send an ndarray or unyt_array over MPI"""
    unyt_array = isinstance(arr, unyt.unyt_array)
    if unyt_array:
        units = arr.units
    else:
        units = None
    comm.send((unyt_array, arr.dtype, arr.shape, units), dest=dest, tag=tag)
    comm.Send(arr, dest=dest, tag=tag)


def recv_array(comm, src, tag):
    """Receive an ndarray or unyt_array over MPI"""
    unyt_array, dtype, shape, units = comm.recv(source=src, tag=tag)
    arr = np.ndarray(shape, dtype=dtype)
    if unyt_array:
        arr = unyt.unyt_array(arr, units=units)
    comm.Recv(arr, source=src, tag=tag)
    return arr


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


class ChunkTaskList:
    """
    Stores a list of ChunkTasks to be executed.
    """

    def __init__(self, cellgrid, so_cat, halo_prop_list):

        # Assign the input halos to chunk tasks
        task_id = so_cat.halo_arrays["task_id"]

        # Sort the halos by task ID
        idx = np.argsort(task_id)
        sorted_halo_arrays = {}
        for name in so_cat.halo_arrays:
            sorted_halo_arrays[name] = so_cat.halo_arrays[name][idx, ...]
        task_id = task_id[idx]

        # Find groups of halos with the same task ID
        unique_ids, offsets, counts = np.unique(
            task_id, return_index=True, return_counts=True
        )
        nr_chunks = len(counts)

        # Create the task list
        tasks = []
        for chunk_nr, (offset, count) in enumerate(zip(offsets, counts)):

            # Make the halo catalogue for this chunk
            task_halo_arrays = {}
            for name in sorted_halo_arrays:
                task_halo_arrays[name] = sorted_halo_arrays[name][
                    offset : offset + count, ...
                ]

            # Add an array to flag halos which have been processed
            task_halo_arrays["done"] = unyt.unyt_array(
                np.zeros(count, dtype=np.int8), dtype=np.int8
            )

            # Create the task for this chunk
            tasks.append(
                ChunkTask(task_halo_arrays, halo_prop_list, chunk_nr, nr_chunks)
            )

            # Report the size of the task
            print(f"Chunk {chunk_nr} has {count} halos")

        # Use number of halos as a rough estimate of cost.
        # Do tasks with the most halos first so we're not waiting for a few big jobs at the end.
        tasks.sort(key=lambda x: -x.halo_arrays["cofp"].shape[0])
        self.tasks = tasks


class ChunkTask:
    """
    Each ChunkTask is a set of halos in a patch of the simulation volume
    for which we want to evaluate spherical overdensity properties.

    Each ChunkTask is called collectively on all of the MPI ranks in one
    compute node.

    centres contains the halo centres
    indexes contains the index of each halo in the input catalogue
    radius  is the radius around each centre which we need to read in
    """

    def __init__(self, halo_arrays=None, halo_prop_list=None, chunk_nr=0, nr_chunks=1):

        self.halo_arrays = halo_arrays
        self.halo_prop_list = halo_prop_list
        self.shared = False
        self.chunk_nr = chunk_nr
        self.nr_chunks = nr_chunks

    def __call__(
        self,
        cellgrid,
        comm,
        inter_node_rank,
        timings,
        max_ranks_reading,
        scratch_file_format,
        xray_calculator):

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

        # Create object to store the results for this chunk
        nr_halos = len(self.halo_arrays["ID"].full)
        results = result_set.ResultSet(initial_size=max(1, nr_halos // comm_size))

        # Unpack arrays we need
        centre = self.halo_arrays["cofp"]
        read_radius = self.halo_arrays["read_radius"]
        done = self.halo_arrays["done"]

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
            nr_halos = len(self.halo_arrays["ID"].full)

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
                xray_calculator)
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

        # Return the names, dimensions and units of the quantities we computed
        # so that we can check they're consistent between chunks
        return results.get_metadata(comm)

    @classmethod
    def bcast(cls, comm, instance):
        """
        Broadcast a class instance over communicator comm.
        instance is only significant on rank 0. Other ranks
        don't have an instance yet, so this is a class method.

        Halo arrays are put in shared memory instead of copying
        to every rank.
        """

        comm_rank = comm.Get_rank()

        # Create a class instance on ranks which don't have one
        if comm_rank == 0:
            self = instance
        else:
            self = cls()

        # Broadcast the halo array names
        if comm_rank == 0:
            names = list(self.halo_arrays.keys())
        else:
            names = None
        names = comm.bcast(names)

        # Copy the arrays to shared memory
        halo_arrays = {}
        for name in names:
            if comm_rank == 0:
                arr = self.halo_arrays[name]
            else:
                arr = None
            halo_arrays[name] = share_array(comm, arr)
        self.halo_arrays = halo_arrays

        # Broadcast quantities to calculate
        self.halo_prop_list = comm.bcast(self.halo_prop_list)
        self.shared = True

        self.chunk_nr = comm.bcast(self.chunk_nr)
        self.nr_chunks = comm.bcast(self.nr_chunks)

        return self

    def send(self, comm, dest, tag):

        # Send quantities to compute
        comm.send(self.halo_prop_list, dest, tag)

        # Send halo array names
        names = list(self.halo_arrays.keys())
        comm.send(names, dest, tag)

        # Send halo arrays
        for name in names:
            send_array(comm, self.halo_arrays[name], dest, tag)

        comm.send(self.chunk_nr, dest, tag)
        comm.send(self.nr_chunks, dest, tag)

    @classmethod
    def recv(cls, comm, src, tag):

        # Receive quantities to compute
        halo_prop_list = comm.recv(source=src, tag=tag)

        # We might receive None instead of a task if there are no more tasks.
        # Just return None in that case.
        if halo_prop_list is None:
            return None

        # Receive halo array names
        names = comm.recv(source=src, tag=tag)

        # Receive halo arrays
        halo_arrays = {}
        for name in names:
            halo_arrays[name] = recv_array(comm, src, tag)

        chunk_nr = comm.recv(source=src, tag=tag)
        nr_chunks = comm.recv(source=src, tag=tag)

        # Construct the class instance
        return cls(halo_arrays, halo_prop_list, chunk_nr, nr_chunks)
