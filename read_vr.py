#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.util
import virgo.mpi.parallel_hdf5
import virgo.mpi.gather_array as ga
import virgo.mpi.parallel_sort as ps


from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def read_vr_datasets(vr_basename, file_type, datasets, return_file_nr=None):
    """
    Parallel read of datasets from VR split over any number of files
    """
    # Make a format string for the filename
    filename_format = vr_basename + "." + file_type + ".%(file_nr)d"

    # Open the file
    vr_file = virgo.mpi.parallel_hdf5.MultiFile(
        filename_format, file_nr_dataset="Num_of_files"
    )

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
    files_on_rank = virgo.mpi.parallel_hdf5.assign_files(nr_files, comm_size)
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
