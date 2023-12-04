#!/bin/env python

import numpy as np
import h5py

import virgo.mpi.parallel_hdf5 as phdf5


def read_vr_group_sizes(basename, suffix, comm):
    """
    Compute number of bound and unbound particles in each group. This is much
    more complicated than it should be because VR doesn't write out bound and
    unbound sizes. Instead we have to compute them in an awkward way.
    
    For groups which are not last in the file, group i has size
    Offset[i+1]-Offset[i]. For the last group in each file we need to know the
    number of particles in the corresponding catalog_particles file to compute
    its size.
    
    basename: VR output filename without trailing .properties[.0] or similar
    suffix: format string to add file number suffix, if necesary
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    def vr_filename(file_type):
        return f"{basename}.{file_type}" + suffix

    # Read numbers of bound and unbound particles in each file
    if comm_rank == 0:
        nr_bound_part_in_file = []
        nr_unbound_part_in_file = []
        nr_groups_in_file = []
        nr_files = 1
        file_nr = 0
        while file_nr < nr_files:
            # Read number of bound particles per file
            filename = vr_filename("catalog_particles") % {"file_nr": file_nr}
            with h5py.File(filename, "r") as infile:
                nr_bound_part_in_file.append(
                    int(infile["Num_of_particles_in_groups"][...])
                )
                nr_files = int(infile["Num_of_files"][...])
            # Read number of unbound particles per file
            filename = vr_filename("catalog_particles.unbound") % {"file_nr": file_nr}
            with h5py.File(filename, "r") as infile:
                nr_unbound_part_in_file.append(
                    int(infile["Num_of_particles_in_groups"][...])
                )
            # Read number of groups per file
            filename = vr_filename("catalog_groups") % {"file_nr": file_nr}
            with h5py.File(filename, "r") as infile:
                nr_groups_in_file.append(int(infile["Num_of_groups"][...]))
            file_nr += 1
        nr_bound_part_in_file = np.asarray(nr_bound_part_in_file, dtype=int)
        nr_unbound_part_in_file = np.asarray(nr_unbound_part_in_file, dtype=int)
        nr_groups_in_file = np.asarray(nr_groups_in_file, dtype=int)
    else:
        nr_files = None
        nr_bound_part_in_file = None
        nr_unbound_part_in_file = None
        nr_groups_in_file = None
    (
        nr_bound_part_in_file,
        nr_unbound_part_in_file,
        nr_groups_in_file,
        nr_files,
    ) = comm.bcast(
        (nr_bound_part_in_file, nr_unbound_part_in_file, nr_groups_in_file, nr_files)
    )

    # Read offsets from the catalog_groups files
    mf = phdf5.MultiFile(vr_filename("catalog_groups"), file_nr_dataset="Num_of_files")
    offset_bound, offset_unbound, group_size = mf.read(
        ("Offset", "Offset_unbound", "Group_Size"), unpack=True
    )

    # Allocate storage for particle numbers
    nr_halos_local = len(offset_bound)
    nr_parts_bound = np.zeros(nr_halos_local, dtype=int)
    nr_parts_unbound = np.zeros(nr_halos_local, dtype=int)

    # Identify groups which are the last in their file and compute their sizes
    is_last = np.zeros(nr_halos_local, dtype=bool)
    last_in_file = (
        np.cumsum(nr_groups_in_file) - 1
    )  # Global index of last halo in each file
    local_offset = (
        comm.scan(nr_halos_local) - nr_halos_local
    )  # Nr. of halos on previous MPI ranks
    for file_nr in range(nr_files):
        if nr_groups_in_file[file_nr] > 0:
            local_index = last_in_file[file_nr] - local_offset
            if local_index >= 0 and local_index < nr_halos_local:
                is_last[local_index] = True
                nr_parts_bound[local_index] = (
                    nr_bound_part_in_file[file_nr] - offset_bound[local_index]
                )
                nr_parts_unbound[local_index] = (
                    nr_unbound_part_in_file[file_nr] - offset_unbound[local_index]
                )

    # Find the offsets for the first halo on each MPI rank
    if nr_halos_local > 0:
        first_offset_bound = offset_bound[0]
        first_offset_unbound = offset_unbound[0]
    else:
        first_offset_bound = None
        first_offset_unbound = None
    first_offset_bound = comm.allgather(first_offset_bound)
    first_offset_unbound = comm.allgather(first_offset_unbound)

    # Find the first offset on the next rank which has nr_halos_local > 0.
    # This will be None if no later ranks have any halos.
    next_offset_bound = None
    next_offset_unbound = None
    for rank in range(comm_rank + 1, comm_size):
        assert (first_offset_bound[rank] is None) == (
            first_offset_unbound[rank] is None
        )
        if first_offset_bound[rank] is not None:
            next_offset_bound = first_offset_bound[rank]
            next_offset_unbound = first_offset_unbound[rank]
            break

    # Now, for halos which are not last in their file and not last on their MPI rank we can compute their size
    for halo_nr in range(nr_halos_local - 1):
        if not (is_last[halo_nr]):
            nr_parts_bound[halo_nr] = offset_bound[halo_nr + 1] - offset_bound[halo_nr]
            nr_parts_unbound[halo_nr] = (
                offset_unbound[halo_nr + 1] - offset_unbound[halo_nr]
            )

    # Finally, compute size of last halo on this rank if we didn't already
    if nr_halos_local > 0:
        if not (is_last[-1]):
            nr_parts_bound[-1] = next_offset_bound - offset_bound[-1]
            nr_parts_unbound[-1] = next_offset_unbound - offset_unbound[-1]

    # Consistency check: we know that the total number of bound+unbound particles should be group_size
    assert len(nr_parts_bound) == len(group_size)
    assert len(nr_parts_unbound) == len(group_size)
    nr_wrong = np.sum(nr_parts_bound + nr_parts_unbound != group_size)
    nr_wrong = comm.allreduce(nr_wrong)
    if nr_wrong > 0:
        index_wrong = np.arange(nr_halos_local, dtype=int)[
            nr_parts_bound + nr_parts_unbound != group_size
        ]
        for rank in range(comm_size):
            if comm_rank == rank:
                print(f"Rank {comm_rank}:")
                for i in index_wrong:
                    print(
                        f"  Halo {i} of {nr_halos_local}: {nr_parts_bound[i]} + {nr_parts_unbound[i]} != {group_size[i]}"
                    )
            comm.barrier()
        raise RuntimeError(
            f"Number of particles in halos has been computed wrongly for {nr_wrong} halos!"
        )

    return nr_parts_bound, nr_parts_unbound


if __name__ == "__main__":

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    import sys

    snap_nr = int(sys.argv[1])

    basename = f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_{snap_nr:04d}/vr_catalogue_{snap_nr:04d}"
    # basename = f"/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/VR/catalogue_{snap_nr:04d}/vr_catalogue_{snap_nr:04d}"
    suffix = ".%(file_nr)d"

    nr_parts_bound, nr_parts_unbound = read_vr_group_sizes(basename, suffix, comm)

    comm.barrier()
    nr_halos_total = comm.allreduce(len(nr_parts_bound))
    if comm_rank == 0:
        print(f"Read {nr_halos_total} halos")
