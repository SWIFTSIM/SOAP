#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
from glob import glob
import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

# Maximum number of particle types
NTYPEMAX = 5

def message(s):
    if comm_rank == 0:
        print(s)


def exchange_array(arr, dest, comm):
    """
    Carry out an alltoallv on the supplied array, given the MPI rank
    to send each element to.
    """
    order = np.argsort(dest)
    sendbuf = arr[order]
    send_count = np.bincount(dest, minlength=comm_size)
    send_offset = np.cumsum(send_count) - send_count
    recv_count = np.zeros_like(send_count)
    comm.Alltoall(send_count, recv_count)
    recv_offset = np.cumsum(recv_count) - recv_count
    recvbuf = np.ndarray(recv_count.sum(), dtype=arr.dtype)
    psort.my_alltoallv(
        sendbuf, send_count, send_offset, recvbuf, recv_count, recv_offset, comm=comm
    )
    return recvbuf

def find_matching_halos(
    cat1_length,
    cat1_offset,
    cat1_ids,
    cat1_types,
    host_index1,
    cat2_length,
    cat2_offset,
    cat2_ids,
    cat2_types,
    host_index2,
    max_nr_particles,
    use_type,
    field_only,
):

    # Decide range of halos in cat1 which we'll store on each rank:
    # This is used to partition the result between MPI ranks.
    nr_cat1_tot = comm.allreduce(len(cat1_length))
    nr_cat1_per_rank = nr_cat1_tot // comm_size
    if comm_rank < comm_size - 1:
        nr_cat1_local = nr_cat1_per_rank
    else:
        nr_cat1_local = nr_cat1_tot - (comm_size - 1) * nr_cat1_per_rank

    # Find group membership for particles in the first catalogue:
    cat1_grnr_in_cat1 = read_vr.vr_group_membership_from_ids(
        cat1_length, cat1_offset, cat1_ids
    )

    # Find group membership for particles in the second catalogue
    cat2_grnr_in_cat2 = read_vr.vr_group_membership_from_ids(
        cat2_length, cat2_offset, cat2_ids
    )

    # Clear group membership for particles of types we're not using in the first catalogue
    discard = (use_type[cat1_types] == False) | (cat1_grnr_in_cat1 < 0)
    cat1_grnr_in_cat1[discard] = -1

    # If we're only matching to field halos, then any particles in the second catalogue which
    # belong to a halo with hostHaloID != -1 need to have their group membership reset to their
    # host halo.
    if field_only:
        # Find particles in halos in cat2
        in_halo = cat2_grnr_in_cat2 >= 0
        # Fetch host halo array index for each particle in cat2, or -1 if not in a halo
        particle_host_index = -np.ones_like(cat2_grnr_in_cat2)
        particle_host_index[in_halo] = psort.fetch_elements(
            host_index2, cat2_grnr_in_cat2[in_halo], comm=comm
        )
        # Where a particle's halo has a host halo, set its group membership to be the host halo
        have_host = particle_host_index >= 0
        cat2_grnr_in_cat2[have_host] = particle_host_index[have_host]

    # Discard particles which are in no halo from each catalogue
    in_group = cat1_grnr_in_cat1 >= 0
    cat1_ids = cat1_ids[in_group]
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[in_group]
    in_group = cat2_grnr_in_cat2 >= 0
    cat2_ids = cat2_ids[in_group]
    cat2_grnr_in_cat2 = cat2_grnr_in_cat2[in_group]

    # Now we need to identify the first max_nr_particles remaining particles for each
    # halo in catalogue 1. First, find the ranking of each particle within the part of
    # its group which is stored on this MPI rank. First particle in a group has rank 0.
    unique_grnr, unique_index, unique_count = np.unique(
        cat1_grnr_in_cat1, return_index=True, return_counts=True
    )
    cat1_rank_in_group = -np.ones_like(cat1_grnr_in_cat1)
    for ui, uc in zip(unique_index, unique_count):
        cat1_rank_in_group[ui : ui + uc] = np.arange(uc, dtype=int)
    assert np.all(cat1_rank_in_group >= 0)

    # Then for the first group on each rank we'll need to add the total number of particles in
    # the same group on all lower numbered ranks. Since the particles are sorted by group this
    # can only ever be the last group on each lower numbered rank.
    if len(unique_grnr) > 0:
        # This rank has at least one particle in a group. Store indexes of first and last groups
        # and the number of particles from the last group which are stored on this rank.
        assert unique_index[0] == 0
        first_grnr = unique_grnr[0]
        last_grnr = unique_grnr[-1]
        last_grnr_count = unique_count[-1]
    else:
        # This rank has no particles in groups
        first_grnr = -1
        last_grnr = -1
        last_grnr_count = 0
    all_last_grnr = comm.allgather(last_grnr)
    all_last_grnr_count = comm.allgather(last_grnr_count)
    # Loop over lower numbered ranks
    for rank_nr in range(comm_rank):
        if first_grnr >= 0 and all_last_grnr[rank_nr] == first_grnr:
            cat1_rank_in_group[: unique_count[0]] += all_last_grnr_count[rank_nr]

    # Only keep the first max_nr_particles remaining particles in each group in catalogue 1
    keep = cat1_rank_in_group < max_nr_particles
    cat1_ids = cat1_ids[keep]
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[keep]

    # For each particle ID in catalogue 1, try to find the same particle ID in catalogue 2
    ptr = psort.parallel_match(cat1_ids, cat2_ids, comm=comm)
    matched = ptr >= 0

    # For each particle ID in catalogue 1, fetch the group membership of the matching ID in catalogue 2
    cat1_grnr_in_cat2 = -np.ones_like(cat1_grnr_in_cat1)
    cat1_grnr_in_cat2[matched] = psort.fetch_elements(cat2_grnr_in_cat2, ptr[matched])

    # Discard unmatched particles
    cat1_grnr_in_cat1 = cat1_grnr_in_cat1[matched]
    cat1_grnr_in_cat2 = cat1_grnr_in_cat2[matched]

    # Get sorted, unique (grnr1, grnr2) combinations and counts of how many instances of each we have
    assert np.all(cat1_grnr_in_cat1 < 2 ** 32)
    assert np.all(cat1_grnr_in_cat1 >= 0)
    assert np.all(cat1_grnr_in_cat2 < 2 ** 32)
    assert np.all(cat1_grnr_in_cat2 >= 0)
    sort_key = (cat1_grnr_in_cat1.astype(np.uint64) << 32) + cat1_grnr_in_cat2.astype(
        np.uint64
    )
    unique_value, cat1_count = psort.parallel_unique(
        sort_key, comm=comm, return_counts=True, repartition_output=True
    )
    cat1_grnr_in_cat1 = (unique_value >> 32).astype(
        int
    )  # Cast to int because mixing signed and unsigned causes numpy to cast to float!
    cat1_grnr_in_cat2 = (unique_value % (1 << 32)).astype(int)

    # Send each (grnr1, grnr2, count) combination to the rank which will store the result for that halo
    if nr_cat1_per_rank > 0:
        dest = (cat1_grnr_in_cat1 // nr_cat1_per_rank).astype(int)
        dest[dest > comm_size - 1] = comm_size - 1
    else:
        dest = np.empty_like(cat1_grnr_in_cat1, dtype=int)
        dest[:] = comm_size - 1
    recv_grnr_in_cat1 = exchange_array(cat1_grnr_in_cat1, dest, comm)
    recv_grnr_in_cat2 = exchange_array(cat1_grnr_in_cat2, dest, comm)
    recv_count = exchange_array(cat1_count, dest, comm)

    # Allocate output arrays:
    # Each rank has nr_cat1_per_rank halos with any extras on the last rank
    first_in_cat1 = comm_rank * nr_cat1_per_rank
    result_grnr_in_cat2 = -np.ones(
        nr_cat1_local, dtype=int
    )  # For each halo in cat1, will store index of match in cat2
    result_count = np.zeros(
        nr_cat1_local, dtype=int
    )  # Will store number of matching particles

    # Update output arrays using the received data.
    for recv_nr in range(len(recv_grnr_in_cat1)):
        # Compute local array index of halo to update
        local_halo_nr = recv_grnr_in_cat1[recv_nr] - first_in_cat1
        assert local_halo_nr >= 0
        assert local_halo_nr < nr_cat1_local
        # Check if the received count is higher than the highest so far
        if recv_count[recv_nr] > result_count[local_halo_nr]:
            # This received combination has the highest count so far
            result_grnr_in_cat2[local_halo_nr] = recv_grnr_in_cat2[recv_nr]
            result_count[local_halo_nr] = recv_count[recv_nr]
        elif recv_count[recv_nr] == result_count[local_halo_nr]:
            # In the event of a tie, go for the lowest group number for reproducibility
            if recv_grnr_in_cat2[recv_nr] < result_grnr_in_cat2[local_halo_nr]:
                result_grnr_in_cat2[local_halo_nr] = recv_grnr_in_cat2[recv_nr]
                result_count[local_halo_nr] = recv_count[recv_nr]

    return result_grnr_in_cat2, result_count


def consistent_match(match_index_12, match_index_21):
    """
    For each halo in catalogue 1, determine if its match in catalogue 2
    points back at it.

    match_index_12 has one entry for each halo in catalogue 1 and
    specifies the matching halo in catalogue 2 (or -1 for not match)

    match_index_21 has one entry for each halo in catalogue 2 and
    specifies the matching halo in catalogue 1 (or -1 for not match)

    Returns an array with 1 for a match and 0 otherwise.
    """

    # Find the global array indexes of halos stored on this rank
    nr_local_halos = len(match_index_12)
    local_halo_offset = comm.scan(nr_local_halos) - nr_local_halos
    local_halo_index = np.arange(
        local_halo_offset, local_halo_offset + nr_local_halos, dtype=int
    )

    # For each halo, find the halo that its match in the other catalogue was matched with
    match_back = -np.ones(nr_local_halos, dtype=int)
    has_match = match_index_12 >= 0
    match_back[has_match] = psort.fetch_elements(
        match_index_21, match_index_12[has_match], comm=comm
    )

    # If we retrieved our own halo index, we have a match
    return np.where(match_back == local_halo_index, 1, 0)




def get_membership_subfile_list(path):
    """
    Function to circumvent the lack of information about subfiles within 
    membership files.
    """

    # If no file_nr is provided, there are no subfiles
    if "{file_nr}" not in path:
        return [path]

    def get_subfile_nr(path):
        return int(path.split('/')[-1].split('.')[1])

    return sorted(glob(path.replace('{file_nr}','*')),key=get_subfile_nr)

def load_particle_subgroup_memberships(membership_path, particles_types_to_use):
    '''
    Loads the subgroup memberships of particles.

    Parameters
    ----------
    membership_path: str
        Location of the memberships file.
    particle_types_to_use: list
        List containing which particle types to load.

    Returns
    -------
    particle_subgroup_memberships: np.ndarray
        Subgroup membership for each particle, sorted in the same way 
        as the snapshots.
    '''

    # Get paths to each subfile, if applicable
    file_paths = get_membership_subfile_list(membership_path)

    file =  phdf5.MultiFile(file_paths, comm=comm)

    # Only load the particle memberships for the particle types we will use to match
    particle_subgroup_memberships = []
    for particle_type in particles_types_to_use:

        # Load information from the snapshot
        particle_subgroup_memberships.append(file.read(f"PartType{particle_type}/GroupNr_bound"))

    del file

    # Return the merged array
    return np.hstack(particle_subgroup_memberships)

def match_halos(first_membership_path, second_membership_path, output_path, centrals_only, dmo, types):

    # Assuming they have both been created from the same simulation, the ordering is the same. We
    # can therefore do a direct match.
    # TODO: add a sorting algorithm based on IDs 

    # We load the particle memberships of both catalogues.
    first_subgroup_particle_memberships  = load_particle_subgroup_memberships(first_membership_path, types)
    second_subgroup_particle_memberships = load_particle_subgroup_memberships(second_membership_path, types)

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    # Read command line parameters
    parser = MPIArgumentParser(comm, description="Match haloes between two specified SOAP catalogues.")

    # Mandatory arguments
    parser.add_argument("first_membership_path", type=str, help="Path to the first SOAP catalogue file.")
    parser.add_argument("second_membership_path", type=str, help="Path to the second SOAP catalogue file.")
    parser.add_argument("output_path", type=str, help="Path where the matching inf ormation is saved to.")

    # Optional arguments
    parser.add_argument("--centrals_only", action="store_true", help="Whether we match centrals only or not.")
    parser.add_argument("--dmo", action="store_true", help="If the simulation is DMO or hydro.")
    parser.add_argument('--types', nargs='+', help='Which particle types are used to match subgroups')

    args = parser.parse_args()

    match_halos(**vars(args))

    comm.barrier()
    message("Done.")
