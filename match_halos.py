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

def assign_task_based_on_id(ids):
    """
    Uses a hash function and modulus operation to assign a
    task given an id.

    Parameters
    ----------
    ids : np.ndarray
        An array of particle IDs

    Returns
    -------
    np.ndarray
        An array with the task rank assigned to each ID
    """

    # Taken from: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    ids = ((ids >> 16) ^ ids) * 0x45d9f3b
    ids = ((ids >> 16) ^ ids) * 0x45d9f3b
    ids = (ids >> 16) ^ ids

    return abs(ids) % comm.size

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
    particle_memberships: np.ndarray
        Subgroup membership for each particle, sorted in the same way 
        as the snapshots.
    '''

    # Get paths to each subfile, if applicable
    file_paths = get_membership_subfile_list(membership_path)

    file =  phdf5.MultiFile(file_paths, comm=comm)

    # Only load the particle memberships for the particle types we will use to match
    particle_memberships = []
    for particle_type in particles_types_to_use:

        # Load information from the snapshot
        particle_memberships.append(file.read(f"PartType{particle_type}/GroupNr_bound"))

    del file

    # Make it a np.ndarray and return
    return np.hstack(particle_memberships)

def load_memberships(catalogue_path_1, catalogue_path_2, types):
    """
    Returns the memberships of particles that are bound and present
    in both catalogues. Entries are sorted in ascending particle ID.

    Parameters
    ----------
    catalogue_path_1: str
        Path to the membership files of the reference catalogue.
    catalogue_path_2: str
        Path to membership files of the catalogue to match to.
    types: list
        Particle types to load and use for matching.

    Returns
    -------
    particle_memberships_1: np.ndarray
        Subgroup memberships of particles from the reference catalogue. They
        are sorted in ascending ID and only have those bound to objects in both
        catalogues.
    particle_memberships_2: np.ndarray
        Subgroup memberships of particles from the catalogue to match to. The entries
        have a one-to-one correspondence to the particles in particle_memberships_1
    """

    particle_memberships_1 = load_particle_subgroup_memberships(catalogue_path_1, types)
    particle_memberships_2 = load_particle_subgroup_memberships(catalogue_path_2, types)

    # Remove particles that are not bound to any subgroup in either dataset
    index_to_keep = (particle_memberships_1 != -1) & (particle_memberships_2 != -1)
    particle_memberships_1 = particle_memberships_1[index_to_keep]
    particle_memberships_2 = particle_memberships_2[index_to_keep]

    # Sanity checks for overflow
    assert np.all(particle_memberships_1 < 2 ** 32)
    assert np.all(particle_memberships_2 < 2 ** 32)

    # Sanity checks for keeping bound particles
    assert np.all(particle_memberships_1 >= 0)
    assert np.all(particle_memberships_2 >= 0)

    return particle_memberships_1, particle_memberships_2

def match_one_way(particle_memberships_one, particle_memberships_two):
    '''
    Obtains the most likely match of subgroups between two SOAP catalogues.

    Parameters
    ----------
    particle_memberships_one: np.ndarray 
        Subgroup memberships of all particles for one SOAP catalogue. 
    particle_memberships_two: np.ndarray 
        Subgroup memberships of all particles for a second SOAP catalogue. 

    Returns
    -------
    np.ndarray
        2D array with the first column corresponding to subgroups in the reference 
        catalogue, and the second column their matched counterparts in the other 
        catalogue.
    '''


    # Get sorted, unique (grnr1, grnr2) combinations and counts of how many instances of each we have
    sort_key = (particle_memberships_one.astype(np.uint64) << 32) + particle_memberships_two.astype(np.uint64)
    unique_value, counts = psort.parallel_unique(sort_key, comm=comm, return_counts=True, repartition_output=True)

    # Cast to int because mixing signed and unsigned causes numpy to cast to float!
    particle_memberships_one = (unique_value >> 32).astype(int)
    particle_memberships_two = (unique_value % (1 << 32)).astype(int)

    # Send each (grnr1, grnr2, count) combination to the rank which will store the result for that halo
    target_task = assign_task_based_on_id(particle_memberships_one)
    reference_subgroups = exchange_array(particle_memberships_one, target_task, comm)
    matched_subgroups = exchange_array(particle_memberships_two, target_task, comm)
    matched_counts = exchange_array(counts, target_task, comm)

    # We now do a sort to place the matched subgroups for each reference subgroup in descending 
    # matched count number. NOTE: primary sort array is the LAST parameter.
    idx_sort = np.lexsort((matched_subgroups, -matched_counts, reference_subgroups))
    reference_subgroups = reference_subgroups[idx_sort]
    matched_subgroups = matched_subgroups[idx_sort]
    matched_counts = matched_counts [idx_sort]

    # Find index corresponding to the first entry of each unique reference subgroup, i.e.
    # the one with the most matches.
    _,  unique_index = np.unique(reference_subgroups, return_index=True)

    return np.vstack([reference_subgroups[unique_index], matched_subgroups[unique_index]]).T





def match_halos(first_membership_path, second_membership_path, output_path, centrals_only, dmo, types):

    # NOTE: Assuming they have both been created from the same simulation, the ordering is the same. We
    # can therefore do a direct match.
    # TODO: add a sorting algorithm based on IDs 

    # We load the membership of particles present in both catalogues, only keeping particles that are bound to 
    # something.
    particle_memberships_1, particle_memberships_2 = load_memberships(first_membership_path, second_membership_path, types)

    results_12 = match_one_way(particle_memberships_1, particle_memberships_2)
    results_21 = match_one_way(particle_memberships_2, particle_memberships_1)
    results = match_bijectively(results_12, results_21)
    
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

    if comm_rank == 0:
        print("Done.")
