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

    # Remove particles that are not bound to any subgroup in either dataset
    index_to_keep = (first_subgroup_particle_memberships != -1) & (second_subgroup_particle_memberships != -1)
    first_subgroup_particle_memberships  = first_subgroup_particle_memberships[index_to_keep]
    second_subgroup_particle_memberships = second_subgroup_particle_memberships[index_to_keep]

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
