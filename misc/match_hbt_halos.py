#!/bin/env python

import os

import h5py
from mpi4py import MPI
import numpy as np
import unyt
import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.core import lustre, swift_cells
from SOAP.catalogue_readers import read_hbtplus

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Maximum number of particle types
NTYPEMAX = 7


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


# Define a placeholder unit system, since everything we do is dimensionless.
# However, it would be better to load this from a snapshot.
def define_unit_system():

    # Create a registry using this base unit system
    reg = unyt.unit_registry.UnitRegistry()

    # Add some units which might be useful for dealing with input halo catalogues
    unyt.define_unit("swift_mpc", 1.0 * unyt.cm, registry=reg)
    unyt.define_unit("swift_msun", 1.0 * unyt.g, registry=reg)
    unyt.define_unit("h", 1.0 * unyt.Hz, registry=reg)

    return reg


def find_matching_halos(
    base_name1,
    base_name2,
    max_nr_particles,
    min_particle_id,
    max_particle_id,
    field_only,
):

    # We only care about dimensionless quantities here, so
    # define a placeholder unit system
    registry = define_unit_system()
    a_unit = unyt.Unit("cm", registry=registry) ** 0
    boxsize = None

    # Load halo data from the second catalogue
    keep_orphans = True
    halo_data2 = read_hbtplus.read_hbtplus_catalogue(
        comm, base_name2, a_unit, registry, boxsize, keep_orphans
    )

    # Catalogue 2 properties
    track_ids2 = halo_data2["TrackId"]
    host_ids2 = halo_data2["HostHaloId"]
    is_central2 = halo_data2["is_central"]

    # Find the track ID of the central halo that has the same hostHaloID
    central_index2 = psort.parallel_match(
        host_ids2, host_ids2[is_central2 == 1], comm=comm
    )
    host_track_ids2 = psort.fetch_elements(
        track_ids2[is_central2 == 1], central_index2, comm=comm
    )

    # Find the index of that central halo (which may differ from the track ID)
    host_index2 = psort.parallel_match(host_track_ids2, track_ids2, comm=comm)

    # Free the other halo data
    del halo_data2

    # Find group membership for particles in the first catalogue:
    total_nr_halos1, cat1_ids, cat1_grnr_in_cat1, rank_bound1 = read_hbtplus.read_hbtplus_groupnr(
        base_name1
    )

    # Find group membership for particles in the second catalogue
    total_nr_halos2, cat2_ids, cat2_grnr_in_cat2, rank_bound2 = read_hbtplus.read_hbtplus_groupnr(
        base_name2
    )

    # Decide range of halos in cat1 which we'll store on each rank:
    # This is used to partition the result between MPI ranks.
    nr_cat1_tot = total_nr_halos1
    nr_cat1_per_rank = nr_cat1_tot // comm_size
    if comm_rank < comm_size - 1:
        nr_cat1_local = nr_cat1_per_rank
    else:
        nr_cat1_local = nr_cat1_tot - (comm_size - 1) * nr_cat1_per_rank

    # Clear group membership for particles with invalid IDs
    if min_particle_id != None:
        discard = cat1_ids < min_particle_id
        cat1_grnr_in_cat1[discard] = -1

    if max_particle_id != None:
        discard = cat1_ids >= max_particle_id
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


def get_match_hbt_halos_args(comm):
    """
    Process command line arguments for halo matching program.

    Returns a dict with the argument values, or None on failure.
    """

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(
        comm, description="Find matching halos between snapshots"
    )
    parser.add_argument("hbt_basename1", help="Base name of the first set of HBT files")
    parser.add_argument(
        "hbt_basename2", help="Base name of the second set of HBT files"
    )
    parser.add_argument(
        "nr_particles",
        metavar="N",
        type=int,
        help="Number of most bound particles to use.",
    )
    parser.add_argument("output_file", help="Output file name")
    parser.add_argument(
        "--min-particle-id",
        nargs="*",
        type=int,
        help="Only use particle with ID >= this",
    )
    parser.add_argument(
        "--max-particle-id",
        nargs="*",
        type=int,
        help="Only use particle with ID < this",
    )
    parser.add_argument(
        "--to-field-halos-only", action="store_true", help="Only match to field halos"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Read command line parameters
    args = get_match_hbt_halos_args(comm)

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm.barrier()

    # For each halo in output 1, find the matching halo in output 2
    message("Matching from first catalogue to second")
    match_index_12, count_12 = find_matching_halos(
        args.hbt_basename1,
        args.hbt_basename2,
        args.nr_particles,
        args.min_particle_id,
        args.max_particle_id,
        args.to_field_halos_only,
    )
    total_nr_halos = comm.allreduce(len(match_index_12))
    total_nr_matched = comm.allreduce(np.sum(match_index_12 >= 0))
    message(f"  Matched {total_nr_matched} of {total_nr_halos} halos")

    # For each halo in output 2, find the matching halo in output 1
    message("Matching from second catalogue to first")
    match_index_21, count_21 = find_matching_halos(
        args.hbt_basename2,
        args.hbt_basename1,
        args.nr_particles,
        args.min_particle_id,
        args.max_particle_id,
        args.to_field_halos_only,
    )
    total_nr_halos = comm.allreduce(len(match_index_21))
    total_nr_matched = comm.allreduce(np.sum(match_index_21 >= 0))
    message(f"  Matched {total_nr_matched} of {total_nr_halos} halos")

    # Check for consistent matches in both directions
    message("Checking for consistent matches")
    consistent_12 = consistent_match(match_index_12, match_index_21)
    consistent_21 = consistent_match(match_index_21, match_index_12)

    # Write the output
    def write_output_field(name, data, description):
        dataset = phdf5.collective_write(outfile, name, data, comm)
        dataset.attrs["Description"] = description

    message("Writing output")
    with h5py.File(args.output_file, "w", driver="mpio", comm=comm) as outfile:
        # Write input parameters
        params = outfile.create_group("Parameters")
        for name, value in vars(args).items():
            if value is not None:
                params.attrs[name] = value
        # Matching from first catalogue to second
        write_output_field(
            "MatchIndex1to2",
            match_index_12,
            "For each halo in the first catalogue, index of the matching halo in the second",
        )
        write_output_field(
            "MatchCount1to2",
            count_12,
            f"How many of the {args.nr_particles} most bound particles from the halo in the first catalogue are in the matched halo in the second",
        )
        write_output_field(
            "Consistent1to2",
            consistent_12,
            "Whether the match from first to second catalogue is consistent with second to first (1) or not (0)",
        )
        # Matching from second catalogue to first
        write_output_field(
            "MatchIndex2to1",
            match_index_21,
            "For each halo in the second catalogue, index of the matching halo in the first",
        )
        write_output_field(
            "MatchCount2to1",
            count_21,
            f"How many of the {args.nr_particles} most bound particles from the halo in the second catalogue are in the matched halo in the first",
        )
        write_output_field(
            "Consistent2to1",
            consistent_21,
            "Whether the match from second to first catalogue is consistent with first to second (1) or not (0)",
        )
    comm.barrier()
    message("Done.")
