#!/bin/env python

import os
import sys
import numpy as np
import h5py

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()


def check_membership(
    membership_filenames,
    nr_memb_files,
    snapshot_filenames,
    nr_snap_files,
    hbt_filenames,
):
    """
    Check that membership files are consistent with HBT particle lists
    """

    # Check on files so we can fail quickly if filenames are wrong
    # Also get number of HBT files
    if comm_rank == 0:
        if not os.path.exists(membership_filenames.format(file_nr=0)):
            raise RuntimeError("Membership files not found")
        if not os.path.exists(snapshot_filenames.format(file_nr=0)):
            raise RuntimeError("Snapshot files not found")
        if not os.path.exists(hbt_filenames.format(file_nr=0)):
            raise RuntimeError("HBT files not found")
        with h5py.File(hbt_filenames.format(file_nr=0)) as infile:
            nr_hbt_files = int(infile["NumberOfFiles"][...])
    else:
        nr_hbt_files = None
    nr_hbt_files = comm.bcast(nr_hbt_files)
    assert nr_memb_files == nr_snap_files

    # Read membership files
    if comm_rank == 0:
        print("Reading membership files")
    memb = phdf5.MultiFile(
        membership_filenames, file_idx=np.arange(nr_memb_files, dtype=int), comm=comm
    )
    memb_grnr = memb.read("PartType1/GroupNr_bound")

    # Read snapshot files
    if comm_rank == 0:
        print("Reading snapshot files")
    snap = phdf5.MultiFile(
        snapshot_filenames, file_idx=np.arange(nr_snap_files, dtype=int), comm=comm
    )
    snap_part_ids = snap.read("PartType1/ParticleIDs")

    # Mask out particles which aren't in a halo
    mask = memb_grnr != -1
    memb_grnr = memb_grnr[mask]
    snap_part_ids = snap_part_ids[mask]

    # Sort by particle id
    if comm_rank == 0:
        print("Sorting particles from snapshot")
    order = psort.parallel_sort(snap_part_ids, return_index=True, comm=comm)
    memb_grnr = psort.fetch_elements(memb_grnr, order)
    del order
    del snap_part_ids

    # Get track id of each particle
    if comm_rank == 0:
        print("Getting track id of each particle")
    cat = phdf5.MultiFile(hbt_filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalo = cat.read("Subhalos")
    hbt_track_id_mapping = subhalo["TrackId"]
    del subhalo
    memb_track_ids = psort.fetch_elements(hbt_track_id_mapping, memb_grnr, comm=comm)
    del memb_grnr
    del hbt_track_id_mapping

    # Assign files to MPI ranks
    files_per_rank = np.zeros(comm_size, dtype=int)
    files_per_rank[:] = nr_hbt_files // comm_size
    files_per_rank[: nr_hbt_files % comm_size] += 1
    assert np.sum(files_per_rank) == nr_hbt_files
    first_file_on_rank = np.cumsum(files_per_rank) - files_per_rank

    # Read in the halos from the HBT output
    if comm_rank == 0:
        print("Reading HBT output")
    hbt_track_ids = []
    hbt_part_ids = []
    for file_nr in range(
        first_file_on_rank[comm_rank],
        first_file_on_rank[comm_rank] + files_per_rank[comm_rank],
    ):
        with h5py.File(hbt_filenames.format(file_nr=file_nr), "r") as infile:
            hbt_part_ids.append(infile["SubhaloParticles"][...])
            hbt_track_ids.append(
                np.repeat(infile["Subhalos"]["TrackId"], infile["Subhalos"]["Nbound"])
            )

    # Combine arrays of particles in halos
    if len(hbt_part_ids) > 0:
        hbt_track_ids = np.concatenate(hbt_track_ids)
        hbt_part_ids = np.concatenate(
            hbt_part_ids
        )  # Combine arrays of halos from different files
        if len(hbt_part_ids) > 0:
            hbt_part_ids = np.concatenate(
                hbt_part_ids
            )  # Combine arrays of particles from different halos
    # TODO: Handle ranks which didn't read files?

    # Sort by particle ID
    if comm_rank == 0:
        print("Sorting particles from HBT")
    order = psort.parallel_sort(hbt_part_ids, return_index=True, comm=comm)
    hbt_track_ids = psort.fetch_elements(hbt_track_ids, order)
    del order
    del hbt_part_ids

    # Ensure both arrays of track ids are partitioned in the same way
    if comm_rank == 0:
        print("Repartitioning particles")
    ndesired = np.asarray(comm.allgather(len(hbt_track_ids)), dtype=int)
    memb_track_ids = psort.repartition(memb_track_ids, ndesired, comm=comm)

    # Check that track ids agree
    if np.any(memb_track_ids != hbt_track_ids):
        raise RuntimeError("HBT catalogues disagree with membership files!")

    comm.barrier()
    if comm_rank == 0:
        print("Track IDs agree.")


if __name__ == "__main__":

    # Read in snapshot as input
    snap_nr = int(sys.argv[1])

    # Location of membership files
    membership_dir = "/cosma8/data/dp004/dc-mcgi1/FLAMINGO/Runs/L2800N10080/DMO_FIDUCIAL/SOAP/HBTplus/"
    membership_filenames = (
        f"{membership_dir}/membership_{snap_nr:04d}/membership_{snap_nr:04d}."
        + "{file_nr}.hdf5"
    )
    nr_memb_files = 1024
    # Location of HBT output
    hbt_dir = "/snap8/scratch/dp004/jch/FLAMINGO/HBT/L2800N10080/DMO_FIDUCIAL/hbt/"
    hbt_filenames = f"{hbt_dir}/{snap_nr:03d}/SubSnap_{snap_nr:03d}." + "{file_nr}.hdf5"
    # Location of snapshot files
    snapshot_dir = (
        "/cosma8/data/dp004/flamingo/Runs/L2800N10080/DMO_FIDUCIAL/snapshots/"
    )
    snapshot_filenames = (
        f"{snapshot_dir}/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}."
        + "{file_nr}.hdf5"
    )
    nr_snap_files = 1024

    check_membership(
        membership_filenames,
        nr_memb_files,
        snapshot_filenames,
        nr_snap_files,
        hbt_filenames,
    )
