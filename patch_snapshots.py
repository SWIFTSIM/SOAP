"""
Script for patching issues with COLIBRE snapshots.
Run from the run base directory, assumes snapshots are found in ./snapshots
Run with $ mpirun -np 16 python -u patch_snapshots.py SNAP_NR, where
SNAP_NR is the snapshot to patch
"""

import os
import sys

import h5py
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def assign_files(nr_files, nr_ranks):
    """
    Assign files to MPI ranks. Lifted from github.com/jchelly/VirgoDC
    """
    files_on_rank = np.zeros(nr_ranks, dtype=int)
    files_on_rank[:] = nr_files // nr_ranks
    remainder = nr_files % nr_ranks
    if remainder > 0:
        step = max(nr_files // (remainder+1), 1)
        for i in range(remainder):
            files_on_rank[(i*step) % nr_ranks] += 1
    assert sum(files_on_rank) == nr_files
    return files_on_rank

def apply_patches(file):
    '''
    Apply all the current patches we have for snapshots
    Patches must be written in a way that then can be reapplied (run on files
    which have already been patched) without issue
    '''

    # Consistent datatypes for ParticleIDs
    for ptype in [4, 5]:
        group = file[f'PartType{ptype}']
        if group['ParticleIDs'].dtype == np.uint64:
            err_msg = f'{str(file)} has ParticleIDs'
            assert group['ParticleIDs'].shape[0] == 0, err_msg
            dset_attrs = dict(group['ParticleIDs'].attrs)
            shape = group['ParticleIDs'].shape

            del group['ParticleIDs']
            dset = group.create_dataset(
                'ParticleIDs',
                shape=shape,
                dtype=np.int64,
            )
            for k, v in dset_attrs.items():
                dset.attrs[k] = v

    return


if __name__ == '__main__':

    # TODO: Set parameters
    args = sys.argv[1:]
    snap_nr = args[0]
    snapshot_basename = args[1]
    # snapshot_basename = './snapshots/colibre_{snap_nr:04}/colibre_{snap_nr:04}{file_nr}.hdf5'
    
    virtual_filename = snapshot_basename.format(snap_nr=snap_nr, file_nr='')
    if os.path.exists(virtual_filename):

        if comm_rank == 0:
            # Determine chunk files each rank should process
            chunk_file = snapshot_basename.format(snap_nr=snap_nr, file_nr='.0')
            with h5py.File(chunk_file, 'r') as file:
                nfile = int(file['Header'].attrs['NumFilesPerSnapshot'][0])
        else:
            nfile = None
        nfile = comm.bcast(nfile)
        files_on_rank = assign_files(nfile, comm_size)
        first_file = np.cumsum(files_on_rank) - files_on_rank

        # Patch chunk files
        for i_chunk in range(
            first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
        ):
            chunk_filename = snapshot_basename.format(snap_nr=snap_nr, file_nr=f'.{i_chunk}')
            with h5py.File(chunk_filename, 'a') as file:
                apply_patches(file)
            print(f'[{comm_rank}] Patched {chunk_filename}')
        comm.barrier()
    else:
        print('Snapshots not found')