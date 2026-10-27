#!/bin/env python

"""
compress_membership.py

Apply lossless GZIP compression to the group membership files. This gives
the same result as running

h5repack -i ${input} -o ${output} -l CHUNK=10000 -f GZIP=4

on each file, but splits the files between MPI ranks. As with h5repack,
datasets with fewer elements than the chunk size are not compressed. Usage:

mpirun -- python compress_membership.py ${input_filename} ${output_filename}

where the filenames should contain {file_nr} if there are multiple
membership files per snapshot.
"""

import argparse
import os
import time

import h5py
from mpi4py import MPI

chunksize = 10000
compression_opts = {"compression": "gzip", "compression_opts": 4}

# Number of elements to copy at once, to limit memory use for large files
copy_block_size = 64 * 1024 * 1024


def compress_file(input_filename, output_filename):
    """
    Write a GZIP compressed copy of a single membership file
    """

    with (
        h5py.File(input_filename, "r") as ifile,
        h5py.File(output_filename, "w") as ofile,
    ):
        for attr in ifile.attrs:
            ofile.attrs[attr] = ifile.attrs[attr]

        def copy_object(name, h5obj):
            if isinstance(h5obj, h5py.Group):
                ofile.create_group(name)
            elif isinstance(h5obj, h5py.Dataset):
                # Match h5repack, which leaves datasets smaller than the
                # chunk size uncompressed
                if (len(h5obj.shape) == 0) or (h5obj.shape[0] < chunksize):
                    ofile.create_dataset(name, data=h5obj[()])
                else:
                    chunks = (chunksize,) + h5obj.shape[1:]
                    dset = ofile.create_dataset(
                        name,
                        shape=h5obj.shape,
                        dtype=h5obj.dtype,
                        chunks=chunks,
                        **compression_opts,
                    )
                    for start in range(0, h5obj.shape[0], copy_block_size):
                        end = min(start + copy_block_size, h5obj.shape[0])
                        dset[start:end] = h5obj[start:end]
            else:
                raise RuntimeError(f"Unknown HDF5 object type: {name}")
            for attr in h5obj.attrs:
                ofile[name].attrs[attr] = h5obj.attrs[attr]

        ifile.visititems(copy_object)


def get_membership_filenames(filename, comm):
    """
    Return a list of all the files in a set of membership files. filename
    should contain {file_nr} if there are multiple files per snapshot.
    """
    if "{file_nr}" not in filename:
        return [filename]

    if comm.Get_rank() == 0:
        with h5py.File(filename.format(file_nr=0), "r") as infile:
            nr_files = int(infile["Header"].attrs["NumFilesPerSnapshot"][0])
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)
    return [filename.format(file_nr=file_nr) for file_nr in range(nr_files)]


def compress_membership(input_filename, output_filename, comm):
    """
    Compress a set of membership files, splitting the files between the ranks
    in comm. The filenames should contain {file_nr} if there are multiple
    membership files per snapshot.

    Returns a list of the input files which were compressed.
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if ("{file_nr}" in input_filename) != ("{file_nr}" in output_filename):
        raise ValueError(
            "Either both or neither of the input and output membership "
            "filenames should contain {file_nr}"
        )

    tic = time.time()
    input_filenames = get_membership_filenames(input_filename, comm)
    output_filenames = [
        output_filename.format(file_nr=file_nr)
        for file_nr in range(len(input_filenames))
    ]
    if comm_rank == 0:
        print(f"Compressing {len(input_filenames)} membership files")
        print(f"Source     : {input_filename}")
        print(f"Destination: {output_filename}")
        output_dir = os.path.dirname(os.path.abspath(output_filenames[0]))
        os.makedirs(output_dir, exist_ok=True)
    comm.barrier()

    for file_nr in range(comm_rank, len(input_filenames), comm_size):
        try:
            compress_file(input_filenames[file_nr], output_filenames[file_nr])
        except Exception as e:
            print(f"Error compressing {input_filenames[file_nr]}: {e}", flush=True)
            comm.Abort(1)

    comm.barrier()
    toc = time.time()
    if comm_rank == 0:
        input_size = sum(os.path.getsize(f) for f in input_filenames)
        output_size = sum(os.path.getsize(f) for f in output_filenames)
        print(
            f"{input_size / 1024**2:.1f}MB -> {output_size / 1024**2:.1f}MB "
            f"({100. * output_size / input_size:.2f}%)"
        )
        print(f"Compressing membership files took {toc-tic:.2f} s.")

    return input_filenames


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description="Apply GZIP compression to SOAP membership files"
    )
    argparser.add_argument(
        "input",
        help="Filename of the uncompressed membership files. Should contain "
        "{file_nr} if there are multiple files per snapshot",
    )
    argparser.add_argument(
        "output",
        help="Filename of the compressed membership files. Should contain "
        "{file_nr} if there are multiple files per snapshot",
    )
    args = argparser.parse_args()

    compress_membership(args.input, args.output, MPI.COMM_WORLD)
