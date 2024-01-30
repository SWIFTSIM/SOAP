#!/bin/env python
#
# Apply lossy compression to SOAP output using parallel IO.
#

import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import lossy_filters
import h5py
import virgo.mpi.parallel_hdf5 as phdf5


def message(m):
    if comm_rank == 0:
        print(m)


def compress_file(input_file, output_file):
    """
    Apply lossy compression to the specified file
    """

    #
    # First rank sets up the structure of the output file
    #
    message("Creating output file")
    if comm_rank == 0:

        infile = h5py.File(input_file, "r")
        outfile = h5py.File(output_file, "w")
        
        # Find names of all objects in the input file
        names = []
        def visit_object(name, obj):
            names.append(name)
        infile.visititems(visit_object)
            
        # Create groups and datasets in the output file
        for name in names:
            obj = infile[name]
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                compression_method = obj.attrs["Lossy Compression Algorithm"]
                if compression_method != "None":
                    dtype, _ = lossy_filters.get_dtype_and_filters(compression_method)
                    enable_compression = True
                else:
                    dtype = obj.dtype
                    enable_compression = False
                lossy_filters.create_compressed_dataset(outfile, name, shape, dtype, compression_method, enable_compression)
            elif isinstance(obj, h5py.Group):
                outfile.require_group(name)
            else:
                raise RuntimeError("Unsupported object type")
                
        outfile.close()
        infile.close()

    else:
        names = None
    names = comm.bcast(names)
        
    comm.barrier()

    #
    # Read all of the datasets into memory
    #
    data = {}
    n = 0
    infile = h5py.File(input_file, "r", driver="mpio", comm=comm)
    for name in sorted(names):
        obj = infile[name]
        if isinstance(obj, h5py.Dataset):
            message(f"Reading: {name}")
            data[name] = phdf5.collective_read(obj, comm=comm)
            n += 1
    infile.close()

    #
    # Write all of the datasets to the new file
    #
    comm.barrier()
    t0 = time.time()
    outfile = h5py.File(output_file, "r+", driver="mpio", comm=comm)
    for i, name in enumerate(sorted(names)):
        obj = outfile[name]
        if isinstance(obj, h5py.Dataset):
            message(f"Writing [{i}/{n}]: {name}")
            lossy_filters.collective_write(outfile, name, data[name], comm=comm)
    outfile.close()
    comm.barrier()
    t1 = time.time()

    elapsed = comm.allreduce(t1-t0, op=MPI.MAX)

    message(f"Time elapsed = {elapsed}s")
    
    
    
if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Compress SOAP output")
    parser.add_argument("input_file", type=str, help="Name of the input file")
    parser.add_argument("output_file", type=str, help="Name of the output file")
    args = parser.parse_args()

    compress_file(args.input_file, args.output_file)
    
