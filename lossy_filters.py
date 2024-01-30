#!/bin/env python
#
# This file contains serialized filters and data types which are not available
# through the h5py interface. Copied from filters.yml used by Bert's standalone
# compression script.
#

import numpy as np
import h5py
from mpi4py import MPI
import yaml

# Default maximum size of I/O operations in bytes.
# This is to avoid MPI issues with buffers >2GB.
BUFFER_SIZE=100*1024*1024


# Read serialized filter data
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
if comm_rank == 0:
    filename = "./compression/filters.yml"
    with open(filename, "r") as filter_file:
        filter_data = filter_file.read()
    filter_dict = yaml.safe_load(filter_data)
else:
    filter_dict = None
filter_dict = comm.bcast(filter_dict)


def get_dtype_and_filters(name):
    """
    Given the name of a lossy compression scheme, return the data type of the
    dataset and a new property list with the necessary filter(s) enabled.
    """
    fprops = filter_dict[name]
    dtype = h5py.h5t.decode(fprops["type"])

    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    for f in fprops["filters"]:
        plist.set_filter(f[0], f[1], tuple(f[2]))

    return dtype, plist


def create_compressed_dataset(loc, name, shape, dtype, compression_method, enable_compression):
    """
    Create a new dataset with the specified compression scheme
    """
    # Determine data type in the file and the dataset creation properties
    if compression_method != "None" and enable_compression:
        file_dtype, dcpl = get_dtype_and_filters(compression_method)
    else:
        file_dtype = h5py.h5t.py_create(dtype)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)

    if enable_compression:
        # Enable lossless compression
        dcpl.set_deflate(9)
        # Enable chunking
        max_chunk_size = 1048576
        chunk_size = min(max_chunk_size, shape[0])
        chunks = (chunk_size,)+shape[1:]
        dcpl.set_chunk(chunks)

    # Create the memory dataspace
    file_space = h5py.h5s.create_simple(shape, shape)
    
    # Create the dataset
    dataset_id = h5py.h5d.create(loc.id, name.encode(), file_dtype, file_space, dcpl=dcpl)

    return h5py.Dataset(dataset_id)
    

def collective_write(group, name, data, comm):
    """
    Do a parallel collective write of a HDF5 dataset by concatenating
    contributions from MPI ranks along the first axis.
    
    File must have been opened in MPI mode.

    This is slightly different from the implementation in
    virgo.mpi.parallel_hdf5 because it explicitly converts the data to the
    type in the file before writing. This allows collective I/O using custom
    types which don't exist in numpy.
    """

    buffer_size = 100*1024*1024
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Store the original dataset shape
    shape = data.shape
    
    # Ensure input is a contiguous numpy array
    data = np.ascontiguousarray(data)
    
    # Determine how many elements to write on each task
    num_on_task = np.asarray(comm.allgather(shape[0]))
    ntot = np.sum(num_on_task)

    # Determine offsets at which to write data from each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    # Find the full shape of the new dataset
    full_shape = (ntot,) + shape[1:]

    # Open the dataset
    dataset = group[name]
    file_dtype = dataset.id.get_type()
    
    # Determine slice to write
    file_offset   = offset_on_task[comm_rank]
    nr_left       = num_on_task[comm_rank]
    memory_offset = 0

    # Determine how many elements we can write per iteration
    element_size = file_dtype.get_size()
    for s in full_shape[1:]:
        element_size *= s
    max_elements = buffer_size // element_size

    # Convert the data to the same type as the dataset in the file:
    # Collective writes fail if any conversion needs to be done.
    # First need to copy to a buffer large enough for the conversion.
    data_dtype = h5py.h5t.py_create(data.dtype)
    nr_elements = 1
    for s in data.shape:
        nr_elements *= s
    bytes_in_memory = nr_elements * data_dtype.get_size()
    bytes_converted = nr_elements * file_dtype.get_size()
    max_bytes = max(bytes_in_memory, bytes_converted)
    buf = np.ndarray(max_bytes, dtype=np.uint8)
    buf[:bytes_in_memory] = data.flatten().view(dtype=np.uint8)[:bytes_in_memory]

    # Then carry out the conversion
    h5py.h5t.convert(data_dtype, file_dtype, nr_elements, buf)
        
    # We need to use the low level interface here because the h5py high
    # level interface omits zero sized writes, which causes a hang in
    # collective mode if a rank has nothing to write.
    dset_id = dataset.id
    file_space = dset_id.get_space()
    mem_space = h5py.h5s.create_simple(shape)
    prop_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
    prop_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    # Loop until all elements have been written
    while comm.allreduce(nr_left) > 0:

        # Decide how many elements to write on this rank
        nr_to_write = min(nr_left, max_elements)

        # Select the region in the file
        if nr_to_write > 0:
            start = (file_offset,)+(0,)*(len(data.shape)-1)
            count = (nr_to_write,)+tuple(full_shape[1:])
            file_space.select_hyperslab(start, count)
        else:
            file_space.select_none()

        # Select the region in memory
        if nr_to_write > 0:
            start = (memory_offset,)+(0,)*(len(data.shape)-1)
            count = (nr_to_write,)+tuple(full_shape[1:])
            mem_space.select_hyperslab(start, count)
        else:
            mem_space.select_none()

        # Write the data
        dset_id.write(mem_space, file_space, buf, dxpl=prop_list, mtype=file_dtype)

        # Advance to next chunk
        file_offset   += nr_to_write
        memory_offset += nr_to_write
        nr_left       -= nr_to_write

    return dataset
