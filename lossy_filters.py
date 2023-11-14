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

filter_data = \
"""
BFloat16:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 4
      - 0
      - 16
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIA8ABAAAAAAAEAAHCAAHfwAAAA==
DMantissa13:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 25
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIBgACAAAAAAAGQANCwAN/wMAAA==
DMantissa9:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 21
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIBQACAAAAAAAFQAJCwAJ/wMAAA==
DScale1:
  filters:
  - - 6
    - 1
    - - 0
      - 1
      - 800648
      - 1
      - 4
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIB8ABAAAAAAAIAAXCAAXfwAAAA==
DScale2:
  filters:
  - - 6
    - 1
    - - 0
      - 2
      - 800648
      - 1
      - 4
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIB8ABAAAAAAAIAAXCAAXfwAAAA==
DScale3:
  filters:
  - - 6
    - 1
    - - 0
      - 3
      - 800648
      - 1
      - 4
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIB8ABAAAAAAAIAAXCAAXfwAAAA==
DScale4:
  filters:
  - - 6
    - 1
    - - 0
      - 4
      - 800648
      - 1
      - 4
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIB8ABAAAAAAAIAAXCAAXfwAAAA==
DScale5:
  filters:
  - - 6
    - 1
    - - 0
      - 5
      - 800648
      - 1
      - 8
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARID8ACAAAAAAAQAA0CwA0/wMAAA==
DScale6:
  filters:
  - - 6
    - 1
    - - 0
      - 6
      - 800648
      - 1
      - 4
      - 0
      - 0
      - 1
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 1818321779
      - 1717989221
      - 7628147
      - 0
    - !!binary |
      c2NhbGVvZmZzZXQ=
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIB8ABAAAAAAAIAAXCAAXfwAAAA==
FMantissa13:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 4
      - 0
      - 22
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIBUABAAAAAAAFgANCAANfwAAAA==
FMantissa9:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 4
      - 0
      - 18
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIBEABAAAAAAAEgAJCAAJfwAAAA==
HalfFloat:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 4
      - 0
      - 16
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwARIA8ABAAAAAAAEAAKBQAKDwAAAA==
Nbit36:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 36
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwAQCAAACAAAAAAAJAA=
Nbit40:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 40
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwAQCAAACAAAAAAAKAA=
Nbit48:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 48
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwAQCAAACAAAAAAAMAA=
Nbit56:
  filters:
  - - 5
    - 1
    - - 8
      - 0
      - 800648
      - 1
      - 8
      - 0
      - 56
      - 0
    - !!binary |
      bmJpdA==
  - - 3
    - 0
    - []
    - !!binary |
      ZmxldGNoZXIzMg==
  type: !!binary |
    AwAQCAAACAAAAAAAOAA=
"""
filter_dict = yaml.safe_load(filter_data)


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
    

def convert_dataset(data, compression_method, enable_compression):
    """
    HDF5 collective writes fall back to independent mode if any data conversion
    is required. Here we explicitly convert the data to the data type in the
    file in advance to prevent this.
    """
    # If we're not compressing, there's nothing to do
    if compression_method == "None" or (not enable_compression):
        return data

    # Get the data type in the file
    file_dtype, dcpl = get_dtype_and_filters(compression_method)

    # If this compression method does not change the dtype, there's nothing to do
    data_dtype = h5py.h5t.py_create(data.dtype)
    if file_dtype == data_dtype:
        return data

    # In this case we need to convert the data. H5Tconvert converts in place so
    # converting to a larger type would cause a buffer overrun. Shouldn't
    # happen if we're trying to make the output smaller!
    if file_dtype.get_size() > data_dtype.get_size():
        raise RuntimeError("Converting to larger data type in convert_dataset()!")

    # Make a contiguous copy of the data array
    buf = np.ascontiguousarray(data.copy()).flatten()

    # Convert the data in place
    h5py.h5t.convert(data_dtype, file_dtype, len(buf), buf)

    return buf
    

def collective_write(group, name, data, comm, compression_method, enable_compression):
    """
    Do a parallel collective write of a HDF5 dataset by concatenating
    contributions from MPI ranks along the first axis.
    
    File must have been opened in MPI mode.

    This is slightly different from the implementation in
    virgo.mpi.parallel_hdf5 because it explicitly converts the data to the
    type in the file before writing.
    """

    buffer_size = 100*1024*1024
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Determine data type in the file
    if compression_method != "None" and enable_compression:
        file_dtype, _ = get_dtype_and_filters(compression_method)
    else:
        file_dtype = h5py.h5t.py_create(data.dtype)
    
    # Store the original dataset shape
    shape = data.shape
    
    # Ensure input is a contiguous numpy array
    data = np.ascontiguousarray(data)
    data = convert_dataset(data, compression_method, enable_compression)
    
    # Determine how many elements to write on each task
    num_on_task = np.asarray(comm.allgather(shape[0]))
    ntot = np.sum(num_on_task)

    # Determine offsets at which to write data from each task
    offset_on_task = np.cumsum(num_on_task) - num_on_task

    # Find the full shape of the new dataset
    full_shape = (ntot,) + shape[1:]

    # Open the dataset
    dataset = group[name]

    # Determine slice to write
    file_offset   = offset_on_task[comm_rank]
    nr_left       = num_on_task[comm_rank]
    memory_offset = 0

    # Determine how many elements we can write per iteration
    element_size = file_dtype.get_size()
    for s in full_shape[1:]:
        element_size *= s
    max_elements = buffer_size // element_size

    # We need to use the low level interface here because the h5py high
    # level interface omits zero sized writes, which causes a hang in
    # collective mode if a rank has nothing to write.
    dset_id = dataset.id
    file_space = dset_id.get_space()
    mem_space = h5py.h5s.create_simple(data.shape)
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
        dset_id.write(mem_space, file_space, data, dxpl=prop_list, mtype=file_dtype)

        # Advance to next chunk
        file_offset   += nr_to_write
        memory_offset += nr_to_write
        nr_left       -= nr_to_write

    return dataset
