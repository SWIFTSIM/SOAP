# Compression scripts

### Membership files
Compression of the membership files should be done
using the [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html) command.

The script `make_virtual_snapshot.py` can be used to create a virtual hdf5 files that
links the membership files to the snapshot files, such that properties from both can be accesed from a single file.

### SOAP catalogues

The script `compress_soap_catalogue.py` creates a compressed output catalogue. It works
by reading the lossy compression filter metadata for each property from the
uncompressed SOAP output, applying it (and GZIP compression),
and updating the metadata to reflect the change. It should be run with MPI, e.g.

`mpirun -- python compress_soap_catalogue.py ${input_filename} ${output_filename} ${scratch_dir}`

This scripts outputs temporary files as it runs to the `scratch_dir`, so preferably
this is a filesystem with fast reads/writes (e.g. `/snap8` on cosma).

The following files in this directory are to help with running this script:
 - Lossy compression filters for datasets which had the wrong filter set when SOAP
   was run can be placed within `wrong_compression.yml`. The compression script will
   use the filters in this file rather than the ones in the original catalogue.
 - The script `create_empty_SOAP_catalogue.py`can be used to generate an empty SOAP
   catalogue for snapshots that have no halos, since SOAP will not run on those snapshots.
 - The file `filters.yml` contains the serialised information for the lossy compression
   filters. These were grabbed from a SWIFT snapshot, since those filters are
   not available in h5py. The script `extract_filters.py` can generate this file.

