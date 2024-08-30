This directory contains a fast compression script that can be used to compress
a SOAP catalogue file.
The script reads the lossy compression filter metadata from the SOAP output,
applies it (and GZIP compression), and updates the metadata to reflect the
change.

The script requires two additional data files, also found in this directory:
 - a file (filters.yml) containing the serialised information for the lossy compression
   filters. These were grabbed from a SWIFT snapshot, since those filters are
   not available in h5py. The script extract_filters.py can generate this file.
 - a file (wrong_compression.yml) that contains updated lossy compression filter
   names for datasets which had the wrong filter set when SOAP was run. This should
   no longer be necessary, but is good to have just in case.

This directory additionally contains a script that can be used to generate an
empty SOAP catalogue for snapshots that have no halos, since SOAP will not run
on those snapshots.
