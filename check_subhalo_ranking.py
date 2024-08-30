#!/bin/env python

import numpy as np
import h5py


# Read VR IDs and positions
filename = "/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL_DATA/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.properties.0"
with h5py.File(filename, "r") as infile:
    vr_id = infile["ID"][...]
    vr_host_id = infile["hostHaloID"][...]

# Read SOAP output
filename = "/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL_DATA/HYDRO_FIDUCIAL/SOAP/halo_properties_0077.hdf5"
with h5py.File(filename, "r") as infile:
    soap_id = infile["VR/ID"][...]
    soap_rank = infile["SOAP/SubhaloRankByBoundMass"][...]
    soap_host_id = infile["VR/HostHaloID"][...]
    soap_mass = infile["BoundSubhaloProperties/TotalMass"][...]

assert np.all(soap_id == vr_id)
assert np.all(soap_host_id == vr_host_id)

# Set field halos to be their own host (they have host=-1 in VR)
field = soap_host_id < 0
soap_host_id[field] = soap_id[field]

# Get ID and mass sorted by host id then rank within a host
order = np.lexsort((soap_rank, soap_host_id))
sorted_id = soap_id[order]
sorted_host_id = soap_host_id[order]
sorted_rank = soap_rank[order]
sorted_mass = soap_mass[order]

# Identify sequences of halos in the same host
unique_host_id, offset, count = np.unique(
    sorted_host_id, return_index=True, return_counts=True
)
for o, c, hid in zip(offset, count, unique_host_id):
    assert np.all(
        sorted_rank[o : o + c] == np.arange(c, dtype=int)
    )  # ranks should be sequential
    assert np.all(sorted_host_id[o : o + c] == hid)  # all halos should be in host hid

# Verify that object with rank=0 has maximum mass and that masses are in descending order
for o, c in zip(offset, count):
    assert sorted_rank[o] == 0  # first subhalo has rank=0
    assert sorted_mass[o] == np.amax(sorted_mass[o : o + c])  # rank 0 has maximum mass
    if c > 1:
        mass = sorted_mass[o : o + c]
        assert np.all(mass[1:] <= mass[:-1])  # subhalos are in descending order of mass
