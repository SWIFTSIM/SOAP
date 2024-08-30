#!/bin/env python3
#
# Find IDs of halos in a corner of the simulation box
#
# Run with e.g. `python3 ./find_halo_ids.py L1000N1800/HYDRO_FIDUCIAL 57 10`
#
import sys
import numpy as np
import h5py


def find_halo_indices(sim, snap_nr, boxsize):
    soap_file = f"/cosma8/data/dp004/flamingo/Runs/{sim}/SOAP-HBT/halo_properties_{snap_nr:04d}.hdf5"
    with h5py.File(soap_file, "r") as f:
        pos = f["InputHalos/HaloCentre"][()]
        mask = np.all(pos < boxsize, axis=1)
        index = f["InputHalos/HaloCatalogueIndex"][()]
        is_central = f["InputHalos/IsCentral"][()]
        if np.sum(is_central[mask]) == 0:
            print('No centrals loaded')
        return index[mask]


if __name__ == "__main__":
    sim = sys.argv[1]
    snap_nr = int(sys.argv[2])
    boxsize = float(sys.argv[3])

    indices = find_halo_indices(sim, snap_nr, boxsize)
    indices_list = " ".join([str(i) for i in indices])
    print(indices_list)
