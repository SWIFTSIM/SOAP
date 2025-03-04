import sys
import warnings

from mpi4py import MPI
import pytest

from SOAP.catalogue_readers.read_vr import read_vr_group_sizes

import helpers

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

required_files = [
    'VR_output/vr_018.catalog_groups.0',
    'VR_output/vr_018.catalog_particles.0',
    'VR_output/vr_018.catalog_particles.unbound.0',
    'VR_output/vr_018.properties.0',
]
@pytest.mark.mpi
@helpers.requires(required_files, comm=comm)
def test_read_vr(filenames):

    basename = filenames[0].split('.')[0]
    suffix = '.' + filenames[0].split('.')[-1]

    # Catch deprecation warning from VirgoDC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        nr_parts_bound, nr_parts_unbound = read_vr_group_sizes(basename, suffix, comm)

    comm.barrier()
    nr_halos_total = comm.allreduce(len(nr_parts_bound))
    if comm_rank == 0:
        print(f"Read {nr_halos_total} halos")


if __name__ == "__main__":
    test_read_vr()
