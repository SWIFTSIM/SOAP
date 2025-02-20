import sys
import warnings

from mpi4py import MPI
import pytest

from SOAP.catalogue_readers.read_vr import read_vr_group_sizes

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


@pytest.mark.mpi
def test_read_vr(snap_nr=57):

    basename = f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_{snap_nr:04d}/vr_catalogue_{snap_nr:04d}"
    suffix = ".%(file_nr)d"

    # Catch deprecation warning from VirgoDC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        nr_parts_bound, nr_parts_unbound = read_vr_group_sizes(basename, suffix, comm)

    comm.barrier()
    nr_halos_total = comm.allreduce(len(nr_parts_bound))
    if comm_rank == 0:
        print(f"Read {nr_halos_total} halos")


if __name__ == "__main__":
    snap_nr = int(sys.argv[1])
    test_read_vr(snap_nr=snap_nr)
