#!/bin/env python

from mpi4py import MPI
import h5py
import numpy as np
import pytest
import unyt
import virgo.mpi.parallel_sort as psort
from virgo.mpi.gather_array import gather_array

from SOAP.core import swift_units
from SOAP.catalogue_readers.read_swift_fof import (
    read_swift_fof_groupnr,
    read_swift_fof_catalogue,
)

import helpers

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

# Snapshot and FOF catalogue (which includes Groups/Radii) from a small DMO run
required_files = [
    "swift_output/snap_0018.hdf5",
    "swift_output/fof_output_0018.hdf5",
]


@pytest.mark.mpi
@helpers.requires(required_files, comm=comm)
def test_read_swift_fof_groupnr(filenames):
    """
    Read FOF group numbers from the snapshot and check the number of
    particles in each group matches the sizes in the FOF catalogue.
    """
    snap_filename, fof_filename = filenames

    _, ids, grnr = read_swift_fof_groupnr(snap_filename)
    assert ids.shape == grnr.shape
    assert np.all(grnr >= 0)
    del ids

    sizes_from_grnr = psort.parallel_bincount(grnr, comm=comm)
    sizes_from_grnr = gather_array(sizes_from_grnr, comm=comm)

    def check():
        with h5py.File(fof_filename, "r") as fof:
            group_ids = fof["Groups/GroupIDs"][...]
            sizes = fof["Groups/Sizes"][...]
        # Every particle in a group must belong to a group in the catalogue
        assert sizes_from_grnr.shape[0] == np.max(group_ids) + 1
        assert np.sum(sizes_from_grnr) == np.sum(sizes)
        assert np.all(sizes_from_grnr[group_ids] == sizes)

    helpers.check_on_rank_zero(check, comm)


@pytest.mark.mpi
@helpers.requires(required_files, comm=comm)
def test_read_swift_fof_catalogue(filenames):
    """
    Read the FOF catalogue and check the values match the input file.
    """
    snap_filename, fof_filename = filenames

    if comm_rank == 0:
        with h5py.File(snap_filename, "r") as snap:
            registry = swift_units.unit_registry_from_snapshot(snap)
            boxsize = snap["Header"].attrs["BoxSize"][0]
    else:
        registry = None
        boxsize = None
    registry, boxsize = comm.bcast((registry, boxsize))
    a_unit = unyt.Unit("a", registry=registry)
    boxsize = unyt.unyt_quantity(
        boxsize, units=unyt.Unit("snap_length", registry=registry) * a_unit
    )

    halo = read_swift_fof_catalogue(comm, fof_filename, a_unit, registry, boxsize)
    for name in ("index", "cofp", "search_radius", "is_central", "nr_bound_part"):
        assert name in halo

    def gather(name):
        return gather_array(halo[name].to_value(halo[name].units), comm=comm)

    index = gather("index")
    nr_bound_part = gather("nr_bound_part")
    is_central = gather("is_central")
    cofp = gather("cofp")
    search_radius = gather("search_radius")
    centres = gather("Centres")
    radii = gather("Radii")
    masses = gather("Masses")

    def check():
        with h5py.File(fof_filename, "r") as fof:
            assert np.all(index == fof["Groups/GroupIDs"][...])
            assert np.all(nr_bound_part == fof["Groups/Sizes"][...])
            assert np.allclose(cofp, fof["Groups/Centres"][...] % boxsize.value)
            # The values passed through to the output are unmodified
            assert np.allclose(centres, fof["Groups/Centres"][...])
            assert np.allclose(radii, fof["Groups/Radii"][...])
            assert np.allclose(masses, fof["Groups/Masses"][...])
            # The search radius must enclose all particles in the group
            assert np.all(search_radius >= fof["Groups/Radii"][...])
        assert np.all(is_central == 1)
        assert np.all(cofp >= 0) and np.all(cofp < boxsize.value)

    helpers.check_on_rank_zero(check, comm)


if __name__ == "__main__":
    # Run with: mpirun -np 8 python -m pytest --with-mpi tests/test_read_swift_fof.py
    test_read_swift_fof_groupnr()
    test_read_swift_fof_catalogue()
