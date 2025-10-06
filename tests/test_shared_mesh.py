import numpy as np
from mpi4py import MPI
import pytest
import unyt

from SOAP.core import shared_array, shared_mesh


def make_test_dataset(boxsize, total_nr_points, centre, radius, box_wrap, comm):
    """
    Make a set of random test points

    boxsize - periodic box size (unyt scalar)
    total_nr_points - number of points in the box over all MPI ranks
    centre          - centre of the particle distribution
    radius          - half side length of the particle distribution
    box_wrap        - True if points should be wrapped into the box
    comm            - MPI communicator to use

    Returns a (total_nr_points,3) SharedArray instance.
    """
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Determine number of points per rank
    nr_points = total_nr_points // comm_size
    if comm_rank < (total_nr_points % comm_size):
        nr_points += 1
    assert comm.allreduce(nr_points) == total_nr_points

    # Make some test data
    pos = shared_array.SharedArray(
        local_shape=(nr_points, 3), dtype=np.float64, units=radius.units, comm=comm
    )
    if comm_rank == 0:
        # Rank 0 initializes all elements to avoid parallel RNG issues
        pos.full[:, :] = 2 * radius * np.random.random_sample(pos.full.shape) - radius
        pos.full[:, :] += centre[None, :].to(radius.units)
        if box_wrap:
            pos.full[:, :] = pos.full[:, :] % boxsize
            assert np.all((pos.full >= 0.0) & (pos.full < boxsize))
    pos.sync()
    comm.barrier()
    return pos


def _test_periodic_box(
    total_nr_points,
    centre,
    radius,
    boxsize,
    box_wrap,
    nr_queries,
    resolution,
    max_search_radius,
):
    """
    Test case where points fill the periodic box.

    Creates a shared mesh from random points, queries for points near random
    centres and checks the results against a simple brute force method.
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        print(
            f"Test with {total_nr_points} points, resolution {resolution} and {nr_queries} queries"
        )
        print(
            f"    Boxsize {boxsize}, centre {centre}, radius {radius}, box_wrap {box_wrap}"
        )

    def periodic_distance_squared(pos, centre):
        dr = pos - centre[None, :]
        dr[dr > 0.5 * boxsize] -= boxsize
        dr[dr < -0.5 * boxsize] += boxsize
        return np.sum(dr**2, axis=1)

    # Generate random test points
    pos = make_test_dataset(boxsize, total_nr_points, centre, radius, box_wrap, comm)

    # Construct the shared mesh
    mesh = shared_mesh.SharedMesh(comm, pos, resolution=resolution)

    # Each MPI rank queries random points and verifies the result
    nr_failures = 0
    for query_nr in range(nr_queries):

        # Pick a centre and radius
        search_centre = (np.random.random_sample((3,)) * 2 * radius) - radius + centre
        search_radius = np.random.random_sample(()) * max_search_radius

        # Query the mesh for point indexes
        idx = mesh.query_radius_periodic(search_centre, search_radius, pos, boxsize)

        # Check that the indexes are unique
        if len(idx) != len(np.unique(idx)):
            print(
                f"    Duplicate IDs for centre={search_centre}, radius={search_radius}"
            )
            nr_failures += 1
        else:
            # Flag the points in the returned index array
            in_idx = np.zeros(pos.full.shape[0], dtype=bool)
            in_idx[idx] = True
            # Find radii of all points
            r2 = periodic_distance_squared(pos.full, search_centre)
            # Check for any flagged points outside the radius
            if np.any(r2[in_idx] > search_radius * search_radius):
                print(
                    f"    Returned point outside radius for centre={search_centre}, radius={search_radius}"
                )
                nr_failures += 1
            # Check for any non-flagged points inside the radius
            missed = (in_idx == False) & (r2 < search_radius * search_radius)
            if np.any(missed):
                print(r2[missed])
                print(
                    f"    Missed point inside radius for centre={search_centre}, radius={search_radius}, rank={comm_rank}"
                )
                nr_failures += 1

    # Tidy up before possibly throwing an exception
    pos.free()
    mesh.free()

    nr_failures = comm.allreduce(nr_failures)

    comm.barrier()
    if comm_rank == 0:
        if nr_failures == 0:
            print(f"    OK")
        else:
            print(f"    {nr_failures} of {nr_queries*comm_size} queries FAILED")
            comm.Abort(1)


@pytest.mark.mpi
def test_shared_mesh():

    # Use a different, reproducible seed on each rank
    comm = MPI.COMM_WORLD
    np.random.seed(comm.Get_rank())

    resolutions = (1, 2, 4, 8, 16, 32)

    # Test a particle distribution which fills the box, searching up to 0.25 box size
    for resolution in resolutions:
        centre = 0.5 * np.ones(3, dtype=np.float64) * unyt.m
        radius = 0.5 * unyt.m
        centre, radius = comm.bcast((centre, radius))
        boxsize = 1.0 * unyt.m
        _test_periodic_box(
            1000,
            centre,
            radius,
            boxsize,
            box_wrap=False,
            nr_queries=100,
            resolution=resolution,
            max_search_radius=0.25 * boxsize,
        )

    # Test populating some random sub-regions, which may extend outside the box or be wrapped back in
    nr_regions = 10
    boxsize = 1.0 * unyt.m
    for box_wrap in (True, False):
        for resolution in resolutions:
            for region_nr in range(nr_regions):
                centre = np.random.random_sample((3,)) * boxsize
                radius = 0.25 * np.random.random_sample(()) * boxsize
                centre, radius = comm.bcast((centre, radius))
                _test_periodic_box(
                    1000,
                    centre,
                    radius,
                    boxsize,
                    box_wrap=box_wrap,
                    nr_queries=10,
                    resolution=resolution,
                    max_search_radius=radius,
                )

    # Zero particles in the box
    for resolution in resolutions:
        centre = 0.5 * np.ones(3, dtype=np.float64) * unyt.m
        radius = 0.5 * unyt.m
        centre, radius = comm.bcast((centre, radius))
        boxsize = 1.0 * unyt.m
        _test_periodic_box(
            0,
            centre,
            radius,
            boxsize,
            box_wrap=False,
            nr_queries=100,
            resolution=resolution,
            max_search_radius=0.25 * boxsize,
        )

    # One particle in the box
    for resolution in resolutions:
        centre = 0.5 * np.ones(3, dtype=np.float64) * unyt.m
        radius = 0.5 * unyt.m
        centre, radius = comm.bcast((centre, radius))
        boxsize = 1.0 * unyt.m
        _test_periodic_box(
            1,
            centre,
            radius,
            boxsize,
            box_wrap=False,
            nr_queries=100,
            resolution=resolution,
            max_search_radius=0.25 * boxsize,
        )


if __name__ == "__main__":
    test_shared_mesh()
