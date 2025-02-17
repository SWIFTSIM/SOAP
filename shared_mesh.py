#!/bin/env python

import numpy as np
import shared_array
import virgo.mpi.parallel_sort as ps
from mpi4py import MPI
import pytest


class SharedMesh:
    def __init__(self, comm, pos, resolution):
        """
        Build a mesh in shared memory which can be used to find
        particles in a particular region. Input is assumed to
        already be wrapped so we don't need to consider the periodic
        boundary.

        Input positions are stored in a SharedArray instance. Setting
        up the mesh is a collective operation over communicator comm.
        """

        comm_rank = comm.Get_rank()

        # Catch the case where there are zero particles on any rank
        if pos.full.shape[0] == 0:
            self.empty = True
            return
        else:
            self.empty = False

        # First, we need to establish a bounding box for the particles.
        # Some ranks might have no particles.
        if pos.local.shape[0] > 0:
            # This rank has particles, so find min and max coords
            pos_min_local = np.amin(pos.local, axis=0)
            pos_max_local = np.amax(pos.local, axis=0)
        else:
            # This rank has no particles so set local min and max coordinates
            # to maximum and minimum possible float values respectively.
            finfo = np.finfo(pos.local.dtype)
            pos_min_local = np.empty_like(pos.local, shape=(3,))
            pos_min_local[:] = finfo.max
            pos_max_local = np.empty_like(pos.local, shape=(3,))
            pos_max_local[:] = finfo.min

        # Then we can evaluate the minimum and maximum coordinates across
        # ranks which have particles with an allreduce.
        # Multiplication by 1 is needed because of https://github.com/SWIFTSIM/SOAP/pull/58
        self.pos_min = np.empty_like(pos_min_local * 1)
        comm.Allreduce(pos_min_local * 1, self.pos_min, op=MPI.MIN)
        self.pos_max = np.empty_like(pos_max_local * 1)
        comm.Allreduce(pos_max_local * 1, self.pos_max, op=MPI.MAX)

        # If all particles are at the same coordinates (e.g. if only one
        # particle exists), impose an arbitrary non-zero cell size.
        for i in range(3):
            if self.pos_min[i] == self.pos_max[i]:
                self.pos_max[i] = self.pos_min[i] + 1.0 * self.pos_min[i].units
        assert np.all(pos.local >= self.pos_min)
        assert np.all(pos.local <= self.pos_max)
        assert np.all(self.pos_max > self.pos_min)

        # Determine the cell size
        self.resolution = int(resolution)
        nr_cells = self.resolution ** 3
        self.cell_size = (self.pos_max - self.pos_min) / self.resolution

        # Determine which cell each particle in the local part of pos belongs to
        cell_idx = np.floor(
            (pos.local - self.pos_min[None, :]) / self.cell_size[None, :]
        ).value.astype(np.int32)
        cell_idx = np.clip(cell_idx, 0, self.resolution - 1)
        cell_idx = (
            cell_idx[:, 0]
            + self.resolution * cell_idx[:, 1]
            + (self.resolution ** 2) * cell_idx[:, 2]
        )

        # Count local particles per cell
        local_count = np.bincount(cell_idx, minlength=nr_cells)
        # Allocate a shared array to store the global count
        shape = (nr_cells,) if comm_rank == 0 else (0,)
        self.cell_count = shared_array.SharedArray(shape, local_count.dtype, comm)
        # Accumulate local counts to the shared array
        if comm_rank == 0:
            global_count = np.empty_like(local_count)
        else:
            global_count = None
        comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)
        if comm_rank == 0:
            self.cell_count.full[:] = global_count
        comm.barrier()
        self.cell_count.sync()

        # Compute offset to each cell
        self.cell_offset = shared_array.SharedArray(shape, local_count.dtype, comm)
        if comm_rank == 0:
            self.cell_offset.full[0] = 0
            if len(self.cell_offset.full) > 1:
                self.cell_offset.full[1:] = np.cumsum(self.cell_count.full[:-1])
        comm.barrier()
        self.cell_offset.sync()

        # Compute sorting index to put particles in order of cell
        sort_idx_local = ps.parallel_sort(cell_idx, comm=comm, return_index=True)
        del cell_idx

        # Merge local sorting indexes into a single shared array
        self.sort_idx = shared_array.SharedArray(
            sort_idx_local.shape, sort_idx_local.dtype, comm
        )
        self.sort_idx.local[:] = sort_idx_local
        comm.barrier()
        self.sort_idx.sync()

    def free(self):
        if not (self.empty):
            self.cell_count.free()
            self.cell_offset.free()
            self.sort_idx.free()

    def query_radius_periodic(self, centre, radius, pos, boxsize):
        """
        Return indexes of particles which are in a sphere defined by
        centre and radius. pos should be the coordinates used to build
        the mesh. This can be called independently on different MPI ranks
        since it only reads the shared data.

        This version takes the periodic boundary into account in the sense
        that it will return a particle's index if any periodic copy of that
        particle is in the specified region.
        """

        # If there are no particles on any rank, we have nothing to do
        if self.empty:
            return np.ndarray(0, dtype=int)

        def periodic_distance_squared(pos, centre):
            dr = pos - centre[None, :]
            dr[dr > 0.5 * boxsize] -= boxsize
            dr[dr < -0.5 * boxsize] += boxsize
            return np.sum(dr ** 2, axis=1)

        # Find the coordinates in the grid to search in each dimension. Here we deal with the
        # periodic box by also considering periodic copies of the search centre and radius.
        cell_coords = [set() for _ in range(3)]
        for dim in (0, 1, 2):

            # Find leftmost periodic copy of the search radius which overlaps the mesh
            min_copy_nr = 0
            while (
                centre[dim] + (min_copy_nr - 1) * boxsize + radius >= self.pos_min[dim]
            ):
                min_copy_nr -= 1

            # Find rightmost periodic copy of the search radius which overlaps the mesh
            max_copy_nr = 0
            while (
                centre[dim] + (max_copy_nr + 1) * boxsize - radius <= self.pos_max[dim]
            ):
                max_copy_nr += 1

            # Store the grid coordinates to search in this dimension
            for copy_nr in range(min_copy_nr, max_copy_nr + 1, 1):
                min_coord = max(
                    self.pos_min[dim], centre[dim] + copy_nr * boxsize - radius
                )
                min_idx = np.floor(
                    (min_coord - self.pos_min[dim]) / self.cell_size[dim]
                ).astype(int)
                max_coord = min(
                    self.pos_max[dim], centre[dim] + copy_nr * boxsize + radius
                )
                max_idx = np.floor(
                    (max_coord - self.pos_min[dim]) / self.cell_size[dim]
                ).astype(int)
                for cell_nr in range(min_idx, max_idx + 1):
                    if cell_nr >= 0 and cell_nr < self.resolution:
                        cell_coords[dim].add(cell_nr)

        # Get the indexes of particles in the required cells
        idx = []
        for k in cell_coords[2]:
            for j in cell_coords[1]:
                for i in cell_coords[0]:
                    cell_nr = i + self.resolution * j + (self.resolution ** 2) * k
                    start = self.cell_offset.full[cell_nr]
                    count = self.cell_count.full[cell_nr]
                    if count > 0:
                        idx_in_cell = self.sort_idx.full[start : start + count]
                        r2 = periodic_distance_squared(pos.full[idx_in_cell, :], centre)
                        keep = r2 <= radius * radius
                        if np.sum(keep) > 0:
                            idx.append(idx_in_cell[keep])

        # Return a single array of indexes
        if len(idx) > 0:
            return np.concatenate(idx)
        else:
            return np.ndarray(0, dtype=int)


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

    from mpi4py import MPI

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
        return np.sum(dr ** 2, axis=1)

    # Generate random test points
    pos = make_test_dataset(boxsize, total_nr_points, centre, radius, box_wrap, comm)

    # Construct the shared mesh
    mesh = SharedMesh(comm, pos, resolution=resolution)

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

    import unyt

    # Use a different, reproducible seed on each rank
    from mpi4py import MPI

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
