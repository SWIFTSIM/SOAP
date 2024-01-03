#!/bin/env python

import numpy as np
import shared_array
import virgo.mpi.parallel_sort as ps
from mpi4py import MPI


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
        assert np.all(pos.local >= self.pos_min)
        assert np.all(pos.local <= self.pos_max)

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

    def query(self, pos_min, pos_max):
        """
        Return indexes of particles which might be in the region defined
        by pos_min and pos_max. This can be called independently on
        different MPI ranks since it only reads the shared data.
        """

        # If there are no particles on any rank, we have nothing to do
        if self.empty:
            return np.ndarray(0, dtype=int)

        # Find range of cells involved
        cell_min_idx = np.floor((pos_min - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )
        cell_min_idx = np.clip(cell_min_idx, 0, self.resolution - 1)
        cell_max_idx = np.floor((pos_max - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )
        cell_max_idx = np.clip(cell_max_idx, 0, self.resolution - 1)

        # Get the indexes of particles in the required cells
        idx = []
        for k in range(cell_min_idx[2], cell_max_idx[2] + 1):
            for j in range(cell_min_idx[1], cell_max_idx[1] + 1):
                for i in range(cell_min_idx[0], cell_max_idx[0] + 1):
                    cell_nr = i + self.resolution * j + (self.resolution ** 2) * k
                    start = self.cell_offset.full[cell_nr]
                    count = self.cell_count.full[cell_nr]
                    if count > 0:
                        idx.append(self.sort_idx.full[start : start + count])

        # Return a single array of indexes
        if len(idx) > 0:
            return np.concatenate(idx)
        else:
            return np.ndarray(0, dtype=int)

    def query_radius(self, centre, radius, pos):
        """
        Return indexes of particles which are in a sphere defined by
        centre and radius. pos should be the coordinates used to build
        the mesh. This can be called independently on different MPI ranks
        since it only reads the shared data.
        """

        # If there are no particles on any rank, we have nothing to do
        if self.empty:
            return np.ndarray(0, dtype=int)

        pos_min = centre - radius
        pos_max = centre + radius

        # Find range of cells involved
        cell_min_idx = np.floor((pos_min - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )
        cell_min_idx = np.clip(cell_min_idx, 0, self.resolution - 1)
        cell_max_idx = np.floor((pos_max - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )
        cell_max_idx = np.clip(cell_max_idx, 0, self.resolution - 1)

        # Get the indexes of particles in the required cells
        idx = []
        for k in range(cell_min_idx[2], cell_max_idx[2] + 1):
            for j in range(cell_min_idx[1], cell_max_idx[1] + 1):
                for i in range(cell_min_idx[0], cell_max_idx[0] + 1):
                    cell_nr = i + self.resolution * j + (self.resolution ** 2) * k
                    start = self.cell_offset.full[cell_nr]
                    count = self.cell_count.full[cell_nr]
                    if count > 0:
                        idx_in_cell = self.sort_idx.full[start : start + count]
                        r2 = np.sum(
                            (pos.full[idx_in_cell, :] - centre[None, :]) ** 2, axis=1
                        )
                        keep = r2 <= radius * radius
                        if np.sum(keep) > 0:
                            idx.append(idx_in_cell[keep])

        # Return a single array of indexes
        if len(idx) > 0:
            return np.concatenate(idx)
        else:
            return np.ndarray(0, dtype=int)

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

        pos_min = centre - radius
        pos_max = centre + radius

        # Find range of cells involved
        cell_min_idx = np.floor((pos_min - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )
        cell_max_idx = np.floor((pos_max - self.pos_min) / self.cell_size).value.astype(
            np.int32
        )

        def wrap_coord(dim, i):
            if i < 0:
                return np.floor(
                    ((i + 0.5) * self.cell_size[dim] + boxsize) / self.cell_size[dim]
                ).value.astype(np.int32)
            elif i >= self.resolution:
                return np.floor(
                    ((i + 0.5) * self.cell_size[dim] - boxsize) / self.cell_size[dim]
                ).value.astype(np.int32)
            else:
                return i

        def periodic_distance_squared(pos, centre):
            dr = pos - centre[None, :]
            dr[dr > 0.5 * boxsize] -= boxsize
            dr[dr < -0.5 * boxsize] += boxsize
            return np.sum(dr ** 2, axis=1)

        # Get the indexes of particles in the required cells
        idx = []
        for k in range(cell_min_idx[2], cell_max_idx[2] + 1):
            kk = wrap_coord(2, k)
            if kk >= 0 and kk < self.resolution:
                for j in range(cell_min_idx[1], cell_max_idx[1] + 1):
                    jj = wrap_coord(1, j)
                    if jj >= 0 and jj < self.resolution:
                        for i in range(cell_min_idx[0], cell_max_idx[0] + 1):
                            ii = wrap_coord(0, i)
                            if ii >= 0 and ii < self.resolution:
                                cell_nr = (
                                    ii
                                    + self.resolution * jj
                                    + (self.resolution ** 2) * kk
                                )
                                start = self.cell_offset.full[cell_nr]
                                count = self.cell_count.full[cell_nr]
                                if count > 0:
                                    idx_in_cell = self.sort_idx.full[
                                        start : start + count
                                    ]
                                    r2 = periodic_distance_squared(
                                        pos.full[idx_in_cell, :], centre
                                    )
                                    keep = r2 <= radius * radius
                                    if np.sum(keep) > 0:
                                        idx.append(idx_in_cell[keep])

        # Return a single array of indexes
        if len(idx) > 0:
            return np.concatenate(idx)
        else:
            return np.ndarray(0, dtype=int)
