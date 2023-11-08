#!/bin/env python

import os.path
import threading

from mpi4py import MPI
import h5py
import numpy as np
import unyt

import virgo.util.match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as g

import domain_decomposition
import read_vr
import read_hbtplus
import read_gadget4

from mpi_tags import HALO_REQUEST_TAG, HALO_RESPONSE_TAG
from sleepy_recv import sleepy_recv


class SOCatalogue:
    def __init__(
        self,
        comm,
        halo_basename,
        halo_format,
        a_unit,
        registry,
        boxsize,
        max_halos,
        centrals_only,
        halo_ids,
        halo_prop_list,
        nr_chunks,
        halo_size_file,
    ):
        """
        This reads in the halo catalogue and stores the halo properties in a
        dict of unyt_arrays, self.local_halo, distributed over all ranks of
        communicator comm.

        self.local_halo["read_radius"] contains the radius to read in about
        the potential minimum of each halo.

        self.local_halo["search_radius"] contains an initial guess for the
        radius we need to search to reach the required overdensity. This will
        be increased up to read_radius if necessary.

        Both read_radius and search_radius will be set to be at least as large
        as the largest physical_radius_mpc specified by the halo property
        calculations.
        """

        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        
        # Get SWIFT's definition of physical and comoving Mpc units
        swift_pmpc = unyt.Unit("swift_mpc", registry=registry)
        swift_cmpc = unyt.Unit(a_unit * swift_pmpc, registry=registry)
        swift_msun = unyt.Unit("swift_msun", registry=registry)

        # Get expansion factor as a float
        a = a_unit.base_value

        # Find minimum physical radius to read in
        physical_radius_mpc = 0.0
        for halo_prop in halo_prop_list:
            physical_radius_mpc = max(
                physical_radius_mpc, halo_prop.physical_radius_mpc
            )
        physical_radius_mpc = unyt.unyt_quantity(physical_radius_mpc, units=swift_pmpc)

        # Read the input halo catalogue
        common_props = (
            "index",
            "cofp",
            "search_radius",
            "is_central",
            "nr_bound_part",
            "nr_unbound_part",
        )
        if halo_format == "VR":
            halo_data = read_vr.read_vr_catalogue(
                comm, halo_basename, a_unit, registry, boxsize
            )
        elif halo_format == "HBTplus":
            halo_data = read_hbtplus.read_hbtplus_catalogue(
                comm, halo_basename, a_unit, registry, boxsize, halo_size_file
            )
        elif halo_format == "Gadget4":
            halo_data = read_gadget4.read_gadget4_catalogue(
                comm, halo_basename, a_unit, registry, boxsize
            )
        else:
            raise RuntimeError(f"Halo format {format} not recognised!")

        # Add halo finder prefix to halo finder specific quantities:
        # This in case different finders use the same property names.
        local_halo = {}
        for name in halo_data:
            if name in common_props:
                local_halo[name] = halo_data[name]
            else:
                local_halo[f"{halo_format}/{name}"] = halo_data[name]
        del halo_data

        # Only keep halos in the supplied list of halo IDs.
        if halo_ids is not None:
            halo_ids = np.asarray(halo_ids, dtype=np.int64)
            keep = np.zeros_like(local_halo["index"], dtype=bool)
            matching_index = virgo.util.match.match(halo_ids, local_halo["index"])
            have_match = matching_index >= 0
            keep[matching_index[have_match]] = True
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # Discard satellites, if necessary
        if centrals_only:
            keep = local_halo["Structuretype"] == 10
            for name in local_halo:
                local_halo[name] = local_halo[name][keep, ...]

        # For testing: limit number of halos processed
        if max_halos > 0:
            nr_halos_local = len(local_halo["index"])
            nr_halos_prev = comm.scan(nr_halos_local) - nr_halos_local
            nr_keep_local = max_halos - nr_halos_prev
            if nr_keep_local < 0:
                nr_keep_local = 0
            if nr_keep_local > nr_halos_local:
                nr_keep_local = nr_halos_local
            for name in local_halo:
                local_halo[name] = local_halo[name][:nr_keep_local, ...]

        # Store total number of halos
        self.nr_local_halos = len(local_halo["index"])
        self.nr_halos = comm.allreduce(self.nr_local_halos, op=MPI.SUM)
        
        # Reduce the number of chunks if necessary so that all chunks have at least one halo
        nr_chunks = min(nr_chunks, self.nr_halos)
        self.nr_chunks = nr_chunks
        
        # Assign halos to chunk tasks:
        # This sorts the halos by chunk across all MPI ranks and returns the size of each chunk.
        chunk_size = domain_decomposition.peano_decomposition(
            boxsize, local_halo, nr_chunks, comm
        )

        # Compute initial radius to read in about each halo
        local_halo["read_radius"] = local_halo["search_radius"].copy()
        min_radius = 5.0 * swift_cmpc
        local_halo["read_radius"] = local_halo["read_radius"].clip(min=min_radius)

        # Ensure that both the initial search radius and the radius to read in
        # are >= the minimum physical radius required by property calculations
        local_halo["read_radius"] = local_halo["read_radius"].clip(
            min=physical_radius_mpc
        )
        local_halo["search_radius"] = local_halo["search_radius"].clip(
            min=physical_radius_mpc
        )

        # Determine what range of halos is stored on each MPI rank
        self.local_halo = local_halo
        self.local_halo_offset = comm.scan(self.nr_local_halos) - self.nr_local_halos

        # Determine global offset to the first halo in each chunk
        self.chunk_size = chunk_size
        self.chunk_offset = np.cumsum(chunk_size) - chunk_size

        # Determine local offset to the first halo in each chunk.
        # This will be different on each MPI rank.
        self.local_chunk_size = np.zeros(nr_chunks, dtype=int)
        self.local_chunk_offset = np.zeros(nr_chunks, dtype=int)
        for chunk_nr in range(nr_chunks):
            # Find the range of local halos which are in this chunk (may be none)
            i1 = self.chunk_offset[chunk_nr] - self.local_halo_offset
            if i1 < 0:
                i1 = 0
            i2 = self.chunk_offset[chunk_nr] + self.chunk_size[chunk_nr] - self.local_halo_offset
            if i2 > self.nr_local_halos:
                i2 = self.nr_local_halos
            # Record the range
            if i2 > i1:
                self.local_chunk_size[chunk_nr] = i2 - i1
                self.local_chunk_offset[chunk_nr] = i1
            else:
                self.local_chunk_size[chunk_nr] = 0
                self.local_chunk_offset[chunk_nr] = 0
        assert np.all(comm.allreduce(self.local_chunk_size) == chunk_size)

        # Now, for each chunk we need to know which MPI ranks have halos from that chunk.
        # Here we make an array with one element per chunk. Each MPI rank enters its own rank
        # index in every chunk for which it has >0 halos. We then find the min and max of
        # each array element over all MPI ranks.
        chunk_min_rank = np.ones(nr_chunks, dtype=int) * comm_size # One more than maximum rank
        chunk_max_rank = np.ones(nr_chunks, dtype=int) -1          # One less than minimum rank
        for chunk_nr in range(nr_chunks):
            if self.local_chunk_size[chunk_nr] > 0:
                chunk_min_rank[chunk_nr] = comm_rank
                chunk_max_rank[chunk_nr] = comm_rank
        comm.Allreduce(MPI.IN_PLACE, chunk_min_rank, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, chunk_max_rank, op=MPI.MAX)
        assert np.all(chunk_min_rank < comm_size)
        assert np.all(chunk_min_rank >= 0)
        assert np.all(chunk_max_rank < comm_size)
        assert np.all(chunk_max_rank >= 0)
        
        # Check that chunk_[min|max]_rank is consistent with local_chunk_size
        for chunk_nr in range(nr_chunks):
            assert (comm_rank >= chunk_min_rank[chunk_nr] and comm_rank <= chunk_max_rank[chunk_nr]) == (self.local_chunk_size[chunk_nr] > 0)
                
        self.chunk_min_rank = chunk_min_rank
        self.chunk_max_rank = chunk_max_rank

        # Store halo property names in an order which is consistent between MPI ranks
        self.prop_names = sorted(self.local_halo.keys())
        self.comm = comm
        
    def process_requests(self):
        """
        Wait for and respond to requests for halo data.
        To be run in a separate thread. Request chunk -1 to terminate.
        """
        comm = self.comm
        
        while True:

            # Receive the requested chunk number and check where the request came form
            status = MPI.Status()
            chunk_nr = int(sleepy_recv(self.comm, HALO_REQUEST_TAG, status=status))
            src_rank = status.Get_source()
            if chunk_nr < 0:
                break
            assert self.local_chunk_size[chunk_nr] > 0 # Should only get requests for chunks we have locally

            # Return our local part of the halo catalogue arrays for the
            # requested chunk.
            for name in self.prop_names:
                i1 = self.local_chunk_offset[chunk_nr]
                i2 = self.local_chunk_offset[chunk_nr] + self.local_chunk_size[chunk_nr]
                sendbuf = self.local_halo[name][i1:i2,...]
                # First send the type, dimensions and units
                comm.send((sendbuf.shape, sendbuf.dtype, sendbuf.units), dest=src_rank, tag=HALO_RESPONSE_TAG)
                # Then send the data
                comm.Send(sendbuf, dest=src_rank, tag=HALO_RESPONSE_TAG)
        
    def start_request_thread(self):
        """
        Start a thread to respond to requests for halos
        """
        self.request_thread = threading.Thread(target=self.process_requests)
        self.request_thread.start()

    def request_chunk(self, chunk_nr):
        """
        Request the halo catalogue for the specified chunk from whichever
        MPI ranks contain the halos.

        TODO: should send off all requests simultaneously in non blocking mode
        to avoid serializing delays waiting for ranks to respond.
        """
        comm = self.comm
        
        # Loop over ranks which have data we need        
        data = {name : [] for name in self.prop_names}
        for rank_nr in range(self.chunk_min_rank[chunk_nr], self.chunk_max_rank[chunk_nr]+1):

            # Request the halos from this rank
            comm.send(chunk_nr, dest=rank_nr, tag=HALO_REQUEST_TAG)

            # Receive the arrays from this rank
            for name in self.prop_names:
                shape, dtype, units = comm.recv(source=rank_nr, tag=HALO_RESPONSE_TAG)
                recvbuf = np.ndarray(shape, dtype=dtype)
                comm.Recv(recvbuf, source=rank_nr, tag=HALO_RESPONSE_TAG)
                recvbuf = unyt.unyt_array(recvbuf, units=units)
                data[name].append(recvbuf)

        # Combine the arrays and return the halo catalogue
        for name in data:
            data[name] = unyt.array.uconcatenate(data[name])
        return data

    def stop_request_thread(self):
        """
        Send a terminate signal to the request thread then join it.
        Need to be sure that no requests are pending before calling this,
        which should be the guaranteed if all chunk tasks have completed.
        """

        # This send should match a pending sleepy_recv() on this rank's halo request thread
        self.comm.send(-1, dest=self.comm.Get_rank(), tag=HALO_REQUEST_TAG)

        # Request thread should now be returning
        self.request_thread.join()
