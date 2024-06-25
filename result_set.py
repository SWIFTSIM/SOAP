#!/bin/env python

import unyt
import numpy as np

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

import create_groups
import swift_units

from mpi4py import MPI


def concatenate(result_sets):
    """
    Concatenate a list of result sets into a single set.
    Also clears the input result sets as a side effect.
    """

    # Make sure all result sets contains the same quantity names
    for rs in result_sets:
        if sorted(list(rs.result_arrays.keys())) != sorted(
            list(result_sets[0].result_arrays.keys())
        ):
            raise Exception("Attempting to concatenate inconsistent ResultSets!")
    names = list(result_sets[0].result_arrays.keys())

    # Make sure all arrays are exactly the right size
    for rs in result_sets:
        rs.trim()

    # Check units, dtype and shape for all quantities
    for name in names:
        arr1 = rs.result_arrays[name][0]
        arr2 = result_sets[0].result_arrays[name][0]
        if arr1.dtype != arr2.dtype:
            raise Exception(
                f"Result arrays for quantity {name} have inconsistent dtypes!"
            )
        if arr1.units != arr2.units:
            raise Exception(
                f"Result arrays for quantity {name} have inconsistent units!"
            )
        if arr1.shape[1:] != arr2.shape[1:]:
            raise Exception(
                f"Result arrays for quantity {name} have inconsistent shapes!"
            )

    # Compute total number of halos in the output
    nr_halos = sum([rs.nr_halos for rs in result_sets])

    # Create the output result set
    output = ResultSet(initial_size=nr_halos)
    output.nr_halos = nr_halos

    # Concatenate arrays
    for name in names:

        # Store concatenated arrays in the output
        list_of_arrays = [rs.result_arrays[name][0] for rs in result_sets]
        concatenated_array = np.concatenate(list_of_arrays, axis=0)
        description = rs.result_arrays[name][1]
        output.result_arrays[name] = [concatenated_array, description]

        # Free input arrays (to avoid storing two copies of the full dataset at once)
        for rs in result_sets:
            del rs.result_arrays[name]
            rs.nr_halos = 0

    return output


class ResultSet:
    """
    Class to store halo properties as a dict of (unyt_array, description)
    tuples.
    """

    def __init__(self, initial_size):

        # Initially we have no data arrays
        self.nr_halos = 0
        self.initial_size = initial_size
        self.result_arrays = {}

    def __len__(self):
        return self.nr_halos

    def append(self, results):
        """
        Append results for a halo

        results - dict of the form results[name] = (description, data)
        where name is the name of the quantity, description is a description
        of the quantity, and data is a unyt_array with the value.
        """

        # Loop over quantities to store
        for result_name, (result_data, result_description, result_physical, result_a_exponent) in results.items():

            # Create a new array to store this property if necessary
            if result_name not in self.result_arrays:
                shape = (self.initial_size,) + result_data.shape
                # need to ensure we don't pass a unyt_quantity to empty_like
                # because unyt_quantities must be scalars only
                arr = np.empty_like(
                    unyt.unyt_array(result_data, registry=result_data.units.registry),
                    shape=shape,
                )
                self.result_arrays[result_name] = [arr, result_description, result_physical, result_a_exponent]

            # Find the array to store this result
            result_array, result_description, result_physical, result_a_exponent = self.result_arrays[result_name]

            # Consistency check: data type, units and shape should match the existing array
            if result_data.units != result_array.units:
                raise Exception(
                    f"Result units are inconsistent for quantity {result_name}"
                )
            if result_data.dtype != result_array.dtype:
                raise Exception(
                    f"Result dtypes are inconsistent for quantity {result_name}"
                )
            if result_data.shape != result_array.shape[1:]:
                raise Exception(
                    f"Result shapes are inconsistent for quantity {result_name}"
                )

            # Ensure the array is large enough
            if self.nr_halos >= result_array.shape[0]:
                new_shape = list(result_array.shape)
                new_shape[0] *= 2
                new_result_array = np.empty_like(
                    unyt.unyt_array(result_array, registry=result_data.units.registry),
                    shape=new_shape,
                )
                new_result_array[0 : result_array.shape[0], ...] = result_array[...]
                result_array = new_result_array
                self.result_arrays[result_name] = [result_array, result_description, result_physical, result_a_exponent]

            # Store this property for this halo to the output array
            result_array[self.nr_halos, ...] = result_data

        # Increment number of halos stored
        self.nr_halos += 1

    def trim(self):
        """
        Trim arrays down to exactly the required size
        """
        for name in self.result_arrays:
            self.result_arrays[name][0] = self.result_arrays[name][0][
                : self.nr_halos, ...
            ]

    def parallel_sort(self, key, comm):
        """
        Do an MPI parallel sort of the results on the specified sort key
        """

        # Ensure arrays are exactly the right size
        self.trim()

        # First, make sure everyone has the same set of array names
        names_local = list(self.result_arrays.keys())
        names_ref = comm.bcast(names_local)
        if names_local != names_ref:
            raise Exception(
                "Names of result arrays are not consistent between MPI ranks!"
            )

        # Make sure units, dtype and shapes also agree between MPI ranks
        for name in names_ref:
            dtype_local = self.result_arrays[name][0].dtype
            units_local = self.result_arrays[name][0].units
            shape_local = self.result_arrays[name][0].shape[1:]
            dtype_ref, units_ref, shape_ref = comm.bcast(
                (dtype_local, units_local, shape_local)
            )
            if dtype_local != dtype_ref:
                raise Exception(
                    f"Results for {name} have inconsistent dtype between MPI ranks!"
                )
            if units_local != units_ref:
                raise Exception(
                    f"Results for {name} have inconsistent units between MPI ranks!"
                )
            if shape_local != shape_ref:
                raise Exception(
                    f"Results for {name} have inconsistent shape between MPI ranks!"
                )

        # Now make a sorting index on the specified key
        idx = psort.parallel_sort(
            self.result_arrays[key][0], comm=comm, return_index=True
        )

        # And reorder the other arrays
        for name in names_ref:
            if name != key:
                self.result_arrays[name][0] = psort.fetch_elements(
                    self.result_arrays[name][0], idx, comm=comm
                )
        del idx

    def collective_write(self, outfile, comm):
        """
        Write the results to a file in collective mode
        """

        # Ensure arrays are exactly the right size
        self.trim()

        # Get names in a consistent order between MPI ranks
        names = comm.bcast(list(self.result_arrays.keys()))

        # Ensure any HDF5 groups we need exist
        group_names = comm.bcast(create_groups.find_groups_to_create(names))
        for group_name in group_names:
            outfile.create_group(group_name)

        # Loop over output arrays
        for name in names:

            # Write this array
            data, description, physical, a_exponent = self.result_arrays[name]
            phdf5.collective_write(outfile, name, data, comm)

            # Attach units metadata and description
            if hasattr(data, "units"):
                attrs = swift_units.attributes_from_units(data.units, physical, a_exponent)
                for attr_name, attr_value in attrs.items():
                    outfile[name].attrs[attr_name] = attr_value
            outfile[name].attrs["Description"] = description

    def get_metadata(self, comm):
        """
        Return a list of (name, size, units, description) for all arrays in
        the result set. Also checks that all ranks have consistent metadata,
        except for those ranks that processed zero halos.

        Returns metadata for this chunk on first rank in comm.
        Other ranks return None.

        comm should be the intra node communicator for this node.
        """

        if len(self) > 0:
            # Make a list of (names, sizes, units, descr) for properties computed
            names = sorted(self.result_arrays.keys())
            sizes = [self.result_arrays[n][0].shape[1:] for n in names]
            units = [self.result_arrays[n][0].units for n in names]
            dtype = [self.result_arrays[n][0].dtype for n in names]
            descr = [self.result_arrays[n][1] for n in names]
            physical = [self.result_arrays[n][2] for n in names]
            a_exponent = [self.result_arrays[n][3] for n in names]
            my_metadata = list(zip(names, sizes, units, dtype, descr, physical, a_exponent))
        else:
            # This rank processed zero halos
            my_metadata = None

        if comm.Get_rank() > 0:
            # Ranks >0 send their metadata to rank 0
            comm.send(my_metadata, 0)
        else:
            # Rank 0 receives and checks for consistency
            ref_metadata = my_metadata
            for other_rank in range(1, comm.Get_size()):
                recv_metadata = comm.recv(source=other_rank)
                if ref_metadata is None:
                    ref_metadata = recv_metadata
                elif recv_metadata is not None:
                    if ref_metadata != recv_metadata:
                        raise RuntimeError(
                            "MPI ranks within a chunk have inconsistent metadata!"
                        )

        # First rank on the node returns the metadata
        if comm.Get_rank() == 0:
            assert ref_metadata is not None  # All chunk tasks contain at least one halo
            return ref_metadata
        else:
            return None


def check_metadata(metadata, comm_inter_node, comm_world):
    """
    Check that the input metadata lists are consistent.
    These take the form

    metadata[chunk_nr] = [(name, size, units, description), ...]

    but are only set on the first rank on each node. Other
    ranks have metadata=None.
    """

    # Each compute node checks consistency of the zero or more chunks it processed
    if comm_inter_node is not MPI.COMM_NULL:
        # First rank on the node carries out the check
        assert metadata is not None
        for md in metadata:
            if md != metadata[0]:
                raise RuntimeError(
                    "Metadata is inconsistent between chunks within a node!"
                )
        # Just keep metadata for one chunk, since they're all the same
        if len(metadata) > 0:
            metadata = metadata[0]
        else:
            # This compute node was assigned no chunks
            metadata = None

    # Check consistency between nodes:
    # Every MPI rank has either a (name, size, units, description) tuple, or None.
    # Need to check that the non-None entries are all identical.
    if comm_world.Get_rank() > 0:
        # Everyone else sends to rank 0
        comm_world.send(metadata, 0)
        ref_metadata = None
    else:
        # Rank 0 in comm_world checks for consistency
        ref_metadata = metadata
        for other_rank in range(1, comm_world.Get_size()):
            recv_metadata = comm_world.recv(source=other_rank)
            if ref_metadata is None:
                ref_metadata = recv_metadata
            elif recv_metadata is not None:
                if ref_metadata != recv_metadata:
                    raise RuntimeError("Metadata is inconsistent between nodes!")

    # Everyone gets a copy of the reference metadata
    ref_metadata = comm_world.bcast(ref_metadata)
    assert ref_metadata is not None
    return ref_metadata
