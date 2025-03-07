#!/bin/env python

import collections

import h5py
from mpi4py import MPI
import numpy as np
import scipy.spatial
import virgo.mpi.parallel_hdf5 as phdf5
import unyt

from . import swift_units
from . import task_queue
from . import shared_array
from .snapshot_datasets import SnapshotDatasets
from SOAP import property_table

# HDF5 chunk cache parameters:
# SWIFT writes datasets with large chunks so the default 1Mb may be too small
# (e.g. one uncompressed chunk of positions is ~24Mb in FLAMINGO 2.8Gpc run)
rdcc_nbytes = 250 * 1024 * 1024

# Type to store information about a SWIFT cell for one particle type
swift_cell_t = np.dtype(
    [
        ("centre", np.float64, 3),  # coordinates of cell centre
        ("count", np.int64),  # number of particles in the cell
        ("offset", np.int64),  # offset to first particle
        ("file", np.int32),  # file containing this cell
        ("order", np.int32),  # ordering of the cells in the snapshot file(s)
    ]
)


class DatasetCache:
    """
    Class to allow h5py File and Dataset objects to persist
    between ReadTask invocations.
    """

    def __init__(self):

        self.file_name = None
        self.infile = None
        self.dataset_name = None
        self.dataset = None

    def open_dataset(self, file_name, dataset_name):

        if file_name != self.file_name:
            self.infile = h5py.File(file_name, "r", rdcc_nbytes=rdcc_nbytes)
            self.file_name = file_name
            self.dataset_name = None
            self.dataset = None

        if dataset_name != self.dataset_name:
            self.dataset = self.infile[dataset_name]
            self.dataset_name = dataset_name

        return self.dataset

    def close(self):
        self.dataset = None
        self.dataset_name = None
        if self.infile is not None:
            self.infile.close()
        self.file_name = None


class ReadTask:
    """
    Class to execute a read of a single contiguous chunk of an array
    """

    def __init__(self, file_name, ptype, dataset, file_offset, mem_offset, count):

        self.file_name = file_name
        self.ptype = ptype
        self.dataset = dataset
        self.file_offset = file_offset
        self.mem_offset = mem_offset
        self.count = count

    def __call__(self, data, cache):

        # Find the dataset
        dataset_name = self.ptype + "/" + self.dataset
        dataset = cache.open_dataset(self.file_name, dataset_name)

        # Read the data
        mem_start = self.mem_offset
        mem_end = self.mem_offset + self.count
        file_start = self.file_offset
        file_end = self.file_offset + self.count

        dataset.read_direct(
            data[self.ptype][self.dataset].full,
            np.s_[file_start:file_end, ...],
            np.s_[mem_start:mem_end, ...],
        )


def identify_datasets(filename, nr_files, ptypes, registry):
    """
    Find units, data type and shape for datasets in snapshot-like files.
    Returns a dict with one entry per particle type. Dict keys are the
    property names and values are (shape, dtype, units) tuples.
    """
    metadata = {ptype: {} for ptype in ptypes}

    # Make a dict of flags of which particle types we still need to find
    to_find = {ptype: True for ptype in ptypes}

    # Scan snapshot files to find shape, type and units for each quantity
    for file_nr in range(nr_files):
        infile = h5py.File(filename.format(file_nr=file_nr), "r")
        nr_left = 0
        for ptype in ptypes:
            if to_find[ptype]:
                group_name = ptype
                if group_name in infile:
                    for name in infile[group_name]:
                        dset = infile[group_name][name]
                        if "a-scale exponent" in dset.attrs:
                            units = swift_units.units_from_attributes(
                                dict(dset.attrs), registry
                            )
                            dtype = dset.dtype
                            shape = dset.shape[1:]
                            metadata[ptype][name] = (units, dtype, shape)
                    to_find[ptype] = False
                else:
                    nr_left += 1
        infile.close()
        if nr_left == 0:
            break

    return metadata


class SWIFTCellGrid:
    def get_unit(self, name):
        return unyt.Unit(name, registry=self.snap_unit_registry)

    def __init__(
        self,
        snap_filename,
        extra_filenames,
        snap_filename_ref=None,
        extra_filenames_ref=None,
    ):

        # Since we require membership files to be passed when running SOAP, the
        # extra_filenames list will always have at least one entry
        self.snap_filename = snap_filename
        self.extra_filenames = extra_filenames

        # Create the SnapshotsDataset object. This is needed for aliases of column
        # names, and also for accessing named columns.
        if snap_filename_ref is None:
            self.snapshot_datasets = SnapshotDatasets([snap_filename] + extra_filenames)
        else:
            self.snapshot_datasets = SnapshotDatasets(
                [snap_filename_ref] + extra_filenames_ref
            )

        # Open the input file
        with h5py.File(snap_filename.format(file_nr=0), "r") as infile:

            # Get the snapshot unit system
            self.snap_unit_registry = swift_units.unit_registry_from_snapshot(infile)
            self.a_unit = self.get_unit("a")
            self.a = self.a_unit.base_value
            self.h_unit = self.get_unit("h")
            self.h = self.h_unit.base_value
            self.z = 1.0 / self.a - 1.0

            # Read cosmology
            self.cosmology = {}
            for name in infile["Cosmology"].attrs:
                self.cosmology[name] = infile["Cosmology"].attrs[name][0]

            # Read parameters
            self.parameters = {}
            for name in infile["Parameters"].attrs:
                self.parameters[name] = infile["Parameters"].attrs[name]

            # Read constants
            self.constants = {}
            for name in infile["PhysicalConstants"]["CGS"].attrs:
                self.constants[name] = infile["PhysicalConstants"]["CGS"].attrs[name][0]
            self.constants_internal = {}
            for name in infile["PhysicalConstants"]["InternalUnits"].attrs:
                self.constants_internal[name] = infile["PhysicalConstants"][
                    "InternalUnits"
                ].attrs[name][0]

            # Store units groups
            self.swift_units_group = {}
            for name in infile["Units"].attrs:
                self.swift_units_group[name] = infile["Units"].attrs[name][0]
            self.swift_internal_units_group = {}
            for name in infile["InternalCodeUnits"].attrs:
                self.swift_internal_units_group[name] = infile[
                    "InternalCodeUnits"
                ].attrs[name][0]

            # Store SWIFT header
            self.swift_header_group = {}
            for name in infile["Header"].attrs:
                self.swift_header_group[name] = infile["Header"].attrs[name]

            # Determine if this is a snapshot or snipshot
            self.snipshot = (
                self.swift_header_group.get("SelectOutput", b"").decode() == "Snipshot"
            )

            # Read the critical density and attach units
            # This is in internal units, which may not be the same as snapshot units.
            critical_density = float(
                self.cosmology["Critical density [internal units]"]
            )
            internal_density_unit = self.get_unit("code_mass") / (
                self.get_unit("code_length") ** 3
            )
            self.critical_density = unyt.unyt_quantity(
                critical_density, units=internal_density_unit
            )

            # Read in the softening lengths, determine whether to use comoving or physical
            self.dark_matter_softening = min(
                float(self.parameters["Gravity:comoving_DM_softening"]) * self.a,
                float(self.parameters["Gravity:max_physical_DM_softening"]),
            ) * self.get_unit("code_length")
            self.baryon_softening = min(
                float(self.parameters.get("Gravity:comoving_baryon_softening", 0))
                * self.a,
                float(self.parameters.get("Gravity:max_physical_baryon_softening", 0)),
            ) * self.get_unit("code_length")
            self.nu_softening = min(
                float(self.parameters.get("Gravity:comoving_nu_softening", 0)) * self.a,
                float(self.parameters.get("Gravity:max_physical_nu_softening", 0)),
            ) * self.get_unit("code_length")

            # Try to read in AGN_delta_T. We assert we have a valid value when we
            # create the recently_heated_gas_filter
            self.AGN_delta_T = float(
                self.parameters.get("EAGLEAGN:AGN_delta_T_K", 0)
            ) * self.get_unit("K")

            # Compute mean density at the redshift of the snapshot:
            # Here we compute the mean density in internal units at z=0 using
            # constants from the snapshot. The comoving mean density is
            # constant so we can then just scale by a**3 to get the physical
            # mean density.
            H0 = self.cosmology["H0 [internal units]"]
            G = self.constants_internal["newton_G"]
            critical_density_z0_internal = 3 * (H0**2) / (8 * np.pi * G)
            # We use non-relativistic neutrinos when we compute the mass with
            # an SO, so consider them when we calculate the reference mean density
            omega_m = self.cosmology["Omega_m"] + self.cosmology.get('Omega_nu_0', 0)
            mean_density_z0_internal = (
                critical_density_z0_internal * omega_m
            )
            mean_density_internal = mean_density_z0_internal / (self.a**3)
            self.mean_density = unyt.unyt_quantity(
                mean_density_internal, units=internal_density_unit
            )

            # Compute the BN98 critical density multiple
            Omega_k = self.cosmology["Omega_k"]
            Omega_Lambda = self.cosmology["Omega_lambda"]
            Omega_m = self.cosmology["Omega_m"]
            bnx = -(Omega_k / self.a**2 + Omega_Lambda) / (
                Omega_k / self.a**2 + Omega_m / self.a**3 + Omega_Lambda
            )
            self.virBN98 = 18.0 * np.pi**2 + 82.0 * bnx - 39.0 * bnx**2
            if self.virBN98 < 50.0 or self.virBN98 > 1000.0:
                raise RuntimeError("Invalid value for virBN98!")

            # Get the box size. Assume it's comoving with no h factors.
            comoving_length_unit = self.get_unit("snap_length") * self.a_unit
            self.boxsize = unyt.unyt_quantity(
                infile["Header"].attrs["BoxSize"][0], units=comoving_length_unit
            )

            # Get the observer position for the first lightcone
            try:
                observer_position_str = (
                    infile["Parameters"]
                    .attrs["Lightcone0:observer_position"]
                    .decode("utf-8")
                )
                observer_position = [
                    float(x) for x in observer_position_str[1:-1].split(",")
                ]
                self.observer_position = unyt.unyt_array(
                    observer_position, units=comoving_length_unit
                )
            except KeyError:
                print(
                    "Could not find lightcone observer position in snapshot file. Defaulting to centre of box."
                )
                self.observer_position = 0.5 * unyt.unyt_array([self.boxsize] * 3)

            # Get the number of files
            self.nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]

            # Read cell meta data
            self.ptypes = []
            self.nr_cells = infile["Cells/Meta-data"].attrs["nr_cells"]
            self.dimension = infile["Cells/Meta-data"].attrs["dimension"]
            self.cell_size = unyt.unyt_array(
                infile["Cells/Meta-data"].attrs["size"], units=comoving_length_unit
            )
            self.cell_centres = infile["Cells/Centres"][...]
            for name in infile["Cells/Counts"]:
                self.ptypes.append(name)

            # Create arrays of cells
            self.cell = {}
            for ptype in self.ptypes:
                self.cell[ptype] = np.ndarray(self.nr_cells, dtype=swift_cell_t)

            # Read cell info
            for ptype in self.ptypes:
                cellgrid = self.cell[ptype]
                cellgrid["centre"] = infile["Cells/Centres"][...]
                cellgrid["count"] = infile["Cells"]["Counts"][ptype][...]
                cellgrid["offset"] = infile["Cells"]["OffsetsInFile"][ptype][...]
                cellgrid["file"] = infile["Cells"]["Files"][ptype][...]

        # Determine ordering of the cells in the snapshot
        for ptype in self.ptypes:
            cellgrid = self.cell[ptype]
            idx = np.lexsort((cellgrid["offset"], cellgrid["file"]))
            for cell_order, cell_index in enumerate(idx):
                cellgrid[cell_index]["order"] = cell_order

        # Reshape into a grid
        for ptype in self.ptypes:
            self.cell[ptype] = self.cell[ptype].reshape(self.dimension)

        # Scan files to find shape and dtype etc for all quantities in the snapshot.
        self.snap_metadata = identify_datasets(
            snap_filename, self.nr_files, self.ptypes, self.snap_unit_registry
        )
        self.extra_metadata_combined = {}
        if extra_filenames is not None:
            # Unlike snap_metadata (which is a dict), extra_metadata is a list of
            # dicts. Each dict corresponds to one of the passed extra-input files
            self.extra_metadata = []
            for extra_filename in extra_filenames:
                file_metadata = identify_datasets(
                    extra_filename, self.nr_files, self.ptypes, self.snap_unit_registry
                )
                self.extra_metadata.append(file_metadata)
                for ptype in file_metadata:
                    if ptype not in self.extra_metadata_combined:
                        self.extra_metadata_combined[ptype] = {}
                    for name in file_metadata[ptype]:
                        if name in self.snap_metadata[ptype]:
                            print(
                                f"Using {ptype}/{name} from extra-input files instead of snapshot"
                            )
                        if name in self.extra_metadata_combined[ptype]:
                            raise Exception(
                                f"{ptype}/{name} is present in multiple extra-input files"
                            )
                    self.extra_metadata_combined[ptype].update(file_metadata[ptype])

        # Scan reference snapshot for missing particle types (e.g. stars or black holes at high z)
        self.ptypes_ref = []
        if snap_filename_ref is not None:
            # Determine any particle types present in the reference snapshot but not in the current snapshot
            with h5py.File(snap_filename_ref.format(file_nr=0), "r") as infile:
                for name in infile["Cells/Counts"]:
                    if name not in self.ptypes:
                        self.ptypes_ref.append(name)
            # Scan reference snapshot for properties of additional particle types
            if len(self.ptypes_ref) > 0:
                self.snap_metadata_ref = identify_datasets(
                    snap_filename_ref,
                    self.nr_files,
                    self.ptypes_ref,
                    self.snap_unit_registry,
                )
                if extra_filenames_ref is not None:
                    self.extra_metadata_ref_combined = {}
                    for extra_filename_ref in extra_filenames_ref:
                        file_metadata = identify_datasets(
                            extra_filename_ref,
                            self.nr_files,
                            self.ptypes_ref,
                            self.snap_unit_registry,
                        )
                        for ptype in file_metadata:
                            if ptype not in self.extra_metadata_ref_combined:
                                self.extra_metadata_ref_combined[ptype] = {}
                            self.extra_metadata_ref_combined[ptype].update(
                                file_metadata[ptype]
                            )

    def verify_extra_input(self, comm):
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        if comm_rank == 0:
            pass

        # Loop over the different extra-input files, checking they have the
        # same number of particles as are in the snapshot
        files_on_rank = phdf5.assign_files(self.nr_files, comm_size)
        first_file = np.cumsum(files_on_rank) - files_on_rank
        for file_nr in range(
            first_file[comm_rank], first_file[comm_rank] + files_on_rank[comm_rank]
        ):
            npart_snapshot = {}
            with h5py.File(
                self.snap_filename.format(file_nr=file_nr), "r"
            ) as snap_file:
                for parttype in self.snap_metadata:
                    # Skip if this particle type is not present in the snapshot
                    if len(self.snap_metadata[parttype]) == 0:
                        continue
                    # Pick the first dataset, assume all others are the same size
                    dset = list(self.snap_metadata[parttype].keys())[0]
                    npart_snapshot[parttype] = snap_file[f"{parttype}/{dset}"].shape[0]

            for extra_filename, extra_metadata in zip(
                self.extra_filenames, self.extra_metadata
            ):
                with h5py.File(
                    extra_filename.format(file_nr=file_nr), "r"
                ) as extra_file:
                    for parttype in self.snap_metadata:
                        # Skip if this particle type is not present in the extra files
                        if len(extra_metadata[parttype]) == 0:
                            continue
                        # Check the first dataset has the correct size
                        dset = list(extra_metadata[parttype].keys())[0]
                        npart_extra = extra_file[f"{parttype}/{dset}"].shape[0]
                        if npart_snapshot[parttype] != npart_extra:
                            print(f"Incorrect number of {parttype} in {extra_filename}")
                            comm.Abort(1)

    def check_datasets_exist(self, required_datasets, halo_prop_list):
        # Check we have all the fields needed for each property
        # Doing it at this point rather than in masked cells since we want
        # to output a list of properties that require the missing fields
        for ptype in set(self.ptypes).intersection(set(required_datasets.keys())):
            for name in required_datasets[ptype]:
                in_extra = (self.extra_filenames is not None) and (
                    name in self.extra_metadata_combined[ptype]
                )
                in_snap = name in self.snap_metadata[ptype]
                if not (in_extra or in_snap):
                    dataset = f"{ptype}/{name}"
                    print(f"The following properties require {dataset}:")
                    full_property_list = property_table.PropertyTable.full_property_list
                    for k, v in full_property_list.items():
                        # Skip property if it doesn't require this dataset
                        if dataset not in v.particle_properties:
                            continue
                        # Only print if the property is being calculated for some halo type
                        for halo_prop in halo_prop_list:
                            if halo_prop.property_filters.get(v.name, False):
                                print(f"  {v.name}")
                                break
                    raise KeyError(
                        f"Can't find required dataset {dataset} in input file(s)!"
                    )

    def prepare_read(self, ptype, mask):
        """
        Determine which ranges of particles we need to read from each file
        to read all of the cells indicated by the mask for the specified
        particle type.

        ptype - which particle type to read
        mask  - 3D boolean array with one element per cell, true if the
                cell is to be read and false otherwise

        Returns a dict where the keys are the unique file numbers to read
        and the values are lists of (offset_in_file, offset_in_memory, count) tuples.
        """

        # Make an array of the selected cells
        cells_to_read = self.cell[ptype][mask].flatten()

        # Discard any empty cells
        cells_to_read = cells_to_read[cells_to_read["count"] > 0]

        # Sort the selected cells by file, and then by offset within the file
        idx = np.lexsort((cells_to_read["offset"], cells_to_read["file"]))
        cells_to_read = cells_to_read[idx]

        # Merge adjacent cells
        max_size = 20 * 1024**2
        nr_to_read = len(cells_to_read)
        for cell_nr in range(nr_to_read - 1):
            cell1 = cells_to_read[cell_nr]
            cell2 = cells_to_read[cell_nr + 1]
            if (
                cell1["file"] == cell2["file"]
                and cell1["offset"] + cell1["count"] == cell2["offset"]
                and (cell1["count"] + cell2["count"]) <= max_size
            ):
                # Merge cells: put the particles in cell2 and empty cell1
                cell2["count"] += cell1["count"]
                cell2["offset"] = cell1["offset"]
                cell1["count"] = 0

        # Discard any cells which are now empty due to the merging
        cells_to_read = cells_to_read[cells_to_read["count"] > 0]

        # Find unique file numbers to read
        unique_file_nrs = np.unique(cells_to_read["file"])

        # Make a list of reads for each file:
        # reads[file_nr] is a list of (file_offset, memory_offset, count) tuples for file file_nr.
        mem_offset = 0
        reads = {file_nr: [] for file_nr in unique_file_nrs}
        for cell in cells_to_read:
            reads[cell["file"]].append((cell["offset"], mem_offset, cell["count"]))
            mem_offset += cell["count"]

        return reads

    def empty_mask(self):

        return np.zeros(self.dimension, dtype=bool)

    def mask_region(self, mask, pos_min, pos_max):
        imin = np.asarray(np.floor(pos_min / self.cell_size), dtype=int)
        imax = np.asarray(np.floor(pos_max / self.cell_size), dtype=int)
        for i in range(imin[0], imax[0] + 1):
            ii = i % self.dimension[0]
            for j in range(imin[1], imax[1] + 1):
                jj = j % self.dimension[1]
                for k in range(imin[2], imax[2] + 1):
                    kk = k % self.dimension[2]
                    mask[ii, jj, kk] = True

    def read_masked_cells_to_shared_memory(
        self, property_names, mask, comm, max_ranks_reading
    ):
        """
        Read in the specified properties for the cells with mask=True
        """

        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        # Make a communicator containing I/O ranks only
        colour = 0 if comm_rank < max_ranks_reading else MPI.UNDEFINED
        comm_io = comm.Split(colour, comm_rank)
        if comm_io != MPI.COMM_NULL:
            comm_io_size = comm_io.Get_size()
            comm_io_rank = comm_io.Get_rank()

        # Make a list of all reads to execute for each particle type in this snapshot
        reads_for_type = {}
        for ptype in property_names:
            if ptype in self.ptypes:
                reads_for_type[ptype] = self.prepare_read(ptype, mask)

        # Find union of file numbers to read for all particle types
        all_file_nrs = []
        for ptype in reads_for_type:
            all_file_nrs += list(reads_for_type[ptype])
        all_file_nrs = np.unique(all_file_nrs)

        # Count particles to read in
        nr_parts = {ptype: 0 for ptype in reads_for_type}
        for file_nr in all_file_nrs:
            for ptype in reads_for_type:
                if file_nr in reads_for_type[ptype]:
                    for file_offset, mem_offset, count in reads_for_type[ptype][
                        file_nr
                    ]:
                        nr_parts[ptype] += count

        # Create read tasks in the required order:
        # By file, then by particle type, then by dataset, then by offset in the file.
        # Skip datasets which exist in the extra files because we should not read them
        # from the snapshot.
        all_tasks = collections.deque()
        for file_nr in all_file_nrs:
            filename = self.snap_filename.format(file_nr=file_nr)
            for ptype in reads_for_type:
                for dataset in property_names[ptype]:
                    if (
                        dataset in self.snap_metadata[ptype]
                        and dataset not in self.extra_metadata_combined[ptype]
                    ):
                        if file_nr in reads_for_type[ptype]:
                            for file_offset, mem_offset, count in reads_for_type[ptype][
                                file_nr
                            ]:
                                all_tasks.append(
                                    ReadTask(
                                        filename,
                                        ptype,
                                        dataset,
                                        file_offset,
                                        mem_offset,
                                        count,
                                    )
                                )

        # Create additional read tasks for the extra data files
        if self.extra_filenames is not None:
            for extra_filename, file_metadata in zip(
                self.extra_filenames, self.extra_metadata
            ):
                for file_nr in all_file_nrs:
                    filename = extra_filename.format(file_nr=file_nr)
                    for ptype in reads_for_type:
                        for dataset in property_names[ptype]:
                            if dataset in file_metadata[ptype]:
                                if file_nr in reads_for_type[ptype]:
                                    for (
                                        file_offset,
                                        mem_offset,
                                        count,
                                    ) in reads_for_type[ptype][file_nr]:
                                        all_tasks.append(
                                            ReadTask(
                                                filename,
                                                ptype,
                                                dataset,
                                                file_offset,
                                                mem_offset,
                                                count,
                                            )
                                        )

        if comm_io != MPI.COMM_NULL:
            # Make one task queue per MPI rank reading
            tasks = [collections.deque() for _ in range(comm_io_size)]

            # Share tasks over the task queues roughly equally by number
            nr_tasks = len(all_tasks)
            tasks_per_rank = nr_tasks // comm_io_size
            for rank in range(comm_io_size):
                for _ in range(tasks_per_rank):
                    tasks[rank].append(all_tasks.popleft())
                if rank < nr_tasks % comm_io_size:
                    tasks[rank].append(all_tasks.popleft())
            assert len(all_tasks) == 0

        # Allocate MPI shared memory for the particle data for types which exist
        # in this snapshot. Note that these allocations could have zero size if
        # there are no particles of a type in the masked cells.
        data = {}
        for ptype in reads_for_type:
            if ptype in self.ptypes:
                data[ptype] = {}
                for name in property_names[ptype]:

                    # Get metadata for array to allocate in memory
                    if (self.extra_filenames is not None) and (
                        name in self.extra_metadata_combined[ptype]
                    ):
                        units, dtype, shape = self.extra_metadata_combined[ptype][name]
                    elif name in self.snap_metadata[ptype]:
                        units, dtype, shape = self.snap_metadata[ptype][name]
                    else:
                        # This shouldn't ever be hit because of check_datasets_exist
                        raise Exception(
                            "Can't find required dataset %s in input file(s)!" % name
                        )

                    # Determine size of local array section
                    nr_local = nr_parts[ptype] // comm_size
                    if comm_rank < (nr_parts[ptype] % comm_size):
                        nr_local += 1
                    # Find global and local shape of the array
                    global_shape = (nr_parts[ptype],) + shape
                    local_shape = (nr_local,) + shape
                    # Allocate storage
                    data[ptype][name] = shared_array.SharedArray(
                        local_shape, dtype, comm, units
                    )

        comm.barrier()

        # Execute the tasks
        if comm_io != MPI.COMM_NULL:
            cache = DatasetCache()
            task_queue.execute_tasks(
                tasks,
                args=(data, cache),
                comm_all=comm_io,
                comm_master=comm_io,
                comm_workers=MPI.COMM_SELF,
                queue_per_rank=True,
            )
            cache.close()

        # Create empty arrays for particle types which exist in the reference snapshot but not this one
        for ptype in property_names:
            if ptype in self.ptypes_ref:
                data[ptype] = {}
                for name, (units, dtype, shape) in self.snap_metadata_ref[
                    ptype
                ].items():
                    local_shape = (0,) + shape
                    data[ptype][name] = shared_array.SharedArray(
                        local_shape, dtype, comm, units
                    )
                for name, (units, dtype, shape) in self.extra_metadata_ref_combined[
                    ptype
                ].items():
                    local_shape = (0,) + shape
                    data[ptype][name] = shared_array.SharedArray(
                        local_shape, dtype, comm, units
                    )

        # Ensure all arrays have been fully written
        comm.barrier()
        for ptype in reads_for_type:
            for name in property_names[ptype]:
                data[ptype][name].sync()
        comm.barrier()

        if comm_io != MPI.COMM_NULL:
            comm_io.Free()

        return data

    def copy_swift_metadata(self, outfile):
        """
        Write simulation information etc to the specified HDF5 group
        """

        group = outfile.create_group("SWIFT")

        # Write header
        header = group.create_group("Header")
        for name, value in self.swift_header_group.items():
            header.attrs[name] = value

        # Write parameters
        params = group.create_group("Parameters")
        for name, value in self.parameters.items():
            params.attrs[name] = value

    def complete_radius_from_mask(self, mask):
        """
        Given a mask of selected cells, for each selected cell compute
        a radius within which we are guaranteed to have read all particles
        around any halo that could exist in the cell.

        Here we assume that cells can contribute particles up to half a cell
        outside their own volume, so the furthest a particle can be from the
        centre of its parent cell is equal to the cell diagonal. In the worst
        case we can have a halo at the corner of its parent cell nearest to
        a cell which has not been read. Then the radius within which we have
        all particles is limited to the distance between the cell centres
        minus 1.5 times the cell diagonal.

        This is used to handle the case where we didn't ask for a large enough
        radius around a halo: it may be that we still have enough particles in
        memory due to reading cells needed for adjacent halos.
        """

        # All types use the same grid, so just use cell arrays for the first type
        ptype = list(self.cell.keys())[0]
        cell_centre = self.cell[ptype]["centre"]
        cell_diagonal = np.sqrt(np.sum(self.cell_size.value**2))

        # Output array
        cell_complete_radius = np.zeros(self.dimension)

        # Make tree with the centers of the cells we did not read
        centre_not_read = cell_centre[mask == False, :]
        tree = scipy.spatial.cKDTree(centre_not_read, boxsize=self.boxsize.value)

        # For each cell, find the nearest cell we didn't read
        distance, index = tree.query(cell_centre, 1)
        cell_complete_radius[:, :, :] = distance.reshape(mask.shape)

        # Get a limit on the radius within which halos in the cell have all particles
        cell_complete_radius -= 1.5 * cell_diagonal
        cell_complete_radius[cell_complete_radius < 0.0] = 0.0

        # Return the result in suitable units
        comoving_length_unit = self.get_unit("snap_length") * self.a_unit
        return unyt.unyt_array(cell_complete_radius, units=comoving_length_unit)
