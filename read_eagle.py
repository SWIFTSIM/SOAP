#!/bin/env python

import os
import glob

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np
import unyt

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

import swift_units


# TODO: argparse

# /cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data
sim_dir = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0025N0376/PE/REFERENCE/data'
output_dir = '/snap7/scratch/dp004/dc-mcgi1'
swift_filename = '/cosma8/data/dp004/colibre/Runs/L0012N0376/Thermal_non_equilibrium/snapshots/colibre_0127/colibre_0127.hdf5'
snap_nr = 28

snap_dir = glob.glob(f'{sim_dir}/snapshot_{snap_nr:03}_*')[0]
z_suffix = os.path.basename(snap_dir).split('_')[2]
snap_filename = f'{snap_dir}/snap_{snap_nr:03}_{z_suffix}' + '.{file_nr}.hdf5'
output_filename = f'{output_dir}/{os.path.basename(snap_dir)}'
os.makedirs(output_filename, exist_ok=True)
output_filename += f'/snap_{snap_nr:03}_{z_suffix}' + '.{file_nr}.hdf5'



# Extract particle types present, hubble param, and box size
if comm_rank == 0:
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        n_file = infile['Header'].attrs['NumFilesPerSnapshot']
        h = infile['Header'].attrs['HubbleParam']
        box_size_cmpc = infile['Header'].attrs['BoxSize'] / h

        ptypes = []
        nr_types = infile["Header"].attrs["NumPart_Total"].shape[0]
        nr_parts = infile["Header"].attrs["NumPart_Total"]
        nr_parts_hw = infile["Header"].attrs["NumPart_Total_HighWord"]
        for i in range(nr_types):
            if nr_parts[i] > 0 or nr_parts_hw[i] > 0:
                ptypes.append(i)

    # TODO: REMOVE
    ptypes = [5]
else:
    n_file = None
    ptypes = None
    h = None
    box_size_cmpc = None
n_file = comm.bcast(n_file)
ptypes = comm.bcast(ptypes)
h = comm.bcast(h)
box_size_cmpc = comm.bcast(box_size_cmpc)

# TODO: Can we run with less ranks than files?
assert comm_size == n_file

# Load units from a reference SWIFT snapshot
if comm_rank == 0:
    with h5py.File(swift_filename, 'r') as file:
        reg = swift_units.unit_registry_from_snapshot(file)
        units_header = dict(file['Units'].attrs)
        constants_header = dict(file['PhysicalConstants/CGS'].attrs)
else:
    reg = None
reg = comm.bcast(reg)


# The information in this dictionary is used to convert the datasets in
# the EAGLE files into datasets in the output SWIFT snapshot. The key
# should be the name of the dataset in the EAGLE file. The values are:
#   'swift_name'  - name of dataset in output swift file
#   'exponents'   - what are the dimensions of the units of this dataset
#   'a_exponent'  - if this dataset is stored in comoving coordinates
#                   what is the scale factor exponent
#   'description' - short description of this dataset
#   'conversion_factor' - factor required to convert the values of this
#                         dataset to the snapshot units of the reference
#                         SWIFT snapshot. Note the resulting units should
#                         be h-free
#
# In the case of EAGLE snapshots most of this information is available
# in the metadata of the snapshots themselves. We therefore leave those
# value blank in the properties dictionary, and read them in directly.

properties = {
    'PartType5':{
        'Coordinates': {
            'swift_name': 'Coordinates',
            'exponents': {'L': 1, 'M': 0, 'T': 0, 't': 0},
            'a_exponent': None,
            'description': None,
            'conversion_factor': None,
        },
        'ParticleIDs': {
            'swift_name': 'ParticleIDs',
            'exponents': {'L': 0, 'M': 0, 'T': 0, 't': 0},
            'a_exponent': None,
            'description': None,
            'conversion_factor': None,
        },
    },
}
# Extract unit metadata from the snapshots
if comm_rank == 0:
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        for ptype in ptypes:
            for k in infile[f'PartType{ptype}']:
                if k not in properties.get(f'PartType{ptype}', {}):
                    continue
                a_exponent = infile[f'PartType{ptype}/{k}'].attrs['aexp-scale-exponent']
                properties[f'PartType{ptype}'][k]['a_exponent'] = a_exponent

                description = infile[f'PartType{ptype}/{k}'].attrs['VarDescription']
                properties[f'PartType{ptype}'][k]['description'] = description

                # Calculate the conversion factor to (h-free) SWIFT internal units
                cgs_factor = infile[f'PartType{ptype}/{k}'].attrs['CGSConversionFactor']
                exponents = properties[f'PartType{ptype}'][k]['exponents']
                swift_cgs_factor = 1
                swift_cgs_factor *= (1 * unyt.Unit('snap_length', registry=reg)).to('cm') ** exponents['L']
                swift_cgs_factor *= (1 * unyt.Unit('snap_mass', registry=reg)).to('g') ** exponents['M']
                swift_cgs_factor *= (1 * unyt.Unit('snap_time', registry=reg)).to('s') ** exponents['t']
                swift_cgs_factor *= (1 * unyt.Unit('snap_temperature', registry=reg)).to('K') ** exponents['T']
                conversion_factor = cgs_factor / swift_cgs_factor

                h_exponent = infile[f'PartType{ptype}/{k}'].attrs['h-scale-exponent']
                if h_exponent != 0:
                    conversion_factor *= h ** h_exponent

                properties[f'PartType{ptype}'][k]['conversion_factor'] = conversion_factor
properties = comm.bcast(properties)

# Create cosmology header for output file
if comm_rank == 0:
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        cosmology = {
            'Omega_b': np.array(infile['Header'].attrs['OmegaBaryon']),
            'Omega_m': np.array(infile['Header'].attrs['Omega0']),
            'Omega_lambda': np.array(infile['Header'].attrs['OmegaLambda']),
            'Redshift': np.array(infile['Header'].attrs['Redshift']),
        }

# Define cell structure required for SWIFT output
n_cell = 16
cell_size_cmpc = box_size_cmpc / n_cell
cell_counts = {}
cell_offsets = {}
cell_files = {}
idx = np.arange(n_cell)
cell_indices = np.meshgrid(idx, idx, idx, indexing='ij')
cell_indices = np.stack(cell_indices, axis=-1).reshape(-1, 3)
cell_centres = cell_indices * cell_size_cmpc
cell_centres += cell_size_cmpc / 2

# Load the snapshot files using multiple ranks
snap_file = phdf5.MultiFile(
    snap_filename, 
    file_nr_attr=("Header","NumFilesPerSnapshot"),
    comm=comm
)

create_file = True
for ptype in ptypes:

    # We sort the particles spatially (based on what cell they are in),
    # and then by particle ID
    pos = snap_file.read(f"PartType{ptype}/Coordinates") / h
    particle_id = snap_file.read(f"PartType{ptype}/ParticleIDs").astype(np.int64)

    cell_indices = (pos // cell_size_cmpc).astype(np.int64)
    assert np.min(cell_indices) >= 0
    assert np.max(cell_indices) < n_cell
    sort_hash_dtype = [("cell_index", np.int64), ("particle_id", np.int64)]
    sort_hash = np.zeros(cell_indices.shape[0], dtype=sort_hash_dtype)
    sort_hash["cell_index"] += cell_indices[:, 0] * n_cell ** 2
    sort_hash["cell_index"] += cell_indices[:, 1] * n_cell
    sort_hash["cell_index"] += cell_indices[:, 2]
    sort_hash["particle_id"] = particle_id
    order = psort.parallel_sort(sort_hash, return_index=True, comm=comm)

    # Calculate local count of particles in each cell
    local_cell_counts = np.bincount(
        sort_hash["cell_index"], minlength=n_cell**3
    ).astype("int64")
    cell_counts[ptype] = comm.allreduce(local_cell_counts)

    # Calculate how to partition particles
    cells_per_file = np.zeros(n_file, dtype=int)
    cells_per_file[:] = cell_counts[ptype].shape[0] // n_file
    remainder = cell_counts[ptype].shape[0] % n_file
    cells_per_file[:remainder] += 1
    assert np.sum(cells_per_file) == n_cell**3
    i_cell = np.cumsum(cells_per_file) - 1
    elements_per_file = np.cumsum(cell_counts[ptype])[i_cell]
    elements_per_file[1:] -= elements_per_file[:-1]
    assert np.sum(elements_per_file) == np.sum(cell_counts[ptype])

    # Calculate cell offsets
    cell_files[ptype] = np.repeat(np.arange(n_file), cells_per_file)
    absolute_offset = np.cumsum(cell_counts[ptype]) - cell_counts[ptype]
    file_offset = (np.cumsum(elements_per_file) - elements_per_file)[cell_files[ptype]]
    cell_offsets[ptype] = absolute_offset - file_offset

    # Loop though the datasets, sort them spatially, convert their units,
    # and write them to the output file
    for prop, prop_info in properties[f'PartType{ptype}'].items():
        if prop_info['conversion_factor'] is None:
            # The property is missing from snapshots, or we 
            # would have calculated the value of 'conversion_factor'
            continue

        # Load data from file and sort according to cell structure
        arr = snap_file.read(f"PartType{ptype}/{prop}")
        arr = psort.fetch_elements(arr, order, comm=comm)

        # Convert to SWIFT internal units
        arr *= prop_info['conversion_factor']

        # Create the units metadata for the SWIFT snapshot
        exponents = prop_info['exponents']
        a_exponent = prop_info['a_exponent']
        unit = 1
        unit *= unyt.Unit('snap_length', registry=reg) ** exponents['L']
        unit *= unyt.Unit('snap_mass', registry=reg) ** exponents['M']
        unit *= unyt.Unit('snap_time', registry=reg) ** exponents['t']
        unit *= unyt.Unit('snap_temperature', registry=reg) ** exponents['T']
        unit *= unyt.Unit('a', registry=reg) ** a_exponent
        unit_attrs = swift_units.attributes_from_units(unit.units, False, a_exponent)

        # Add some extra metadata
        attrs = {
            'original_name': prop,
            'Description': prop_info['description'],
        }
        attrs.update(unit_attrs)

        # Write to the output file
        arr = psort.repartition(arr, elements_per_file, comm=comm)
        if create_file:
            mode = "w"
            create_file = False
        else:
            mode = "r+"
        snap_file.write(
            {prop_info['swift_name']: arr},
            elements_per_file,
            filenames=output_filename,
            mode=mode,
            group=f'PartType{ptype}',
            attrs=attrs,
        )

    # TODO: Write GroupNr_bound to a membership file



if comm_rank != 0:
    exit()

# TODO: Parallel
for i_file in range(n_file):
    with h5py.File(output_filename.format(file_nr=i_file), "r+") as outfile:
        # Write cosmology
        cosmo = outfile.create_group("Cosmology")
        for name, value in cosmology.items():
            cosmo.attrs[name] = [value]

        # Write units
        units = outfile.create_group("Units")
        for name, value in units_header.items():
            units.attrs[name] = value

        # Write physical constants
        const = outfile.create_group("PhysicalConstants")
        const = const.create_group("CGS")
        for name, value in constants_header.items():
            const.attrs[name] = value

        # TODO: 
        params = outfile.create_group("Parameters")
        for name, value in {
                'Gravity:comoving_DM_softening': 0,
                'Gravity:max_physical_DM_softening': 0,
            }.items():
            params.attrs[name] = value

        # Write cell information
        cells = outfile.create_group("Cells")
        cells_metadata = cells.create_group("Meta-data")
        cells_metadata.attrs["dimension"] = np.array([n_cell, n_cell, n_cell])
        cells_metadata.attrs["nr_cells"] = n_cell ** 3
        cells_metadata.attrs["size"] = np.array([cell_size_cmpc, cell_size_cmpc, cell_size_cmpc])
        cells.create_dataset("Centres", data=cell_centres)

        for ptype in ptypes:
            cells.create_dataset(f"Counts/PartType{ptype}", data=cell_counts[ptype])
            cells.create_dataset(f"Files/PartType{ptype}", data=cell_files[ptype])
            cells.create_dataset(f"OffsetsInFile/PartType{ptype}", data=cell_offsets[ptype])




