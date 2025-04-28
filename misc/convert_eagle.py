#!/bin/env python

"""
convert_eagle.py

This script converts EAGLE GADGET snapshots into SWIFT snapshots.
Usage:

  mpirun -- python convert_eagle.py \
          --snapshot-basename=SNAPSHOT \
          --output-basename=OUTPUT \
          --membership-basename=MEMBERSHIP

where SNAPSHOT is the EAGLE snapshot (use the particledata_*** files,
since the normal snapshots don't store SubGroupNumber), and OUTPUT &
MEMBERSHIP are the names of the output files. You must run with
the same number of ranks as input files.

This script has three mains parts:
  We first have to determine the units system of the input snapshots,
  and specify the units system of the output snapshots. This step
  also includes copying over relevant cosmology information. This
  is all done by reading the headers of the input snapshots

  The second part is conversion of the datasets in the input snapshot
  to the units system of the output snapshot. This is done by
  creating a "properties" dictionary, which contains information
  about how to convert every property. More information about the
  structure of the dictionary can be found below in the code comments.

  The third part is to sort the new snapshots spatially based on
  the SWIFT cell structure. This step requires reading the coordinates
  and IDs of every particle before we can save all the other properties.

Since there are many different GADGET snapshot formats this script
is not guaranteed to work for other simulations, but the basic
structure should be the same.

For EAGLE the information about which subhalo each particle is bound
to can be found in the particledata_*** files. We therefore use these
files for this script rather than the full snapshots. This allows us
to easily output the membership files that are also required to run
SOAP. If you wanted to use the full snapshots you could obtain the
the SubGroup number by matching with the particledata_*** files based
on ParticleID.
"""

import argparse
import collections
import os
import glob

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import astropy.cosmology
import h5py
import numpy as np
import unyt

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.core import swift_units

# Parse arguments
parser = argparse.ArgumentParser(
    description=(
        "Script to convert EAGLE snapshots to SWIFT format. "
        "Also outputs membership files directly"
    )
)
parser.add_argument(
    "--snap-basename",
    type=str,
    required=True,
    help=(
        "The basename for the snapshot files (the snapshot "
        "name without the .{file_nr}.hdf5 suffix)"
    ),
)
parser.add_argument(
    "--subfind-basename",
    type=str,
    required=True,
    help=(
        "The basename for the subfind files"
    ),
)
parser.add_argument(
    "--output-basename",
    type=str,
    required=True,
    help="The basename for the output files",
)
parser.add_argument(
    "--membership-basename",
    type=str,
    required=True,
    help="The basename for the membership files",
)
args = parser.parse_args()
snap_filename = args.snap_basename + ".{file_nr}.hdf5"
subfind_filename = args.subfind_basename + ".{file_nr}.hdf5"
output_filename = args.output_basename + ".{file_nr}.hdf5"
membership_filename = args.membership_basename + ".{file_nr}.hdf5"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
os.makedirs(os.path.dirname(membership_filename), exist_ok=True)

if comm_rank == 0:
    print("Extracting basic information from EAGLE snapshot header")
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        n_file = infile["Header"].attrs["NumFilesPerSnapshot"]
        h = infile["Header"].attrs["HubbleParam"]
        a = infile["Header"].attrs["ExpansionFactor"]
        box_size_cmpc = infile["Header"].attrs["BoxSize"] / h

        ptypes = []
        nr_types = infile["Header"].attrs["NumPart_Total"].shape[0]
        nr_parts = infile["Header"].attrs["NumPart_Total"]
        nr_parts_hw = infile["Header"].attrs["NumPart_Total_HighWord"]
        for i in range(nr_types):
            if nr_parts[i] > 0 or nr_parts_hw[i] > 0:
                ptypes.append(i)

else:
    n_file, ptypes, a, h, box_size_cmpc = None, None, None, None, None
n_file = comm.bcast(n_file)
ptypes = comm.bcast(ptypes)
a = comm.bcast(a)
h = comm.bcast(h)
box_size_cmpc = comm.bcast(box_size_cmpc)

assert comm_size == n_file

# Specify the unit system of the output SWIFT snapshot
if comm_rank == 0:
    units_header = {
        "Unit current in cgs (U_I)": np.array([1.0]),
        "Unit length in cgs (U_L)": np.array([3.08567758e24]),
        "Unit mass in cgs (U_M)": np.array([1.98841e43]),
        "Unit temperature in cgs (U_T)": np.array([1.0]),
        "Unit time in cgs (U_t)": np.array([3.08567758e19]),
    }
    const_cgs_header = {
        "T_CMB_0": np.array([2.7255]),
        "newton_G": np.array([6.6743e-08]),
        "parsec": np.array([3.08567758e18]),
        "solar_mass": np.array([1.98841e33]),
        "speed_light_c": np.array([2.99792458e10]),
    }
    const_internal_header = {
        "T_CMB_0": np.array([2.7255]),
        "newton_G": np.array([43.00917552]),
        "parsec": np.array([1.0e-06]),
        "solar_mass": np.array([1.0e-10]),
        "speed_light_c": np.array([299792.458]),
    }

    # Pretend these are in an HDF5 file so we can create a unyt
    # registry using an existing SOAP function
    MockGroup = collections.namedtuple("MockGroup", ["attrs"])
    mock_swift_snap = {
        "Units": MockGroup(attrs=units_header),
        "InternalCodeUnits": MockGroup(attrs=units_header),
        "PhysicalConstants/CGS": MockGroup(attrs=const_cgs_header),
        "Cosmology": MockGroup(
            attrs={
                "Scale-factor": [a],
                "h": [h],
            }
        ),
    }
    reg = swift_units.unit_registry_from_snapshot(mock_swift_snap)

else:
    reg = None
reg = comm.bcast(reg)

# Extract information required for headers of output SWIFT snapshot
if comm_rank == 0:
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        runtime = infile["RuntimePars"]
        parameters_header = {
            "Gravity:comoving_DM_softening": runtime.attrs["SofteningHalo"],
            "Gravity:max_physical_DM_softening": runtime.attrs["SofteningHaloMaxPhys"],
            "Gravity:comoving_baryon_softening": runtime.attrs["SofteningGas"],
            "Gravity:max_physical_baryon_softening": runtime.attrs[
                "SofteningGasMaxPhys"
            ],
            "EAGLE:InitAbundance_Hydrogen": runtime.attrs['InitAbundance_Hydrogen'],
            "EAGLE:InitAbundance_Helium": runtime.attrs['InitAbundance_Helium'],
            "EAGLE:EOS_Jeans_GammaEffective": runtime.attrs['EOS_Jeans_GammaEffective'],
            "EAGLE:EOS_Jeans_TempNorm_K": runtime.attrs['EOS_Jeans_TempNorm_K'],
        }

        # Check units are indeed what we are assuming below, since HO,
        # BoxSize, etc must match with SWIFT internal units
        snap_L_in_cm = (1 * unyt.Mpc).to("cm").value
        assert np.isclose(units_header["Unit length in cgs (U_L)"][0], snap_L_in_cm)
        snap_M_in_g = (10**10 * unyt.Msun).to("g").value
        assert np.isclose(units_header["Unit mass in cgs (U_M)"][0], snap_M_in_g)
        snap_V = (1 * unyt.km / unyt.s).to("cm/s").value
        snap_t_in_s = snap_L_in_cm / snap_V
        assert np.isclose(units_header["Unit time in cgs (U_t)"][0], snap_t_in_s)

        # Cosmology
        header = infile["Header"]
        z = header.attrs["Redshift"]
        H = astropy.cosmology.Planck13.H(z).value
        G = const_internal_header["newton_G"][0]
        critical_density = 3 * (H**2) / (8 * np.pi * G)
        cosmology_header = {
            "Omega_b": header.attrs["OmegaBaryon"],
            "Omega_m": header.attrs["Omega0"],
            "Omega_k": 0,
            "Omega_nu_0": 0,
            "Omega_r": astropy.cosmology.Planck13.Ogamma(z),
            "Omega_g": astropy.cosmology.Planck13.Ogamma(z),
            "Omega_lambda": header.attrs["OmegaLambda"],
            "Redshift": z,
            "H0 [internal units]": h * 100,
            "H [internal units]": H,
            "Critical density [internal units]": critical_density,
            "Scale-factor": header.attrs["ExpansionFactor"],
            "h": h,
            "w": -1,
            "w_0": -1,
            "w_a": 0,
        }
        swift_header = {
            "BoxSize": [box_size_cmpc, box_size_cmpc, box_size_cmpc],
            "NumFilesPerSnapshot": [n_file],
            "NumPartTypes": [max(ptypes) + 1],
            "Scale-factor": [header.attrs["ExpansionFactor"]],
            "Dimension": [3],
            "Redshift": [z],
            "RunName": snap_filename.encode(),
        }


# The information in this dictionary is used to convert the datasets in
# the EAGLE files into datasets in the output SWIFT snapshot. The key
# should be the name of the dataset in the EAGLE file. The values are:
#
#   'swift_name'  - name of the dataset in output swift file
#   'exponents'   - the dimensions of the units of the dataset
#   'a_exponent'  - the scale factor exponent if the dataset is stored
#                   in comoving coordinates
#   'description' - short description of the dataset
#   'conversion_factor' - factor required to convert the values of the
#                         dataset to the units system of the output
#                         SWIFT snapshot. Note the resulting units
#                         should be h-free
#
# In the case of EAGLE snapshots most of this information is available
# in the metadata of the snapshots themselves. We therefore set those
# values as None in the properties dictionary, and read them in directly.
properties = {
    "PartType0": {
        "Coordinates": {
            "swift_name": "Coordinates",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Velocity": {
            "swift_name": "Velocities",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Mass": {
            "swift_name": "Masses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "GroupNumber": {
            "swift_name": "FOFGroupIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "ParticleIDs": {
            "swift_name": "ParticleIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Density": {
            "swift_name": "Densities",
            "exponents": {"L": -3, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "InternalEnergy": {
            "swift_name": "InternalEnergies",
            "exponents": {"L": 2, "M": 0, "T": 0, "t": -2},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "IronMassFracFromSNIa": {
            "swift_name": "IronMassFractionsFromSNIa",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Metallicity": {
            "swift_name": "MetalMassFractions",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "StarFormationRate": {
            "swift_name": "StarFormationRates",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Temperature": {
            "swift_name": "Temperatures",
            "exponents": {"L": 0, "M": 0, "T": 1, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        # Note this is handled differently to normal properties.
        # EAGLE stores each element in its own group, SWIFT expects
        # all elements to be in a 2D array
        "ElementMassFractions": {
            "swift_name": "ElementMassFractions",
            "exponents": None,
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
    },
    "PartType1": {
        "Coordinates": {
            "swift_name": "Coordinates",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Velocity": {
            "swift_name": "Velocities",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        # Note this is handled differently to normal properties.
        # All particles have the same mass in EAGLE, so there
        # is no mass dataset for dark matter.
        "Mass": {
            "swift_name": "Masses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": 0,
            "description": "Particle mass",
            "conversion_factor": None,
        },
        "GroupNumber": {
            "swift_name": "FOFGroupIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "ParticleIDs": {
            "swift_name": "ParticleIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
    },
    "PartType4": {
        "Coordinates": {
            "swift_name": "Coordinates",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Velocity": {
            "swift_name": "Velocities",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Mass": {
            "swift_name": "Masses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "GroupNumber": {
            "swift_name": "FOFGroupIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "ParticleIDs": {
            "swift_name": "ParticleIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "BirthDensity": {
            "swift_name": "BirthDensities",
            "exponents": {"L": -3, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "InitialMass": {
            "swift_name": "InitialMasses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "IronMassFracFromSNIa": {
            "swift_name": "IronMassFractionsFromSNIa",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Metallicity": {
            "swift_name": "MetalMassFractions",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "StellarFormationTime": {
            "swift_name": "BirthScaleFactors",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        # Note this is handled differently to normal properties.
        # EAGLE stores each element in its own group, SWIFT expects
        # all elements to be in a 2D array
        "ElementMassFractions": {
            "swift_name": "ElementMassFractions",
            "exponents": None,
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
    },
    "PartType5": {
        "Coordinates": {
            "swift_name": "Coordinates",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Velocity": {
            "swift_name": "Velocities",
            "exponents": {"L": 1, "M": 0, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "BH_Mass": {
            "swift_name": "SubgridMasses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "Mass": {
            "swift_name": "DynamicalMasses",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "GroupNumber": {
            "swift_name": "FOFGroupIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "ParticleIDs": {
            "swift_name": "ParticleIDs",
            "exponents": {"L": 0, "M": 0, "T": 0, "t": 0},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
        "BH_Mdot": {
            "swift_name": "AccretionRates",
            "exponents": {"L": 0, "M": 1, "T": 0, "t": -1},
            "a_exponent": None,
            "description": None,
            "conversion_factor": None,
        },
    },
}
if comm_rank == 0:
    print("Calculating conversion factor for each property using dataset metadata")
    with h5py.File(snap_filename.format(file_nr=0), "r") as infile:
        for ptype in ptypes:
            for k in infile[f"PartType{ptype}"]:
                if k not in properties.get(f"PartType{ptype}", {}):
                    continue
                a_exponent = infile[f"PartType{ptype}/{k}"].attrs["aexp-scale-exponent"]
                properties[f"PartType{ptype}"][k]["a_exponent"] = a_exponent

                description = infile[f"PartType{ptype}/{k}"].attrs["VarDescription"]
                properties[f"PartType{ptype}"][k]["description"] = description

                # Calculate the conversion factor to (h-free) SWIFT internal units
                cgs_factor = infile[f"PartType{ptype}/{k}"].attrs["CGSConversionFactor"]
                exponents = properties[f"PartType{ptype}"][k]["exponents"]
                swift_cgs_factor = 1
                swift_cgs_factor *= (1 * unyt.Unit("snap_length", registry=reg)).to(
                    "cm"
                ) ** exponents["L"]
                swift_cgs_factor *= (1 * unyt.Unit("snap_mass", registry=reg)).to(
                    "g"
                ) ** exponents["M"]
                swift_cgs_factor *= (1 * unyt.Unit("snap_time", registry=reg)).to(
                    "s"
                ) ** exponents["t"]
                swift_cgs_factor *= (
                    1 * unyt.Unit("snap_temperature", registry=reg)
                ).to("K") ** exponents["T"]
                conversion_factor = cgs_factor / swift_cgs_factor

                h_exponent = infile[f"PartType{ptype}/{k}"].attrs["h-scale-exponent"]
                if h_exponent != 0:
                    conversion_factor *= h**h_exponent

                properties[f"PartType{ptype}"][k][
                    "conversion_factor"
                ] = conversion_factor

        # Load DM mass from mass table
        if "Mass" in properties.get(f"PartType1", {}):
            dm_mass = infile["Header"].attrs["MassTable"][1] / h
            properties["PartType1"]["Mass"]["conversion_factor"] = dm_mass

        # Get list of elements for ElementMassFractions
        if "ElementMassFractions" in properties.get(f"PartType0", {}):
            elements = sorted(infile['PartType0/ElementAbundance'].keys())
        else:
            elements = None
        if "ElementMassFractions" in properties.get(f"PartType4", {}):
            star_elements = sorted(infile['PartType4/ElementAbundance'].keys())
            if elements is not None:
                for i in range(len(elements)):
                    assert elements[i] == star_elements[i]
            elements = star_elements
properties = comm.bcast(properties)


# Define cell structure required for SWIFT output
n_cell = 16
cell_size_cmpc = box_size_cmpc / n_cell
cell_counts = {}
cell_offsets = {}
cell_files = {}
idx = np.arange(n_cell)
cell_indices = np.meshgrid(idx, idx, idx, indexing="ij")
cell_indices = np.stack(cell_indices, axis=-1).reshape(-1, 3)
cell_centres = cell_indices * cell_size_cmpc
cell_centres += cell_size_cmpc / 2

# Load and process the snapshot files using multiple ranks
snap_file = phdf5.MultiFile(
    snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
)
# Load the SubFind catalogue and create an array that links the GroupNumber
# and SubGroupNumber of a subhalo to its index within the subhalo catalogue
# (for creating the membership files)
subfind_file = phdf5.MultiFile(
    subfind_filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm
)
subfind_id = subfind_file.read("Subhalo/GroupNumber").astype(np.int64)
subfind_id <<= 32
subfind_id += subfind_file.read("Subhalo/SubGroupNumber").astype(np.int64)

create_output_file = True
create_membership_file = True
for ptype in ptypes:

    if comm_rank == 0:
        print(f"Sorting PartType{ptype} arrays")
    # We sort the particles spatially (based on what cell they are in),
    # and then by particle ID
    pos = snap_file.read(f"PartType{ptype}/Coordinates") / h
    particle_id = snap_file.read(f"PartType{ptype}/ParticleIDs").astype(np.int64)

    cell_indices = (pos // cell_size_cmpc).astype(np.int64)
    assert np.min(cell_indices) >= 0
    assert np.max(cell_indices) < n_cell
    sort_hash_dtype = [("cell_index", np.int64), ("particle_id", np.int64)]
    sort_hash = np.zeros(cell_indices.shape[0], dtype=sort_hash_dtype)
    sort_hash["cell_index"] += cell_indices[:, 0] * n_cell**2
    sort_hash["cell_index"] += cell_indices[:, 1] * n_cell
    sort_hash["cell_index"] += cell_indices[:, 2]
    sort_hash["particle_id"] = particle_id
    order = psort.parallel_sort(sort_hash, return_index=True, comm=comm)

    # Calculate count of particles in each cell
    local_cell_counts = np.bincount(
        sort_hash["cell_index"], minlength=n_cell**3
    ).astype("int64")
    cell_counts[ptype] = comm.allreduce(local_cell_counts)

    # Calculate how to partition particles across files
    cells_per_file = np.zeros(n_file, dtype=int)
    cells_per_file[:] = cell_counts[ptype].shape[0] // n_file
    remainder = cell_counts[ptype].shape[0] % n_file
    cells_per_file[:remainder] += 1
    assert np.sum(cells_per_file) == n_cell**3
    i_cell = np.cumsum(cells_per_file) - 1
    elements_per_file = np.cumsum(cell_counts[ptype])[i_cell]
    elements_per_file[1:] -= elements_per_file[:-1]
    assert np.sum(elements_per_file) == np.sum(cell_counts[ptype])

    # Calculate offsets of the first particle in each cell
    cell_files[ptype] = np.repeat(np.arange(n_file), cells_per_file)
    absolute_offset = np.cumsum(cell_counts[ptype]) - cell_counts[ptype]
    file_offset = (np.cumsum(elements_per_file) - elements_per_file)[cell_files[ptype]]
    cell_offsets[ptype] = absolute_offset - file_offset

    # Loop though the datasets, sort them spatially, convert their units,
    # and write them to the output file
    for prop, prop_info in properties[f"PartType{ptype}"].items():
        if prop_info["conversion_factor"] is None:
            # The property is missing from snapshots, or we
            # would have calculated the value of 'conversion_factor'
            continue

        if comm_rank == 0:
            print(f"Converting PartType{ptype}/{prop}")

        # DM particles all have the same mass, so are not saved in the snapshots
        if (ptype == 1) and (prop == "Mass"):
            arr = np.ones(pos.shape[0])
        else:
            # Load data from file and sort according to cell structure
            arr = snap_file.read(f"PartType{ptype}/{prop}")
            arr = psort.fetch_elements(arr, order, comm=comm)

        # Convert to SWIFT internal units
        arr *= prop_info["conversion_factor"]

        # Create the units metadata for the SWIFT snapshot
        exponents = prop_info["exponents"]
        a_exponent = prop_info["a_exponent"]
        unit = 1
        unit *= unyt.Unit("snap_length", registry=reg) ** exponents["L"]
        unit *= unyt.Unit("snap_mass", registry=reg) ** exponents["M"]
        unit *= unyt.Unit("snap_time", registry=reg) ** exponents["t"]
        unit *= unyt.Unit("snap_temperature", registry=reg) ** exponents["T"]
        unit *= unyt.Unit("a", registry=reg) ** a_exponent
        unit_attrs = swift_units.attributes_from_units(unit.units, False, a_exponent)

        # Add some extra metadata
        attrs = {
            "original_name": prop,
            "Description": prop_info["description"],
        }
        attrs.update(unit_attrs)

        # Write to the output file
        arr = psort.repartition(arr, elements_per_file, comm=comm)
        if create_output_file:
            mode = "w"
            create_output_file = False
        else:
            mode = "r+"
        snap_file.write(
            {prop_info["swift_name"]: arr},
            elements_per_file,
            filenames=output_filename,
            mode=mode,
            group=f"PartType{ptype}",
            attrs={prop_info["swift_name"]: attrs},
        )

    # Handle ElementMassFractions as a special case
    if 'ElementMassFractions' in properties[f"PartType{ptype}"]:
        if comm_rank == 0:
            print(f"Converting PartType{ptype}/ElementMassFractions")
        else:
            elements = None
        elements = comm.bcast(elements)

        # Read in the array for each element
        arr = np.zeros((pos.shape[0], len(elements)))
        for i_element, element in enumerate(elements):
            element_arr = snap_file.read(f"PartType{ptype}/ElementAbundance/{element}")
            element_arr = psort.fetch_elements(element_arr, order, comm=comm)
            arr[:, i_element] = element_arr

        # Create attributes
        exponents = {"L": 0, "M": 0, "T": 0, "t": 0}
        a_exponent = 0
        unit = 1
        unit *= unyt.Unit("snap_length", registry=reg) ** exponents["L"]
        unit *= unyt.Unit("snap_mass", registry=reg) ** exponents["M"]
        unit *= unyt.Unit("snap_time", registry=reg) ** exponents["t"]
        unit *= unyt.Unit("snap_temperature", registry=reg) ** exponents["T"]
        unit *= unyt.Unit("a", registry=reg) ** a_exponent
        unit_attrs = swift_units.attributes_from_units(unit.units, False, a_exponent)
        attrs = {
            "original_name": 'ElementAbundance',
            "Description": 'Fraction of mass in each element',
        }
        attrs.update(unit_attrs)

        # Write to the output file
        arr = psort.repartition(arr, elements_per_file, comm=comm)
        if create_output_file:
            mode = "w"
            create_output_file = False
        else:
            mode = "r+"
        snap_file.write(
            {'ElementMassFractions': arr},
            elements_per_file,
            filenames=output_filename,
            mode=mode,
            group=f"PartType{ptype}",
            attrs={'ElementMassFractions': attrs},
        )

    # Create a subhalo id for each particle by combining the
    # group number and subgroup number
    sub_group = snap_file.read(f"PartType{ptype}/SubGroupNumber")
    subhalo = snap_file.read(f"PartType{ptype}/GroupNumber").astype(np.int64)
    subhalo <<= 32
    subhalo += sub_group.astype(np.int64)
    # Indicate unbound particles with -1
    bound = sub_group != 1073741824
    subhalo[np.logical_not(bound)] = -1
    # Get SubFind index of bound particles
    subhalo[bound] = psort.parallel_match(
        subhalo[bound], subfind_id, comm=comm
    )
    assert np.all(subhalo[bound] != -1)

    # Sort, add units, and write to file (same as for other properties)
    subhalo = psort.fetch_elements(subhalo, order, comm=comm)
    units = unyt.Unit("dimensionless", registry=reg)
    unit_attrs = swift_units.attributes_from_units(units, False, 0)
    attrs = {
        "Description": (
            "Unique identifier of the subhalo this particle is "
            "bound to. This is a combination of the GroupNumber and"
            "the SubGroupNumber. -1 if the particle is not bound"
        )
    }
    attrs.update(unit_attrs)
    subhalo = psort.repartition(subhalo, elements_per_file, comm=comm)
    if create_membership_file:
        mode = "w"
        create_membership_file = False
    else:
        mode = "r+"
    snap_file.write(
        {"GroupNr_bound": subhalo},
        elements_per_file,
        filenames=membership_filename,
        mode=mode,
        group=f"PartType{ptype}",
        attrs={"GroupNr_bound": attrs},
    )

# Add headers to the snapshots
if comm_rank == 0:
    print("Writing output snapshot headers")
    for i_file in range(n_file):
        with h5py.File(output_filename.format(file_nr=i_file), "r+") as outfile:
            header = outfile.create_group("Header")
            for name, value in swift_header.items():
                header.attrs[name] = value
                n_part = np.zeros(max(ptypes) + 1)
                for ptype in ptypes:
                    n_part[ptype] = outfile[f"PartType{ptype}/Coordinates"].shape[0]
                header.attrs["NumPart_ThisFile"] = n_part
            header.attrs["ThisFile"] = [i_file]
            header.attrs['NumPart_Total'] = nr_parts
            header.attrs['NumPart_Total_HighWord'] = nr_parts_hw

            cosmo = outfile.create_group("Cosmology")
            for name, value in cosmology_header.items():
                cosmo.attrs[name] = [value]

            units = outfile.create_group("Units")
            for name, value in units_header.items():
                units.attrs[name] = value
            units = outfile.create_group("InternalCodeUnits")
            for name, value in units_header.items():
                units.attrs[name] = value

            const = outfile.create_group("PhysicalConstants")
            const_cgs = const.create_group("CGS")
            for name, value in const_cgs_header.items():
                const_cgs.attrs[name] = value
            const_internal = const.create_group("InternalUnits")
            for name, value in const_internal_header.items():
                const_internal.attrs[name] = value

            params = outfile.create_group("Parameters")
            for name, value in parameters_header.items():
                params.attrs[name] = value

            cells = outfile.create_group("Cells")
            cells_metadata = cells.create_group("Meta-data")
            cells_metadata.attrs["dimension"] = np.array([n_cell, n_cell, n_cell])
            cells_metadata.attrs["nr_cells"] = [n_cell**3]
            cells_metadata.attrs["size"] = np.array(
                [cell_size_cmpc, cell_size_cmpc, cell_size_cmpc]
            )
            cells.create_dataset("Centres", data=cell_centres)
            for ptype in ptypes:
                cells.create_dataset(f"Counts/PartType{ptype}", data=cell_counts[ptype])
                cells.create_dataset(f"Files/PartType{ptype}", data=cell_files[ptype])
                cells.create_dataset(
                    f"OffsetsInFile/PartType{ptype}", data=cell_offsets[ptype]
                )

            # Store the order used for ElementMassFractions
            if elements is not None:
                subgrid_scheme = outfile.create_group("SubgridScheme")
                named_columns = subgrid_scheme.create_group("NamedColumns")
                encoded_elements = [element.encode() for element in elements]
                named_columns.create_dataset(
                    'ElementMassFractions',
                    data=encoded_elements,
                )

if comm_rank == 0:
    print("Done!")
