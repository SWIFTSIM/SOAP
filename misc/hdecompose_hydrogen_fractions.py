#!/bin/env python

"""
approximate_hydrogen_fractions.py

This script calculates HI and H2 species fractions using the
Hdecompose package (https://github.com/kyleaoman/Hdecompose).
Usage:

  mpirun -- python approximate_hydrogen_fractions.py \
          --snapshot-basename=SNAPSHOT \
          --output-basename=OUTPUT

where SNAPSHOT is the basename of the input Swift snapshots and
OUTPUT is basename of the output files.
"""

import argparse
import collections
import os
import glob

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import astropy.units
import h5py
import numpy as np
import unyt

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from SOAP.core import swift_units

# TODO: Install Hdecompose from github, pypi version is out of date
from Hdecompose.RahmatiEtal2013 import neutral_frac as calculate_neutral_frac
from Hdecompose.BlitzRosolowsky2006 import molecular_frac as calculate_molecular_frac

# Parse arguments
parser = argparse.ArgumentParser(
    description=("Script to estimate HI and H2 species fractions.")
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
    "--output-basename",
    type=str,
    required=True,
    help="The basename for the output files",
)
args = parser.parse_args()
snap_filename = args.snap_basename + ".{file_nr}.hdf5"
output_filename = args.output_basename + ".{file_nr}.hdf5"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

# List of gas properties we require from the input snapshots
property_list = [
    "Densities",
    "StarFormationRates",
    "ElementMassFractions",
    "Temperatures",
]

if comm_rank == 0:
    print("Reading in run parameters and units")
    params = {}
    units = {}
    with h5py.File(snap_filename.format(file_nr=0), "r") as file:
        params["z"] = file["Cosmology"].attrs["Redshift"]
        params["fH"] = file["Parameters"].attrs["EAGLE:InitAbundance_Hydrogen"]
        params["fHe"] = file["Parameters"].attrs["EAGLE:InitAbundance_Helium"]
        params["gamma"] = file["Parameters"].attrs["EAGLE:EOS_Jeans_GammaEffective"]
        params["T0"] = (
            file["Parameters"].attrs["EAGLE:EOS_Jeans_TempNorm_K"] * astropy.units.K
        )

        reg = swift_units.unit_registry_from_snapshot(file)
        for prop in property_list:
            attrs = file[f"PartType0/{prop}"].attrs
            units[prop] = swift_units.units_from_attributes(attrs, reg)
        elements = [
            e.decode()
            for e in file["SubgridScheme/NamedColumns/ElementMassFractions"][:]
        ]
        i_H = elements.index("Hydrogen")

        n_file = file["Header"].attrs["NumFilesPerSnapshot"][0]
else:
    params = None
    units = None
    i_H = None
params = comm.bcast(params)
units = comm.bcast(units)
i_H = comm.bcast(i_H)

if comm_rank == 0:
    print("Loading data")

# Load raw arrays from file
snap_file = phdf5.MultiFile(
    snap_filename, file_nr_attr=("Header", "NumFilesPerSnapshot")
)
raw_properties = snap_file.read(property_list, "PartType0")

# Convert to astropy arrays (physical not comoving)
densities = (
    (raw_properties["Densities"] * units["Densities"]).to("g/cm**3").to_astropy()
)
temperatures = (
    (raw_properties["Temperatures"] * units["Temperatures"]).to("K").to_astropy()
)
# SFR units don't matter, we just need to know if they are nonzero
sfr = raw_properties["StarFormationRates"]
Habundance = raw_properties["ElementMassFractions"][:, i_H]

# Free up some memory
del raw_properties

if comm_rank == 0:
    print("Calculating neutral fraction")
mu = 1 / (params["fH"] + 0.25 * params["fHe"])
neutral_frac = calculate_neutral_frac(
    params["z"],
    densities * Habundance / (mu * astropy.constants.m_p),
    temperatures,
    onlyA1=True,
    noCol=False,
    onlyCol=False,
    SSH_Thresh=False,
    local=False,
    EAGLE_corrections=True,
    TNG_corrections=False,
    SFR=sfr,
    mu=mu,
    gamma=params["gamma"],
    fH=params["fH"],
    Habundance=Habundance,
    T0=params["T0"],
    rho=densities,
)

if comm_rank == 0:
    print("Calculating molecular fraction")
molecular_mass_frac = calculate_molecular_frac(
    temperatures,
    densities,
    EAGLE_corrections=True,
    SFR=sfr,
    mu=mu,
    gamma=params["gamma"],
    fH=params["fH"],
    T0=params["T0"],
)

if comm_rank == 0:
    print("Writing output")
species_names = ["HI", "H2"]
species_fractions = np.zeros((neutral_frac.shape[0], 2))
species_fractions[:, 0] = (1.0 - molecular_mass_frac) * neutral_frac
species_fractions[:, 1] = molecular_mass_frac / 2
attrs = {
    "SpeciesFractions": {
        "Description": "The fraction of species i in terms of its number density relative to hydrogen, i.e. n_i / n_H_tot.",
        "Conversion factor to CGS (not including cosmological corrections)": [1.0],
        "Conversion factor to physical CGS (including cosmological corrections)": [1.0],
        "U_I exponent": [0.0],
        "U_L exponent": [0.0],
        "U_M exponent": [0.0],
        "U_t exponent": [0.0],
        "U_T exponent": [0.0],
        "a-scale exponent": [0.0],
        "h-scale exponent": [0.0],
        "Property can be converted to comoving": [1],
        "Value stored as physical": [0],
    }
}
elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group="PartType0")
snap_file.write(
    {"SpeciesFractions": species_fractions},
    elements_per_file,
    filenames=output_filename,
    mode="w",
    group="PartType0",
    attrs=attrs,
)
comm.barrier()

# Write the NamedColumns using rank 0
if comm_rank == 0:
    for i_file in range(n_file):
        with h5py.File(output_filename.format(file_nr=i_file), "a") as file:
            subgrid_scheme = file.create_group("SubgridScheme")
            named_columns = subgrid_scheme.create_group("NamedColumns")
            encoded_species = [species.encode() for species in species_names]
            named_columns.create_dataset(
                "SpeciesFractions",
                data=encoded_species,
            )
    print("Done!")
