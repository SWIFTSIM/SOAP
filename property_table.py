#!/bin/env python

"""
property_table.py

This file contains all the properties that can be calculated by SOAP, and some
functionality to automatically generate the documentation (PDF) containing these
properties.

The rationale for having all of this in one file (and what is essentially one
big dictionary) is consistency: every property is defined exactly once, with
one data type, one unit, one description... Every type of halo still
implements its own calculation of each property, but everything that is exposed
to the user is guaranteed to be consistent for all halo types. To change the
documentation, you need to change the dictionary, so you will automatically
change the code as well. If you remember to regenerate the documentation, the
code will hence always be consistent with its documentation. The documentation
includes a version string to help identify it.

When a specific type of halo wants to implement a property, it should import the
property table from this file and grab all of the information for the
corresponding dictionary element, e.g. (taken from aperture_properties.py)

    from property_table import PropertyTable
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in [
            "Mtot",
            "Mgas",
            "Mdm",
            "Mstar",
        ]
    ]

The elements of each row are documented later in this file.

Note that this file contains some code that helps to regenerate the dictionary
itself. That is useful for adding additional rows to the table.
"""

import numpy as np
import unyt
import subprocess
import datetime
import os
from typing import Dict, List
from halo_properties import HaloProperty


def get_version_string() -> str:
    """
    Generate a version string that uniquely identifies the documentation file.

    The version string will have the format
      SOAP version a7baa6e -- Compiled by user ``vandenbroucke'' on winkel
       on Tuesday 15 November 2022, 10:49:10
    or
      Unknown SOAP version -- Compiled by user ``vandenbroucke'' on winkel
       on Tuesday 15 November 2022, 10:49:10
    if no git version string can be obtained.
    """

    handle = subprocess.run("git describe --always", shell=True, stdout=subprocess.PIPE)
    if handle.returncode != 0:
        git_version = "Unknown SOAP version"
    else:
        git_version = handle.stdout.decode("utf-8").strip()
        git_version = f"SOAP version ``{git_version}''"
    timestamp = datetime.datetime.now().strftime("%A %-d %B %Y, %H:%M:%S")
    username = os.getlogin()
    hostname = os.uname().nodename
    return f"{git_version} -- Compiled by user ``{username}'' on {hostname} on {timestamp}."


class PropertyTable:
    """
    Auxiliary object to manipulate the property table.

    You should only create a PropertyTable object if you actually want to use
    it to generate an updated version of the internal property dictionary or
    to generate the documentation. If you just want to grab the information for
    a particular property from the table, you should directly access the
    static table, e.g.
      Mstar_info = PropertyTable.full_property_list["Mstar"]
    """

    # categories: the first 6 are used for filtering, 'VR' is an extra
    # category used for properties copied over directly from the Velociraptor
    # output. VR properties should not be included by any of the halo types,
    # they are only there to complete the documentation!
    categories = ["basic", "general", "gas", "dm", "star", "baryon", "VR"]
    # some properties require an additional explanation in the form of a
    # footnote. These footnotes are .tex files in the 'documentation' folder
    # (that should exist). The name of the file acts as a key in the dictionary
    # below; the corresponding value is a list of all properties that should
    # include a footnote link to this particular explanation.
    explanation = {
        "footnote_MBH.tex": ["BHmaxM"],
        "footnote_com.tex": ["com", "vcom"],
        "footnote_AngMom.tex": ["Lgas", "Ldm", "Lstar", "Lbaryons"],
        "footnote_kappa.tex": [
            "kappa_corot_gas",
            "kappa_corot_star",
            "kappa_corot_baryons",
        ],
        "footnote_SF.tex": [
            "SFR",
            "gasFefrac_SF",
            "gasOfrac_SF",
            "Mgas_SF",
            "gasmetalfrac_SF",
        ],
        "footnote_Tgas.tex": [
            "Tgas",
            "Tgas_no_agn",
            "Tgas_no_cool",
            "Tgas_no_cool_no_agn",
        ],
        "footnote_lum.tex": ["StellarLuminosity"],
        "footnote_circvel.tex": ["R_vmax", "Vmax"],
        "footnote_spin.tex": ["spin_parameter"],
        "footnote_veldisp_matrix.tex": [
            "veldisp_matrix_gas",
            "veldisp_matrix_dm",
            "veldisp_matrix_star",
        ],
        "footnote_proj_veldisp.tex": [
            "proj_veldisp_gas",
            "proj_veldisp_dm",
            "proj_veldisp_star",
        ],
        "footnote_elements.tex": [
            "gasOfrac",
            "gasOfrac_SF",
            "gasFefrac",
            "gasFefrac_SF",
            "gasmetalfrac",
            "gasmetalfrac_SF",
        ],
        "footnote_halfmass.tex": [
            "HalfMassRadiusTot",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
        ],
        "footnote_satfrac.tex": ["Mfrac_satellites"],
        "footnote_Ekin.tex": ["Ekin_gas", "Ekin_star"],
        "footnote_Etherm.tex": ["Etherm_gas"],
        "footnote_Mnu.tex": ["Mnu", "MnuNS"],
        "footnote_Xray.tex": [
            "Xraylum",
            "Xraylum_no_agn",
            "Xrayphlum",
            "Xrayphlum_no_agn",
        ],
        "footnote_compY.tex": ["compY", "compY_no_agn"],
        "footnote_dopplerB.tex": ["DopplerB"],
        "footnote_dust.tex": [
            "DustGraphiteMass",
            "DustGraphiteMassInMolecularGas",
            "DustGraphiteMassInAtomicGas",
            "DustSilicatesMass",
            "DustSilicatesMassInMolecularGas",
            "DustSilicatesMassInAtomicGas",
            "DustLargeGrainMass",
            "DustLargeGrainMassInMolecularGas",
            "DustSmallGrainMass",
            "DustSmallGrainMassInMolecularGas",
        ],
        "footnote_diffuse.tex": [
            "DiffuseCarbonMass",
            "DiffuseOxygenMass",
            "DiffuseMagnesiumMass",
            "DiffuseSiliconMass",
            "DiffuseIronMass",
        ],
    }

    # dictionary with human-friendly descriptions of the various lossy
    # compression filters that can be applied to data.
    # The key is the name of a lossy compression filter (same names as used
    # by SWIFT), the value is the corresponding description, which can be either
    # an actual description or a representative example.
    compression_description = {
        "FMantissa9": "$1.36693{\\rm{}e}10 \\rightarrow{} 1.367{\\rm{}e}10$",
        "DMantissa9": "$1.36693{\\rm{}e}10 \\rightarrow{} 1.367{\\rm{}e}10$",
        "DScale5": "10 pc accurate",
        "DScale1": "0.1 km/s accurate",
        "Nbit40": "Store less bits",
        "None": "no compression",
    }

    # List of properties that get computed
    # The key for each property is the name that is used internally in SOAP
    # For each property, we have the following columns:
    #  - name: Name of the property within the output file
    #  - shape: Shape of this property for a single halo (1: scalar,
    #      3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to
    #      avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the
    #      property in the output.
    #  - category: Category used to decide if this property should be calculated
    #      for a particular halo (filtering), or 'VR' for properties that are
    #      copied over from the Velociraptor output.
    #  - lossy compression filter: Lossy compression filter used in the output
    #      to reduce the file size. Note that SOAP does not actually compress
    #      the output; this is done by a separate script. We support all lossy
    #      compression filters available in SWIFT.
    #  - DMO property: Should this property be calculated for a DMO run?
    #  - Particle properties: Particle fields that are required to compute this
    #      property. Used to determine which particle fields to read for a
    #      particular SOAP configuration (as defined in the parameter file).
    #
    # Note that there is no good reason to have a diffent internal name and
    # output name; this was mostly done for historical reasons. This means that
    # you can easily change the name in the output without having to change all
    # of the other .py files that use this property.
    full_property_list = {
        "AtomicHydrogenMass": (
            "AtomicHydrogenMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in atomic hydrogen.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "BHlasteventa": (
            "BlackHolesLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event.",
            "general",
            "FMantissa9",
            False,
            ["PartType5/LastAGNFeedbackScaleFactors"],
        ),
        "BHmaxAR": (
            "MostMassiveBlackHoleAccretionRate",
            1,
            np.float32,
            "Msun/yr",
            "Gas accretion rate of most massive black hole.",
            "general",
            "FMantissa9",
            False,
            ["PartType5/SubgridMasses", "PartType5/AccretionRates"],
        ),
        "BHmaxID": (
            "MostMassiveBlackHoleID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive black hole.",
            "basic",
            "Nbit40",
            False,
            ["PartType5/SubgridMasses", "PartType5/ParticleIDs"],
        ),
        "BHmaxM": (
            "MostMassiveBlackHoleMass",
            1,
            np.float32,
            "Msun",
            "Mass of most massive black hole.",
            "basic",
            "FMantissa9",
            False,
            ["PartType5/SubgridMasses"],
        ),
        "BHmaxlasteventa": (
            "MostMassiveBlackHoleLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event for most massive black hole.",
            "general",
            "FMantissa9",
            False,
            ["PartType5/SubgridMasses", "PartType5/LastAGNFeedbackScaleFactors"],
        ),
        "BHmaxpos": (
            "MostMassiveBlackHolePosition",
            3,
            np.float64,
            "kpc",
            "Position of most massive black hole.",
            "general",
            "DScale5",
            False,
            ["PartType5/Coordinates", "PartType5/SubgridMasses"],
        ),
        "BHmaxvel": (
            "MostMassiveBlackHoleVelocity",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive black hole relative to the simulation volume.",
            "general",
            "FMantissa9",
            False,
            ["PartType5/SubgridMasses", "PartType5/Velocities"],
        ),
        "BaryonAxisLengths": (
            "BaryonAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the baryonic (gas and stars) mass distribution, computed from the 3D baryon inertia tensor, relative to the centre of potential.",
            "baryon",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
            ],
        ),
        "DMAxisLengths": (
            "DarkMatterAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the dark matter mass distribution, computed from the 3D DM inertia tensor, relative to the centre of potential.",
            "dm",
            "FMantissa9",
            True,
            ["PartType1/Coordinates", "PartType1/Masses"],
        ),
        "DM_R_vmax": (
            "MaximumDarkMatterCircularVelocityRadius",
            1,
            np.float32,
            "kpc",
            "Radius at which Vmax is reached for dark matter particles.",
            "basic",
            "FMantissa9",
            False,
            ["PartType1/Coordinates", "PartType1/Masses"],
        ),
        "DM_Vmax": (
            "MaximumDarkMatterCircularVelocity",
            1,
            np.float32,
            "km/s",
            "Maximum circular velocity.",
            "basic",
            "FMantissa9",
            False,
            ["PartType1/Coordinates", "PartType1/Masses"],
        ),
        "DiffuseCarbonMass": (
            "DiffuseCarbonMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in carbon that is not contained in dust.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
            ],
        ),
        "DiffuseIronMass": (
            "DiffuseIronMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron that is not contained in dust.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
            ],
        ),
        "DiffuseMagnesiumMass": (
            "DiffuseMagnesiumMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in magnesium that is not contained in dust.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
            ],
        ),
        "DiffuseOxygenMass": (
            "DiffuseOxygenMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen that is not contained in dust.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
            ],
        ),
        "DiffuseSiliconMass": (
            "DiffuseSiliconMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in silicon that is not contained in dust.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
            ],
        ),
        "DopplerB": (
            "DopplerB",
            1,
            np.float32,
            "dimensionless",
            "Kinetic Sunyaey-Zel'dovich effect, assuming a line of sight towards the position of the first lightcone observer.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Velocities",
                "PartType0/ElectronNumberDensities",
                "PartType0/Densities",
            ],
        ),
        "DtoTgas": (
            "DiscToTotalGasMassFraction",
            1,
            np.float32,
            "dimensionless",
            "Fraction of the total gas mass that is co-rotating.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses", "PartType0/Velocities"],
        ),
        "DtoTstar": (
            "DiscToTotalStellarMassFraction",
            1,
            np.float32,
            "dimensionless",
            "Fraction of the total stellar mass that is co-rotating.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Velocities", "PartType4/Masses"],
        ),
        "DustGraphiteMass": (
            "DustGraphiteMass",
            1,
            np.float32,
            "Msun",
            "Total dust mass in graphite grains.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/DustMassFractions"],
        ),
        "DustGraphiteMassInAtomicGas": (
            "DustGraphiteMassInAtomicGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in graphite grains in atomic gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/ElementMassFractions",
                "PartType0/SpeciesFractions",
            ],
        ),
        "DustGraphiteMassInMolecularGas": (
            "DustGraphiteMassInMolecularGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in graphite grains in molecular gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustGraphiteMassInColdDenseGas": (
            "DustGraphiteMassInColdDenseGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in graphite grains in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/Densities",
                "PartType0/Temperatures",
            ],
        ),
        "DustLargeGrainMass": (
            "DustLargeGrainMass",
            1,
            np.float32,
            "Msun",
            "Total dust mass in large grains.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/DustMassFractions"],
        ),
        "DustLargeGrainMassInMolecularGas": (
            "DustLargeGrainMassInMolecularGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in large grains in molecular gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustLargeGrainMassInColdDenseGas": (
            "DustLargeGrainMassInColdDenseGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in large grains in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/Densities",
                "PartType0/Temperatures",
            ],
        ),
        "DustSilicatesMass": (
            "DustSilicatesMass",
            1,
            np.float32,
            "Msun",
            "Total dust mass in silicate grains.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/DustMassFractions"],
        ),
        "DustSilicatesMassInAtomicGas": (
            "DustSilicatesMassInAtomicGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in silicate grains in atomic gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustSilicatesMassInMolecularGas": (
            "DustSilicatesMassInMolecularGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in silicate grains in molecular gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustSilicatesMassInColdDenseGas": (
            "DustSilicatesMassInColdDenseGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in silicate grains in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/Densities",
                "PartType0/Temperatures",
            ],
        ),
        "DustSmallGrainMass": (
            "DustSmallGrainMass",
            1,
            np.float32,
            "Msun",
            "Total dust mass in small grains.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustSmallGrainMassInMolecularGas": (
            "DustSmallGrainMassInMolecularGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in small grains in molecular gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "DustSmallGrainMassInColdDenseGas": (
            "DustSmallGrainMassInColdDenseGas",
            1,
            np.float32,
            "Msun",
            "Total dust mass in small grains in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/DustMassFractions",
                "PartType0/Densities",
                "PartType0/Temperatures",
            ],
        ),
        "Ekin_gas": (
            "KineticEnergyGas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas, relative to the gas centre of mass velocity.",
            "gas",
            "DMantissa9",
            False,
            ["PartType0/Masses", "PartType0/Velocities"],
        ),
        "Ekin_star": (
            "KineticEnergyStars",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the stars, relative to the stellar centre of mass velocity.",
            "star",
            "DMantissa9",
            False,
            ["PartType4/Masses", "PartType4/Velocities"],
        ),
        "Etherm_gas": (
            "ThermalEnergyGas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas.",
            "gas",
            "DMantissa9",
            False,
            ["PartType0/Densities", "PartType0/Pressures", "PartType0/Masses"],
        ),
        "GasAxisLengths": (
            "GasAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the gas mass distribution, computed from the 3D gas inertia tensor, relative to the centre of potential.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses"],
        ),
        "HalfMassRadiusBaryon": (
            "HalfMassRadiusBaryons",
            1,
            np.float32,
            "kpc",
            "Baryonic (gas and stars) half mass radius.",
            "baryon",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
            ],
        ),
        "HalfMassRadiusDM": (
            "HalfMassRadiusDarkMatter",
            1,
            np.float32,
            "kpc",
            "Dark matter half mass radius.",
            "dm",
            "FMantissa9",
            True,
            ["PartType1/Coordinates", "PartType1/Masses"],
        ),
        "HalfMassRadiusGas": (
            "HalfMassRadiusGas",
            1,
            np.float32,
            "kpc",
            "Gas half mass radius.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses"],
        ),
        "HalfMassRadiusStar": (
            "HalfMassRadiusStars",
            1,
            np.float32,
            "kpc",
            "Stellar half mass radius.",
            "basic",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Masses"],
        ),
        "HalfMassRadiusTot": (
            "HalfMassRadiusTotal",
            1,
            np.float32,
            "kpc",
            "Total half mass radius.",
            "general",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
            ],
        ),
        "HeliumMass": (
            "HeliumMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in helium.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/ElementMassFractions"],
        ),
        "HydrogenMass": (
            "HydrogenMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in hydrogen.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/ElementMassFractions"],
        ),
        "IonisedHydrogenMass": (
            "IonisedHydrogenMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in ionised hydrogen.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "LastSupernovaEventMaximumGasDensity": (
            "LastSupernovaEventMaximumGasDensity",
            1,
            np.float32,
            "g/cm**3",
            "Maximum gas density at the last supernova event for the last supernova event of each gas particle.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/LastSNIIThermalFeedbackDensities",
                "PartType0/LastSNIIKineticFeedbackDensities",
            ],
        ),
        "Lbaryons": (
            "AngularMomentumBaryons",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of baryons (gas and stars), relative to the centre of potential and baryonic centre of mass velocity.",
            "baryon",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType0/Velocities",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType4/Velocities",
            ],
        ),
        "Ldm": (
            "AngularMomentumDarkMatter",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the dark matter, relative to the centre of potential and DM centre of mass velocity.",
            "dm",
            "FMantissa9",
            True,
            ["PartType1/Coordinates", "PartType1/Masses", "PartType1/Velocities"],
        ),
        "Lgas": (
            "AngularMomentumGas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the gas, relative to the centre of potential and gas centre of mass velocity.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses", "PartType0/Velocities"],
        ),
        "MedianStellarBirthDensity": (
            "MedianStellarBirthDensity",
            1,
            np.float32,
            "g/cm**3",
            "Median density of gas particles that were converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthDensities"],
        ),
        "MedianStellarBirthTemperature": (
            "MedianStellarBirthTemperature",
            1,
            np.float32,
            "K",
            "Median temperature of gas particles that were converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures"],
        ),
        "MedianStellarBirthPressure": (
            "MedianStellarBirthPressure",
            1,
            np.float32,
            "K/cm**3",
            "Median pressure of gas particles that were converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures", "PartType4/BirthDensities"],
        ),
        "Lstar": (
            "AngularMomentumStars",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of the stars, relative to the centre of potential and stellar centre of mass velocity.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Masses", "PartType4/Velocities"],
        ),
        "MaximumStellarBirthDensity": (
            "MaximumStellarBirthDensity",
            1,
            np.float32,
            "g/cm**3",
            "Maximum density of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthDensities"],
        ),
        "MaximumStellarBirthTemperature": (
            "MaximumStellarBirthTemperature",
            1,
            np.float32,
            "K",
            "Maximum temperature of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures"],
        ),
        "MaximumStellarBirthPressure": (
            "MaximumStellarBirthPressure",
            1,
            np.float32,
            "K/cm**3",
            "Maximum pressure of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures", "PartType4/BirthDensities"],
        ),
        "Mbh_dynamical": (
            "BlackHolesDynamicalMass",
            1,
            np.float32,
            "Msun",
            "Total BH dynamical mass.",
            "basic",
            "FMantissa9",
            False,
            ["PartType5/DynamicalMasses"],
        ),
        "Mbh_subgrid": (
            "BlackHolesSubgridMass",
            1,
            np.float32,
            "Msun",
            "Total BH subgrid mass.",
            "basic",
            "FMantissa9",
            False,
            ["PartType5/SubgridMasses"],
        ),
        "Mdm": (
            "DarkMatterMass",
            1,
            np.float32,
            "Msun",
            "Total DM mass.",
            "basic",
            "FMantissa9",
            True,
            ["PartType1/Masses"],
        ),
        "Mfrac_satellites": (
            "MassFractionSatellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite.",
            "general",
            "FMantissa9",
            True,
            [
                "PartType0/Masses",
                "PartType1/Masses",
                "PartType4/Masses",
                "PartType5/DynamicalMasses",
                "PartType0/GroupNr_bound",
                "PartType1/GroupNr_bound",
                "PartType4/GroupNr_bound",
                "PartType5/GroupNr_bound",
            ],
        ),
        "Mgas": (
            "GasMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass.",
            "basic",
            "FMantissa9",
            False,
            ["PartType0/Masses"],
        ),
        "Mgas_SF": (
            "StarFormingGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of star-forming gas.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/StarFormationRates"],
        ),
        "Mhotgas": (
            "HotGasMass",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with a temperature above 1e5 K.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/Temperatures"],
        ),
        "GasMassInColdDenseGas": (
            "GasMassInColdDenseGas",
            1,
            np.float32,
            "Msun",
            "Total mass of gas in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/Densities", "PartType0/Temperatures"],
        ),
        "MinimumStellarBirthDensity": (
            "MinimumStellarBirthDensity",
            1,
            np.float32,
            "g/cm**3",
            "Minimum density of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthDensities"],
        ),
        "MinimumStellarBirthTemperature": (
            "MinimumStellarBirthTemperature",
            1,
            np.float32,
            "K",
            "Minimum temperature of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures"],
        ),
        "MinimumStellarBirthPressure": (
            "MinimumStellarBirthPressure",
            1,
            np.float32,
            "K/cm**3",
            "Minimum pressure of gas that was converted into a star particle.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/BirthTemperatures", "PartType4/BirthDensities"],
        ),
        "Mnu": (
            "RawNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Total neutrino particle mass.",
            "basic",
            "FMantissa9",
            True,
            ["PartType6/Masses"],
        ),
        "MnuNS": (
            "NoiseSuppressedNeutrinoMass",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass.",
            "basic",
            "FMantissa9",
            True,
            ["PartType6/Masses", "PartType6/Weights"],
        ),
        "MolecularHydrogenMass": (
            "MolecularHydrogenMass",
            1,
            np.float32,
            "Msun",
            "Total gas mass in molecular hydrogen.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/SpeciesFractions",
                "PartType0/ElementMassFractions",
            ],
        ),
        "Mstar": (
            "StellarMass",
            1,
            np.float32,
            "Msun",
            "Total stellar mass.",
            "basic",
            "FMantissa9",
            False,
            ["PartType4/Masses"],
        ),
        "Mstar_init": (
            "StellarInitialMass",
            1,
            np.float32,
            "Msun",
            "Total stellar initial mass.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/InitialMasses"],
        ),
        "Mtot": (
            "TotalMass",
            1,
            np.float32,
            "Msun",
            "Total mass.",
            "basic",
            "FMantissa9",
            True,
            [
                "PartType0/Masses",
                "PartType1/Masses",
                "PartType4/Masses",
                "PartType5/DynamicalMasses",
            ],
        ),
        "Nbh": (
            "NumberOfBlackHoleParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of black hole particles.",
            "basic",
            "None",
            False,
            [],
        ),
        "Ndm": (
            "NumberOfDarkMatterParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of dark matter particles.",
            "basic",
            "None",
            True,
            [],
        ),
        "Ngas": (
            "NumberOfGasParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of gas particles.",
            "basic",
            "None",
            False,
            [],
        ),
        "Nnu": (
            "NumberOfNeutrinoParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of neutrino particles.",
            "basic",
            "None",
            False,
            [],
        ),
        "Nstar": (
            "NumberOfStarParticles",
            1,
            np.uint32,
            "dimensionless",
            "Number of star particles.",
            "basic",
            "None",
            False,
            [],
        ),
        "ProjectedBaryonAxisLengths": (
            "ProjectedBaryonAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected baryon (gas and stars) mass distribution, computed from the 2D baryon inertia tensor, relative to the centre of potential.",
            "baryon",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
            ],
        ),
        "ProjectedGasAxisLengths": (
            "ProjectedGasAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected gas mass distribution, computed from the 2D gas inertia tensor, relative to the centre of potential.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses"],
        ),
        "ProjectedStellarAxisLengths": (
            "ProjectedStellarAxisLengths",
            2,
            np.float32,
            "kpc",
            "Axis lengths of the projected stellar mass distribution, computed from the 2D stellar inertia tensor, relative to the centre of potential.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Masses"],
        ),
        "R_vmax": (
            "MaximumCircularVelocityRadius",
            1,
            np.float32,
            "kpc",
            "Radius at which Vmax is reached.",
            "basic",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
            ],
        ),
        "SFR": (
            "StarFormationRate",
            1,
            np.float32,
            "Msun/yr",
            "Total star formation rate.",
            "basic",
            "FMantissa9",
            False,
            ["PartType0/StarFormationRates"],
        ),
        "StellarAxisLengths": (
            "StellarAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the stellar mass distribution, computed from the 3D stellar inertia tensor, relative to the centre of potential.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Masses"],
        ),
        "StellarLuminosity": (
            "StellarLuminosity",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity in the 9 GAMA bands.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Luminosities"],
        ),
        "Tgas": (
            "GasTemperature",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Temperatures"],
        ),
        "Tgas_no_agn": (
            "GasTemperatureWithoutRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Temperatures", "PartType0/LastAGNFeedbackScaleFactors"],
        ),
        "Tgas_no_cool": (
            "GasTemperatureWithoutCoolGas",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Temperatures"],
        ),
        "Tgas_no_cool_no_agn": (
            "GasTemperatureWithoutCoolGasAndRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding cool gas with a temperature below 1e5 K and gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Temperatures", "PartType0/LastAGNFeedbackScaleFactors"],
        ),
        "TotalAxisLengths": (
            "TotalAxisLengths",
            3,
            np.float32,
            "kpc",
            "Axis lengths of the total mass distribution, computed from the 3D inertia tensor, relative to the centre of potential.",
            "general",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
            ],
        ),
        "TotalSNIaRate": (
            "TotalSNIaRate",
            1,
            np.float32,
            "1/Gyr",
            "Total SNIa rate.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/SNIaRates"],
        ),
        "VRID": (
            "ID",
            1,
            np.uint64,
            "dimensionless",
            "ID assigned to this halo by VR.",
            "VR",
            "None",
            True,
            [],
        ),
        "VRParent_halo_ID": (
            "ParentHaloID",
            1,
            np.int64,
            "dimensionless",
            "VR/ID of the direct parent of this halo. -1 for field halos.",
            "VR",
            "None",
            True,
            [],
        ),
        "VRStructuretype": (
            "StructureType",
            1,
            np.int32,
            "dimensionless",
            "Structure type identified by VR. Field halos are 10, higher numbers are for satellites.",
            "VR",
            "None",
            True,
            [],
        ),
        "VRcofp": (
            "CentreOfPotential",
            3,
            np.float64,
            "Mpc",
            "Centre of potential, as identified by VR. Used as reference for all relative positions. Equal to the position of the most bound particle in the subhalo.",
            "VR",
            "DScale5",
            True,
            [],
        ),
        "VRhostHaloID": (
            "HostHaloID",
            1,
            np.int64,
            "dimensionless",
            "VR/ID of the top level parent of this halo. -1 for field halos.",
            "VR",
            "None",
            True,
            [],
        ),
        "VRindex": (
            "Index",
            1,
            np.int64,
            "dimensionless",
            "Index of this halo in the original VR output.",
            "VR",
            "None",
            True,
            [],
        ),
        "VRnumSubStruct": (
            "NumberOfSubstructures",
            1,
            np.uint64,
            "dimensionless",
            "Number of sub-structures within this halo.",
            "VR",
            "None",
            True,
            [],
        ),
        "Vmax": (
            "MaximumCircularVelocity",
            1,
            np.float32,
            "km/s",
            "Maximum circular velocity.",
            "basic",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
            ],
        ),
        "Xraylum": (
            "XRayLuminosity",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
            ["PartType0/XrayLuminosities"],
        ),
        "Xraylum_no_agn": (
            "XRayLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
            [
                "PartType0/XrayLuminosities",
                "PartType0/LastAGNFeedbackScaleFactors",
                "PartType0/Temperatures",
            ],
        ),
        "Xrayphlum": (
            "XRayPhotonLuminosity",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
            ["PartType0/XrayPhotonLuminosities"],
        ),
        "Xrayphlum_no_agn": (
            "XRayPhotonLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
            [
                "PartType0/XrayPhotonLuminosities",
                "PartType0/LastAGNFeedbackScaleFactors",
                "PartType0/Temperatures",
            ],
        ),
        "com": (
            "CentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass.",
            "basic",
            "DScale5",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
            ],
        ),
        "com_gas": (
            "GasCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of gas.",
            "gas",
            "DScale5",
            False,
            ["PartType0/Coordinates", "PartType0/Masses"],
        ),
        "com_star": (
            "StellarCentreOfMass",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars.",
            "star",
            "DScale5",
            False,
            ["PartType4/Coordinates", "PartType4/Masses"],
        ),
        "compY": (
            "ComptonY",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter.",
            "gas",
            "DMantissa9",
            False,
            ["PartType0/ComptonYParameters"],
        ),
        "compY_no_agn": (
            "ComptonYWithoutRecentAGNHeating",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter. Excludes gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
            [
                "PartType0/ComptonYParameters",
                "PartType0/LastAGNFeedbackScaleFactors",
                "PartType0/Temperatures",
            ],
        ),
        "gasFefrac": (
            "GasMassFractionInIron",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass fraction in iron.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/ElementMassFractions"],
        ),
        "gasFefrac_SF": (
            "StarFormingGasMassFractionInIron",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass fraction in iron for gas that is star-forming.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractions",
                "PartType0/StarFormationRates",
            ],
        ),
        "gasOfrac": (
            "GasMassFractionInOxygen",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass in oxygen.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/ElementMassFractions"],
        ),
        "gasOfrac_SF": (
            "StarFormingGasMassFractionInOxygen",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass fraction in oxygen for gas that is star-forming.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractions",
                "PartType0/StarFormationRates",
            ],
        ),
        "gasmetalfrac": (
            "GasMassFractionInMetals",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass fraction in metals.",
            "basic",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/MetalMassFractions"],
        ),
        "gasmetalfrac_SF": (
            "StarFormingGasMassFractionInMetals",
            1,
            np.float32,
            "dimensionless",
            "Total gas mass fraction in metals for gas that is star-forming.",
            "basic",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/MetalMassFractions",
                "PartType0/StarFormationRates",
            ],
        ),
        "kappa_corot_baryons": (
            "KappaCorotBaryons",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for baryons (gas and stars), relative to the centre of potential and the centre of mass velocity of the baryons.",
            "baryon",
            "FMantissa9",
            False,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType0/Velocities",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType4/Velocities",
            ],
        ),
        "kappa_corot_gas": (
            "KappaCorotGas",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for gas, relative to the centre of potential and the centre of mass velocity of the gas.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Coordinates", "PartType0/Masses", "PartType0/Velocities"],
        ),
        "kappa_corot_star": (
            "KappaCorotStars",
            1,
            np.float32,
            "dimensionless",
            "Kappa-corot for stars, relative to the centre of potential and the centre of mass velocity of the stars.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Coordinates", "PartType4/Masses", "PartType4/Velocities"],
        ),
        "proj_veldisp_dm": (
            "DarkMatterProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the DM along the projection axis, relative to the DM centre of mass velocity.",
            "dm",
            "FMantissa9",
            True,
            ["PartType1/Velocities"],
        ),
        "proj_veldisp_gas": (
            "GasProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the gas along the projection axis, relative to the gas centre of mass velocity.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Velocities"],
        ),
        "proj_veldisp_star": (
            "StellarProjectedVelocityDispersion",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the stars along the projection axis, relative to the stellar centre of mass velocity.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Velocities"],
        ),
        "r": (
            "SORadius",
            1,
            np.float32,
            "Mpc",
            "Radius of a sphere {label}",
            "basic",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
                "PartType6/Coordinates",
                "PartType6/Masses",
                "PartType6/Weights",
            ],
        ),
        "spin_parameter": (
            "SpinParameter",
            1,
            np.float32,
            "dimensionless",
            "Bullock et al. (2001) spin parameter.",
            "general",
            "FMantissa9",
            True,
            [
                "PartType0/Coordinates",
                "PartType0/Masses",
                "PartType0/Velocities",
                "PartType1/Coordinates",
                "PartType1/Masses",
                "PartType1/Velocities",
                "PartType4/Coordinates",
                "PartType4/Masses",
                "PartType4/Velocities",
                "PartType5/Coordinates",
                "PartType5/DynamicalMasses",
                "PartType5/Velocities",
            ],
        ),
        "starFefrac": (
            "StellarMassFractionInIron",
            1,
            np.float32,
            "dimensionless",
            "Total stellar mass fraction in iron.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/ElementMassFractions"],
        ),
        "starMgfrac": (
            "StellarMassFractionInMagnesium",
            1,
            np.float32,
            "dimensionless",
            "Total stellar mass fraction in magnesium.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/ElementMassFractions"],
        ),
        "starOfrac": (
            "StellarMassFractionInOxygen",
            1,
            np.float32,
            "dimensionless",
            "Total stellar mass fraction in oxygen.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/ElementMassFractions"],
        ),
        "starmetalfrac": (
            "StellarMassFractionInMetals",
            1,
            np.float32,
            "dimensionless",
            "Total stellar mass fraction in metals.",
            "basic",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/MetalMassFractions"],
        ),
        "stellar_age_lw": (
            "LuminosityWeightedMeanStellarAge",
            1,
            np.float32,
            "Myr",
            "Luminosity weighted mean stellar age. The weight is the r band luminosity.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Luminosities", "PartType4/BirthScaleFactors"],
        ),
        "stellar_age_mw": (
            "MassWeightedMeanStellarAge",
            1,
            np.float32,
            "Myr",
            "Mass weighted mean stellar age.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/BirthScaleFactors"],
        ),
        "vcom": (
            "CentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity.",
            "basic",
            "DScale1",
            True,
            [
                "PartType0/Masses",
                "PartType0/Velocities",
                "PartType1/Masses",
                "PartType1/Velocities",
                "PartType4/Masses",
                "PartType4/Velocities",
                "PartType5/DynamicalMasses",
                "PartType5/Velocities",
            ],
        ),
        "vcom_gas": (
            "GasCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas.",
            "gas",
            "DScale1",
            False,
            ["PartType0/Masses", "PartType0/Velocities"],
        ),
        "vcom_star": (
            "StellarCentreOfMassVelocity",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars.",
            "star",
            "DScale1",
            False,
            ["PartType4/Masses", "PartType4/Velocities"],
        ),
        "veldisp_matrix_dm": (
            "DarkMatterVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the dark matter. Measured relative to the DM centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "dm",
            "FMantissa9",
            True,
            ["PartType1/Masses", "PartType1/Velocities"],
        ),
        "veldisp_matrix_gas": (
            "GasVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the gas. Measured relative to the gas centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "gas",
            "FMantissa9",
            False,
            ["PartType0/Masses", "PartType0/Velocities"],
        ),
        "veldisp_matrix_star": (
            "StellarVelocityDispersionMatrix",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the stars. Measured relative to the stellar centre of mass velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "star",
            "FMantissa9",
            False,
            ["PartType4/Masses", "PartType4/Velocities"],
        ),
        "LinearMassWeightedOxygenOverHydrogenOfGas": (
            "LinearMassWeightedOxygenOverHydrogenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the oxygen over hydrogen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedNitrogenOverOxygenOfGas": (
            "LinearMassWeightedNitrogenOverOxygenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the total nitrogen over oxygen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedCarbonOverOxygenOfGas": (
            "LinearMassWeightedCarbonOverOxygenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the total carbon over oxygen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedDiffuseNitrogenOverOxygenOfGas": (
            "LinearMassWeightedDiffuseNitrogenOverOxygenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the diffuse nitrogen over oxygen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedDiffuseCarbonOverOxygenOfGas": (
            "LinearMassWeightedDiffuseCarbonOverOxygenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the diffuse carbon over oxygen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedDiffuseOxygenOverHydrogenOfGas": (
            "LinearMassWeightedDiffuseOxygenOverHydrogenOfGas",
            1,
            np.float32,
            "Msun",
            "Linear sum of the diffuse oxygen over hydrogen ratio of gas, multiplied with the gas mass.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasLowLimit": (
            "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse nitrogen over oxygen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-4 times solar N/O.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasHighLimit": (
            "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse nitrogen over oxygen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-3 times solar N/O.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasLowLimit": (
            "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse carbon over oxygen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-4 times solar C/O.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasHighLimit": (
            "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse carbon over oxygen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-3 times solar C/O.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-4 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of gas, multiplied with the gas mass. Imposes a lower limit of 1.e-3 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasLowLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of atomic gas, multiplied with the gas mass. Imposes a lower limit of 1.e-4 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/ElementMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasHighLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of atomic gas, multiplied with the gas mass. Imposes a lower limit of 1.e-3 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/ElementMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasLowLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of molecular gas, multiplied with the gas mass. Imposes a lower limit of 1.e-4 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/ElementMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasHighLimit": (
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the diffuse oxygen over hydrogen ratio of molecular gas, multiplied with the gas mass. Imposes a lower limit of 1.e-3 times solar O/H.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/ElementMassFractionsDiffuse",
                "PartType0/ElementMassFractions",
                "PartType0/SpeciesFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LinearMassWeightedIronOverHydrogenOfStars": (
            "LinearMassWeightedIronOverHydrogenOfStars",
            1,
            np.float32,
            "Msun",
            "Linear sum of the iron over hydrogen ratio of stars, multiplied with the stellar mass.",
            "star",
            "FMantissa9",
            False,
            [
                "PartType4/Masses",
                "PartType4/ElementMassFractions",
            ],
        ),
        "LogarithmicMassWeightedIronOverHydrogenOfStarsLowLimit": (
            "LogarithmicMassWeightedIronOverHydrogenOfStarsLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the iron over hydrogen ratio of stars, multiplied with the stellar mass. Imposes a lower limit of 1.e-4 times solar Fe/H.",
            "star",
            "FMantissa9",
            False,
            [
                "PartType4/Masses",
                "PartType4/ElementMassFractions",
            ],
        ),
        "LogarithmicMassWeightedIronOverHydrogenOfStarsHighLimit": (
            "LogarithmicMassWeightedIronOverHydrogenOfStarsHighLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the iron over hydrogen ratio of stars, multiplied with the stellar mass. Imposes a lower limit of 1.e-3 times solar Fe/H.",
            "star",
            "FMantissa9",
            False,
            [
                "PartType4/Masses",
                "PartType4/ElementMassFractions",
            ],
        ),
        "GasMassInColdDenseDiffuseMetals": (
            "GasMassInColdDenseDiffuseMetals",
            1,
            np.float32,
            "Msun",
            "Sum of the diffuse metal mass in cold, dense gas.",
            "gas",
            "FMantissa9",
            False,
            [
                "PartType0/Masses",
                "PartType0/MetalMassFractions",
                "PartType0/DustMassFractions",
                "PartType0/Temperatures",
                "PartType0/Densities",
            ],
        ),
        "LogarithmicMassWeightedIronFromSNIaOverHydrogenOfStarsLowLimit": (
            "LogarithmicMassWeightedIronFromSNIaOverHydrogenOfStarsLowLimit",
            1,
            np.float32,
            "Msun",
            "Logarithmic sum of the iron over hydrogen ratio of stars, multiplied with the stellar mass, where only iron from SNIa is included. Imposes a lower limit of 1.e-4 times solar Fe/H.",
            "star",
            "FMantissa9",
            False,
            [
                "PartType4/Masses",
                "PartType4/ElementMassFractions",
                "PartType4/IronMassFractionsFromSNIa",
            ],
        ),
        "LinearMassWeightedIronFromSNIaOverHydrogenOfStars": (
            "LinearMassWeightedIronFromSNIaOverHydrogenOfStars",
            1,
            np.float32,
            "Msun",
            "Sum of the iron over hydrogen ratio of stars, multiplied with the stellar mass, where only iron from SNIa is included.",
            "star",
            "FMantissa9",
            False,
            [
                "PartType4/Masses",
                "PartType4/ElementMassFractions",
                "PartType4/IronMassFractionsFromSNIa",
            ],
        ),
    }

    # list of properties in the 'VR' category
    # we should really use removeprefix("VR") instead of [2:], but that only
    # exists since Python 3.9
    vr_properties = [
        vrname[2:] for vrname in full_property_list.keys() if vrname.startswith("VR")
    ]

    # object member variables
    properties: Dict[str, Dict]
    footnotes: List[str]

    def get_footnotes(self, name: str):
        """
        List all of the footnotes for a particular property. Returns an empty
        string for properties that have no footnotes.
        """
        footnotes = []
        for fnote in self.explanation.keys():
            names = self.explanation[fnote]
            if name in names:
                try:
                    i = self.footnotes.index(fnote)
                except ValueError:
                    i = len(self.footnotes)
                    self.footnotes.append(fnote)
                footnotes.append(i + 1)
        if len(footnotes) > 0:
            return f'$^{{{",".join([f"{i}" for i in footnotes])}}}$'
        else:
            return ""

    def __init__(self):
        """
        Constructor.
        """
        self.properties = {}
        self.footnotes = []

    def add_properties(self, halo_property: HaloProperty):
        """
        Add all the properties calculated for a particular halo type to the
        internal dictionary.
        """
        halo_type = halo_property.__name__
        props = halo_property.property_list
        for i, (
            prop_name,
            prop_outputname,
            prop_shape,
            prop_dtype,
            prop_units,
            prop_description,
            prop_cat,
            prop_comp,
            prop_dmo,
            prop_partprops,
        ) in enumerate(props):
            prop_units = (
                unyt.unyt_quantity(1, units=prop_units)
                .units.latex_repr.replace(
                    "\\rm{km} \\cdot \\rm{kpc}", "\\rm{kpc} \\cdot \\rm{km}"
                )
                .replace(
                    "\\frac{\\rm{km}^{2}}{\\rm{s}^{2}}", "\\rm{km}^{2} / \\rm{s}^{2}"
                )
            )
            prop_dtype = prop_dtype.__name__
            if prop_name in self.properties:
                if not prop_cat in self.categories:
                    print(f"Unknown category: {prop_cat}!")
                    exit()
                # run some checks
                if prop_shape != self.properties[prop_name]["shape"]:
                    print("Shape mismatch!")
                    print(halo_type, prop_name, prop_shape, self.properties[prop_name])
                    exit()
                if prop_dtype != self.properties[prop_name]["dtype"]:
                    print("dtype mismatch!")
                    print(halo_type, prop_name, prop_dtype, self.properties[prop_name])
                    exit()
                if prop_units != self.properties[prop_name]["units"]:
                    print("Unit mismatch!")
                    print(halo_type, prop_name, prop_units, self.properties[prop_name])
                    exit()
                if prop_description != self.properties[prop_name]["description"]:
                    print("Description mismatch!")
                    print(
                        halo_type,
                        prop_name,
                        prop_description,
                        self.properties[prop_name],
                    )
                    exit()
                if prop_cat != self.properties[prop_name]["category"]:
                    print("Category mismatch!")
                    print(halo_type, prop_name, prop_cat, self.properties[prop_name])
                    exit()
                assert prop_outputname == self.properties[prop_name]["name"]
                self.properties[prop_name]["types"].append(halo_type)
            else:
                self.properties[prop_name] = {
                    "name": prop_outputname,
                    "shape": prop_shape,
                    "dtype": prop_dtype,
                    "units": prop_units,
                    "description": prop_description,
                    "category": prop_cat,
                    "compression": prop_comp,
                    "dmo": prop_dmo,
                    "types": [halo_type],
                    "raw": props[i],
                }

    def print_dictionary(self):
        """
        Print the internal list of properties. Useful for regenerating the
        property dictionary with additional information for each property.

        Note that his will sort the dictionary alphabetically.
        """
        names = sorted(list(self.properties.keys()))
        print("full_property_list = {")
        for name in names:
            (
                raw_name,
                raw_outputname,
                raw_shape,
                raw_dtype,
                raw_units,
                raw_description,
                raw_cat,
                raw_comp,
                raw_dmo,
                raw_partprops,
            ) = self.properties[name]["raw"]
            raw_dtype = f"np.{raw_dtype.__name__}"
            print(
                f'  "{raw_name}": ("{raw_outputname}", {raw_shape}, {raw_dtype}, "{raw_units}", "{raw_description}", "{raw_cat}", "{raw_comp}", {raw_dmo}, {raw_partprops}),'
            )
        print("}")

    def print_table(self, tablefile: str, footnotefile: str, timestampfile: str):
        """
        Print the table in .tex format and generate the documentation.

        The documentation consists of
          - a hand-written SOAP.tex file.
          - a table .tex file, with the name given by 'tablefile'
          - a footnote .tex file, with the name given by 'footnotefile', which
            will contain the contents of the various hand-written footnote*.tex
            files
          - a version and time stamp .tex file, with the name given by
            'timestampfile'

        This function regenerates the last 3 files, based on the contents of
        the internal property dictionary.
        """

        # sort the properties by category and then alphabetically within each
        # category
        prop_names = sorted(
            self.properties.keys(),
            key=lambda key: (
                self.categories.index(self.properties[key]["category"]),
                self.properties[key]["name"].lower(),
            ),
        )

        # generate the LaTeX header for a standalone table file
        headstr = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{longtable}
\\usepackage{pifont}
\\usepackage{pdflscape}
\\usepackage{a4wide}
\\usepackage{multirow}
\\usepackage{xcolor}

\\begin{document}"""

        # property table string: table header
        tablestr = """\\begin{landscape}
\\begin{longtable}{lllllllllll}
Name & Shape & Type & Units & SH & ES & IS & EP & SO & Category & Compression\\\\
\\multicolumn{11}{l}{\\rule{30pt}{0pt}Description}\\\\
\\hline{}\\endhead{}"""
        # keep track of the previous category to draw a line when a category
        # is finished
        prev_cat = None
        for prop_name in prop_names:
            prop = self.properties[prop_name]
            footnotes = self.get_footnotes(prop_name)
            prop_outputname = f"{prop['name'].replace('_','')}{footnotes}"
            prop_shape = f'{prop["shape"]}'
            prop_dtype = prop["dtype"]
            prop_units = f'${prop["units"]}$' if prop["units"] != "" else "(no unit)"
            prop_cat = prop["category"]
            prop_comp = self.compression_description[prop["compression"]]
            prop_description = prop["description"].format(
                label="satisfying a spherical overdensity criterion."
            )
            checkmark = "\\ding{51}"
            xmark = "\\ding{53}"
            prop_subhalo = checkmark if "SubhaloProperties" in prop["types"] else xmark
            prop_exclusive = (
                checkmark if "ExclusiveSphereProperties" in prop["types"] else xmark
            )
            prop_inclusive = (
                checkmark if "InclusiveSphereProperties" in prop["types"] else xmark
            )
            prop_projected = (
                checkmark if "ProjectedApertureProperties" in prop["types"] else xmark
            )
            prop_SO = checkmark if "SOProperties" in prop["types"] else xmark
            table_props = [
                prop_outputname,
                prop_shape,
                prop_dtype,
                prop_units,
                prop_subhalo,
                prop_exclusive,
                prop_inclusive,
                prop_projected,
                prop_SO,
                prop_cat,
                prop_comp,
            ]
            if prop["dmo"]:
                print_table_props = [f"{{\\color{{violet}}{v}}}" for v in table_props]
                prop_description = f"{{\\color{{violet}}{prop_description}}}"
            else:
                print_table_props = list(table_props)
            if prev_cat is None:
                prev_cat = prop_cat
            if prop_cat != prev_cat:
                prev_cat = prop_cat
                tablestr += "\\hline{}"
            tablestr += "\\rule{0pt}{4ex}"
            tablestr += " & ".join([v for v in print_table_props]) + "\\\\*\n"
            tablestr += f"\\multicolumn{{11}}{{p{{24cm}}}}{{\\rule{{30pt}}{{0pt}}{prop_description}}}\\\\\n"
        tablestr += """\\end{longtable}
\\end{landscape}"""
        # standalone table file footer
        tailstr = "\\end{document}"

        # generate the documentation files
        with open(timestampfile, "w") as ofile:
            ofile.write(get_version_string())
        with open(tablefile, "w") as ofile:
            ofile.write(tablestr)
        with open(footnotefile, "w") as ofile:
            for i, fnote in enumerate(self.footnotes):
                with open(f"documentation/{fnote}", "r") as ifile:
                    fnstr = ifile.read()
                fnstr = fnstr.replace("$FOOTNOTE_NUMBER$", f"{i+1}")
                ofile.write(f"{fnstr}\n\n")

        # print the standalone table to the stdout
        print(f"{headstr}\n{tablestr}\n{tailstr}")


class DummyProperties:
    """
    Dummy HaloProperty object used to ensure all properties are in the property
    table, even if some of them are not computed for any halo type (e.g. the
    'VR' properties).
    """

    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in PropertyTable.full_property_list.keys()
    ]


if __name__ == "__main__":
    """
    Standalone script execution:
    Create a PropertyTable object will all the properties from all the halo
    types and print the property table or the documentation. The latter is the
    default; the former can be achieved by changing the boolean in the condition
    below.
    """

    # get all the halo types
    # we only import them here to avoid circular imports when this script is
    # imported from another script
    from aperture_properties import ExclusiveSphereProperties, InclusiveSphereProperties
    from projected_aperture_properties import ProjectedApertureProperties
    from SO_properties import SOProperties
    from subhalo_properties import SubhaloProperties

    table = PropertyTable()
    table.add_properties(ExclusiveSphereProperties)
    table.add_properties(InclusiveSphereProperties)
    table.add_properties(ProjectedApertureProperties)
    table.add_properties(SOProperties)
    table.add_properties(SubhaloProperties)
    table.add_properties(DummyProperties)

    # set to 'True' to print the internal property table
    # the resulting stdout output can be directly copy-pasted above to replace
    # the full_property_list (please run 'python3 -m black property_table.py'
    # after that to reformat the table).
    if False:
        table.print_dictionary()
    else:
        table.print_table(
            "documentation/table.tex",
            "documentation/footnotes.tex",
            "documentation/timestamp.tex",
        )
