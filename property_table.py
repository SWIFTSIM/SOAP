#!/bin/env python

import numpy as np
import unyt
import subprocess
import datetime
import os


def get_version_string():
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


def word_wrap_name(name):
    """
    Put a line break in if a name gets too long
    """
    maxlen = 20
    count = 0
    output = []
    last_was_lower = False
    for i in range(len(name)):
        next_char = name[i]
        count += 1
        if count > maxlen and next_char.isupper() and last_was_lower:
            output.append(r"\-")
        output.append(next_char)
        last_was_lower = next_char.isupper() == False
    return "".join(output)


class PropertyTable:
    categories = ["basic", "general", "gas", "dm", "star", "baryon", "VR", "SOAP"]
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
            "Xraylum_restframe",
            "Xrayphlum",
            "Xrayphlum_restframe",
        ],
        "footnote_compY.tex": ["compY", "compY_no_agn"],
        "footnote_dopplerB.tex": ["DopplerB"],
        "footnote_coreexcision.tex": [
            "Tgas_cy_weighted_core_excision",
            "Tgas_cy_weighted_core_excision_no_agn",
            "Tgas_core_excision",
            "Tgas_no_cool_core_excision",
            "Tgas_no_agn_core_excision",
            "Tgas_no_cool_no_agn_core_excision",
            "Xraylum_core_excision",
            "Xraylum_no_agn_core_excision",
            "Xrayphlum_core_excision",
            "Xrayphlum_no_agn_core_excision",
            "SpectroscopicLikeTemperature_core_excision",
            "SpectroscopicLikeTemperature_no_agn_core_excision",
        ],
        "footnote_cytemp.tex": [
            "Tgas_cy_weighted",
            "Tgas_cy_weighted_no_agn",
            "Tgas_cy_weighted_core_excision",
            "Tgas_cy_weighted_core_excision_no_agn",
        ],
        "footnote_spectroscopicliketemperature.tex": [
            "SpectroscopicLikeTemperature",
            "SpectroscopicLikeTemperature_core_excision",
            "SpectroscopicLikeTemperature_no_agn",
            "SpectroscopicLikeTemperature_no_agn_core_excision",
        ],
    }

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
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    #  - category: Category used to decide if this property should be calculated for a halo
    #  - lossy compression filter: Lossy compression filter used in the output to reduce the file size
    #  - DMO property: Should this property be calculated for a DMO run?
    full_property_list = {
        "BHlasteventa": (
            "BlackHolesLastEventScalefactor",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event.",
            "general",
            "FMantissa9",
            False,
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
        ),
        "BaryonInertiaTensor": (
            "BaryonInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "3D baryon inertia tensor computed from the baryon (gas and stars) mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "baryon",
            "FMantissa9",
            False,
        ),
        "ReducedBaryonInertiaTensor": (
            "ReducedBaryonInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "Reduced 3D baryon inertia tensor computed from the baryon (gas and stars) mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "baryon",
            "FMantissa9",
            False,
        ),
        "DMInertiaTensor": (
            "DarkMatterInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "3D dark matter inertia tensor computed from the DM mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "dm",
            "FMantissa9",
            True,
        ),
        "ReducedDMInertiaTensor": (
            "ReducedDarkMatterInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "Reduced 3D dark matter inertia tensor computed from the DM mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "dm",
            "FMantissa9",
            True,
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
        ),
        "GasInertiaTensor": (
            "GasInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "3D gas inertia tensor computed from the gas mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "gas",
            "FMantissa9",
            False,
        ),
        "ReducedGasInertiaTensor": (
            "ReducedGasInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "Reduced 3D gas inertia tensor computed from the gas mass distribution, relative to the centre of potential. Stores diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "gas",
            "FMantissa9",
            False,
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
        ),
        "ProjectedBaryonInertiaTensor": (
            "ProjectedBaryonInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "2D inertia tensor of the projected baryon (gas and stars) mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "baryon",
            "FMantissa9",
            False,
        ),
        "ReducedProjectedBaryonInertiaTensor": (
            "ReducedProjectedBaryonInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "Reduced 2D inertia tensor of the projected baryon (gas and stars) mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "baryon",
            "FMantissa9",
            False,
        ),
        "ProjectedGasInertiaTensor": (
            "ProjectedGasInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "2D inertia tensor of the projected gas mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "gas",
            "FMantissa9",
            False,
        ),
        "ReducedProjectedGasInertiaTensor": (
            "ReducedProjectedGasInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "Reduced 2D inertia tensor of the projected gas mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "gas",
            "FMantissa9",
            False,
        ),
        "ProjectedStellarInertiaTensor": (
            "ProjectedStellarInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "2D inertia tensor of the projected stellar mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "star",
            "FMantissa9",
            False,
        ),
        "ReducedProjectedStellarInertiaTensor": (
            "ReducedProjectedStellarInertiaTensor",
            3,
            np.float32,
            "kpc**2",
            "Reduced 2D inertia tensor of the projected stellar mass distribution, relative to the centre of potential. Diagonal and one off-diagonal component as (1,1), (2,2), (1,2).",
            "star",
            "FMantissa9",
            False,
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
        ),
        "StellarInertiaTensor": (
            "StellarInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "3D stellar inertia tensor computed from the stellar mass distribution, relative to the centre of potential. Diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "star",
            "FMantissa9",
            False,
        ),
        "ReducedStellarInertiaTensor": (
            "ReducedStellarInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "Reduced 3D stellar inertia tensor computed from the stellar mass distribution, relative to the centre of potential. Diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "star",
            "FMantissa9",
            False,
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
        ),
        "Tgas_cy_weighted": (
            "GasComptonYTemperature",
            1,
            np.float32,
            "K",
            "ComptonY-weighted mean gas temperature.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_cy_weighted_no_agn": (
            "GasComptonYTemperatureWithoutRecentAGNHeating",
            1,
            np.float32,
            "K",
            "ComptonY-weighted mean gas temperature, excluding gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_cy_weighted_core_excision": (
            "GasComptonYTemperatureCoreExcision",
            1,
            np.float32,
            "K",
            "ComptonY-weighted mean gas temperature, excluding the inner {core_excision}.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_cy_weighted_core_excision_no_agn": (
            "GasComptonYTemperatureWithoutRecentAGNHeatingCoreExcision",
            1,
            np.float32,
            "K",
            "ComptonY-weighted mean gas temperature, excluding the inner {core_excision} and gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_core_excision": (
            "GasTemperatureCoreExcision",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding the inner {core_excision}.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_no_cool_core_excision": (
            "GasTemperatureWithoutCoolGasCoreExcision",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding the inner {core_excision} and gas below 1e5 K.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_no_agn_core_excision": (
            "GasTemperatureWithoutRecentAGNHeatingCoreExcision",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding the inner {core_excision}, and gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
        ),
        "Tgas_no_cool_no_agn_core_excision": (
            "GasTemperatureWithoutCoolGasAndRecentAGNHeatingCoreExcision",
            1,
            np.float32,
            "K",
            "Mass-weighted mean gas temperature, excluding the inner {core_excision}, gas below 1e5 K and gas that was recently heated by AGN.",
            "gas",
            "FMantissa9",
            False,
        ),
        "TotalInertiaTensor": (
            "TotalInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "3D inertia tensor computed from the total mass distribution, relative to the centre of potential. Diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "general",
            "FMantissa9",
            True,
        ),
        "ReducedTotalInertiaTensor": (
            "ReducedTotalInertiaTensor",
            6,
            np.float32,
            "kpc**2",
            "Reduced 3D inertia tensor computed from the total mass distribution, relative to the centre of potential. Diagonal components and one off-diagonal triangle as (1,1), (2,2), (3,3), (1,2), (1,3), (2,3).",
            "general",
            "FMantissa9",
            True,
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
        ),
        "Xraylum": (
            "XRayLuminosity",
            3,
            np.float64,
            "erg/s",
            "Total observer-frame Xray luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_restframe": (
            "XRayLuminosityInRestframe",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_no_agn": (
            "XRayLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "erg/s",
            "Total observer-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_restframe_no_agn": (
            "XRayLuminosityInRestframeWithoutRecentAGNHeating",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_core_excision": (
            "XRayLuminosityCoreExcision",
            3,
            np.float64,
            "erg/s",
            "Total observer-frame Xray luminosity in three bands. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_restframe_core_excision": (
            "XRayLuminosityInRestframeCoreExcision",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_no_agn_core_excision": (
            "XRayLuminosityWithoutRecentAGNHeatingCoreExcision",
            3,
            np.float64,
            "erg/s",
            "Total observer-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xraylum_restframe_no_agn_core_excision": (
            "XRayLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was recently heated by AGN. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum": (
            "XRayPhotonLuminosity",
            3,
            np.float64,
            "1/s",
            "Total observer-frame Xray photon luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_restframe": (
            "XRayPhotonLuminosityInRestframe",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_no_agn": (
            "XRayPhotonLuminosityWithoutRecentAGNHeating",
            3,
            np.float64,
            "1/s",
            "Total observer-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_restframe_no_agn": (
            "XRayPhotonLuminosityInRestframeWithoutRecentAGNHeating",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN.",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_core_excision": (
            "XRayPhotonLuminosityCoreExcision",
            3,
            np.float64,
            "1/s",
            "Total observer-frame Xray photon luminosity in three bands. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_restframe_core_excision": (
            "XRayPhotonLuminosityInRestframeCoreExcision",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_no_agn_core_excision": (
            "XRayPhotonLuminosityWithoutRecentAGNHeatingCoreExcision",
            3,
            np.float64,
            "1/s",
            "Total observer-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "Xrayphlum_restframe_no_agn_core_excision": (
            "XRayPhotonLuminosityInRestframeWithoutRecentAGNHeatingCoreExcision",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was recently heated by AGN. Excludes gas in the inner {core_excision}",
            "gas",
            "DMantissa9",
            False,
        ),
        "SpectroscopicLikeTemperature": (
            "SpectroscopicLikeTemperature",
            1,
            np.float32,
            "K",
            "Spectroscopic-like gas temperature.",
            "gas",
            "FMantissa9",
            False,
        ),
        "SpectroscopicLikeTemperature_no_agn": (
            "SpectroscopicLikeTemperatureWithoutRecentAGNHeating",
            1,
            np.float32,
            "K",
            "Spectroscopic-like gas temperature. Exclude gas that was recently heated by AGN",
            "gas",
            "FMantissa9",
            False,
        ),
        "SpectroscopicLikeTemperature_core_excision": (
            "SpectroscopicLikeTemperatureCoreExcision",
            1,
            np.float32,
            "K",
            "Spectroscopic-like gas temperature. Excludes gas in the inner {core_excision}",
            "gas",
            "FMantissa9",
            False,
        ),
        "SpectroscopicLikeTemperature_no_agn_core_excision": (
            "SpectroscopicLikeTemperatureWithoutRecentAGNHeatingCoreExcision",
            1,
            np.float32,
            "K",
            "Spectroscopic-like gas temperature. Exclude gas that was recently heated by AGN. Excludes gas in the inner {core_excision}",
            "gas",
            "FMantissa9",
            False,
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
        ),
        "SOAPSubhaloRankByBoundMass": (
            "SubhaloRankByBoundMass",
            1,
            np.int32,
            "dimensionless",
            "Ranking by mass of the halo within its parent field halo. Zero for the most massive halo in the field halo.",
            "SOAP",
            "None",
            True,
        ),
    }

    # we should really use removeprefix("VR") instead of [2:], but that only
    # exists since Python 3.9
    vr_properties = [
        vrname[2:] for vrname in full_property_list.keys() if vrname.startswith("VR")
    ]

    # halo properties derived from other properties by SOAP
    soap_properties = [
        soapname[4:]
        for soapname in full_property_list.keys()
        if soapname.startswith("SOAP")
    ]

    def get_footnotes(self, name):
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
        self.properties = {}
        self.footnotes = []

    def add_properties(self, halo_property):
        halo_type = halo_property.__name__
        props = halo_property.property_list
        for (
            i,
            (
                prop_name,
                prop_outputname,
                prop_shape,
                prop_dtype,
                prop_units,
                prop_description,
                prop_cat,
                prop_comp,
                prop_dmo,
            ),
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
            ) = self.properties[name]["raw"]
            raw_dtype = f"np.{raw_dtype.__name__}"
            print(
                f'  "{raw_name}": ("{raw_outputname}", {raw_shape}, {raw_dtype}, "{raw_units}", "{raw_description}", "{raw_cat}", "{raw_comp}", {raw_dmo}),'
            )
        print("}")

    def print_table(self, tablefile, footnotefile, timestampfile):
        prop_names = sorted(
            self.properties.keys(),
            key=lambda key: (
                self.categories.index(self.properties[key]["category"]),
                self.properties[key]["name"].lower(),
            ),
        )
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

        tablestr = """\\begin{landscape}
\\begin{longtable}{p{15em}llllllllll}
Name & Shape & Type & Units & SH & ES & IS & EP & SO & Category & Compression\\\\
\\multicolumn{11}{l}{\\rule{30pt}{0pt}Description}\\\\
\\hline{}\\endhead{}"""
        prev_cat = None
        for prop_name in prop_names:
            prop = self.properties[prop_name]
            footnotes = self.get_footnotes(prop_name)
            prop_outputname = f"{prop['name'].replace('_','')}{footnotes}"
            prop_outputname = word_wrap_name(prop_outputname)
            prop_shape = f'{prop["shape"]}'
            prop_dtype = prop["dtype"]
            prop_units = f'${prop["units"]}$' if prop["units"] != "" else "(no unit)"
            prop_cat = prop["category"]
            prop_comp = self.compression_description[prop["compression"]]
            prop_description = prop["description"].format(
                label="satisfying a spherical overdensity criterion.",
                core_excision="excised core",
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
            prop_SO = checkmark if "CoreExcisedSOProperties" in prop["types"] else xmark
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
            tablestr += f"\\multicolumn{{10}}{{p{{15cm}}}}{{\\rule{{30pt}}{{0pt}}{prop_description}}}\\\\\n"
        tablestr += """\\end{longtable}
\\end{landscape}"""
        tailstr = "\\end{document}"
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
        print(f"{headstr}\n{tablestr}\n{tailstr}")


class DummyProperties:
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in PropertyTable.full_property_list.keys()
    ]


if __name__ == "__main__":
    from aperture_properties import ExclusiveSphereProperties, InclusiveSphereProperties
    from projected_aperture_properties import ProjectedApertureProperties
    from SO_properties import CoreExcisedSOProperties
    from subhalo_properties import SubhaloProperties

    table = PropertyTable()
    table.add_properties(ExclusiveSphereProperties)
    table.add_properties(InclusiveSphereProperties)
    table.add_properties(ProjectedApertureProperties)
    table.add_properties(CoreExcisedSOProperties)
    table.add_properties(SubhaloProperties)
    table.add_properties(DummyProperties)

    if False:
        table.print_dictionary()
    else:
        table.print_table(
            "documentation/table.tex",
            "documentation/footnotes.tex",
            "documentation/timestamp.tex",
        )
