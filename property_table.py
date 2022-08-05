#!/bin/env python

import numpy as np
import unyt


class PropertyTable:

    categories = ["basic", "general", "gas", "dm", "star", "baryon"]
    explanation = [
        ["BHmaxM"],
        ["com", "vcom"],
        ["Lgas", "Ldm", "Lstar", "Lbaryons"],
        ["kappa_corot_gas", "kappa_corot_star", "kappa_corot_baryons"],
        ["SFR", "MgasFe_SF", "MgasO_SF", "Mgas_SF", "Mgasmetal_SF"],
        ["Tgas", "Tgas_no_agn", "Tgas_no_cool", "Tgas_no_cool_no_agn"],
        ["StellarLuminosity"],
        ["R_vmax", "Vmax"],
        ["spin_parameter"],
        ["veldisp_matrix_gas", "veldisp_matrix_dm", "veldisp_matrix_star"],
        ["proj_veldisp_gas", "proj_veldisp_dm", "proj_veldisp_star"],
        ["MgasO", "MgasO_SF", "MgasFe", "MgasFe_SF", "Mgasmetal", "Mgasmetal_SF"],
        [
            "HalfMassRadiusTot",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
        ],
        ["Mfrac_satellites"],
        ["Ekin_gas", "Ekin_star"],
        ["Etherm_gas"],
        ["Mnu", "MnuNS"],
        ["Xraylum", "Xraylum_no_agn", "Xrayphlum", "Xrayphlum_no_agn"],
        ["compY", "compY_no_agn"],
    ]

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    full_property_list = {
        "BHlasteventa": (
            "BHlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event.",
            "general",
        ),
        "BHmaxAR": (
            "BHmaxAR",
            1,
            np.float32,
            "Msun/yr",
            "Accretion rate of most massive black hole.",
            "general",
        ),
        "BHmaxID": (
            "BHmaxID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive black hole.",
            "general",
        ),
        "BHmaxM": (
            "BHmaxM",
            1,
            np.float32,
            "Msun",
            "Mass of most massive black hole.",
            "general",
        ),
        "BHmaxlasteventa": (
            "BHmaxlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Scale-factor of last AGN event for most massive black hole.",
            "general",
        ),
        "BHmaxpos": (
            "BHmaxpos",
            3,
            np.float64,
            "kpc",
            "Position of most massive black hole.",
            "general",
        ),
        "BHmaxvel": (
            "BHmaxvel",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive black hole.",
            "general",
        ),
        "Ekin_gas": (
            "Ekin_gas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas, relative w.r.t. the gas bulk velocity.",
            "gas",
        ),
        "Ekin_star": (
            "Ekin_star",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the stars, relative w.r.t. the stellar bulk velocity.",
            "star",
        ),
        "Etherm_gas": (
            "Etherm_gas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas.",
            "gas",
        ),
        "HalfMassRadiusDM": (
            "HalfMassRadiusDM",
            1,
            np.float32,
            "kpc",
            "Total DM half mass radius.",
            "dm",
        ),
        "HalfMassRadiusGas": (
            "HalfMassRadiusGas",
            1,
            np.float32,
            "kpc",
            "Total gas half mass radius.",
            "gas",
        ),
        "HalfMassRadiusStar": (
            "HalfMassRadiusStar",
            1,
            np.float32,
            "kpc",
            "Total stellar half mass radius.",
            "star",
        ),
        "HalfMassRadiusTot": (
            "HalfMassRadiusTot",
            1,
            np.float32,
            "kpc",
            "Total half mass radius.",
            "general",
        ),
        "Lbaryons": (
            "Lbaryons",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of baryons (gas and stars), relative w.r.t. the centre of potential and baryonic bulk velocity.",
            "baryon",
        ),
        "Ldm": (
            "Ldm",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the dark matter, relative w.r.t. the centre of potential and DM bulk velocity.",
            "dm",
        ),
        "Lgas": (
            "Lgas",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the gas, relative w.r.t. the centre of potential and gas bulk velocity.",
            "gas",
        ),
        "Lstar": (
            "Lstar",
            3,
            np.float32,
            "Msun*km*kpc/s",
            "Total angular momentum of the stars, relative w.r.t. the centre of potential and stellar bulk velocity.",
            "star",
        ),
        "Mbh_dynamical": (
            "Mbh_dynamical",
            1,
            np.float32,
            "Msun",
            "Total BH dynamical mass.",
            "basic",
        ),
        "Mbh_subgrid": (
            "Mbh_subgrid",
            1,
            np.float32,
            "Msun",
            "Total BH subgrid mass.",
            "basic",
        ),
        "Mdm": ("Mdm", 1, np.float32, "Msun", "Total DM mass.", "basic"),
        "Mfrac_satellites": (
            "Mfrac_satellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite.",
            "general",
        ),
        "Mgas": ("Mgas", 1, np.float32, "Msun", "Total gas mass.", "basic"),
        "MgasFe": ("MgasFe", 1, np.float32, "Msun", "Total gas mass in iron.", "gas"),
        "MgasFe_SF": (
            "MgasFe_SF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron for gas that is star-forming.",
            "gas",
        ),
        "MgasFe_noSF": (
            "MgasFe_noSF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in iron for gas that is non star-forming.",
            "gas",
        ),
        "MgasO": ("MgasO", 1, np.float32, "Msun", "Total gas mass in oxygen.", "gas"),
        "MgasO_SF": (
            "MgasO_SF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen for gas that is star-forming.",
            "gas",
        ),
        "MgasO_noSF": (
            "MgasO_noSF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in oxygen for gas that is non star-forming.",
            "gas",
        ),
        "Mgas_SF": (
            "Mgas_SF",
            1,
            np.float32,
            "Msun",
            "Total mass of star-forming gas.",
            "gas",
        ),
        "Mgas_noSF": (
            "Mgas_noSF",
            1,
            np.float32,
            "Msun",
            "Total mass of non star-forming gas.",
            "gas",
        ),
        "Mgasmetal": (
            "Mgasmetal",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals.",
            "gas",
        ),
        "Mgasmetal_SF": (
            "Mgasmetal_SF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals for gas that is star-forming.",
            "gas",
        ),
        "Mgasmetal_noSF": (
            "Mgasmetal_noSF",
            1,
            np.float32,
            "Msun",
            "Total gas mass in metals for gas that is non star-forming.",
            "gas",
        ),
        "Mhotgas": (
            "Mhotgas",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with a temperature above 1e5 K.",
            "gas",
        ),
        "Mnu": ("Mnu", 1, np.float32, "Msun", "Total neutrino particle mass.", "basic"),
        "MnuNS": (
            "MnuNS",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass.",
            "basic",
        ),
        "Mstar": ("Mstar", 1, np.float32, "Msun", "Total stellar mass.", "basic"),
        "Mstar_init": (
            "Mstar_init",
            1,
            np.float32,
            "Msun",
            "Total stellar initial mass.",
            "star",
        ),
        "Mstarmetal": (
            "Mstarmetal",
            1,
            np.float32,
            "Msun",
            "Total stellar mass in metals.",
            "star",
        ),
        "Mtot": ("Mtot", 1, np.float32, "Msun", "Total mass.", "basic"),
        "Nbh": (
            "Nbh",
            1,
            np.uint32,
            "dimensionless",
            "Number of black hole particles.",
            "basic",
        ),
        "Ndm": (
            "Ndm",
            1,
            np.uint32,
            "dimensionless",
            "Number of dark matter particles.",
            "basic",
        ),
        "Ngas": (
            "Ngas",
            1,
            np.uint32,
            "dimensionless",
            "Number of gas particles.",
            "basic",
        ),
        "Nnu": (
            "Nnu",
            1,
            np.uint32,
            "dimensionless",
            "Number of neutrino particles.",
            "basic",
        ),
        "Nstar": (
            "Nstar",
            1,
            np.uint32,
            "dimensionless",
            "Number of star particles.",
            "basic",
        ),
        "R_vmax": (
            "R_vmax",
            1,
            np.float32,
            "kpc",
            "Radius at which Vmax is reached.",
            "general",
        ),
        "SFR": ("SFR", 1, np.float32, "Msun/yr", "Total SFR.", "gas"),
        "StellarLuminosity": (
            "StellarLuminosity",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity in the 9 GAMA bands.",
            "star",
        ),
        "Tgas": ("Tgas", 1, np.float32, "K", "Mass-weighted gas temperature.", "gas"),
        "Tgas_no_agn": (
            "Tgas_no_agn",
            1,
            np.float32,
            "K",
            "Mass-weighted gas temperature, excluding gas that was recently heated by AGN.",
            "gas",
        ),
        "Tgas_no_cool": (
            "Tgas_no_cool",
            1,
            np.float32,
            "K",
            "Mass-weighted gas temperature, excluding cool gas with a temperature below 1e5 K.",
            "gas",
        ),
        "Tgas_no_cool_no_agn": (
            "Tgas_no_cool_no_agn",
            1,
            np.float32,
            "K",
            "Mass-weighted gas temperature, excluding cool gas with a temperature below 1e5 K and gas that was recently heated by AGN.",
            "gas",
        ),
        "Vmax": (
            "Vmax",
            1,
            np.float32,
            "km/s",
            "Maximum circular velocity.",
            "general",
        ),
        "Xraylum": (
            "Xraylum",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands.",
            "gas",
        ),
        "Xraylum_no_agn": (
            "Xraylum_no_agn",
            3,
            np.float64,
            "erg/s",
            "Total rest-frame Xray luminosity in three bands. Excludes gas that was heated by AGN less than 15 Myr ago.",
            "gas",
        ),
        "Xrayphlum": (
            "Xrayphlum",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands.",
            "gas",
        ),
        "Xrayphlum_no_agn": (
            "Xrayphlum_no_agn",
            3,
            np.float64,
            "1/s",
            "Total rest-frame Xray photon luminosity in three bands. Exclude gas that was heated by AGN less than 15 Myr ago.",
            "gas",
        ),
        "com": ("com", 3, np.float32, "kpc", "Centre of mass.", "basic"),
        "com_gas": ("com_gas", 3, np.float32, "Mpc", "Centre of mass of gas.", "gas"),
        "com_star": (
            "com_star",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars.",
            "star",
        ),
        "compY": (
            "compY",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter.",
            "gas",
        ),
        "compY_no_agn": (
            "compY_no_agn",
            1,
            np.float64,
            "cm**2",
            "Total Compton y parameter. Excludes gas that was heated by AGN less than 15 Myr ago.",
            "gas",
        ),
        "kappa_corot_baryons": (
            "kappa_corot_baryons",
            1,
            np.float32,
            "dimensionless",
            "Kappa corot for baryons (gas and stars), relative w.r.t. the centre of potential and the bulk velocity of the baryons.",
            "baryon",
        ),
        "kappa_corot_gas": (
            "kappa_corot_gas",
            1,
            np.float32,
            "dimensionless",
            "Kappa corot for gas, relative w.r.t. the centre of potential and the bulk velocity of the gas.",
            "gas",
        ),
        "kappa_corot_star": (
            "kappa_corot_star",
            1,
            np.float32,
            "dimensionless",
            "Kappa corot for stars, relative w.r.t. the centre of potential and the bulk velocity of the stars.",
            "star",
        ),
        "proj_veldisp_dm": (
            "proj_veldisp_dm",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the DM along the projection axis, relative w.r.t. the DM bulk velocity.",
            "dm",
        ),
        "proj_veldisp_gas": (
            "proj_veldisp_gas",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the gas along the projection axis, relative w.r.t. the gas bulk velocity.",
            "gas",
        ),
        "proj_veldisp_star": (
            "proj_veldisp_star",
            1,
            np.float32,
            "km/s",
            "Mass-weighted velocity dispersion of the stars along the projection axis, relative w.r.t. the stellar bulk velocity.",
            "star",
        ),
        "r": ("r", 1, np.float32, "Mpc", "Radius of a sphere {label}", "basic"),
        "spin_parameter": (
            "spin_parameter",
            1,
            np.float32,
            "dimensionless",
            "Bullock et al. (2001) spin parameter.",
            "general",
        ),
        "vcom": ("vcom", 3, np.float32, "km/s", "Centre of mass velocity.", "basic"),
        "vcom_gas": (
            "vcom_gas",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas.",
            "gas",
        ),
        "vcom_star": (
            "vcom_star",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars.",
            "star",
        ),
        "veldisp_matrix_dm": (
            "veldisp_matrix_dm",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the dark matter. Measured relative w.r.t. the DM bulk velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "dm",
        ),
        "veldisp_matrix_gas": (
            "veldisp_matrix_gas",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the gas. Measured relative w.r.t. the gas bulk velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "gas",
        ),
        "veldisp_matrix_star": (
            "veldisp_matrix_star",
            6,
            np.float32,
            "km**2/s**2",
            "Mass-weighted velocity dispersion of the stars. Measured relative w.r.t. the stellar bulk velocity. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
            "star",
        ),
    }

    def get_footnotes(self, name):
        footnotes = []
        for i, names in enumerate(self.explanation):
            if name in names:
                footnotes.append(i + 1)
        if len(footnotes) > 0:
            return f'$^{{{",".join([f"{i}" for i in footnotes])}}}$'
        else:
            return ""

    def __init__(self):
        self.properties = {}

    def add_properties(self, halo_property):
        halo_type = halo_property.__name__
        props = halo_property.property_list
        for i, (
            prop_name,
            prop_shape,
            prop_dtype,
            prop_units,
            prop_description,
            prop_cat,
        ) in enumerate(props):
            prop_units = unyt.unyt_quantity(1, units=prop_units).units.latex_repr
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
                self.properties[prop_name]["types"].append(halo_type)
            else:
                self.properties[prop_name] = {
                    "shape": prop_shape,
                    "dtype": prop_dtype,
                    "units": prop_units,
                    "description": prop_description,
                    "category": prop_cat,
                    "types": [halo_type],
                    "raw": props[i],
                }

    def print_dictionary(self):
        names = sorted(list(self.properties.keys()))
        print("full_property_list = {")
        for name in names:
            (
                raw_name,
                raw_shape,
                raw_dtype,
                raw_units,
                raw_description,
                raw_cat,
            ) = self.properties[name]["raw"]
            raw_dtype = f"np.{raw_dtype.__name__}"
            print(
                f'  "{raw_name}": ("{raw_name}", {raw_shape}, {raw_dtype}, "{raw_units}", "{raw_description}", "{raw_cat}"),'
            )
        print("}")

    def print_table(self):
        prop_names = sorted(
            self.properties.keys(),
            key=lambda key: (
                self.categories.index(self.properties[key]["category"]),
                key.lower(),
            ),
        )
        tablestr = """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{longtable}
\\usepackage{pifont}
\\usepackage{pdflscape}
\\usepackage{a4wide}
\\begin{document}

\\begin{landscape}
\\begin{longtable}{lllllllllp{10cm}}
Name & Shape & Type & Units & SH & EX & PA & SO & Category & Description\\\\
\\hline{}\\endhead{}"""
        prev_cat = None
        for prop_name in prop_names:
            prop = self.properties[prop_name]
            footnotes = self.get_footnotes(prop_name)
            prop_name = f"\\verb+{prop_name}+{footnotes}"
            prop_shape = f'{prop["shape"]}'
            prop_dtype = prop["dtype"]
            prop_units = f'${prop["units"]}$' if prop["units"] != "" else "(no unit)"
            prop_cat = prop["category"]
            prop_description = prop["description"].format(
                label="satisfying a spherical overdensity criterion."
            )
            checkmark = "\\ding{51}"
            xmark = "\\ding{53}"
            prop_subhalo = checkmark if "SubhaloProperties" in prop["types"] else xmark
            prop_exclusive = (
                checkmark if "ExclusiveSphereProperties" in prop["types"] else xmark
            )
            prop_projected = (
                checkmark if "ProjectedApertureProperties" in prop["types"] else xmark
            )
            prop_SO = checkmark if "SOProperties" in prop["types"] else xmark
            if prev_cat is None:
                prev_cat = prop_cat
            if prop_cat != prev_cat:
                prev_cat = prop_cat
                tablestr += "\\hline{}"
            tablestr += (
                " & ".join(
                    [
                        v
                        for v in [
                            prop_name,
                            prop_shape,
                            prop_dtype,
                            prop_units,
                            prop_subhalo,
                            prop_exclusive,
                            prop_projected,
                            prop_SO,
                            prop_cat,
                            prop_description,
                        ]
                    ]
                )
                + "\\\\\n"
            )
        tablestr += """\\end{longtable}
\\end{landscape}
\\end{document}"""
        print(tablestr)


if __name__ == "__main__":

    from exclusive_sphere_properties import ExclusiveSphereProperties
    from projected_aperture_properties import ProjectedApertureProperties
    from SO_properties import SOProperties
    from subhalo_properties import SubhaloProperties

    table = PropertyTable()
    table.add_properties(ExclusiveSphereProperties)
    table.add_properties(ProjectedApertureProperties)
    table.add_properties(SOProperties)
    table.add_properties(SubhaloProperties)

    #    table.print_dictionary()
    table.print_table()