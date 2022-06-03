#!/bin/env python

import numpy as np
import unyt
from halo_properties import HaloProperty, ReadRadiusTooSmallError

from dataset_names import mass_dataset


class SOProperties(HaloProperty):

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0": [
            "ComptonYParameters",
            "Coordinates",
            "Densities",
            "GroupNr_bound",
            "Masses",
            "MetalMassFractions",
            "Pressures",
            "Temperatures",
            "Velocities",
            "XrayLuminosities",
            "XrayPhotonLuminosities",
        ],
        "PartType1": ["Coordinates", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "MetalMassFractions",
            "Velocities",
        ],
        "PartType5": [
            "AccretionRates",
            "Coordinates",
            "DynamicalMasses",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "ParticleIDs",
            "SubgridMasses",
            "Velocities",
        ],
        "PartType6": ["Coordinates", "Masses", "Weights"],
    }

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    #                 Should contain a "{label}" entry that will be adjusted to describe the sphere for this
    #                 particular type of SO.
    SO_properties = [
        # global properties
        ("r", 1, np.float32, "Mpc", "Radius of a sphere {label}"),
        ("mass", 1, np.float32, "Msun", "Mass within a sphere {label}"),
        (
            "N",
            1,
            np.uint32,
            "dimensionless",
            "Number of particles within a sphere {label}",
        ),
        ("com", 3, np.float32, "Mpc", "Centre of mass within a sphere {label}"),
        (
            "vcom",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity within a sphere {label}",
        ),
        (
            "Mfrac_satellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite within a sphere {label}",
        ),
        # gas properties
        ("mass_gas", 1, np.float32, "Msun", "Total gas mass within a sphere {label}"),
        (
            "Jgas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of gas within a sphere {label}",
        ),
        (
            "com_gas",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of gas within a sphere {label}",
        ),
        (
            "vcom_gas",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas within a sphere {label}",
        ),
        (
            "veldisp_gas",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of gas within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "Mgasmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of gas within a sphere {label}",
        ),
        (
            "Mhotgas",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Tgas",
            1,
            np.float32,
            "K",
            "Mass-weighted average temperature of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Xraylum",
            3,
            np.float64,
            "erg/s",
            "Total Xray luminosity within a sphere {label}",
        ),
        (
            "Xrayphlum",
            3,
            np.float64,
            "1/s",
            "Total Xray photon luminosity within a sphere {label}",
        ),
        ("compY", 1, np.float64, "cm**2", "Total Compton y within a sphere {label}"),
        (
            "Ekin_gas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas within a sphere {label}. Measured relative to the centre of mass velocity of all particles.",
        ),
        (
            "Etherm_gas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas within a sphere {label}",
        ),
        # DM properties
        ("mass_dm", 1, np.float32, "Msun", "Total DM mass within a sphere {label}"),
        (
            "JDM",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of DM within a sphere {label}",
        ),
        (
            "veldisp_dm",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of DM within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        # stellar properties
        (
            "mass_star",
            1,
            np.float32,
            "Msun",
            "Total stellar mass within a sphere {label}",
        ),
        (
            "com_star",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars within a sphere {label}",
        ),
        (
            "vcom_star",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars within a sphere {label}",
        ),
        (
            "veldisp_star",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of stars within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "Jstar",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of stars within a sphere {label}",
        ),
        (
            "Mstarinit",
            1,
            np.float32,
            "Msun",
            "Total initial stellar mass with a sphere {label}",
        ),
        (
            "Mstarmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of stars within a sphere {label}",
        ),
        (
            "Lstar",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity within a sphere {label}",
        ),
        # BH properties
        (
            "MBHdyn",
            1,
            np.float32,
            "Msun",
            "Total dynamical BH mass within a sphere {label}",
        ),
        (
            "MBHsub",
            1,
            np.float32,
            "Msun",
            "Total sub-grid BH mass within a sphere {label}",
        ),
        (
            "BHlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event within a sphere {label}",
        ),
        ("BHmaxM", 1, np.float32, "Msun", "Maximum BH mass within a sphere {label}"),
        (
            "BHmaxID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxpos",
            3,
            np.float32,
            "Mpc",
            "Position of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxvel",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxAR",
            1,
            np.float32,
            "Msun/yr",
            "Accretion rate of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event of the most massive BH within a sphere {label}",
        ),
        # Neutrino properties
        (
            "MnuNS",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass within a sphere {label}",
        ),
        (
            "Mnu",
            1,
            np.float32,
            "Msun",
            "Total neutrino particle mass within a sphere {label}",
        ),
    ]

    def __init__(self, cellgrid, SOval, type="mean"):
        super().__init__(cellgrid)

        if not type in ["mean", "crit", "physical", "BN98"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        # in the neutrino model, the mean neutrino density is implicitly
        # assumed to be based on Omega_nu_0 and critical_density_0
        # here, critical_density_0 = critical_density * (H0/H)**2
        # however, we need to scale this to the appropriate redshift,
        # hence an additional factor 1/a**3
        self.nu_density = (
            cellgrid.cosmology["Omega_nu_0"]
            * cellgrid.critical_density
            * (
                cellgrid.cosmology["H0 [internal units]"]
                / cellgrid.cosmology["H [internal units]"]
            )
            ** 2
            / cellgrid.a ** 3
        )

        # This specifies how large a sphere is read in:
        # we use default values that are sufficiently small/large to avoid reading in too many particles
        self.mean_density_multiple = 1000.0
        self.critical_density_multiple = 1000.0
        self.physical_radius_mpc = 0.0
        if type == "mean":
            self.mean_density_multiple = SOval
        elif type == "crit":
            self.critical_density_multiple = SOval
        elif type == "BN98":
            self.critical_density_multiple = cellgrid.virBN98
        elif type == "physical":
            self.physical_radius_mpc = 0.001 * SOval

        # Give this calculation a name so we can select it on the command line
        if type in ["mean", "crit"]:
            self.name = f"SO_{SOval:.0f}_{type}"
        elif type == "physical":
            self.name = f"SO_{SOval:.0f}_kpc"
        elif type == "BN98":
            self.name = "SO_BN98"

        # set some variables that are used during the calculation and that do not change
        if self.type == "crit":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = f"{self.critical_density_multiple:.0f}_crit"
            self.label = f"within which the density is {self.critical_density_multiple:.0f} times the critical value"
        elif self.type == "mean":
            self.reference_density = self.mean_density_multiple * self.mean_density
            self.SO_name = f"{self.mean_density_multiple:.0f}_mean"
            self.label = f"within which the density is {self.mean_density_multiple:.0f} times the mean value"
        elif self.type == "physical":
            self.reference_density = 0.0
            self.SO_name = f"{1000. * self.physical_radius_mpc:.0f}_kpc"
            self.label = f"with a radius of {1000. * self.physical_radius_mpc:.0f} kpc"
        elif self.type == "BN98":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = "BN98"
            self.label = f"within which the density is {self.critical_density_multiple:.2f} times the critical value"

        if self.reference_density > 0.0:
            self.log_reference_density = np.log10(self.reference_density)
        else:
            self.log_reference_density = 0.0

    def calculate(self, input_halo, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Find the halo centre of potential
        centre = input_halo["cofp"]

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        groupnr = []
        for ptype in data:
            if ptype == "PartType6":
                continue
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos ** 2, axis=1))
            radius.append(r)
            velocity.append(data[ptype]["Velocities"])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)
            groupnr.append(data[ptype]["GroupNr_bound"])
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        position = unyt.array.uconcatenate(position)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)
        groupnr = unyt.array.uconcatenate(groupnr)

        # figure out which particles in the list are bound to a halo that is not the
        # central halo
        is_bound_to_satellite = (groupnr >= 0) & (groupnr != input_halo["index"])

        # add neutrinos
        if "PartType6" in data:
            numass = data["PartType6"]["Masses"] * data["PartType6"]["Weights"]
            pos = data["PartType6"]["Coordinates"] - centre[None, :]
            nur = np.sqrt(np.sum(pos ** 2, axis=1))
            all_mass = unyt.array.uconcatenate([mass, numass])
            all_r = unyt.array.uconcatenate([radius, nur])
        else:
            all_mass = mass
            all_r = radius

        # Sort by radius
        order = np.argsort(all_r)
        ordered_radius = all_r[order]
        cumulative_mass = np.cumsum(all_mass[order], dtype=np.float64).astype(
            mass.dtype
        )
        # add mean neutrino mass
        cumulative_mass += self.nu_density * 4.0 / 3.0 * np.pi * ordered_radius ** 3

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        # Note that because of the definition of the centre of potential, the first
        # particle *should* be at r=0. We need to manually exclude it, in case round
        # off error places it at a very small non-zero radius.
        nskip = 1
        while nskip < len(ordered_radius) and ordered_radius[nskip] == 0:
            nskip += 1
        ordered_radius = ordered_radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(ordered_radius)
        density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius ** 3)

        reg = mass.units.registry

        SO = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _ in self.SO_properties:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            SO[name] = unyt.unyt_array(val, dtype=dtype, units=unit, registry=reg)

        # Check if we ever reach the density threshold
        if self.reference_density > 0.0 * self.reference_density:
            log_density = np.log10(density.to(self.reference_density.units))
            if nr_parts > 0:
                if np.any(log_density > self.log_reference_density):
                    # Find smallest radius where the density is below the threshold
                    i = np.argmax(log_density <= self.log_reference_density)
                    if i == 0:
                        # we know that there are points above the threshold
                        # unfortunately, the centre is not
                        # find the next one that is:
                        i = np.argmax(density[1:] <= self.reference_density)
                        # +1 because i is now relative w.r.t. 1
                        i += 1
                else:
                    # all non-zero radius particles are below the threshold
                    # find a bin with negative slope to interpolate
                    i = 1
                # deal with the pathological case where we have one particle
                # below the threshold
                while (
                    i < len(log_density) - 1
                    and log_density[i + 1] > self.log_reference_density
                ):
                    i += 2
                if i >= len(log_density):
                    if ordered_radius[-1] > 20.0 * unyt.Mpc:
                        raise RuntimeError(
                            "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                        )
                    # trick the code into increasing the radius a bit
                    self.mean_density_multiple *= 0.9
                    self.critical_density_multiple *= 0.9
                    raise ReadRadiusTooSmallError(
                        "SO radius multiple estimate was too small!"
                    )
                # Interpolate to get the actual radius
                r1 = ordered_radius[i - 1]
                r2 = ordered_radius[i]
                logrho1 = log_density[i - 1]
                logrho2 = log_density[i]
                slope = (r2 - r1) / (logrho2 - logrho1)
                while slope > 0 or logrho2 > self.log_reference_density:
                    i += 1
                    if i >= len(log_density):
                        if ordered_radius[-1] > 20.0 * unyt.Mpc:
                            raise RuntimeError(
                                "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                            )
                        # trick the code into increasing the radius a bit
                        self.mean_density_multiple *= 0.9
                        self.critical_density_multiple *= 0.9
                        raise ReadRadiusTooSmallError(
                            "SO radius multiple estimate was too small!"
                        )
                    r1 = r2
                    r2 = ordered_radius[i]
                    logrho1 = logrho2
                    logrho2 = log_density[i]
                    slope = (r2 - r1) / (logrho2 - logrho1)

                # preserve the unyt_array dtype and units by using '+=' instead of assignment
                SO["r"] += r2 + slope * (self.log_reference_density - logrho2)
                SO["mass"] += 4.0 / 3.0 * np.pi * SO["r"] ** 3 * self.reference_density
        elif self.physical_radius_mpc > 0.0:
            SO["r"] += self.physical_radius_mpc * unyt.Mpc
            if nr_parts > 0:
                # find the enclosed mass using interpolation
                i = np.argmax(ordered_radius > SO["r"])
                if i == 0:
                    # we only have particles in the centre, so we cannot interpolate
                    SO["mass"] += cumulative_mass[i]
                else:
                    r1 = ordered_radius[i - 1]
                    r2 = ordered_radius[i]
                    M1 = cumulative_mass[i - 1]
                    M2 = cumulative_mass[i]
                    SO["mass"] += M1 + (SO["r"] - r1) / (r2 - r1) * (M2 - M1)

        else:
            # if we get here, we must be in the case where physical_radius_mpc is supposed to be 0
            # that can only happen if we are looking at a multiple of some radius
            # in that case, SO["r"] should remain 0
            # in any other case, something went wrong
            if not hasattr(self, "multiple"):
                raise ("Physical radius was set to 0! This should not happen!")

        """
        if self.name == "SO_2500_crit":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as pl
            fig, ax = pl.subplots(1, 1)
            rnew = SO["r"].to("Mpc")
            rVR = input_halo["SO_R_2500_rhocrit"].to("Mpc")
            rlim = 2. * max(rnew, rVR)
            ax.semilogy(ordered_radius.to("Mpc"), density.to("g/cm**3"), "o-")
            ax.axhline(y = self.reference_density.to("g/cm**3"), linestyle="--", color="C1")
            ax.axvline(x = rnew, linestyle="--", color="C1")
            ax.axvline(x = rVR, linestyle="--", color="C2")
            ax.set_xlim(0., rlim)
            ax.set_xlabel("$r$ (Mpc)")
            ax.set_ylabel("$\\rho$ (g cm$^{-3}$)")
            pl.tight_layout()
            pl.savefig(f"SO_R_2500_crit_{input_halo['index']}.png", dpi=300)
            fig.clear()
            pl.close(fig)
        """

        if SO["r"] > 0.0 * radius.units:
            gas_selection = radius[types == "PartType0"] < SO["r"]
            dm_selection = radius[types == "PartType1"] < SO["r"]
            star_selection = radius[types == "PartType4"] < SO["r"]
            bh_selection = radius[types == "PartType5"] < SO["r"]

            all_selection = radius < SO["r"]
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]
            is_bound_to_satellite = is_bound_to_satellite[all_selection]

            # we have to set the value like this to avoid conversion to dtype=np.float32...
            SO["N"] = unyt.unyt_array(
                all_selection.sum(),
                dtype=np.uint32,
                units="dimensionless",
                registry=reg,
            )

            # note that we cannot divide by mSO here, since that was based on an interpolation
            SO["com"][:] = (mass[:, None] * position).sum(axis=0) / mass.sum()
            SO["com"][:] += centre
            SO["vcom"][:] = (mass[:, None] * velocity).sum(axis=0) / mass.sum()

            SO["Mfrac_satellites"] += mass[is_bound_to_satellite].sum() / SO["mass"]

            gas_masses = mass[types == "PartType0"]
            gas_pos = position[types == "PartType0"]
            gas_relpos = gas_pos[:, :] - SO["com"][None, :]
            gas_vel = velocity[types == "PartType0"]
            gas_relvel = gas_vel[:, :] - SO["vcom"][None, :]
            SO["mass_gas"] += gas_masses.sum()
            SO["Jgas"][:] = (
                gas_masses[:, None] * unyt.array.ucross(gas_relpos, gas_relvel)
            ).sum(axis=0)

            dm_masses = mass[types == "PartType1"]
            dm_relpos = position[types == "PartType1"][:, :] - SO["com"][None, :]
            dm_vel = velocity[types == "PartType1"]
            dm_relvel = dm_vel[:, :] - SO["vcom"][None, :]
            SO["mass_dm"] += dm_masses.sum()
            SO["JDM"][:] = (
                dm_masses[:, None] * unyt.array.ucross(dm_relpos, dm_relvel)
            ).sum(axis=0)

            star_masses = mass[types == "PartType4"]
            star_pos = position[types == "PartType4"]
            star_relpos = star_pos[:, :] - SO["com"][None, :]
            star_vel = velocity[types == "PartType4"]
            star_relvel = star_vel[:, :] - SO["vcom"][None, :]
            SO["mass_star"] += star_masses.sum()
            SO["Jstar"][:] = (
                star_masses[:, None] * unyt.array.ucross(star_relpos, star_relvel)
            ).sum(axis=0)

            SO["MBHdyn"] += mass[types == "PartType5"].sum()

            # gas specific properties. We (can) only do these if we have gas.
            # (remember that "PartType0" might not be part of 'data' at all)
            if np.any(gas_selection):
                SO["com_gas"][:] = (gas_masses[:, None] * gas_pos).sum(
                    axis=0
                ) / gas_masses.sum()
                SO["com_gas"][:] += centre
                SO["vcom_gas"][:] = (gas_masses[:, None] * gas_vel).sum(
                    axis=0
                ) / gas_masses.sum()
                vrel = gas_vel - SO["vcom"][None, :]
                SO["veldisp_gas"][0] += (
                    gas_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][1] += (
                    gas_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][2] += (
                    gas_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][3] += (
                    gas_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][4] += (
                    gas_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][5] += (
                    gas_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / gas_masses.sum()

                SO["Mgasmetal"] += (
                    gas_masses * data["PartType0"]["MetalMassFractions"][gas_selection]
                ).sum()

                gas_temperatures = data["PartType0"]["Temperatures"][gas_selection]
                Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
                SO["Mhotgas"] += gas_masses[Tgas_selection].sum()

                if np.any(Tgas_selection):
                    SO["Tgas"] += (
                        gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                    ).sum() / SO["Mhotgas"]

                SO["Xraylum"] += data["PartType0"]["XrayLuminosities"][
                    gas_selection
                ].sum()
                SO["Xrayphlum"] += data["PartType0"]["XrayPhotonLuminosities"][
                    gas_selection
                ].sum()

                SO["compY"] += data["PartType0"]["ComptonYParameters"][
                    gas_selection
                ].sum()

                # below we need to force conversion to np.float64 before summing up particles
                # to avoid overflow
                ekin_gas = gas_masses * (
                    (velocity[types == "PartType0"] - SO["vcom"][None, :]) ** 2
                ).sum(axis=1)
                ekin_gas = unyt.unyt_array(
                    ekin_gas.value, dtype=np.float64, units=ekin_gas.units
                )
                SO["Ekin_gas"] += 0.5 * ekin_gas.sum()
                etherm_gas = (
                    1.5
                    * gas_masses
                    * data["PartType0"]["Pressures"][gas_selection]
                    / data["PartType0"]["Densities"][gas_selection]
                )
                etherm_gas = unyt.unyt_array(
                    etherm_gas.value, dtype=np.float64, units=etherm_gas.units
                )
                SO["Etherm_gas"] += etherm_gas.sum()

            # DM specific properties
            if np.any(dm_selection):
                vrel = dm_vel - SO["vcom"][None, :]
                SO["veldisp_dm"][0] += (
                    dm_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][1] += (
                    dm_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][2] += (
                    dm_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][3] += (
                    dm_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][4] += (
                    dm_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][5] += (
                    dm_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / dm_masses.sum()

            # star specific properties
            if np.any(star_selection):
                SO["com_star"][:] = (star_masses[:, None] * star_pos).sum(
                    axis=0
                ) / star_masses.sum()
                SO["com_star"][:] += centre
                SO["vcom_star"][:] = (star_masses[:, None] * star_vel).sum(
                    axis=0
                ) / star_masses.sum()
                vrel = star_vel - SO["vcom"][None, :]
                SO["veldisp_star"][0] += (
                    star_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][1] += (
                    star_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][2] += (
                    star_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][3] += (
                    star_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][4] += (
                    star_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][5] += (
                    star_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / star_masses.sum()

                SO["Mstarinit"] += data["PartType4"]["InitialMasses"][
                    star_selection
                ].sum()
                SO["Mstarmetal"] += (
                    star_masses
                    * data["PartType4"]["MetalMassFractions"][star_selection]
                ).sum()
                SO["Lstar"][:] = data["PartType4"]["Luminosities"][star_selection].sum()

            # BH specific properties
            if np.any(bh_selection):
                SO["MBHsub"] += data["PartType5"]["SubgridMasses"][bh_selection].sum()
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_selection
                ]

                SO["BHlasteventa"] += np.max(agn_eventa)

                iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_selection])
                SO["BHmaxM"] += data["PartType5"]["SubgridMasses"][bh_selection][iBHmax]
                # unyt annoyingly converts to a floating point type if you use '+='
                # the only way to avoid this is by directly setting the data for the unyt_array
                SO["BHmaxID"].data = data["PartType5"]["ParticleIDs"][bh_selection][
                    iBHmax
                ].data
                SO["BHmaxpos"] += data["PartType5"]["Coordinates"][bh_selection][iBHmax]
                SO["BHmaxvel"] += data["PartType5"]["Velocities"][bh_selection][iBHmax]
                SO["BHmaxAR"] += data["PartType5"]["AccretionRates"][bh_selection][
                    iBHmax
                ]
                SO["BHmaxlasteventa"] += agn_eventa[iBHmax]

            # Neutrino specific properties
            if "PartType6" in data:
                pos = data["PartType6"]["Coordinates"] - centre[None, :]
                nur = np.sqrt(np.sum(pos ** 2, axis=1))
                nu_selection = nur < SO["r"]
                SO["Mnu"] += data["PartType6"]["Masses"][nu_selection].sum()
                SO["MnuNS"] += (
                    data["PartType6"]["Masses"][nu_selection]
                    * data["PartType6"]["Weights"][nu_selection]
                ).sum()
                SO["MnuNS"] += self.nu_density * 4.0 / 3.0 * np.pi * SO["r"] ** 3

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for name, _, _, _, description in self.SO_properties:
            halo_result.update(
                {
                    f"SO/{self.SO_name}/{name}": (
                        SO[name],
                        description.format(label=self.label),
                    )
                }
            )

        return


class RadiusMultipleSOProperties(SOProperties):
    def __init__(self, cellgrid, SOval, multiple, type="mean"):
        if not type in ["mean", "crit"]:
            raise AttributeError(
                "SOs with a radius that is a multiple of another SO radius are only allowed for type mean or crit!"
            )

        # initialise the SOProperties object using a conservative physical radius estimate
        super().__init__(cellgrid, 3000.0, "physical")

        # overwrite the name, SO_name and label
        self.SO_name = f"{multiple:.0f}xR_{SOval:.0f}_{type}"
        self.label = f"with a radius that is {self.SO_name}"
        self.name = f"SO_{self.SO_name}"

        self.requested_type = type
        self.requested_SOval = SOval
        self.multiple = multiple

    def calculate(self, input_halo, data, halo_result):

        # find the actual physical radius we want
        key = f"SO/{self.requested_SOval:.0f}_{self.requested_type}/r"
        if not key in halo_result:
            raise RuntimeError(
                f"Trying to obtain {key}, but the corresponding SO radius has not been calculated!"
            )
        self.physical_radius_mpc = self.multiple * (halo_result[key][0].to("Mpc").value)

        # Check that we read in a large enough radius
        if self.multiple * halo_result[key][0] > input_halo["read_radius"]:
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")

        super().calculate(input_halo, data, halo_result)
        return
