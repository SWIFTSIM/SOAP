#!/bin/env python

import numpy as np
import unyt
from scipy.optimize import brentq

from halo_properties import HaloProperty, ReadRadiusTooSmallError
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_axis_lengths,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from property_table import PropertyTable
from dataset_names import mass_dataset
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile

# index of elements O and Fe in the SmoothedElementMassFractions dataset
indexO = 4
indexFe = 8


def cumulative_mass_intersection(r, rho_dim, slope_dim):
    """
    Function used to find the intersection of the cumulative mass curve at fixed
    mean density, and the actual cumulative mass as obtained from linear
    interpolation on a cumulative mass profile.

    The equation we want to solve is:
      4*pi/3*rho * r^3 - (M2-M1)/(r2-r1) * r + (M2-M1)/(r2-r1)*r1 - M1 = 0
    Since all quantities have units and scipy cannot handle those, we actually
    solve
      4*pi/3*rho_d * u^3 - S_d * u + S_d - 1 = 0,
    with
      rho_d = rho * r1**3 / M1
      S_d = (M2-M1)/(r2-r1) * (r1/M1)
    The result then needs to be multiplied with r1 to get the intersection radius
    """
    return 4.0 * np.pi / 3.0 * rho_dim * r**3 - slope_dim * r + slope_dim - 1.0


def find_SO_radius_and_mass(
    ordered_radius, density, cumulative_mass, reference_density
):
    """
    Find the radius and mass of an SO from the ordered density and cumulative
    mass profiles.

    The profiles are constructed by sorting the particles within the spherical
    region and then summing their masses in that order (assigning the full
    mass of the particle to the particle's radius). The density for every radius
    is then computed by dividing the cumulative mass profile by the volume of a
    sphere with that radius.

    The SO radius is defined as the radius at which the density profile dips
    below the given reference density. Unfortunately, real density profiles
    are noisy and can sometimes fluctuate around the threshold. We therefore
    define the SO radius as the first radius for which this happens, at machine
    precision.
    If no particles are below the threshold, then we raise an error and force an
    increase of the search radius.
    If all particles are below the threshold, we assume that the cumulative
    mass profile of the halo is linear out to the radius of the first particle,
    and then use the corresponding density profile to find the intersection.
    In all other cases, we find the actual SO radius by assuming a linear
    cumulative mass profile in the bin where the density dips below the
    threshold, and intersecting the corresponding density profile with the
    threshold. This approach requires a root finding algorithm and does not
    yield exactly the same result as linear interpolation in r-log(rho) space
    for the same bin (which is for example used by VELOCIraptor). It however
    guarantees that the SO mass we find is contained within the intersecting
    bin, which would otherwise not necessarily be true (especially if the
    intersecting bin is relatively wide). We could also interpolate both the
    radius and mass, but then the mean density of the SO would not necessarily
    match the target density, which is also weird.
    """

    # Compute a mask that marks particles above the threshold. We do this
    # exactly once.
    above_mask = density > reference_density
    if above_mask[0]:
        # Get the complementary mask of particles below the threshold.
        # By using the complementary, we avoid any ambiguity about '>' vs '<='
        below_mask = ~above_mask
        # Find smallest radius where the density is below the threshold
        i = np.argmax(below_mask)
        if i == 0:
            # There are no particles below the threshold
            # We need to increase the search radius
            if ordered_radius[-1] > 20.0 * unyt.Mpc:
                raise RuntimeError(
                    "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                )
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")
    else:
        # all non-zero radius particles are below the threshold
        # we linearly interpolate the mass from 0 to the particle radius
        # and determine the radius at which this interpolation matches the
        # target density
        # This is simply the solution of
        #    4*pi/3*r^3*rho = M[0]/r[0]*r
        # Note that if masses are allowed to be negative, the first cumulative
        # mass value could be negative. We make sure to avoid this problem
        ipos = 0
        while ipos < len(cumulative_mass) and cumulative_mass[ipos] < 0.0:
            ipos += 1
        if ipos == len(cumulative_mass):
            raise RuntimeError("Should never happen!")
        SO_r = np.sqrt(
            0.75
            * cumulative_mass[ipos]
            / (np.pi * ordered_radius[ipos] * reference_density)
        )
        SO_mass = cumulative_mass[ipos] * SO_r / ordered_radius[ipos]
        return SO_r, SO_mass, 4.0 * np.pi / 3.0 * SO_r**3

    # We now have the intersecting interval. Get the limits.
    r1 = ordered_radius[i - 1]
    r2 = ordered_radius[i]
    M1 = cumulative_mass[i - 1]
    M2 = cumulative_mass[i]
    # deal with the pathological case where r1==r2
    # we also need an interval where the density intersects
    while r1 == r2 or (above_mask[i - 1] == above_mask[i]):
        i += 1
        # if we run out of 'i', we need to increase the search radius
        if i >= len(density):
            if ordered_radius[-1] > 20.0 * unyt.Mpc:
                raise RuntimeError(
                    "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                )
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")
        # take the next interval
        r1 = r2
        r2 = ordered_radius[i]
        M1 = M2
        M2 = cumulative_mass[i]

    # compute the dimensionless quantities that enter the intersection equation
    # remember, we are simply solving
    #  4*pi/3*r^3*rho = M1 + (M2-M1)/(r2-r1)*(r-r1)
    rho_dim = reference_density * r1**3 / M1
    slope_dim = (M2 - M1) / (r2 - r1) * (r1 / M1)
    SO_r = r1 * brentq(
        cumulative_mass_intersection, 1.0, r2 / r1, args=(rho_dim, slope_dim)
    )

    SO_volume = 4.0 / 3.0 * np.pi * SO_r**3
    # compute the SO mass by requiring that the mean density in the SO is the
    # target density
    SO_mass = SO_volume * reference_density

    return SO_r, SO_mass, SO_volume


class SOParticleData:
    def __init__(
        self,
        input_halo,
        data,
        types_present,
        recently_heated_gas_filter,
        nu_density,
        observer_position,
    ):
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.nu_density = nu_density
        self.observer_position = observer_position
        self.compute_basics()

    def compute_basics(self):
        self.centre = self.input_halo["cofp"]
        self.index = self.input_halo["index"]

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        groupnr = []
        for ptype in self.types_present:
            if ptype == "PartType6":
                # add neutrinos separately, since we need to treat them
                # differently
                continue
            mass.append(self.data[ptype][mass_dataset(ptype)])
            pos = self.data[ptype]["Coordinates"] - self.centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos**2, axis=1))
            radius.append(r)
            velocity.append(self.data[ptype]["Velocities"])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)
            groupnr.append(self.data[ptype]["GroupNr_bound"])
        self.mass = unyt.array.uconcatenate(mass)
        self.radius = unyt.array.uconcatenate(radius)
        self.position = unyt.array.uconcatenate(position)
        self.velocity = unyt.array.uconcatenate(velocity)
        self.types = np.concatenate(types)
        self.groupnr = unyt.array.uconcatenate(groupnr)

        # figure out which particles in the list are bound to a halo that is not the
        # central halo
        self.is_bound_to_satellite = (self.groupnr >= 0) & (self.groupnr != self.index)

    def compute_SO_radius_and_mass(self, reference_density, physical_radius):
        # add neutrinos
        if "PartType6" in self.data:
            numass = (
                self.data["PartType6"]["Masses"] * self.data["PartType6"]["Weights"]
            )
            pos = self.data["PartType6"]["Coordinates"] - self.centre[None, :]
            nur = np.sqrt(np.sum(pos**2, axis=1))
            all_mass = unyt.array.uconcatenate([self.mass, numass])
            all_r = unyt.array.uconcatenate([self.radius, nur])
        else:
            all_mass = self.mass
            all_r = self.radius

        # Sort by radius
        order = np.argsort(all_r)
        ordered_radius = all_r[order]
        cumulative_mass = np.cumsum(all_mass[order], dtype=np.float64).astype(
            self.mass.dtype
        )
        # add mean neutrino mass
        cumulative_mass += self.nu_density * 4.0 / 3.0 * np.pi * ordered_radius**3

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        # Note that because of the definition of the centre of potential, the first
        # particle *should* be at r=0. We need to manually exclude it, in case round
        # off error places it at a very small non-zero radius.
        nskip = max(1, np.argmax(ordered_radius > 0.0 * ordered_radius.units))
        ordered_radius = ordered_radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(ordered_radius)
        density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius**3)

        # Check if we ever reach the density threshold
        if reference_density > 0:
            if nr_parts > 0:
                try:
                    self.SO_r, self.SO_mass, self.SO_volume = find_SO_radius_and_mass(
                        ordered_radius,
                        density,
                        cumulative_mass,
                        reference_density,
                    )
                except ReadRadiusTooSmallError:
                    raise ReadRadiusTooSmallError("SO radius multiple was too small!")
            else:
                self.SO_volume = 0 * ordered_radius.units**3
        elif physical_radius > 0:
            self.SO_r = physical_radius
            self.SO_volume = 4.0 * np.pi / 3.0 * self.SO_r**3
            if nr_parts > 0:
                # find the enclosed mass using interpolation
                outside_radius = ordered_radius > self.SO_r
                if not np.any(outside_radius):
                    # all particles are within the radius, we cannot interpolate
                    self.SO_mass = cumulative_mass[-1]
                else:
                    i = np.argmax(outside_radius)
                    if i == 0:
                        # we only have particles in the centre, so we cannot interpolate
                        self.SO_mass = cumulative_mass[i]
                    else:
                        r1 = ordered_radius[i - 1]
                        r2 = ordered_radius[i]
                        M1 = cumulative_mass[i - 1]
                        M2 = cumulative_mass[i]
                        self.SO_mass = M1 + (self.SO_r - r1) / (r2 - r1) * (M2 - M1)

        # check if we were successful. We only compute SO properties if we
        # have both a radius and mass (the mass criterion covers the case where
        # the radius is set to a physical size but we have no mass nonetheless)
        SO_exists = self.SO_r > 0 and self.SO_mass > 0

        if SO_exists:
            self.gas_selection = self.radius[self.types == "PartType0"] < self.SO_r
            self.dm_selection = self.radius[self.types == "PartType1"] < self.SO_r
            self.star_selection = self.radius[self.types == "PartType4"] < self.SO_r
            self.bh_selection = self.radius[self.types == "PartType5"] < self.SO_r

            self.all_selection = self.radius < self.SO_r
            self.mass = self.mass[self.all_selection]
            self.radius = self.radius[self.all_selection]
            self.position = self.position[self.all_selection]
            self.velocity = self.velocity[self.all_selection]
            self.types = self.types[self.all_selection]
            self.is_bound_to_satellite = self.is_bound_to_satellite[self.all_selection]

        return SO_exists

    @property
    def r(self):
        return self.SO_r

    @property
    def Mtot(self):
        return self.SO_mass

    @lazy_property
    def Mtotpart(self):
        return self.mass.sum()

    @lazy_property
    def mass_fraction(self):
        # note that we cannot divide by mSO here, since that was based on an interpolation
        return self.mass / self.Mtotpart

    @lazy_property
    def com(self):
        return (self.mass_fraction[:, None] * self.position).sum(axis=0) + self.centre

    @lazy_property
    def vcom(self):
        return (self.mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def spin_parameter(self):
        if self.Mtotpart == 0:
            return None
        _, vmax = get_vmax(self.mass, self.radius)
        if vmax > 0:
            vrel = self.velocity - self.vcom[None, :]
            Ltot = unyt.array.unorm(
                (self.mass[:, None] * unyt.array.ucross(self.position, vrel)).sum(
                    axis=0
                )
            )
            return Ltot / (np.sqrt(2.0) * self.Mtotpart * self.SO_r * vmax)
        return None

    @lazy_property
    def TotalAxisLengths(self):
        if self.Mtotpart == 0:
            return None
        return get_axis_lengths(self.mass, self.position)

    @lazy_property
    def Mfrac_satellites(self):
        return self.mass[self.is_bound_to_satellite].sum() / self.SO_mass

    @lazy_property
    def gas_masses(self):
        return self.mass[self.types == "PartType0"]

    @lazy_property
    def gas_pos(self):
        return self.position[self.types == "PartType0"]

    @lazy_property
    def gas_vel(self):
        return self.velocity[self.types == "PartType0"]

    @lazy_property
    def Mgas(self):
        return self.gas_masses.sum()

    @lazy_property
    def gas_mass_fraction(self):
        if self.Mgas == 0:
            return None
        return self.gas_masses / self.Mgas

    @lazy_property
    def com_gas(self):
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.gas_pos).sum(
            axis=0
        ) + self.centre

    @lazy_property
    def vcom_gas(self):
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.gas_vel).sum(axis=0)

    def compute_Lgas_props(self):
        (
            self.internal_Lgas,
            _,
            self.internal_Mcountrot_gas,
        ) = get_angular_momentum_and_kappa_corot(
            self.gas_masses,
            self.gas_pos,
            self.gas_vel,
            ref_velocity=self.vcom_gas,
            do_counterrot_mass=True,
        )

    @lazy_property
    def Lgas(self):
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Lgas"):
            self.compute_Lgas_props()
        return self.internal_Lgas

    @lazy_property
    def DtoTgas(self):
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_gas"):
            self.compute_Lgas_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_gas / self.Mgas

    @lazy_property
    def GasAxisLengths(self):
        if self.Mgas == 0:
            return None
        return get_axis_lengths(self.gas_masses, self.gas_pos)

    @lazy_property
    def dm_masses(self):
        return self.mass[self.types == "PartType1"]

    @lazy_property
    def dm_pos(self):
        return self.position[self.types == "PartType1"]

    @lazy_property
    def dm_vel(self):
        return self.velocity[self.types == "PartType1"]

    @lazy_property
    def Mdm(self):
        return self.dm_masses.sum()

    @lazy_property
    def dm_mass_fraction(self):
        if self.Mdm == 0:
            return None
        return self.dm_masses / self.Mdm

    @lazy_property
    def vcom_dm(self):
        if self.Mdm == 0:
            return None
        return (self.dm_mass_fraction[:, None] * self.dm_vel).sum(axis=0)

    @lazy_property
    def Ldm(self):
        if self.Mdm == 0:
            return None
        return get_angular_momentum(
            self.dm_masses, self.dm_pos, self.dm_vel, ref_velocity=self.vcom_dm
        )

    @lazy_property
    def DMAxisLengths(self):
        if self.Mdm == 0:
            return None
        return get_axis_lengths(self.dm_masses, self.dm_pos)

    @lazy_property
    def star_masses(self):
        return self.mass[self.types == "PartType4"]

    @lazy_property
    def star_pos(self):
        return self.position[self.types == "PartType4"]

    @lazy_property
    def star_vel(self):
        return self.velocity[self.types == "PartType4"]

    @lazy_property
    def Mstar(self):
        return self.star_masses.sum()

    @lazy_property
    def star_mass_fraction(self):
        if self.Mstar == 0:
            return None
        return self.star_masses / self.Mstar

    @lazy_property
    def com_star(self):
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.star_pos).sum(
            axis=0
        ) + self.centre

    @lazy_property
    def vcom_star(self):
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.star_vel).sum(axis=0)

    def compute_Lstar_props(self):
        (
            self.internal_Lstar,
            _,
            self.internal_Mcountrot_star,
        ) = get_angular_momentum_and_kappa_corot(
            self.star_masses,
            self.star_pos,
            self.star_vel,
            ref_velocity=self.vcom_star,
            do_counterrot_mass=True,
        )

    @lazy_property
    def Lstar(self):
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Lstar"):
            self.compute_Lstar_props()
        return self.internal_Lstar

    @lazy_property
    def DtoTstar(self):
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_star"):
            self.compute_Lstar_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_star / self.Mstar

    @lazy_property
    def StellarAxisLengths(self):
        if self.Mstar == 0:
            return None
        return get_axis_lengths(self.star_masses, self.star_pos)

    @lazy_property
    def baryon_masses(self):
        return self.mass[(self.types == "PartType0") | (self.types == "PartType4")]

    @lazy_property
    def baryon_pos(self):
        return self.position[(self.types == "PartType0") | (self.types == "PartType4")]

    @lazy_property
    def baryon_vel(self):
        return self.velocity[(self.types == "PartType0") | (self.types == "PartType4")]

    @lazy_property
    def Mbaryons(self):
        return self.baryon_masses.sum()

    @lazy_property
    def baryon_mass_fraction(self):
        if self.Mbaryons == 0:
            return None
        return self.baryon_masses / self.Mbaryons

    @lazy_property
    def baryon_vcom(self):
        if self.Mbaryons == 0:
            return None
        return (self.baryon_mass_fraction[:, None] * self.baryon_vel).sum(axis=0)

    @lazy_property
    def Lbaryons(self):
        if self.Mbaryons == 0:
            return None
        baryon_relvel = self.baryon_vel - self.baryon_vcom[None, :]
        return (
            self.baryon_masses[:, None]
            * unyt.array.ucross(self.baryon_pos, baryon_relvel)
        ).sum(axis=0)

    @lazy_property
    def BaryonAxisLengths(self):
        if self.Mbaryons == 0:
            return None
        return get_axis_lengths(self.baryon_masses, self.baryon_pos)

    @lazy_property
    def Mbh_dynamical(self):
        return self.mass[self.types == "PartType5"].sum()

    @lazy_property
    def Ngas(self):
        return self.gas_selection.sum()

    @lazy_property
    def gas_metal_masses(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.data["PartType0"]["MetalMassFractions"][self.gas_selection]
        )

    @lazy_property
    def gasmetalfrac(self):
        if self.Ngas == 0:
            return None
        return self.gas_metal_masses.sum() / self.Mgas

    @lazy_property
    def gasOfrac(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.data["PartType0"]["SmoothedElementMassFractions"][
                self.gas_selection
            ][:, indexO]
        ).sum() / self.Mgas

    @lazy_property
    def gasFefrac(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.data["PartType0"]["SmoothedElementMassFractions"][
                self.gas_selection
            ][:, indexFe]
        ).sum() / self.Mgas

    @lazy_property
    def gas_temperatures(self):
        if self.Ngas == 0:
            return None
        return self.data["PartType0"]["Temperatures"][self.gas_selection]

    @lazy_property
    def Tgas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_temperatures * (self.gas_masses / self.Mgas)).sum()

    @lazy_property
    def gas_no_cool(self):
        if self.Ngas == 0:
            return None
        return self.gas_temperatures > 1.0e5 * unyt.K

    @lazy_property
    def Mhotgas(self):
        if self.Ngas == 0:
            return None
        return self.gas_masses[self.gas_no_cool].sum()

    @lazy_property
    def Tgas_no_cool(self):
        if self.Ngas == 0:
            return None
        if np.any(self.gas_no_cool):
            return (
                self.gas_temperatures[self.gas_no_cool]
                * self.gas_masses[self.gas_no_cool]
            ).sum() / self.Mhotgas

    @lazy_property
    def gas_SFR(self):
        if self.Ngas == 0:
            return None
        SFR = self.data["PartType0"]["StarFormationRates"][self.gas_selection]
        is_SFR = SFR > 0.0
        SFR[~is_SFR] = 0.0
        return SFR

    @lazy_property
    def SFR(self):
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def Mgas_SF(self):
        if self.Ngas == 0:
            return None
        return self.gas_masses[self.gas_SFR > 0.0].sum()

    @lazy_property
    def gasmetalfrac_SF(self):
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_metal_masses[self.gas_SFR > 0.0].sum() / self.Mgas_SF

    @lazy_property
    def gas_xraylum(self):
        if self.Ngas == 0:
            return None
        return self.data["PartType0"]["XrayLuminosities"][self.gas_selection]

    @lazy_property
    def Xraylum(self):
        if self.Ngas == 0:
            return None
        return self.gas_xraylum.sum()

    @lazy_property
    def gas_xrayphlum(self):
        if self.Ngas == 0:
            return None
        return self.data["PartType0"]["XrayPhotonLuminosities"][self.gas_selection]

    @lazy_property
    def Xrayphlum(self):
        if self.Ngas == 0:
            return None
        return self.gas_xrayphlum.sum()

    @lazy_property
    def gas_compY(self):
        if self.Ngas == 0:
            return None
        return self.data["PartType0"]["ComptonYParameters"][self.gas_selection]

    @lazy_property
    def compY_unit(self):
        # We need to manually convert the units for compY to avoid numerical
        # overflow inside unyt
        if self.Ngas == 0:
            return None
        unit = 1.0 * self.gas_compY.units
        new_unit = unit.to(PropertyTable.full_property_list["compY"][3])
        return new_unit

    @lazy_property
    def compY(self):
        if self.Ngas == 0:
            return None
        return self.gas_compY.sum().value * self.compY_unit

    @lazy_property
    def gas_no_agn(self):
        if self.Ngas == 0:
            return None
        last_agn_gas = self.data["PartType0"]["LastAGNFeedbackScaleFactors"][
            self.gas_selection
        ]
        return ~self.recently_heated_gas_filter.is_recently_heated(
            last_agn_gas, self.gas_temperatures
        )

    @lazy_property
    def Xraylum_no_agn(self):
        if self.Ngas == 0:
            return None
        return self.gas_xraylum[self.gas_no_agn].sum()

    @lazy_property
    def Xrayphlum_no_agn(self):
        if self.Ngas == 0:
            return None
        return self.gas_xrayphlum[self.gas_no_agn].sum()

    @lazy_property
    def compY_no_agn(self):
        if self.Ngas == 0:
            return None
        return self.gas_compY[self.gas_no_agn].sum().value * self.compY_unit

    @lazy_property
    def Tgas_no_agn(self):
        if self.Ngas == 0:
            return None
        mass_gas_no_agn = self.gas_masses[self.gas_no_agn]
        Mgas_no_agn = mass_gas_no_agn.sum()
        if Mgas_no_agn > 0:
            return (
                (mass_gas_no_agn / Mgas_no_agn) * self.gas_temperatures[self.gas_no_agn]
            ).sum()

    @lazy_property
    def gas_no_cool_no_agn(self):
        if self.Ngas == 0:
            return None
        return self.gas_no_cool & self.gas_no_agn

    @lazy_property
    def Tgas_no_cool_no_agn(self):
        if self.Ngas == 0:
            return None
        mass_gas_no_cool_no_agn = self.gas_masses[self.gas_no_cool_no_agn]
        Mgas_no_cool_no_agn = mass_gas_no_cool_no_agn.sum()
        if Mgas_no_cool_no_agn > 0:
            return (
                (mass_gas_no_cool_no_agn / Mgas_no_cool_no_agn)
                * self.gas_temperatures[self.gas_no_cool_no_agn]
            ).sum()

    @lazy_property
    def Ekin_gas(self):
        if self.Ngas == 0:
            return None
        # below we need to force conversion to np.float64 before summing up particles
        # to avoid overflow
        ekin_gas = self.gas_masses * ((self.gas_vel - self.vcom_gas[None, :]) ** 2).sum(
            axis=1
        )
        ekin_gas = unyt.unyt_array(
            ekin_gas.value, dtype=np.float64, units=ekin_gas.units
        )
        return 0.5 * ekin_gas.sum()

    @lazy_property
    def gas_densities(self):
        if self.Ngas == 0:
            return None
        return self.data["PartType0"]["Densities"][self.gas_selection]

    @lazy_property
    def Etherm_gas(self):
        if self.Ngas == 0:
            return None
        etherm_gas = (
            1.5
            * self.gas_masses
            * self.data["PartType0"]["Pressures"][self.gas_selection]
            / self.gas_densities
        )
        etherm_gas = unyt.unyt_array(
            etherm_gas.value, dtype=np.float64, units=etherm_gas.units
        )
        return etherm_gas.sum()

    @lazy_property
    def DopplerB(self):
        if self.Ngas == 0:
            return None
        ne = self.data["PartType0"]["ElectronNumberDensities"][self.gas_selection]
        # note: the positions where relative to the centre, so we have
        # to make them absolute again before subtracting the observer
        # position
        relpos = self.gas_pos + self.centre[None, :] - self.observer_position[None, :]
        distance = np.sqrt((relpos**2).sum(axis=1))
        # we need to exclude particles at zero distance
        # (we assume those have no relative velocity)
        vr = unyt.unyt_array(
            np.zeros(self.gas_vel.shape[0]),
            dtype=self.gas_vel.dtype,
            units=self.gas_vel.units,
        )
        has_distance = distance > 0.0
        vr[has_distance] = (
            self.gas_vel[has_distance, 0] * relpos[has_distance, 0]
            + self.gas_vel[has_distance, 1] * relpos[has_distance, 1]
            + self.gas_vel[has_distance, 2] * relpos[has_distance, 2]
        ) / distance[has_distance]
        fac = unyt.sigma_thompson / unyt.c
        volumes = self.gas_masses / self.gas_densities
        area = np.pi * self.SO_r**2
        return (fac * ne * vr * (volumes / area)).sum()

    @lazy_property
    def Ndm(self):
        return self.dm_selection.sum()

    @lazy_property
    def Nstar(self):
        return self.star_selection.sum()

    @lazy_property
    def Mstar_init(self):
        if self.Nstar == 0:
            return None
        return self.data["PartType4"]["InitialMasses"][self.star_selection].sum()

    @lazy_property
    def starmetalfrac(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.data["PartType4"]["MetalMassFractions"][self.star_selection]
        ).sum() / self.Mstar

    @lazy_property
    def starOfrac(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.data["PartType4"]["SmoothedElementMassFractions"][
                self.star_selection
            ][:, indexO]
        ).sum() / self.Mstar

    @lazy_property
    def starFefrac(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.data["PartType4"]["SmoothedElementMassFractions"][
                self.star_selection
            ][:, indexFe]
        ).sum() / self.Mstar

    @lazy_property
    def StellarLuminosity(self):
        if self.Nstar == 0:
            return None
        return self.data["PartType4"]["Luminosities"][self.star_selection].sum(axis=0)

    @lazy_property
    def Ekin_star(self):
        if self.Nstar == 0:
            return None
        # below we need to force conversion to np.float64 before summing up particles
        # to avoid overflow
        ekin_star = self.star_masses * (
            (self.star_vel - self.vcom_star[None, :]) ** 2
        ).sum(axis=1)
        ekin_star = unyt.unyt_array(
            ekin_star.value, dtype=np.float64, units=ekin_star.units
        )
        return 0.5 * ekin_star.sum()

    @lazy_property
    def Nbh(self):
        return self.bh_selection.sum()

    @lazy_property
    def Mbh_subgrid(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["SubgridMasses"][self.bh_selection].sum()

    @lazy_property
    def agn_eventa(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["LastAGNFeedbackScaleFactors"][self.bh_selection]

    @lazy_property
    def BHlasteventa(self):
        if self.Nbh == 0:
            return None
        return np.max(self.agn_eventa)

    @lazy_property
    def iBHmax(self):
        if self.Nbh == 0:
            return None
        return np.argmax(self.data["PartType5"]["SubgridMasses"][self.bh_selection])

    @lazy_property
    def BHmaxM(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["SubgridMasses"][self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxID(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["ParticleIDs"][self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxpos(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["Coordinates"][self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxvel(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["Velocities"][self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxAR(self):
        if self.Nbh == 0:
            return None
        return self.data["PartType5"]["AccretionRates"][self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxlasteventa(self):
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def Nnu(self):
        if "PartType6" in self.data:
            pos = self.data["PartType6"]["Coordinates"] - self.centre[None, :]
            nur = np.sqrt(np.sum(pos**2, axis=1))
            self.nu_selection = nur < self.SO_r
            return self.nu_selection.sum()
        else:
            return unyt.unyt_array(0, dtype=np.uint32, units="dimensionless")

    @lazy_property
    def Mnu(self):
        if self.Nnu == 0:
            return None
        return self.data["PartType6"]["Masses"][self.nu_selection].sum()

    @lazy_property
    def MnuNS(self):
        if self.Nnu == 0:
            return None
        return (
            self.data["PartType6"]["Masses"][self.nu_selection]
            * self.data["PartType6"]["Weights"][self.nu_selection]
        ).sum() + self.nu_density * self.SO_volume


class SOProperties(HaloProperty):

    # get the properties we want from the table
    property_list = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in [
            "r",
            "Mtot",
            "Ngas",
            "Ndm",
            "Nstar",
            "Nbh",
            "Nnu",
            "com",
            "vcom",
            "Mfrac_satellites",
            "Mgas",
            "Lgas",
            "com_gas",
            "vcom_gas",
            #            "veldisp_matrix_gas",
            "gasmetalfrac",
            "Mhotgas",
            "Tgas",
            "Tgas_no_cool",
            "Tgas_no_agn",
            "Tgas_no_cool_no_agn",
            "Xraylum",
            "Xrayphlum",
            "compY",
            "Xraylum_no_agn",
            "Xrayphlum_no_agn",
            "compY_no_agn",
            "Ekin_gas",
            "Etherm_gas",
            "Mdm",
            "Ldm",
            #            "veldisp_matrix_dm",
            "Mstar",
            "com_star",
            "vcom_star",
            #            "veldisp_matrix_star",
            "Lstar",
            "Mstar_init",
            "starmetalfrac",
            "StellarLuminosity",
            "Ekin_star",
            "Lbaryons",
            "Mbh_dynamical",
            "Mbh_subgrid",
            "BHlasteventa",
            "BHmaxM",
            "BHmaxID",
            "BHmaxpos",
            "BHmaxvel",
            "BHmaxAR",
            "BHmaxlasteventa",
            "MnuNS",
            "Mnu",
            "spin_parameter",
            "SFR",
            "TotalAxisLengths",
            "GasAxisLengths",
            "DMAxisLengths",
            "StellarAxisLengths",
            "BaryonAxisLengths",
            "DopplerB",
            "gasOfrac",
            "gasFefrac",
            "DtoTgas",
            "DtoTstar",
            "starOfrac",
            "starFefrac",
            "gasmetalfrac_SF",
        ]
    ]

    def __init__(
        self,
        cellgrid,
        parameters,
        recently_heated_gas_filter,
        category_filter,
        SOval,
        type="mean",
    ):
        super().__init__(cellgrid)

        self.property_mask = parameters.get_property_mask(
            "SOProperties", [prop[1] for prop in self.property_list]
        )

        if not type in ["mean", "crit", "physical", "BN98"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        self.filter = recently_heated_gas_filter
        self.category_filter = category_filter

        self.observer_position = cellgrid.observer_position

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
            / cellgrid.a**3
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

        # Arrays which must be read in for this calculation.
        # Note that if there are no particles of a given type in the
        # snapshot, that type will not be read in and will not have
        # an entry in the data argument to calculate(), below.
        # (E.g. gas, star or BH particles in DMO runs)
        self.particle_properties = {
            "PartType0": [
                "Coordinates",
                "GroupNr_bound",
                "Masses",
                "Velocities",
            ],
            "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
            "PartType4": [
                "Coordinates",
                "GroupNr_bound",
                "Masses",
                "Velocities",
            ],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "GroupNr_bound",
                "Velocities",
            ],
            "PartType6": ["Coordinates", "Masses", "Weights"],
        }
        for prop in self.property_list:
            outputname = prop[1]
            if not self.property_mask[outputname]:
                continue
            is_dmo = prop[8]
            if self.category_filter.dmo and not is_dmo:
                continue
            partprops = prop[9]
            for partprop in partprops:
                pgroup, dset = partprop.split("/")
                if not pgroup in self.particle_properties:
                    self.particle_properties[pgroup] = []
                if not dset in self.particle_properties[pgroup]:
                    self.particle_properties[pgroup].append(dset)

    def calculate(self, input_halo, search_radius, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius in which we have all particles
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        registry = input_halo["cofp"].units.registry

        do_calculation = self.category_filter.get_filters(halo_result)

        SO = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for prop in self.property_list:
            outputname = prop[1]
            # skip properties that are masked
            if not self.property_mask[outputname]:
                continue
            # skip non-DMO properties in DMO run mode
            is_dmo = prop[8]
            if do_calculation["DMO"] and not is_dmo:
                continue
            name = prop[0]
            shape = prop[2]
            dtype = prop[3]
            unit = prop[4]
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            SO[name] = unyt.unyt_array(val, dtype=dtype, units=unit, registry=registry)

        # SOs only exist for central galaxies
        if input_halo["Structuretype"] == 10:

            types_present = [type for type in self.particle_properties if type in data]

            part_props = SOParticleData(
                input_halo,
                data,
                types_present,
                self.filter,
                self.nu_density,
                self.observer_position,
            )

            # we need to make sure the physical radius uses the correct unit
            # registry, since we use it to set the SO radius in some cases
            physical_radius = unyt.unyt_quantity(
                self.physical_radius_mpc, units=unyt.Mpc, registry=registry
            )
            try:
                SO_exists = part_props.compute_SO_radius_and_mass(
                    self.reference_density, physical_radius
                )
            except ReadRadiusTooSmallError:
                raise ReadRadiusTooSmallError(
                    f"Need more particles to determine SO radius and mass!"
                )

            if SO_exists:

                for prop in self.property_list:
                    outputname = prop[1]
                    # skip properties that are masked
                    if not self.property_mask[outputname]:
                        continue
                    # skip non-DMO properties in DMO run mode
                    is_dmo = prop[8]
                    if do_calculation["DMO"] and not is_dmo:
                        continue
                    name = prop[0]
                    dtype = prop[3]
                    unit = prop[4]
                    category = prop[6]
                    if do_calculation[category]:
                        val = getattr(part_props, name)
                        if val is not None:
                            if unit == "dimensionless":
                                SO[name] = unyt.unyt_array(
                                    val.astype(dtype),
                                    dtype=dtype,
                                    units=unit,
                                    registry=registry,
                                )
                            else:
                                SO[name] += val

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for prop in self.property_list:
            outputname = prop[1]
            # skip properties that are masked
            if not self.property_mask[outputname]:
                continue
            # skip non-DMO properties in DMO run mode
            is_dmo = prop[8]
            if do_calculation["DMO"] and not is_dmo:
                continue
            name = prop[0]
            description = prop[5]
            halo_result.update(
                {
                    f"SO/{self.SO_name}/{outputname}": (
                        SO[name],
                        description.format(label=self.label),
                    )
                }
            )

        return


class RadiusMultipleSOProperties(SOProperties):

    # since the halo_result dictionary contains the name of the dataset as it
    # appears in the output, we have to get that name from the property table
    # to access the radius
    radius_name = PropertyTable.full_property_list["r"][0]

    def __init__(
        self,
        cellgrid,
        parameters,
        recently_heated_gas_filter,
        category_filter,
        SOval,
        multiple,
        type="mean",
    ):
        if not type in ["mean", "crit"]:
            raise AttributeError(
                "SOs with a radius that is a multiple of another SO radius are only allowed for type mean or crit!"
            )

        # initialise the SOProperties object using a conservative physical radius estimate
        super().__init__(
            cellgrid,
            parameters,
            recently_heated_gas_filter,
            category_filter,
            3000.0,
            "physical",
        )

        # overwrite the name, SO_name and label
        self.SO_name = f"{multiple:.0f}xR_{SOval:.0f}_{type}"
        self.label = f"with a radius that is {self.SO_name}"
        self.name = f"SO_{self.SO_name}"

        self.requested_type = type
        self.requested_SOval = SOval
        self.multiple = multiple

    def calculate(self, input_halo, search_radius, data, halo_result):

        # find the actual physical radius we want
        key = f"SO/{self.requested_SOval:.0f}_{self.requested_type}/{self.radius_name}"
        if not key in halo_result:
            raise RuntimeError(
                f"Trying to obtain {key}, but the corresponding SO radius has not been calculated!"
            )
        self.physical_radius_mpc = self.multiple * (halo_result[key][0].to("Mpc").value)

        # Check that we read in a large enough radius
        if self.multiple * halo_result[key][0] > search_radius:
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")

        super().calculate(input_halo, search_radius, data, halo_result)
        return


def test_SO_properties():

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(4251)
    filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())
    cat_filter = CategoryFilter()
    parameters = ParameterFile()
    parameters.get_halo_type_variations(
        "SOProperties",
        {
            "50_kpc": {"value": 50.0, "type": "physical"},
            "2500_mean": {"value": 2500.0, "type": "mean"},
            "2500_crit": {"value": 2500.0, "type": "crit"},
            "BN98": {"value": 0.0, "type": "BN98"},
            "5xR2500_mean": {"value": 2500.0, "type": "mean", "radius_multiple": 5.0},
        },
    )

    property_calculator_50kpc = SOProperties(
        dummy_halos.get_cell_grid(), parameters, filter, cat_filter, 50.0, "physical"
    )
    property_calculator_2500mean = SOProperties(
        dummy_halos.get_cell_grid(), parameters, filter, cat_filter, 2500.0, "mean"
    )
    property_calculator_2500crit = SOProperties(
        dummy_halos.get_cell_grid(), parameters, filter, cat_filter, 2500.0, "crit"
    )
    property_calculator_BN98 = SOProperties(
        dummy_halos.get_cell_grid(), parameters, filter, cat_filter, 0.0, "BN98"
    )
    property_calculator_5x2500mean = RadiusMultipleSOProperties(
        dummy_halos.get_cell_grid(), parameters, filter, cat_filter, 2500.0, 5.0, "mean"
    )

    parameters.write_parameters("SO.used_values.yml")

    for i in range(100):
        (
            input_halo,
            data,
            rmax,
            Mtot,
            Npart,
            particle_numbers,
        ) = dummy_halos.get_random_halo([2, 10, 100, 1000, 10000], has_neutrinos=True)
        halo_result_template = {
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ngas'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType0"],
                    dtype=PropertyTable.full_property_list["Ngas"][2],
                    units="dimensionless",
                ),
                "Dummy Ngas for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ndm'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType1"],
                    dtype=PropertyTable.full_property_list["Ndm"][2],
                    units="dimensionless",
                ),
                "Dummy Ndm for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nstar'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType4"],
                    dtype=PropertyTable.full_property_list["Nstar"][2],
                    units="dimensionless",
                ),
                "Dummy Nstar for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nbh'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType5"],
                    dtype=PropertyTable.full_property_list["Nbh"][2],
                    units="dimensionless",
                ),
                "Dummy Nbh for filter",
            ),
        }
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax**3)

        # force the SO radius to be outside the search sphere and check that
        # we get a ReadRadiusTooSmallError
        property_calculator_2500mean.reference_density = 0.01 * rho_ref
        property_calculator_2500crit.reference_density = 0.01 * rho_ref
        property_calculator_BN98.reference_density = 0.01 * rho_ref
        for prop_calc in [
            property_calculator_2500mean,
            property_calculator_2500crit,
            property_calculator_BN98,
        ]:
            fail = False
            try:
                halo_result = dict(halo_result_template)
                prop_calc.calculate(input_halo, rmax, data, halo_result)
            except ReadRadiusTooSmallError:
                fail = True
            # 1 particle halos don't fail, since we always assume that the first
            # particle is at the centre of potential (which means we exclude it
            # in the SO calculation)
            # non-centrals don't fail, since we do not calculate any SO
            # properties and simply return zeros in this case
            assert (Npart == 1) or input_halo["Structuretype"] != 10 or fail

        # force the radius multiple to trip over not having computed the
        # required radius
        fail = False
        try:
            halo_result = dict(halo_result_template)
            property_calculator_5x2500mean.calculate(
                input_halo, rmax, data, halo_result
            )
        except RuntimeError:
            fail = True
        assert fail

        # force the radius multiple to trip over the search radius
        fail = False
        try:
            halo_result = dict(halo_result_template)
            halo_result.update(
                {
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}": (
                        0.1 * rmax,
                        "Dummy value.",
                    )
                }
            )
            property_calculator_5x2500mean.calculate(
                input_halo, 0.2 * rmax, data, halo_result
            )
        except ReadRadiusTooSmallError:
            fail = True
        assert fail

        # force the SO radius to be within the search sphere
        property_calculator_2500mean.reference_density = 2.0 * rho_ref
        property_calculator_2500crit.reference_density = 2.0 * rho_ref
        property_calculator_BN98.reference_density = 2.0 * rho_ref

        for SO_name, prop_calc in [
            ("50_kpc", property_calculator_50kpc),
            ("2500_mean", property_calculator_2500mean),
            ("2500_crit", property_calculator_2500crit),
            ("BN98", property_calculator_BN98),
            ("5xR_2500_mean", property_calculator_5x2500mean),
        ]:

            halo_result = dict(halo_result_template)
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (
                    0.1 * rmax,
                    "Dummy value to force correct behaviour",
                )
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list:
                outputname = prop[1]
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["SOProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["SOProperties"]["properties"]:
            single_property["SOProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)

        property_calculator_50kpc = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            filter,
            cat_filter,
            50.0,
            "physical",
        )
        property_calculator_2500mean = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            filter,
            cat_filter,
            2500.0,
            "mean",
        )
        property_calculator_2500crit = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            filter,
            cat_filter,
            2500.0,
            "crit",
        )
        property_calculator_BN98 = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            filter,
            cat_filter,
            0.0,
            "BN98",
        )
        property_calculator_5x2500mean = RadiusMultipleSOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            filter,
            cat_filter,
            2500.0,
            5.0,
            "mean",
        )

        halo_result_template = {
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ngas'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType0"],
                    dtype=PropertyTable.full_property_list["Ngas"][2],
                    units="dimensionless",
                ),
                "Dummy Ngas for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ndm'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType1"],
                    dtype=PropertyTable.full_property_list["Ndm"][2],
                    units="dimensionless",
                ),
                "Dummy Ndm for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nstar'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType4"],
                    dtype=PropertyTable.full_property_list["Nstar"][2],
                    units="dimensionless",
                ),
                "Dummy Nstar for filter",
            ),
            f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nbh'][0]}": (
                unyt.unyt_array(
                    particle_numbers["PartType5"],
                    dtype=PropertyTable.full_property_list["Nbh"][2],
                    units="dimensionless",
                ),
                "Dummy Nbh for filter",
            ),
        }
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax**3)

        # force the SO radius to be within the search sphere
        property_calculator_2500mean.reference_density = 2.0 * rho_ref
        property_calculator_2500crit.reference_density = 2.0 * rho_ref
        property_calculator_BN98.reference_density = 2.0 * rho_ref

        for SO_name, prop_calc in [
            ("50_kpc", property_calculator_50kpc),
            ("2500_mean", property_calculator_2500mean),
            ("2500_crit", property_calculator_2500crit),
            ("BN98", property_calculator_BN98),
            ("5xR_2500_mean", property_calculator_5x2500mean),
        ]:

            halo_result = dict(halo_result_template)
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (
                    0.1 * rmax,
                    "Dummy value to force correct behaviour",
                )
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list:
                outputname = prop[1]
                if not outputname == property:
                    continue
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone mode. Just run test_SO_properties().
    """
    print("Calling test_SO_properties()...")
    test_SO_properties()
    print("Test passed.")
