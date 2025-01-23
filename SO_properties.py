#!/bin/env python

"""
SO_properties.py

Halo properties within spherical overdensities. These include
all particles within a radius that is set by the halo density
profile and some threshold density.

Just like the other HaloProperty implementations, the calculation of the
properties is done lazily: only calculations that are actually needed are
performed. A fully documented explanation can be found in
aperture_properties.py.

Apart from the usual halo property calculations, this file also
contains the spherical overdensity radius calculation, which is
somewhat more involved than simply using a fixed aperture.

Contrary to the other halo types, spherical overdensities are only
calculated for central halos. SO properties are also only calculated if 
an SO radius could be determined.
"""

import time
import numpy as np
import unyt
from scipy.optimize import brentq

from halo_properties import HaloProperty, SearchRadiusTooSmallError
from kinematic_properties import (
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_inertia_tensor,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from property_table import PropertyTable
from dataset_names import mass_dataset
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile
from snapshot_datasets import SnapshotDatasets
from swift_cells import SWIFTCellGrid

from typing import Tuple, Dict, List
from numpy.typing import NDArray


def cumulative_mass_intersection(r: float, rho_dim: float, slope_dim: float) -> float:
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

    Parameters:
     - r: float
       Dimensionless radius variable.
     - rho_dim: float
       Dimensionless density variable.
     - slope_dim: float
       Dimensionless slope variable.

    Returns the value of the intersection equation for the given r (and using
    the given rho_dim and slope_dim boundary conditions).
    """
    return 4.0 * np.pi / 3.0 * rho_dim * r ** 3 - slope_dim * r + slope_dim - 1.0


def find_SO_radius_and_mass(
    ordered_radius: unyt.unyt_array,
    density: unyt.unyt_array,
    cumulative_mass: unyt.unyt_array,
    reference_density: unyt.unyt_quantity,
) -> Tuple[unyt.unyt_quantity, unyt.unyt_quantity, unyt.unyt_quantity]:
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

    Parameters:
     - ordered_radius: unyt.unyt_array
       Sorted radius of the particles.
     - density: unyt.unyt_array
       Density of the particles (also sorted on radius).
     - cumulative_mass: unyt.unyt_array
       Cumulative mass profile of the particles (also sorted on radius).
     - reference_density: unyt.unyt_quantity
       Target threshold density.

    Returns the radius, mass and volume of the sphere where the density
    reaches the target value.

    Throws a SearchRadiusTooSmallError if the intersection point is outside
    the range of the particles that were passed on.

    Throws a RuntimeError if the intersection point is outside the range and
    the current search radius is larger than 20 Mpc.
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
            raise SearchRadiusTooSmallError(
                "SO radius multiple estimate was too small!"
            )
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
        return SO_r, SO_mass, 4.0 * np.pi / 3.0 * SO_r ** 3

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
            raise SearchRadiusTooSmallError(
                "SO radius multiple estimate was too small!"
            )
        # take the next interval
        r1 = r2
        r2 = ordered_radius[i]
        M1 = M2
        M2 = cumulative_mass[i]

    # compute the dimensionless quantities that enter the intersection equation
    # remember, we are simply solving
    #  4*pi/3*r^3*rho = M1 + (M2-M1)/(r2-r1)*(r-r1)
    rho_dim = reference_density * r1 ** 3 / M1
    slope_dim = (M2 - M1) / (r2 - r1) * (r1 / M1)
    SO_r = r1 * brentq(
        cumulative_mass_intersection, 1.0, r2 / r1, args=(rho_dim, slope_dim)
    )

    SO_volume = 4.0 / 3.0 * np.pi * SO_r ** 3
    # compute the SO mass by requiring that the mean density in the SO is the
    # target density
    SO_mass = SO_volume * reference_density

    return SO_r, SO_mass, SO_volume


class SOParticleData:
    """
    Halo calculation class.

    All properties we want to compute are implemented as lazy methods of this
    class.

    Note that unlike other halo properties that use apertures, SO calculations
    only require a single mask, since they are always inclusive.
    That said, we still require a types==PartTypeX mask
    (see aperture_properties.py) to access some arrays that have been
    precomputed for all particles.

    Note that SOs are the only halo types that can include neutrino particles
    (these are never bound to a subhalo). They are however only included in
    the spherical overdensity radius calculation and in the calculation of
    neutrino specific properties (i.e. neutrino masses), and are not taken into
    account for other properties, like the total particle mass, velocity
    dispersion...
    """

    def __init__(
        self,
        input_halo: Dict,
        data: Dict,
        types_present: List[str],
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        observer_position: unyt.unyt_array,
        snapshot_datasets: SnapshotDatasets,
        core_excision_fraction: float,
        softening_of_parttype: unyt.unyt_array,
        virial_definition: bool,
        search_radius: unyt.unyt_quantity,
        cosmology: dict,
        boxsize: unyt.unyt_quantity,
    ):
        """
        Constructor.

        Parameters:
         - input_halo: Dict
           Dictionary containing properties of the halo read from the VR catalogue.
         - data: Dict
           Dictionary containing particle data.
         - types_present: List
           List of all particle types (e.g. 'PartType0') that are present in the data
           dictionary.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - observer_position: unyt.unyt_array
           Position of an observer, used to determine the observer direction for
           Doppler B calculations.
         - snapshot_datasets: SnapshotDatasets
           Object containing metadata about the datasets in the snapshot, like
           appropriate aliases and column names.
         - core_excision_fraction: float
           Ignore particles within a sphere of core_excision_fraction * SORadius
           when calculating CoreExcision properties
         - softening_of_parttype: unyt.unyt_array
           Softening length of each particle types
         - virial_definition: bool
           Whether to calculate the properties that are only valid for virial SO
           definitions
         - search_radius: unyt.unyt_quantity
           Current search radius. Particles are guaranteed to be included up to
           this radius.
         - cosmology: dict
           Cosmological parameters required for SO calculation
         - boxsize: unyt.unyt_quantity
           Boxsize for correcting periodic boundary conditions
        """
        self.input_halo = input_halo
        self.data = data
        self.has_neutrinos = "PartType6" in data
        self.types_present = types_present
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.observer_position = observer_position
        self.snapshot_datasets = snapshot_datasets
        self.core_excision_fraction = core_excision_fraction
        self.softening_of_parttype = softening_of_parttype
        self.virial_definition = virial_definition
        self.search_radius = search_radius
        self.cosmology = cosmology
        self.boxsize = boxsize
        self.compute_basics()

    def get_dataset(self, name: str) -> unyt.unyt_array:
        """
        Local wrapper for SnapshotDatasets.get_dataset().
        """
        return self.snapshot_datasets.get_dataset(name, self.data)

    def compute_basics(self):
        """
        Compute some properties that are always needed, regardless of which
        properties we actually want to compute.
        """
        self.centre = self.input_halo["cofp"]
        self.index = self.input_halo["index"]

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        groupnr = []
        fofid = []
        softening = []
        for ptype in self.types_present:
            if ptype == "PartType6":
                # add neutrinos separately, since we need to treat them
                # differently
                continue
            mass.append(self.get_dataset(f"{ptype}/{mass_dataset(ptype)}"))
            pos = self.get_dataset(f"{ptype}/Coordinates") - self.centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos ** 2, axis=1))
            radius.append(r)
            velocity.append(self.get_dataset(f"{ptype}/Velocities"))
            typearr = int(ptype[-1]) * np.ones(r.shape, dtype=np.int32)
            types.append(typearr)
            groupnr.append(self.get_dataset(f"{ptype}/GroupNr_bound"))
            fofid.append(self.get_dataset(f"{ptype}/FOFGroupIDs"))
            s = np.ones(r.shape, dtype=np.float64) * self.softening_of_parttype[ptype]
            softening.append(s)
        self.mass = np.concatenate(mass)
        self.radius = np.concatenate(radius)
        self.position = np.concatenate(position)
        self.velocity = np.concatenate(velocity)
        self.types = np.concatenate(types)
        self.groupnr = np.concatenate(groupnr)
        self.fofid = np.concatenate(fofid)
        self.softening = np.concatenate(softening)

    def compute_SO_radius_and_mass(
        self, reference_density: unyt.unyt_quantity, physical_radius: unyt.unyt_quantity
    ) -> bool:
        """
        Compute the SO radius from the density profile of the particles.

        Adds the contribution from neutrinos (if present) to the masses and
        radii. Sorts the particles by radius and computes the cumulative mass
        profile. Calls find_SO_radius_and_mass(), unless a radius multiple is
        used as aperture radius.

        Parameters:
         - reference_density: unyt.unyt_quantity
           Threshold density value that determines the SO radius.
         - physical_radius: unyt.unyt_quantity
           Physical radius that determines the SO radius in case a radius
           multiple is used (e.g. 5xR500_crit).

        Returns True if an SO radius was found, i.e. when both SO_radius and
        SO_mass are non-zero.

        Rethrows any SearchRadiusTooSmallError thrown by find_SO_radius_and_mass().
        """
        # add neutrinos
        if self.has_neutrinos:
            numass = self.get_dataset("PartType6/Masses") * self.get_dataset(
                "PartType6/Weights"
            )
            pos = self.get_dataset("PartType6/Coordinates") - self.centre[None, :]
            nur = np.sqrt(np.sum(pos ** 2, axis=1))
            self.nu_mass = numass
            self.nu_radius = nur
            self.nu_softening = (
                np.ones_like(nur) * self.softening_of_parttype["PartType6"]
            )
            all_mass = np.concatenate([self.mass, numass / unyt.dimensionless])
            all_r = np.concatenate([self.radius, nur])
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
        cumulative_mass += (
            self.cosmology["nu_density"] * 4.0 / 3.0 * np.pi * ordered_radius ** 3
        )
        # Determine FOF ID of object using the central non-neutrino particle
        non_neutrino_order = order[order < self.radius.shape[0]]
        fofid = self.fofid[non_neutrino_order[0]]

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        # Note that because of the definition of the centre of potential, the first
        # particle *should* be at r=0. We need to manually exclude it, in case round
        # off error places it at a very small non-zero radius.
        nskip = max(1, np.argmax(ordered_radius > 0.0 * ordered_radius.units))
        ordered_radius = ordered_radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(ordered_radius)
        density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius ** 3)

        # Check if we ever reach the density threshold
        if reference_density > 0:
            if nr_parts > 0:
                try:
                    self.SO_r, self.SO_mass, self.SO_volume = find_SO_radius_and_mass(
                        ordered_radius, density, cumulative_mass, reference_density
                    )
                except SearchRadiusTooSmallError:
                    raise SearchRadiusTooSmallError("SO radius multiple was too small!")
            else:
                self.SO_volume = 0 * ordered_radius.units ** 3
        elif physical_radius > 0:
            self.SO_r = physical_radius
            self.SO_volume = 4.0 * np.pi / 3.0 * self.SO_r ** 3
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

        # figure out which particles in the list are bound to a halo that is not the
        # central halo
        self.is_bound_to_satellite = (
            (self.groupnr >= 0) & (self.groupnr != self.index) & (self.fofid == fofid)
        )
        self.is_bound_to_external = (
            (self.groupnr >= 0) & (self.groupnr != self.index) & (self.fofid != fofid)
        )

        if SO_exists:
            # Estimate DMO mass fraction found at SO_r
            # This is used when computing concentration_dmo
            dm_r = self.radius[self.types == 1]
            dm_m = self.mass[self.types == 1]
            outside_radius = dm_r > self.SO_r
            self.dm_missed_mass = 0 * self.mass.units
            if np.any(outside_radius):
                inside_radius = np.logical_not(outside_radius)
                if np.any(inside_radius):
                    r1 = np.max(dm_r[inside_radius])
                    i = np.argmin(dm_r[outside_radius])
                    r2 = dm_r[outside_radius][i]
                    m2 = dm_m[outside_radius][i]
                    self.dm_missed_mass = m2 * (self.SO_r - r1) / (r2 - r1)

            # Removing particles outside SO radius
            self.all_selection = self.radius < self.SO_r
            self.gas_selection = self.radius[self.types == 0] < self.SO_r
            self.dm_selection = self.radius[self.types == 1] < self.SO_r
            self.star_selection = self.radius[self.types == 4] < self.SO_r
            self.bh_selection = self.radius[self.types == 5] < self.SO_r

            # Save particles outside SO radius for inertia tensor calculations
            self.surrounding_mass = self.mass[np.logical_not(self.all_selection)]
            self.surrounding_position = self.position[
                np.logical_not(self.all_selection)
            ]
            self.surrounding_types = self.types[np.logical_not(self.all_selection)]

            self.mass = self.mass[self.all_selection]
            self.radius = self.radius[self.all_selection]
            self.position = self.position[self.all_selection]
            self.velocity = self.velocity[self.all_selection]
            self.types = self.types[self.all_selection]
            self.is_bound_to_satellite = self.is_bound_to_satellite[self.all_selection]
            self.is_bound_to_external = self.is_bound_to_external[self.all_selection]
            self.softening = self.softening[self.all_selection]

            if self.has_neutrinos:
                self.nu_selection = self.nu_radius < self.SO_r
                self.nu_mass = self.nu_mass[self.nu_selection]
                self.nu_radius = self.nu_radius[self.nu_selection]
                self.nu_softening = self.nu_softening[self.nu_selection]

        return SO_exists

    @property
    def r(self) -> unyt.unyt_quantity:
        """
        SO radius.
        """
        return self.SO_r

    @property
    def Mtot(self) -> unyt.unyt_quantity:
        """
        SO mass. Unlike other halo types, this is not simply the sum of all the
        particle masses, but the extrapolated cumulative mass up to the SO
        radius (i.e. SO_mass = 4*pi/3 * SO_radius**3 * SO_density).
        """
        return self.SO_mass

    @lazy_property
    def Mtotpart(self) -> unyt.unyt_quantity:
        """
        Total particle mass of all particles in the SO radius.

        This is the equivalent of the total mass for other halo types.
        """
        return self.mass.sum()

    @lazy_property
    def mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of all particles.

        Used to avoid numerical overflow in calculations like
          com = (mass * position).sum() / Mtot
        by rewriting it as
          com = ((mass / Mtot) * position).sum()
              = (mass_fraction * position).sum()
        This is more accurate, since the mass fractions are numbers
        of the order of 1e-5 or so, while the masses themselves can be much
        larger, if expressed in the wrong units (and that is up to unyt).
        """
        # note that we cannot divide by mSO here, since that was based on an interpolation
        return self.mass / self.Mtotpart

    @lazy_property
    def com(self) -> unyt.unyt_array:
        """
        Centre of mass of all particles in the spherical overdensity.
        """
        return ((self.mass_fraction[:, None] * self.position).sum(
            axis=0
        ) + self.centre) % self.boxsize

    @lazy_property
    def vcom(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles in the spherical overdensity.
        """
        return (self.mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def R_vmax_soft(self) -> unyt.unyt_quantity:
        """
        Radius at which the maximum circular velocity of the halo is reached.
        Particles are set to have minimum radius equal to their softening length.

        This includes contributions from all particle types.
        """
        if self.Mtotpart == 0:
            return None
        if not hasattr(self, "vmax_soft"):
            soft_r = np.maximum(self.softening, self.radius)
            self.r_vmax_soft, self.vmax_soft = get_vmax(self.mass, soft_r)
        return self.r_vmax_soft

    @lazy_property
    def Vmax_soft(self):
        """
        Maximum circular velocity of the halo.
        Particles are set to have minimum radius equal to their softening length.
        This includes contributions from all particle types.
        """
        if self.Mtotpart == 0:
            return None
        if not hasattr(self, "vmax_soft"):
            soft_r = np.maximum(self.softening, self.radius)
            self.r_vmax_soft, self.vmax_soft = get_vmax(self.mass, soft_r)
        return self.vmax_soft

    @lazy_property
    def spin_parameter(self) -> unyt.unyt_quantity:
        """
        Spin parameter of all particles in the spherical overdensity.

        Computed as in Bullock et al. (2021):
          lambda = |Ltot| / (sqrt(2) * M * v_max * R)
        """
        if self.Mtotpart == 0:
            return None
        if self.Vmax_soft > 0:
            vrel = self.velocity - self.vcom[None, :]
            Ltot = np.linalg.norm(
                (self.mass[:, None] * np.cross(self.position, vrel)).sum(axis=0)
            )
            return Ltot / (np.sqrt(2.0) * self.Mtotpart * self.SO_r * self.Vmax_soft)
        return None

    @lazy_property
    def TotalInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mtotpart == 0:
            return None
        mass = np.concatenate([self.mass, self.surrounding_mass], axis=0)
        position = np.concatenate([self.position, self.surrounding_position], axis=0)
        return get_inertia_tensor(
            mass, position, self.SO_r, search_radius=self.search_radius
        )

    @lazy_property
    def TotalInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mtotpart == 0:
            return None
        mass = np.concatenate([self.mass, self.surrounding_mass], axis=0)
        position = np.concatenate([self.position, self.surrounding_position], axis=0)
        return get_inertia_tensor(
            mass, position, self.SO_r, search_radius=self.search_radius, reduced=True
        )

    @lazy_property
    def TotalInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mtotpart == 0:
            return None
        return get_inertia_tensor(self.mass, self.position, self.SO_r, max_iterations=1)

    @lazy_property
    def TotalInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mtotpart == 0:
            return None
        return get_inertia_tensor(
            self.mass, self.position, self.SO_r, reduced=True, max_iterations=1
        )

    @lazy_property
    def Mfrac_satellites(self) -> unyt.unyt_quantity:
        """
        Mass fraction contributed by particles that are bound to subhalos other
        than the main subhalo. Excludes particles from hostless subhalos and
        from subhalos in other FOF groups.

        Note that this function is only called when we are guaranteed to have
        an SO, so we do not need to check SO_mass > 0.
        """
        return self.mass[self.is_bound_to_satellite].sum() / self.SO_mass

    @lazy_property
    def Mfrac_external(self) -> unyt.unyt_quantity:
        """
        Mass fraction contributed by particles that are bound to subhalos, but
        are outside this FOF group. Includes particles from hostless subhalos.
        """
        return self.mass[self.is_bound_to_external].sum() / self.SO_mass

    @lazy_property
    def gas_masses(self) -> unyt.unyt_array:
        """
        Masses of gas particles.
        """
        return self.mass[self.types == 0]

    @lazy_property
    def gas_pos(self) -> unyt.unyt_array:
        """
        Positions of gas particles.
        """
        return self.position[self.types == 0]

    @lazy_property
    def gas_vel(self) -> unyt.unyt_array:
        """
        Velocities of gas particles.
        """
        return self.velocity[self.types == 0]

    @lazy_property
    def Mgas(self) -> unyt.unyt_quantity:
        """
        Total mass of gas within the spherical overdensity.
        """
        return self.gas_masses.sum()

    @lazy_property
    def gas_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of gas particles.

        See mass_fraction() for the rationale behind this property.
        """
        if self.Mgas == 0:
            return None
        return self.gas_masses / self.Mgas

    @lazy_property
    def com_gas(self) -> unyt.unyt_array:
        """
        Centre of mass of gas particles.
        """
        if self.Mgas == 0:
            return None
        return ((self.gas_mass_fraction[:, None] * self.gas_pos).sum(
            axis=0
        ) + self.centre) % self.boxsize

    @lazy_property
    def vcom_gas(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of gas particles.
        """
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.gas_vel).sum(axis=0)

    def compute_Lgas_props(self):
        """
        Auxiliary function used to compute Lgas related properties.

        We use this function instead of a single property since it is cheaper
        to compute Lgas and Mcountrot_gas together.
        """
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
    def Lgas(self) -> unyt.unyt_array:
        """
        Total angular momentum of gas particles.

        Calls compute_Lgas_props() if needed.
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Lgas"):
            self.compute_Lgas_props()
        return self.internal_Lgas

    @lazy_property
    def DtoTgas(self) -> unyt.unyt_quantity:
        """
        Disc to total mass ratio of gas.

        Calls compute_Lgas_props() if needed.
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_gas"):
            self.compute_Lgas_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_gas / self.Mgas

    def gas_inertia_tensor(self, **kwargs) -> unyt.unyt_array:
        """
        Helper function for calculating gas inertia tensors
        """
        surrounding_mass = self.surrounding_mass[self.surrounding_types == 0]
        surrounding_position = self.surrounding_position[self.surrounding_types == 0]
        mass = np.concatenate([self.gas_masses, surrounding_mass], axis=0)
        position = np.concatenate([self.gas_pos, surrounding_position], axis=0)
        return get_inertia_tensor(
            mass, position, self.SO_r, search_radius=self.search_radius, **kwargs
        )

    @lazy_property
    def GasInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mgas == 0:
            return None
        return self.gas_inertia_tensor()

    @lazy_property
    def GasInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mgas == 0:
            return None
        return self.gas_inertia_tensor(reduced=True)

    @lazy_property
    def GasInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(
            self.gas_masses, self.gas_pos, self.SO_r, max_iterations=1
        )

    @lazy_property
    def GasInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(
            self.gas_masses, self.gas_pos, self.SO_r, reduced=True, max_iterations=1
        )

    @lazy_property
    def dm_masses(self) -> unyt.unyt_array:
        """
        Masses of dark matter particles.
        """
        return self.mass[self.types == 1]

    @lazy_property
    def dm_pos(self) -> unyt.unyt_array:
        """
        Positions of dark matter particles.
        """
        return self.position[self.types == 1]

    @lazy_property
    def dm_vel(self) -> unyt.unyt_array:
        """
        Velocities of dark matter particles.
        """
        return self.velocity[self.types == 1]

    @lazy_property
    def Mdm(self) -> unyt.unyt_quantity:
        """
        Total mass of dark matter particles in the spherical overdensity.
        """
        return self.dm_masses.sum()

    @lazy_property
    def dm_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of dark matter particles.

        See mass_fractions() for the rationale behind this property.
        """
        if self.Mdm == 0:
            return None
        return self.dm_masses / self.Mdm

    @lazy_property
    def vcom_dm(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of dark matter particles in the spherical overdensity.
        """
        if self.Mdm == 0:
            return None
        return (self.dm_mass_fraction[:, None] * self.dm_vel).sum(axis=0)

    @lazy_property
    def Ldm(self) -> unyt.unyt_array:
        """
        Total angular momentum of dark matter particles.
        """
        if self.Mdm == 0:
            return None
        return get_angular_momentum(
            self.dm_masses, self.dm_pos, self.dm_vel, ref_velocity=self.vcom_dm
        )

    def dm_inertia_tensor(self, **kwargs) -> unyt.unyt_array:
        """
        Helper function for calculating dm inertia tensors
        """
        surrounding_mass = self.surrounding_mass[self.surrounding_types == 1]
        surrounding_position = self.surrounding_position[self.surrounding_types == 1]
        mass = np.concatenate([self.dm_masses, surrounding_mass], axis=0)
        position = np.concatenate([self.dm_pos, surrounding_position], axis=0)
        return get_inertia_tensor(
            mass, position, self.SO_r, search_radius=self.search_radius, **kwargs
        )

    @lazy_property
    def DarkMatterInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the dark matter mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mdm == 0:
            return None
        return self.dm_inertia_tensor()

    @lazy_property
    def DarkMatterInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the dark matter mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mdm == 0:
            return None
        return self.dm_inertia_tensor(reduced=True)

    @lazy_property
    def DarkMatterInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the dark matter mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(
            self.dm_masses, self.dm_pos, self.SO_r, max_iterations=1
        )

    @lazy_property
    def DarkMatterInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the dark matter mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(
            self.dm_masses, self.dm_pos, self.SO_r, reduced=True, max_iterations=1
        )

    @lazy_property
    def star_masses(self) -> unyt.unyt_array:
        """
        Masses of star particles.
        """
        return self.mass[self.types == 4]

    @lazy_property
    def star_pos(self) -> unyt.unyt_array:
        """
        Positions of star particles.
        """
        return self.position[self.types == 4]

    @lazy_property
    def star_vel(self) -> unyt.unyt_array:
        """
        Velocities of star particles.
        """
        return self.velocity[self.types == 4]

    @lazy_property
    def Mstar(self) -> unyt.unyt_quantity:
        """
        Total mass of star particles in the spherical overdensity.
        """
        return self.star_masses.sum()

    @lazy_property
    def star_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of star particles.

        See mass_fractions() for the rationale behind this property.
        """
        if self.Mstar == 0:
            return None
        return self.star_masses / self.Mstar

    @lazy_property
    def com_star(self) -> unyt.unyt_array:
        """
        Centre of mass of star particles.
        """
        if self.Mstar == 0:
            return None
        return ((self.star_mass_fraction[:, None] * self.star_pos).sum(
            axis=0
        ) + self.centre) % self.boxsize

    @lazy_property
    def vcom_star(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of star particles.
        """
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.star_vel).sum(axis=0)

    def compute_Lstar_props(self):
        """
        Auxiliary function used to compute properties related to Lstar.

        We do this because computing these properties together is cheaper.
        """
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
    def Lstar(self) -> unyt.unyt_array:
        """
        Total angular momentum of star particles in the spherical overdensity.

        Calls compute_Lstar_props() if required.
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Lstar"):
            self.compute_Lstar_props()
        return self.internal_Lstar

    @lazy_property
    def DtoTstar(self) -> unyt.unyt_quantity:
        """
        Disc to total mass ratio for star particles.

        Calls compute_Lstar_props() if required.
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_star"):
            self.compute_Lstar_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_star / self.Mstar

    def stellar_inertia_tensor(self, **kwargs) -> unyt.unyt_array:
        """
        Helper function for calculating stellar inertia tensors
        """
        surrounding_mass = self.surrounding_mass[self.surrounding_types == 4]
        surrounding_position = self.surrounding_position[self.surrounding_types == 4]
        mass = np.concatenate([self.star_masses, surrounding_mass], axis=0)
        position = np.concatenate([self.star_pos, surrounding_position], axis=0)
        return get_inertia_tensor(
            mass, position, self.SO_r, search_radius=self.search_radius, **kwargs
        )

    @lazy_property
    def StellarInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mstar == 0:
            return None
        return self.stellar_inertia_tensor()

    @lazy_property
    def StellarInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius SORadius.
        """
        if self.Mstar == 0:
            return None
        return self.stellar_inertia_tensor(reduced=True)

    @lazy_property
    def StellarInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.star_masses, self.star_pos, self.SO_r, max_iterations=1
        )

    @lazy_property
    def StellarInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution.
        Computed using all particles within the SORadius.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.star_masses, self.star_pos, self.SO_r, reduced=True, max_iterations=1
        )

    @lazy_property
    def baryon_masses(self) -> unyt.unyt_array:
        """
        Masses of baryon (gas + star) particles.
        """
        return self.mass[(self.types == 0) | (self.types == 4)]

    @lazy_property
    def baryon_pos(self) -> unyt.unyt_array:
        """
        Positions of baryon (gas + star) particles.
        """
        return self.position[(self.types == 0) | (self.types == 4)]

    @lazy_property
    def baryon_vel(self) -> unyt.unyt_array:
        """
        Velocities of baryon (gas + star) particles.
        """
        return self.velocity[(self.types == 0) | (self.types == 4)]

    @lazy_property
    def Mbaryons(self) -> unyt.unyt_quantity:
        """
        Total mass of baryon (gas + star) particles in the spherical overdensity.
        """
        return self.baryon_masses.sum()

    @lazy_property
    def baryon_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of baryon (gas + star) particles.

        See mass_fractions() for the rationale behind this function.
        """
        if self.Mbaryons == 0:
            return None
        return self.baryon_masses / self.Mbaryons

    @lazy_property
    def baryon_vcom(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of baryon (gas + star) particles.
        """
        if self.Mbaryons == 0:
            return None
        return (self.baryon_mass_fraction[:, None] * self.baryon_vel).sum(axis=0)

    @lazy_property
    def Lbaryons(self) -> unyt.unyt_array:
        """
        Total angular momentum of baryon (gas + star) particles.
        """
        if self.Mbaryons == 0:
            return None
        baryon_relvel = self.baryon_vel - self.baryon_vcom[None, :]
        return (
            self.baryon_masses[:, None] * np.cross(self.baryon_pos, baryon_relvel)
        ).sum(axis=0)

    @lazy_property
    def Mbh_dynamical(self) -> unyt.unyt_quantity:
        """
        Total dynamical mass of black hole particles in the spherical overdensity.
        """
        return self.mass[self.types == 5].sum()

    @lazy_property
    def Ngas(self) -> int:
        """
        Number of gas particles in the spherical overdensity.
        """
        return self.gas_selection.sum()

    @lazy_property
    def Ngas_no_agn(self) -> int:
        """Number of gas particles, excluding those recently heated by AGN."""
        if self.Ngas == 0:
            return 0
        return self.gas_no_agn.sum()

    @lazy_property
    def Ngas_no_cool(self) -> int:
        """
        Number of non-cool gas particles (i.e. temperature > 1.e5 K).
        """
        if self.Ngas == 0:
            return 0
        return self.gas_no_cool.sum()

    @lazy_property
    def Ngas_no_cool_no_agn(self) -> int:
        """
        Number of gas particles that are not cold (temperature > 1.e5 K) and that
        were not recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return 0
        return self.gas_no_cool_no_agn.sum()

    @lazy_property
    def Ngas_core_excision(self) -> int:
        """
        Number of gas particles, exluding those in the inner core.
        """
        if self.Ngas == 0:
            return 0
        return self.gas_selection_core_excision.sum()

    @lazy_property
    def Ngas_no_agn_core_excision(self) -> int:
        """
        Number of gas particles, excluding those which are in the inner core or were recently heated by AGN.
        """
        if self.Ngas == 0:
            return 0
        return self.gas_selection_no_agn_core_excision.sum()

    @lazy_property
    def Ngas_no_cool_core_excision(self) -> int:
        """
        Number of gas particles, excluding those which are in the inner core or are cold (temperature > 1.e5 K).
        """
        if self.Ngas == 0:
            return 0
        return self.gas_selection_core_excision_no_cool.sum()

    @lazy_property
    def Ngas_no_cool_no_agn_core_excision(self) -> int:
        """
        Number of gas particles, excluding those which are in the inner core, are cold (temperature > 1.e5 K), or were recently heated by AGN.
        """
        if self.Ngas == 0:
            return 0
        return self.gas_selection_core_excision_no_cool_no_agn.sum()

    @lazy_property
    def Ngas_xray_temperature(self) -> int:
        if self.Ngas == 0:
            return 0
        return self.gas_selection_xray_temperature.sum()

    @lazy_property
    def Ngas_xray_temperature_no_agn(self) -> int:
        if self.Ngas == 0:
            return 0
        return self.gas_no_agn_xray_temperature.sum()

    @lazy_property
    def Ngas_core_excision_xray_temperature(self) -> int:
        if self.Ngas == 0:
            return 0
        return self.gas_selection_core_excision_xray_temperature.sum()

    @lazy_property
    def Ngas_core_excision_xray_temperature_no_agn(self) -> int:
        if self.Ngas == 0:
            return 0
        return self.gas_selection_core_excision_no_agn_xray_temperature.sum()

    @lazy_property
    def gas_metal_masses(self) -> unyt.unyt_array:
        """
        Metal masses in gas particles.

        Contains the total metal mass, including dust contributions.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.get_dataset("PartType0/MetalMassFractions")[self.gas_selection]
        )

    @lazy_property
    def gasmetalfrac(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction in gas particles.

        As a fraction of the total gas mass.
        Contains the total metal mass, including dust contributions.
        """
        if self.Ngas == 0:
            return None
        return self.gas_metal_masses.sum() / self.Mgas

    @lazy_property
    def gasOfrac(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction in oxygen.

        As a fraction of the total gas mass.
        Contains the total oxygen mass, including dust contributions.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.get_dataset("PartType0/ElementMassFractions")[self.gas_selection][
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
        ).sum() / self.Mgas

    @lazy_property
    def gasFefrac(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction in iron.

        As a fraction of the total gas mass.
        Contains the total iron mass, including dust contributions.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_masses
            * self.get_dataset("PartType0/ElementMassFractions")[self.gas_selection][
                :,
                self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron"),
            ]
        ).sum() / self.Mgas

    @lazy_property
    def gas_temperatures(self) -> unyt.unyt_array:
        """
        Temperatures of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Temperatures")[self.gas_selection]

    @lazy_property
    def Tgas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average gas temperature.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_temperatures * self.gas_mass_fraction).sum()

    @lazy_property
    def gas_no_cool(self) -> NDArray[bool]:
        """
        Mask for non-cool gas particles (i.e. temperature > 1.e5 K).
        """
        if self.Ngas == 0:
            return None
        return self.gas_temperatures > 1.0e5 * unyt.K

    @lazy_property
    def Tgas_cy_weighted(self) -> unyt.unyt_quantity:
        """
        ComptonY-weighted average gas temperature.
        """
        if self.Ngas == 0:
            return None
        gas_compY_sum = self.gas_compY.sum()
        if gas_compY_sum == 0:
            return None
        return (
            self.gas_temperatures * (self.gas_compY.value / gas_compY_sum.value)
        ).sum()

    @lazy_property
    def Tgas_cy_weighted_no_agn(self) -> unyt.unyt_quantity:
        """
        ComptonY-weighted average gas temperature, excluding gas recently heated by AGN.
        """
        if self.Ngas == 0:
            return None
        gas_compY_sum = self.gas_compY[self.gas_no_agn].sum()
        if gas_compY_sum == 0:
            return None
        return (
            self.gas_temperatures[self.gas_no_agn]
            * (self.gas_compY[self.gas_no_agn].value / gas_compY_sum.value)
        ).sum()

    @lazy_property
    def Tgas_cy_weighted_core_excision(self) -> unyt.unyt_quantity:
        """
        ComptonY-weighted average gas temperature, excluding the inner core.
        """
        if self.Ngas == 0:
            return None
        gas_compY_sum = self.gas_compY[self.gas_selection_core_excision].sum()
        if gas_compY_sum == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_core_excision]
            * (
                self.gas_compY[self.gas_selection_core_excision].value
                / gas_compY_sum.value
            )
        ).sum()

    @lazy_property
    def Tgas_cy_weighted_core_excision_no_agn(self) -> unyt.unyt_quantity:
        """
        ComptonY-weighted average gas temperature, excluding the inner core and 
        gas recently heated by AGN.
        """
        if self.Ngas == 0:
            return None
        gas_compY_sum = self.gas_compY[self.gas_selection_no_agn_core_excision].sum()
        if gas_compY_sum == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_no_agn_core_excision]
            * (
                self.gas_compY[self.gas_selection_no_agn_core_excision].value
                / gas_compY_sum.value
            )
        ).sum()

    @lazy_property
    def Tgas_core_excision(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average gas temperature, excluding the inner core.
        """
        if self.Ngas_core_excision == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_core_excision]
            * (
                self.gas_masses[self.gas_selection_core_excision]
                / self.gas_masses[self.gas_selection_core_excision].sum()
            )
        ).sum()

    @lazy_property
    def Tgas_no_agn_core_excision(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average gas temperature, excluding the inner core and gas
        recently heated by AGN.
        """
        if self.Ngas_no_agn_core_excision == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_no_agn_core_excision]
            * (
                self.gas_masses[self.gas_selection_no_agn_core_excision]
                / self.gas_masses[self.gas_selection_no_agn_core_excision].sum()
            )
        ).sum()

    @lazy_property
    def Tgas_no_cool_core_excision(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average gas temperature, excluding the inner core and cool
        gas. 
        """
        if self.Ngas_no_cool_core_excision == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_core_excision_no_cool]
            * (
                self.gas_masses[self.gas_selection_core_excision_no_cool]
                / self.gas_masses[self.gas_selection_core_excision_no_cool].sum()
            )
        ).sum()

    @lazy_property
    def Tgas_no_cool_no_agn_core_excision(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average gas temperature, excluding the inner core, cool
        gas and gas recently heated by AGN.
        """
        if self.Ngas_no_cool_no_agn_core_excision == 0:
            return None
        return (
            self.gas_temperatures[self.gas_selection_core_excision_no_cool_no_agn]
            * (
                self.gas_masses[self.gas_selection_core_excision_no_cool_no_agn]
                / self.gas_masses[self.gas_selection_core_excision_no_cool_no_agn].sum()
            )
        ).sum()

    @lazy_property
    def gas_selection_core_excision(self):
        return self.radius[self.types == 0] > self.core_excision_fraction * self.SO_r

    @lazy_property
    def gas_selection_no_agn_core_excision(self):
        return self.gas_no_agn & self.gas_selection_core_excision

    @lazy_property
    def Mhotgas(self) -> unyt.unyt_quantity:
        """
        Total gas mass in hot (temperature > 1.e5 K) gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_masses[self.gas_no_cool].sum()

    @lazy_property
    def Tgas_no_cool(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of hot (temperature > 1.e5 K) gas.
        """
        if self.Ngas == 0:
            return None
        if np.any(self.gas_no_cool):
            return (
                self.gas_temperatures[self.gas_no_cool]
                * self.gas_masses[self.gas_no_cool]
            ).sum() / self.Mhotgas

    @lazy_property
    def gas_SFR(self) -> unyt.unyt_array:
        """
        Star formation rate of gas particles.

        Filters out negative values that were used to store the last star
        formation time/scale factor in older SWIFT versions.
        """
        if self.Ngas == 0:
            return None
        SFR = self.get_dataset("PartType0/StarFormationRates")[self.gas_selection]
        is_SFR = SFR > 0.0
        SFR[~is_SFR] = 0.0
        return SFR

    @lazy_property
    def SFR(self) -> unyt.unyt_quantity:
        """
        Total star formation rate of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def AveragedStarFormationRate(self) -> unyt.unyt_array:
        """
        Averaged star formation rates of gas particles. Averaging times are
        set by the value of 'recording_triggers' in the SWIFT parameter file.
        """
        if self.Ngas == 0:
            return None
        avg_SFR = self.get_dataset("PartType0/AveragedStarFormationRates")[
            self.gas_selection
        ]
        return np.sum(avg_SFR, axis=0)

    @lazy_property
    def Mgas_SF(self) -> unyt.unyt_quantity:
        """
        Total mass of star-forming (SFR > 0) gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_masses[self.gas_SFR > 0.0].sum()

    @lazy_property
    def gasmetalfrac_SF(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction of star-forming gas.

        As a fraction of the total mass of star-forming gas.
        Includes contributions from dust.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_metal_masses[self.gas_SFR > 0.0].sum() / self.Mgas_SF

    @lazy_property
    def gas_xraylum(self) -> unyt.unyt_array:
        """
        Observer-frame X-ray luminosities of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/XrayLuminosities")[self.gas_selection]

    @lazy_property
    def gas_xraylum_restframe(self):
        """
        Rest-frame X-ray luminosities of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/XrayLuminositiesRestframe")[
            self.gas_selection
        ]

    @lazy_property
    def Xraylum(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray luminosities of gas.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xraylum.sum(axis=0)

    @lazy_property
    def Xraylum_restframe(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray luminosities of gas particles.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xraylum_restframe.sum(axis=0)

    @lazy_property
    def gas_xrayphlum(self) -> unyt.unyt_array:
        """
        Observer-frame X-ray photon luminosities of gas.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/XrayPhotonLuminosities")[self.gas_selection]

    @lazy_property
    def gas_xrayphlum_restframe(self) -> unyt.unyt_array:
        """
        Rest-frame X-ray photon luminosities of gas.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/XrayPhotonLuminositiesRestframe")[
            self.gas_selection
        ]

    @lazy_property
    def Xrayphlum(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray photon luminosities of gas.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xrayphlum.sum(axis=0)

    @lazy_property
    def Xrayphlum_restframe(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray photon luminosities of gas.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xrayphlum_restframe.sum(axis=0)

    @lazy_property
    def gas_compY(self) -> unyt.unyt_array:
        """
        Compton Y parameter for gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/ComptonYParameters")[self.gas_selection]

    @lazy_property
    def compY_unit(self) -> unyt.unyt_quantity:
        """
        Unit for Compton Y variables.

        We cannot rely on unyt single precision to do unit conversions for
        Compton Y, since the values have such a large range in unyt default
        units.
        """
        # We need to manually convert the units for compY to avoid numerical
        # overflow inside unyt
        if self.Ngas == 0:
            return None
        unit = 1.0 * self.gas_compY.units
        new_unit = unit.to(PropertyTable.full_property_list["compY"].unit)
        return new_unit

    @lazy_property
    def compY(self) -> unyt.unyt_quantity:
        """
        Total Compton Y parameter for gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.gas_compY.sum().value * self.compY_unit

    @lazy_property
    def gas_no_agn(self) -> NDArray[bool]:
        """
        Mask for gas particles that were not recently heated by AGN feedback.

        The mask is obtained from the RecentlyHeatedGasFilter.
        """
        if self.Ngas == 0:
            return None
        last_agn_gas = self.get_dataset("PartType0/LastAGNFeedbackScaleFactors")[
            self.gas_selection
        ]
        return ~self.recently_heated_gas_filter.is_recently_heated(
            last_agn_gas, self.gas_temperatures
        )

    @lazy_property
    def Xraylum_no_agn(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray luminosities of gas particles, excluding 
        contributions from gas particles that were recently heated by AGN feedback.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xraylum[self.gas_no_agn].sum(axis=0)

    @lazy_property
    def Xrayphlum_no_agn(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles that were recently heated by AGN feedback.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas == 0:
            return None
        return self.gas_xrayphlum[self.gas_no_agn].sum(axis=0)

    @lazy_property
    def Xraylum_restframe_no_agn(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray luminosities of gas particles, excluding contributions
        from gas particles that were recently heated by AGN feedback.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn == 0:
            return None
        return self.gas_xraylum_restframe[self.gas_no_agn].sum(axis=0)

    @lazy_property
    def Xrayphlum_restframe_no_agn(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles that were recently heated by AGN feedback.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn == 0:
            return None
        return self.gas_xrayphlum_restframe[self.gas_no_agn].sum(axis=0)

    @lazy_property
    def compY_no_agn(self) -> unyt.unyt_quantity:
        """
        Total Compton Y parameter for gas particles, excluding gas particles
        that were recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return None
        if np.any(self.gas_no_agn):
            return self.gas_compY[self.gas_no_agn].sum().value * self.compY_unit
        else:
            return None

    @lazy_property
    def Tgas_no_agn(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of gas particles, excluding gas
        particles that were recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return None
        mass_gas_no_agn = self.gas_masses[self.gas_no_agn]
        Mgas_no_agn = mass_gas_no_agn.sum()
        if Mgas_no_agn > 0:
            return (
                (mass_gas_no_agn / Mgas_no_agn) * self.gas_temperatures[self.gas_no_agn]
            ).sum()

    @lazy_property
    def gas_no_cool_no_agn(self) -> NDArray[bool]:
        """
        Mask for gas particles that are not cold (temperature > 1.e5 K) and that
        were not recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return np.zeros(0, dtype=bool)
        return self.gas_no_cool & self.gas_no_agn

    @lazy_property
    def Tgas_no_cool_no_agn(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of gas particles, excluding cool
        (temperature <= 1.e5 K) gas particles and particles that were recently
        heated by AGN feedback.
        """
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
    def Xraylum_core_excision(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray luminosities of gas particles,
        excluding contributions from gas particles in the inner core.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_core_excision == 0:
            return None
        return self.gas_xraylum[self.gas_selection_core_excision].sum(axis=0)

    @lazy_property
    def Xrayphlum_core_excision(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles in the inner core.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_core_excision == 0:
            return None
        return self.gas_xrayphlum[self.gas_selection_core_excision].sum(axis=0)

    @lazy_property
    def Xraylum_no_agn_core_excision(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray luminosities of gas particles, 
        excluding contributions from gas particles in the inner core and those
        recently heated by AGN.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn_core_excision == 0:
            return None
        return self.gas_xraylum[self.gas_selection_no_agn_core_excision].sum(axis=0)

    @lazy_property
    def Xrayphlum_no_agn_core_excision(self) -> unyt.unyt_array:
        """
        Total observer-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles in the inner core and those
        recently heated by AGN.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn_core_excision == 0:
            return None
        return self.gas_xrayphlum[self.gas_selection_no_agn_core_excision].sum(axis=0)

    @lazy_property
    def Xraylum_restframe_core_excision(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray luminosities of gas particles, 
        excluding contributions from gas particles in the inner core.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_core_excision == 0:
            return None
        return self.gas_xraylum_restframe[self.gas_selection_core_excision].sum(axis=0)

    @lazy_property
    def Xrayphlum_restframe_core_excision(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles in the inner core.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_core_excision == 0:
            return None
        return self.gas_xrayphlum_restframe[self.gas_selection_core_excision].sum(
            axis=0
        )

    @lazy_property
    def Xraylum_restframe_no_agn_core_excision(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray luminosities of gas particles, 
        excluding contributions from gas particles in the inner core and those
        recently heated by AGN.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn_core_excision == 0:
            return None
        return self.gas_xraylum_restframe[self.gas_selection_no_agn_core_excision].sum(
            axis=0
        )

    @lazy_property
    def Xrayphlum_restframe_no_agn_core_excision(self) -> unyt.unyt_array:
        """
        Total rest-frame X-ray photon luminosities of gas particles, 
        excluding contributions from gas particles in the inner core and those
        recently heated by AGN.

        Note that this is an array, since we have multiple luminosity bands.
        """
        if self.Ngas_no_agn_core_excision == 0:
            return None
        return self.gas_xrayphlum_restframe[
            self.gas_selection_no_agn_core_excision
        ].sum(axis=0)

    @lazy_property
    def gas_selection_xray_temperature(self):
        if self.Ngas == 0:
            return None
        return self.gas_temperatures > 1.16e6 * unyt.K

    @lazy_property
    def gas_no_agn_xray_temperature(self):
        if self.Ngas == 0:
            return None
        return self.gas_no_agn & self.gas_selection_xray_temperature

    @lazy_property
    def gas_selection_core_excision_xray_temperature(self):
        if self.Ngas == 0:
            return None
        return self.gas_selection_core_excision & self.gas_selection_xray_temperature

    @lazy_property
    def gas_selection_core_excision_no_cool(self):
        if self.Ngas == 0:
            return None
        return self.gas_selection_core_excision & self.gas_no_cool

    @lazy_property
    def gas_selection_core_excision_no_cool_no_agn(self):
        if self.Ngas == 0:
            return None
        return self.gas_selection_core_excision & self.gas_no_cool & self.gas_no_agn

    @lazy_property
    def gas_selection_core_excision_no_agn_xray_temperature(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_selection_core_excision
            & self.gas_no_agn
            & self.gas_selection_xray_temperature
        )

    @lazy_property
    def SpectroscopicLikeTemperature(self) -> unyt.unyt_quantity:
        """
        Temperature of the gas, as inferred from spectroscopic-like estimates.
        """
        if self.Ngas == 0:
            return None
        numerator = np.sum(
            self.gas_densities[self.gas_selection_xray_temperature]
            * self.gas_masses[self.gas_selection_xray_temperature]
            * self.gas_temperatures[self.gas_selection_xray_temperature] ** (1 / 4)
        )
        denominator = np.sum(
            self.gas_densities[self.gas_selection_xray_temperature]
            * self.gas_masses[self.gas_selection_xray_temperature]
            * self.gas_temperatures[self.gas_selection_xray_temperature] ** (-3 / 4)
        )
        if denominator == 0:
            return None
        return numerator / denominator

    @lazy_property
    def SpectroscopicLikeTemperature_no_agn(self) -> unyt.unyt_quantity:
        """
        Temperature of the gas, as inferred from spectroscopic-like estimates,
        excluding gas particles that were recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return None
        numerator = np.sum(
            self.gas_densities[self.gas_no_agn_xray_temperature]
            * self.gas_masses[self.gas_no_agn_xray_temperature]
            * self.gas_temperatures[self.gas_no_agn_xray_temperature] ** (1 / 4)
        )
        denominator = np.sum(
            self.gas_densities[self.gas_no_agn_xray_temperature]
            * self.gas_masses[self.gas_no_agn_xray_temperature]
            * self.gas_temperatures[self.gas_no_agn_xray_temperature] ** (-3 / 4)
        )
        if denominator == 0:
            return None
        return numerator / denominator

    @lazy_property
    def SpectroscopicLikeTemperature_core_excision(self) -> unyt.unyt_quantity:
        """
        Temperature of the gas, as inferred from spectroscopic-like estimates,
        excluding those in the inner core.
        """
        if self.Ngas == 0:
            return None
        numerator = np.sum(
            self.gas_densities[self.gas_selection_core_excision_xray_temperature]
            * self.gas_masses[self.gas_selection_core_excision_xray_temperature]
            * self.gas_temperatures[self.gas_selection_core_excision_xray_temperature]
            ** (1 / 4)
        )
        denominator = np.sum(
            self.gas_densities[self.gas_selection_core_excision_xray_temperature]
            * self.gas_masses[self.gas_selection_core_excision_xray_temperature]
            * self.gas_temperatures[self.gas_selection_core_excision_xray_temperature]
            ** (-3 / 4)
        )
        if denominator == 0:
            return None
        return numerator / denominator

    @lazy_property
    def SpectroscopicLikeTemperature_no_agn_core_excision(self) -> unyt.unyt_quantity:
        """
        Temperature of the gas, as inferred from spectroscopic-like estimates,
        excluding those in the inner core and recently heated by AGN.
        """
        if self.Ngas == 0:
            return None
        numerator = np.sum(
            self.gas_densities[self.gas_selection_core_excision_no_agn_xray_temperature]
            * self.gas_masses[self.gas_selection_core_excision_no_agn_xray_temperature]
            * self.gas_temperatures[
                self.gas_selection_core_excision_no_agn_xray_temperature
            ]
            ** (1 / 4)
        )
        denominator = np.sum(
            self.gas_densities[self.gas_selection_core_excision_no_agn_xray_temperature]
            * self.gas_masses[self.gas_selection_core_excision_no_agn_xray_temperature]
            * self.gas_temperatures[
                self.gas_selection_core_excision_no_agn_xray_temperature
            ]
            ** (-3 / 4)
        )
        if denominator == 0:
            return None
        return numerator / denominator

    @lazy_property
    def Ekin_gas(self) -> unyt.unyt_quantity:
        """
        Total kinetic energy of gas particles.

        Note that we need to be careful with the units here to avoid numerical
        overflow.
        """
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
    def gas_densities(self) -> unyt.unyt_array:
        """
        Densities of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Densities")[self.gas_selection]

    @lazy_property
    def gas_electron_number_densities(self):
        return self.get_dataset("PartType0/ElectronNumberDensities")[self.gas_selection]

    @lazy_property
    def Etherm_gas(self) -> unyt.unyt_array:
        """
        Total thermal energy of gas particles.

        While this could be computed from PartType0/InternalEnergies, we use
        the equation of state
         P = (gamma-1) * rho * u
        (with gamma=5/3) because some simulations (read: FLAMINGO) do not output
        the internal energies.

        Note that we need to be careful with the units here to avoid numerical
        overflow.
        """
        if self.Ngas == 0:
            return None
        etherm_gas = (
            1.5
            * self.gas_masses
            * self.get_dataset("PartType0/Pressures")[self.gas_selection]
            / self.gas_densities
        )
        etherm_gas = unyt.unyt_array(
            etherm_gas.value, dtype=np.float64, units=etherm_gas.units
        )
        return etherm_gas.sum()

    @lazy_property
    def DopplerB(self) -> unyt.unyt_quantity:
        """
        Total Doppler B of gas in the spherical overdensity.

        Since Doppler B requires a line of sight, it is calculated relative to
        the given observer position.

        We need to be careful with units to avoid numerical overflow.
        We also need to be careful with halos very close to the observer
        position. While the line of sight velocities for particles at zero
        relative distance are zero, we would still be dividing by the zero
        distance in the expression, which leads to numerical problems.
        """
        if self.Ngas == 0:
            return None
        ne = self.get_dataset("PartType0/ElectronNumberDensities")[self.gas_selection]
        # note: the positions where relative to the centre, so we have
        # to make them absolute again before subtracting the observer
        # position
        relpos = self.gas_pos + self.centre[None, :] - self.observer_position[None, :]
        distance = np.sqrt((relpos ** 2).sum(axis=1))
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
        area = np.pi * self.SO_r ** 2
        return (fac * ne * vr * (volumes / area)).sum().to("dimensionless")

    @lazy_property
    def Ndm(self) -> int:
        """
        Number of dark matter particles in the spherical overdensity.
        """
        return self.dm_selection.sum()

    @lazy_property
    def Nstar(self) -> int:
        """
        Number of star particles in the spherical overdensity.
        """
        return self.star_selection.sum()

    @lazy_property
    def Mstar_init(self) -> unyt.unyt_quantity:
        """
        Total initial stellar mass of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/InitialMasses")[self.star_selection].sum()

    @lazy_property
    def starmetalfrac(self) -> unyt.unyt_quantity:
        """
        Total metal mass fraction of star particles.

        Given as a fraction of the total mass in star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.get_dataset("PartType4/MetalMassFractions")[self.star_selection]
        ).sum() / self.Mstar

    @lazy_property
    def starOfrac(self) -> unyt.unyt_quantity:
        """
        Total oxygen mass fraction of star particles.

        Given as a fraction of the total mass in star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.get_dataset("PartType4/ElementMassFractions")[self.star_selection][
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
        ).sum() / self.Mstar

    @lazy_property
    def starFefrac(self) -> unyt.unyt_quantity:
        """
        Total iron mass fraction of star particles.

        Given as a fraction of the total mass in star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_masses
            * self.get_dataset("PartType4/ElementMassFractions")[self.star_selection][
                :,
                self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron"),
            ]
        ).sum() / self.Mstar

    @lazy_property
    def StellarLuminosity(self) -> unyt.unyt_array:
        """
        Total luminosity of star particles.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/Luminosities")[self.star_selection].sum(
            axis=0
        )

    @lazy_property
    def Ekin_star(self) -> unyt.unyt_quantity:
        """
        Total kinetic energy of star particles.

        Note that we need to be careful with units here to avoid numerical
        overflow.
        """
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
    def Nbh(self) -> int:
        """
        Number of black hole particles in the spherical overdensity.
        """
        return self.bh_selection.sum()

    @lazy_property
    def BH_subgrid_masses(self) -> unyt.unyt_array:
        """
        Sub-grid masses of black hole particles.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_selection]

    @lazy_property
    def Mbh_subgrid(self) -> unyt.unyt_quantity:
        """
        Total sub-grid mass of black hole particles in the spherical overdensity.
        """
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses.sum()

    @lazy_property
    def agn_eventa(self) -> unyt.unyt_array:
        """
        Last AGN feedback scale factor of black hole particles.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
            self.bh_selection
        ]

    @lazy_property
    def BHlasteventa(self) -> unyt.unyt_quantity:
        """
        Maximum AGN feedback scale factor among all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.max(self.agn_eventa)

    @lazy_property
    def BlackHolesTotalInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Total thermal energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.get_dataset("PartType5/AGNTotalInjectedEnergies")[self.bh_selection]
        )

    @lazy_property
    def BlackHolesTotalInjectedJetEnergy(self) -> unyt.unyt_quantity:
        """
        Total jet energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.get_dataset("PartType5/InjectedJetEnergies")[self.bh_selection]
        )

    @lazy_property
    def iBHmax(self) -> int:
        """
        Index of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return np.argmax(self.BH_subgrid_masses)

    @lazy_property
    def BHmaxM(self) -> unyt.unyt_quantity:
        """
        Sub-grid mass of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses[self.iBHmax]

    @lazy_property
    def BHmaxID(self) -> int:
        """
        ID of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/ParticleIDs")[self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxpos(self) -> unyt.unyt_array:
        """
        Position of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Coordinates")[self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxvel(self) -> unyt.unyt_array:
        """
        Velocity of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Velocities")[self.bh_selection][self.iBHmax]

    @lazy_property
    def BHmaxAR(self) -> unyt.unyt_quantity:
        """
        Accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionRates")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleAveragedAccretionRate(self) -> unyt.unyt_quantity:
        """
        Averaged accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AveragedAccretionRates")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Thermal energy injected into gas particles by the most massive
        BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AGNTotalInjectedEnergies")[
            self.bh_selection
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleAccretionMode(self) -> unyt.unyt_quantity:
        """
        Accretion flow regime of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionModes")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleGWMassLoss(self) -> unyt.unyt_quantity:
        """
        Cumulative mass lost to GW of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/GWMassLosses")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleInjectedJetEnergyByMode(self) -> unyt.unyt_quantity:
        """
        Total energy injected in the kinetic jet AGN feedback mode, split by accretion mode,
        of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/InjectedJetEnergiesByMode")[
            self.bh_selection
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleLastJetEventScalefactor(self) -> unyt.unyt_quantity:
        """
        Scale-factor of last jet event of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNJetScaleFactors")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNEvents(self) -> unyt.unyt_quantity:
        """
        Number of AGN events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNEvents")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNJetEvents(self) -> unyt.unyt_quantity:
        """
        Number of jet events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNJetEvents")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfMergers(self) -> unyt.unyt_quantity:
        """
        Number of mergers the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfMergers")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleRadiatedEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into radiation by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/RadiatedEnergiesByMode")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMassesByMode(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/TotalAccretedMassesByMode")[
            self.bh_selection
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleWindEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into accretion disc winds by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/WindEnergiesByMode")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleSpin(self) -> unyt.unyt_quantity:
        """
        The spin of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Spins")[self.bh_selection][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMass(self) -> unyt.unyt_quantity:
        """
        The total accreted mass of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/TotalAccretedMasses")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleFormationScalefactor(self) -> unyt.unyt_quantity:
        """
        The formation scale factor of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/FormationScaleFactors")[self.bh_selection][
            self.iBHmax
        ]

    @lazy_property
    def BHmaxlasteventa(self) -> unyt.unyt_quantity:
        """
        Last feedback scale factor of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def Nnu(self) -> int:
        """
        Number of neutrino particles in the spherical overdensity.
        """
        if self.has_neutrinos:
            return self.nu_selection.sum()
        else:
            return unyt.unyt_array(0, dtype=np.uint32, units="dimensionless")

    @lazy_property
    def Mnu(self) -> unyt.unyt_quantity:
        """
        Total mass of neutrino particles.

        This uses the raw masses of the particles, without weights.
        """
        if self.Nnu == 0:
            return None
        return self.get_dataset("PartType6/Masses")[self.nu_selection].sum()

    @lazy_property
    def MnuNS(self) -> unyt.unyt_quantity:
        """
        Total noise-suppressed mass of neutrino particles.

        This uses the weighted masses of the neutrino particles and adds the
        total mass due to the average neutrino density inside the SO volume.
        """
        if self.Nnu == 0:
            return None
        return (
            self.get_dataset("PartType6/Masses")[self.nu_selection]
            * self.get_dataset("PartType6/Weights")[self.nu_selection]
        ).sum() + self.cosmology["nu_density"] * self.SO_volume

    @staticmethod
    def concentration_from_R1(R1):
        # This is a higher degree polynomial fit than was used in Wang+23
        # Obtained by fitting R1-concentration radius for 1<c<1000
        polynomial = [-79.71, -222.46, -250.14, -140.17, -43.59, -5.07]
        c = 0
        for i, b in enumerate(polynomial[::-1]):
            # Ensure scale factors have been removed
            c += b * np.log10(R1.to("dimensionless")) ** i
        # Cap concentration values, as polynomial is only valid for 1<c<1000
        c = max(min(c, 3), 0)
        return unyt.unyt_quantity(10 ** c, dtype=np.float32, units="dimensionless")

    def calculate_concentration(self, r):
        if r.shape[0] < 10:
            return None
        R1 = np.sum(self.mass * r)
        missed_mass = self.Mtot - np.sum(self.mass)
        if self.Nnu != 0:
            # Neutrino particles within self.r
            R1 += np.sum(self.nu_mass * self.nu_radius)
            missed_mass -= np.sum(self.nu_mass)
        # Neutrino background
        R1 += np.pi * self.cosmology["nu_density"] * self.r ** 4
        missed_mass -= self.cosmology["nu_density"] * 4.0 / 3.0 * np.pi * self.r ** 3
        R1 += missed_mass * self.r
        # Normalize
        R1 /= self.r * self.Mtot
        return self.concentration_from_R1(R1)

    @lazy_property
    def concentration_unsoft(self):
        if not self.virial_definition:
            return None
        return self.calculate_concentration(self.radius)

    @lazy_property
    def concentration_soft(self):
        if not self.virial_definition:
            return None
        soft_r = np.maximum(self.softening, self.radius)
        return self.calculate_concentration(soft_r)

    def calculate_concentration_dmo(self, r):
        if r.shape[0] < 10:
            return None
        R1 = np.sum(self.mass[self.types == 1] * r)
        R1 += self.dm_missed_mass * self.r
        R1 /= self.r * (self.Mdm + self.dm_missed_mass)
        return self.concentration_from_R1(R1)

    @lazy_property
    def concentration_dmo_unsoft(self):
        if not self.virial_definition:
            return None
        r = self.radius[self.types == 1]
        return self.calculate_concentration_dmo(r)

    @lazy_property
    def concentration_dmo_soft(self):
        if not self.virial_definition:
            return None
        soft_r = np.maximum(
            self.softening[self.types == 1], self.radius[self.types == 1]
        )
        return self.calculate_concentration_dmo(soft_r)

    @lazy_property
    def vcom_ten_percent(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles within 0.1 R_SO.
        """
        mask = self.radius < 0.1 * self.SO_r
        return (self.mass[mask, None] * self.velocity[mask]).sum(axis=0) / self.mass[mask].sum()

    @lazy_property
    def vcom_thirty_percent(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles within 0.3 R_SO.
        """
        mask = self.radius < 0.3 * self.SO_r
        return (self.mass[mask, None] * self.velocity[mask]).sum(axis=0) / self.mass[mask].sum()

    def calculate_flow_rate(
        self,
        flow_type,
        positions,
        masses,
        velocities,
        internal_energies=None,
        fast_outflows=False,
        pseudo_evolve=True,
        hubble=False,
    ) -> unyt.unyt_array:
        """
        Calculate the flowrate through 3 spherical shells with radius 0.1R_SO, 0.3R_SO,
        and R_SO. Three flow types can be calculated: mass, energy, or momentum.
        Inflow and outflow rates are computed separately. If fast_outflows=True then
        fast outflows rates are also computed.
        Internal energies are required if the flow_type is 'energy' or 'momentum'.
        This function can raise a SearchRadiusTooSmallError since the spherical shell
        centered at R_SO extends beyond R_SO.
        """
        # Calculate particle radii
        radii = np.sqrt(np.sum(positions ** 2, axis=1))

        # Specify radii to calculate flow rates for
        R_fracs = [0.1, 0.3, 1]

        # Initialise a list for saving the results since
        # the units will depend on flow_type
        flow_rates = 2 * len(R_fracs) * [0]
        if fast_outflows:
            flow_rates += len(R_fracs) * [0]

        for i_R, R_frac in enumerate(R_fracs):
            # Remove particles outside spherical shell
            R = R_frac * self.SO_r
            dR = 0.1 * R
            if R + (dR / 2) > self.search_radius:
                raise SearchRadiusTooSmallError
            r_mask = (R - (dR / 2) < radii) & (radii < R + (dR / 2))

            # Use CoM velocity of all particles with R_frac * self.SO_r
            vcom = {
                0.1: self.vcom_ten_percent,
                0.3: self.vcom_thirty_percent,
                1: self.vcom,
            }[R_frac]

            # Calculate radial velocity by subtracting CoM velocity and
            # taking dot product with r_hat
            r_hat = positions[r_mask] / np.stack(3 * [radii[r_mask]], axis=1)
            v_r = np.sum((velocities[r_mask] - vcom[None, :]) * r_hat, axis=1)
            # Adding Hubble flow term
            if hubble:
                v_r += radii[r_mask] * self.cosmology["H"]
            # Account for expansion of R_SO
            if pseudo_evolve:
                G = unyt.Unit("newton_G", registry=masses.units.registry)
                R_dot = (2 / 3) * (G * self.SO_mass * self.cosmology["H"] / 100) ** (
                    1 / 3
                )
                R_dot *= (
                    2 * self.cosmology["Omega_g"] + (3 / 2) * self.cosmology["Omega_m"]
                )
                R_dot *= R_frac
                v_r -= R_dot

            # Calculate different flow types
            # We want both the inflow and outflow rates to be positive values
            if flow_type == "mass":
                flow_rate = masses[r_mask] * np.abs(v_r)
            elif flow_type == "energy":
                # Subtract CoM velocity
                proper_vel = velocities[r_mask] - vcom[None, :]
                kinetic = 0.5 * np.sqrt(np.sum(proper_vel ** 2, axis=1)) ** 2
                flow_rate = (
                    masses[r_mask] * np.abs(v_r) * (kinetic + internal_energies[r_mask])
                )
            elif flow_type == "momentum":
                # Calculate sound speed squared assuming gamma = 5/3
                gamma = 5.0 / 3.0
                sq_sound_speed = (gamma - 1) * gamma * internal_energies[r_mask]
                # Calculate momentum flux, second term accounts for pressure
                flow_rate = masses[r_mask] * (v_r ** 2 + (sq_sound_speed / gamma))

            # Determine total outflow/inflow rates
            inflow = np.sum(flow_rate[v_r < 0]) / dR
            outflow = np.sum(flow_rate[v_r > 0]) / dR
            flow_rates[i_R] = inflow
            flow_rates[i_R + len(R_fracs)] = outflow

            # Save fast outflow rates
            if fast_outflows:
                fast_mask = v_r > 0.25 * self.Vmax_soft
                fast_outflow = np.sum(flow_rate[fast_mask]) / dR
                flow_rates[i_R + 2 * len(R_fracs)] = fast_outflow

        return unyt.unyt_array(flow_rates)

    @lazy_property
    def DarkMatterMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of dark matter through 3 spherical shells
        """
        if (self.Ndm == 0) or (not self.virial_definition):
            return None

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        mass = self.get_dataset("PartType1/Masses")
        pos = self.get_dataset(f"PartType1/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType1/Velocities")

        return self.calculate_flow_rate("mass", pos, mass, vel)

    @lazy_property
    def StellarMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of stars through 3 spherical shells
        """
        if (self.Nstar == 0) or (not self.virial_definition):
            return None

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        mass = self.get_dataset("PartType4/Masses")
        pos = self.get_dataset(f"PartType4/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType4/Velocities")

        return self.calculate_flow_rate("mass", pos, mass, vel)

    @lazy_property
    def HIMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of HI through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        pos = self.get_dataset(f"PartType0/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType0/Velocities")
        i_H = self.snapshot_datasets.get_column_index(
            "ElementMassFractions", "Hydrogen"
        )
        i_HI = self.snapshot_datasets.get_column_index("SpeciesFractions", "HI")
        mass = (
            self.get_dataset("PartType0/Masses")
            * self.get_dataset("PartType0/ElementMassFractions")[:, i_H]
            * self.get_dataset("PartType0/SpeciesFractions")[:, i_HI]
        )

        return self.calculate_flow_rate("mass", pos, mass, vel)

    @lazy_property
    def H2MassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of H2 through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        pos = self.get_dataset(f"PartType0/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType0/Velocities")
        i_H = self.snapshot_datasets.get_column_index(
            "ElementMassFractions", "Hydrogen"
        )
        i_H2 = self.snapshot_datasets.get_column_index("SpeciesFractions", "H2")
        # Factor of two needed since we want mass fraction not number
        mass = (
            self.get_dataset("PartType0/Masses")
            * self.get_dataset("PartType0/ElementMassFractions")[:, i_H]
            * self.get_dataset("PartType0/SpeciesFractions")[:, i_H2]
            * 2
        )

        return self.calculate_flow_rate("mass", pos, mass, vel)

    @lazy_property
    def MetalMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of metals through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        pos = self.get_dataset(f"PartType0/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType0/Velocities")
        mass = self.get_dataset("PartType0/Masses") * self.get_dataset("PartType0/MetalMassFractions")

        return self.calculate_flow_rate("mass", pos, mass, vel)

    def _calculate_temperature_constrained_gas_flow_rate(
        self, flow_type, Tmin=None, Tmax=None
    ):
        """
        Helper function for calculating the flow rate of gas particles 
        masked based on their temperature.
        """

        # Particles outside the SO radius are required to calculate the
        # flow through the SO radius
        mass = self.get_dataset("PartType0/Masses")
        pos = self.get_dataset(f"PartType0/Coordinates") - self.centre[None, :]
        vel = self.get_dataset("PartType0/Velocities")
        temp = self.get_dataset("PartType0/Temperatures")
        internal_energies = None

        mask = temp > 0 * unyt.K  # Initialise mask with all true
        if Tmin is not None:
            mask &= temp > Tmin
        if Tmax is not None:
            mask &= temp < Tmax

        if flow_type in ["energy", "momentum"]:
            internal_energies = self.get_dataset("PartType0/InternalEnergies")[mask]

        return self.calculate_flow_rate(
            flow_type,
            pos[mask],
            mass[mask],
            vel[mask],
            fast_outflows=True,
            internal_energies=internal_energies,
        )

    @lazy_property
    def ColdGasMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of cold gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmax = 1.0e3 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate("mass", Tmax=Tmax)

    @lazy_property
    def CoolGasMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of cool gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e3 * unyt.K
        Tmax = 1.0e5 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "mass", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def WarmGasMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of warm gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e5 * unyt.K
        Tmax = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "mass", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def HotGasMassFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the mass flow rate of hot gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate("mass", Tmin=Tmin)

    @lazy_property
    def ColdGasEnergyFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the energy flow rate of cold gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmax = 1.0e3 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "energy", Tmax=Tmax
        )

    @lazy_property
    def CoolGasEnergyFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the energy flow rate of cool gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e3 * unyt.K
        Tmax = 1.0e5 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "energy", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def WarmGasEnergyFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the energy flow rate of warm gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e5 * unyt.K
        Tmax = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "energy", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def HotGasEnergyFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the energy flow rate of hot gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "energy", Tmin=Tmin
        )

    @lazy_property
    def ColdGasMomentumFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the momentum flow rate of cold gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmax = 1.0e3 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "momentum", Tmax=Tmax
        )

    @lazy_property
    def CoolGasMomentumFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the momentum flow rate of cool gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e3 * unyt.K
        Tmax = 1.0e5 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "momentum", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def WarmGasMomentumFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the momentum flow rate of warm gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e5 * unyt.K
        Tmax = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "momentum", Tmin=Tmin, Tmax=Tmax
        )

    @lazy_property
    def HotGasMomentumFlowRate(self) -> unyt.unyt_array:
        """
        Calculate the momentum flow rate of hot gas through 3 spherical shells
        """
        if (self.Ngas == 0) or (not self.virial_definition):
            return None

        Tmin = 1.0e7 * unyt.K
        return self._calculate_temperature_constrained_gas_flow_rate(
            "momentum", Tmin=Tmin
        )


class SOProperties(HaloProperty):
    """
    Compute SO properties for halos.

    Spherical overdensities have a radius that is determined by intersecting
    the density profile of all the particles in a sphere centred on the halo
    centre with a target density value, or a multiple of such a radius.

    Spherical overdensities are only calculated for central halos, not for
    sattelites.
    """

    """
    List of properties from the table that we want to compute.
    Each property should have a corresponding method/property/lazy_property in
    the SOParticleData class above.
    """
    property_list = {
        name: PropertyTable.full_property_list[name]
        for name in [
            "r",
            "Mtot",
            "Ngas",
            "Ndm",
            "Nstar",
            "Nbh",
            "Nnu",
            # Calculate inertia tensors and flow rates first as they can
            # throw a SearchRadiusTooSmallError
            "GasInertiaTensor",
            "DarkMatterInertiaTensor",
            "StellarInertiaTensor",
            "TotalInertiaTensor",
            "GasInertiaTensorReduced",
            "DarkMatterInertiaTensorReduced",
            "StellarInertiaTensorReduced",
            "TotalInertiaTensorReduced",
            "GasInertiaTensorNoniterative",
            "DarkMatterInertiaTensorNoniterative",
            "StellarInertiaTensorNoniterative",
            "TotalInertiaTensorNoniterative",
            "GasInertiaTensorReducedNoniterative",
            "DarkMatterInertiaTensorReducedNoniterative",
            "StellarInertiaTensorReducedNoniterative",
            "TotalInertiaTensorReducedNoniterative",
            "DarkMatterMassFlowRate",
            "ColdGasMassFlowRate",
            "CoolGasMassFlowRate",
            "WarmGasMassFlowRate",
            "HotGasMassFlowRate",
            "HIMassFlowRate",
            "H2MassFlowRate",
            "MetalMassFlowRate",
            "StellarMassFlowRate",
            "ColdGasEnergyFlowRate",
            "CoolGasEnergyFlowRate",
            "WarmGasEnergyFlowRate",
            "HotGasEnergyFlowRate",
            "ColdGasMomentumFlowRate",
            "CoolGasMomentumFlowRate",
            "WarmGasMomentumFlowRate",
            "HotGasMomentumFlowRate",
            "com",
            "vcom",
            "Vmax_soft",
            "R_vmax_soft",
            "Mfrac_satellites",
            "Mfrac_external",
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
            "Tgas_cy_weighted",
            "Tgas_cy_weighted_no_agn",
            "Xraylum",
            "Xraylum_restframe",
            "SpectroscopicLikeTemperature",
            "SpectroscopicLikeTemperature_no_agn",
            "Xrayphlum",
            "Xrayphlum_restframe",
            "compY",
            "Xraylum_no_agn",
            "Xrayphlum_no_agn",
            "Xraylum_restframe_no_agn",
            "Xrayphlum_restframe_no_agn",
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
            "BlackHolesTotalInjectedThermalEnergy",
            "BlackHolesTotalInjectedJetEnergy",
            "MostMassiveBlackHoleAveragedAccretionRate",
            "MostMassiveBlackHoleInjectedThermalEnergy",
            "MostMassiveBlackHoleNumberOfAGNEvents",
            "MostMassiveBlackHoleAccretionMode",
            "MostMassiveBlackHoleGWMassLoss",
            "MostMassiveBlackHoleInjectedJetEnergyByMode",
            "MostMassiveBlackHoleLastJetEventScalefactor",
            "MostMassiveBlackHoleNumberOfAGNJetEvents",
            "MostMassiveBlackHoleNumberOfMergers",
            "MostMassiveBlackHoleRadiatedEnergyByMode",
            "MostMassiveBlackHoleTotalAccretedMassesByMode",
            "MostMassiveBlackHoleWindEnergyByMode",
            "MostMassiveBlackHoleSpin",
            "MostMassiveBlackHoleTotalAccretedMass",
            "MostMassiveBlackHoleFormationScalefactor",
            "MnuNS",
            "Mnu",
            "spin_parameter",
            "SFR",
            "AveragedStarFormationRate",
            "DopplerB",
            "gasOfrac",
            "gasFefrac",
            "DtoTgas",
            "DtoTstar",
            "starOfrac",
            "starFefrac",
            "gasmetalfrac_SF",
            "concentration_unsoft",
            "concentration_soft",
            "concentration_dmo_unsoft",
            "concentration_dmo_soft",
        ]
    }

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        category_filter: CategoryFilter,
        halo_filter: str,
        SOval: float,
        type: str = "mean",
    ):
        """
        Constructor.

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology and the dataset metadata.
         - parameters: ParameterFile
           Parameter file object containing the parameters from the parameter
           file.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
         - SOval: float
           SO threshold value. The precise meaning of this parameter depends on
           the selected SO type.
         - type: str
           SO type. Possible values are "mean", "crit", "physical" or "BN98",
           for respectively a multiple of the mean density, a multiple of the
           critical density, a physical SO radius that is the multiple of another
           SO radius, or the Brian & Norman (1998) density multiple.
        """

        super().__init__(cellgrid)

        self.property_filters = parameters.get_property_filters(
            "SOProperties", [prop.name for prop in self.property_list.values()]
        )

        if not type in ["mean", "crit", "physical", "BN98"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type
        if not (hasattr(self, "core_excision_fraction")):
            self.core_excision_fraction = None
        self.filter = recently_heated_gas_filter
        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets
        self.halo_filter = halo_filter
        self.record_timings = parameters.record_property_timings
        self.observer_position = cellgrid.observer_position
        self.boxsize = cellgrid.boxsize

        self.cosmology = {}
        # in the neutrino model, the mean neutrino density is implicitly
        # assumed to be based on Omega_nu_0 and critical_density_0
        # here, critical_density_0 = critical_density * (H0/H)**2
        # however, we need to scale this to the appropriate redshift,
        # hence an additional factor 1/a**3
        self.cosmology["nu_density"] = (
            cellgrid.cosmology["Omega_nu_0"]
            * cellgrid.critical_density
            * (
                cellgrid.cosmology["H0 [internal units]"]
                / cellgrid.cosmology["H [internal units]"]
            )
            ** 2
            / cellgrid.a ** 3
        )

        # We need the following for inflow/outflow calculations
        self.cosmology["H"] = cellgrid.cosmology[
            "H [internal units]"
        ] / cellgrid.get_unit("code_time")
        self.cosmology["Omega_g"] = cellgrid.cosmology["Omega_g"]
        self.cosmology["Omega_m"] = cellgrid.cosmology["Omega_m"]

        # This specifies how large a sphere is read in:
        # we use default values that are sufficiently small/large to avoid reading in too many particles
        self.mean_density_multiple = 1000.0
        self.critical_density_multiple = 1000.0
        self.physical_radius_mpc = 0.0
        self.virial_definition = False
        if type == "mean":
            self.mean_density_multiple = SOval
            if SOval == 200:
                self.virial_definition = True
        elif type == "crit":
            self.critical_density_multiple = SOval
            if SOval == 200:
                self.virial_definition = True
        elif type == "BN98":
            self.critical_density_multiple = cellgrid.virBN98
            self.virial_definition = True
        elif type == "physical":
            self.physical_radius_mpc = 0.001 * SOval

        # Give this calculation a name so we can select it on the command line
        if type in ["mean", "crit"]:
            self.name = f"SO_{SOval:.0f}_{type}"
        elif type == "physical":
            self.name = f"SO_{SOval:.0f}_kpc"
        elif type == "BN98":
            self.name = "SO_BN98"

        # Make a string describing the excised core radius, if any.
        if self.core_excision_fraction is not None:
            self.core_excision_string = (
                f"{self.core_excision_fraction}*R{self.name[3:]}"
            )
        else:
            self.core_excision_string = None

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

        # Save name of group in the final output file
        self.group_name = f"SO/{self.SO_name}"
        self.mask_metadata = self.category_filter.get_filter_metadata(self.halo_filter)

        # Arrays which must be read in for this calculation.
        # Note that if there are no particles of a given type in the
        # snapshot, that type will not be read in and will not have
        # an entry in the data argument to calculate(), below.
        # (E.g. gas, star or BH particles in DMO runs)
        self.particle_properties = {
            "PartType0": [
                "Coordinates",
                "FOFGroupIDs",
                "GroupNr_bound",
                "Masses",
                "Velocities",
            ],
            "PartType1": [
                "Coordinates",
                "FOFGroupIDs",
                "GroupNr_bound",
                "Masses",
                "Velocities",
            ],
            "PartType4": [
                "Coordinates",
                "FOFGroupIDs",
                "GroupNr_bound",
                "Masses",
                "Velocities",
            ],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "FOFGroupIDs",
                "GroupNr_bound",
                "Velocities",
            ],
            "PartType6": ["Coordinates", "Masses", "Weights"],
        }
        # add additional particle properties based on the selected halo
        # properties in the parameter file
        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            if not self.property_filters[outputname]:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            partprops = prop.particle_properties
            for partprop in partprops:
                pgroup, dset = parameters.get_particle_property(partprop)
                if not pgroup in self.particle_properties:
                    self.particle_properties[pgroup] = []
                if not dset in self.particle_properties[pgroup]:
                    self.particle_properties[pgroup].append(dset)

    def calculate(
        self,
        input_halo: Dict,
        search_radius: unyt.unyt_quantity,
        data: Dict,
        halo_result: Dict,
    ):
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

        SO = {}
        timings = {}

        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            filter_name = self.property_filters[outputname]
            if not filter_name:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            shape = prop.shape
            dtype = prop.dtype
            unit = unyt.Unit(prop.unit, registry=registry)
            physical = prop.output_physical
            a_exponent = prop.a_scale_exponent
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            if not physical:
                unit = unit * unyt.Unit("a", registry=registry) ** a_exponent
            SO[name] = unyt.unyt_array(val, dtype=dtype, units=unit, registry=registry)

        # Get do_calculation to determine whether to skip halo
        do_calculation = self.category_filter.get_do_calculation(halo_result)

        # SOs only exist for central galaxies
        # Determine whether to skip this halo because of filter
        if input_halo["is_central"] and do_calculation[self.halo_filter]:
            types_present = [type for type in self.particle_properties if type in data]

            part_props = SOParticleData(
                input_halo,
                data,
                types_present,
                self.filter,
                self.observer_position,
                self.snapshot_datasets,
                self.core_excision_fraction,
                self.softening_of_parttype,
                self.virial_definition,
                search_radius,
                self.cosmology,
                self.boxsize,
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
            except SearchRadiusTooSmallError:
                raise SearchRadiusTooSmallError(
                    f"Need more particles to determine SO radius and mass!"
                )

            if SO_exists:
                for name, prop in self.property_list.items():
                    outputname = prop.name
                    # Skip if this property is disabled in the parameter file
                    filter_name = self.property_filters[outputname]
                    if not filter_name:
                        continue
                    # Skip non-DMO properties when in DMO run mode
                    if self.category_filter.dmo and not prop.dmo_property:
                        continue
                    dtype = prop.dtype
                    unit = unyt.Unit(prop.unit, registry=registry)
                    physical = prop.output_physical
                    a_exponent = prop.a_scale_exponent
                    if not physical:
                        unit = unit * unyt.Unit("a", registry=registry) ** a_exponent
                    if do_calculation[filter_name]:
                        t0_calc = time.time()
                        val = getattr(part_props, name)
                        if val is not None:
                            assert (
                                SO[name].shape == val.shape
                            ), f"Attempting to store {name} with wrong dimensions"
                            if unit == unyt.Unit("dimensionless"):
                                if hasattr(val, "units"):
                                    assert (
                                        val.units == unyt.dimensionless
                                    ), f"{name} is not dimensionless"
                                SO[name] = unyt.unyt_array(
                                    val.astype(dtype),
                                    dtype=dtype,
                                    units=unit,
                                    registry=registry,
                                )
                            else:
                                err = f'Overflow for halo {input_halo["index"]} when'
                                err += f"calculating {name} in SO_properties"
                                assert np.max(np.abs(val.to(unit).value)) < float(
                                    "inf"
                                ), err
                                SO[name] += val
                            timings[name] = time.time() - t0_calc

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            if not self.property_filters[outputname]:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            halo_result.update(
                {
                    f"{self.group_name}/{outputname}": (
                        SO[name],
                        prop.description.format(
                            label=self.label, core_excision=self.core_excision_string
                        ),
                        prop.output_physical,
                        prop.a_scale_exponent,
                    )
                }
            )
            if self.record_timings:
                arr = unyt.unyt_array(
                        timings.get(name, 0),
                        dtype=np.float32,
                        units=unyt.dimensionless,
                        registry=registry,
                    )
                halo_result.update(
                    {
                        f"{self.group_name}/{outputname}_time": (
                            arr,
                            'Time taken in seconds',
                            True,
                            None,
                        )
                    }
                )

        return


class CoreExcisedSOProperties(SOProperties):
    # Add the extra core excised properties we want from the table
    property_list = SOProperties.property_list | {
        name: PropertyTable.full_property_list[name]
        for name in [
            "Tgas_core_excision",
            "Tgas_no_cool_core_excision",
            "Tgas_no_agn_core_excision",
            "Tgas_no_cool_no_agn_core_excision",
            "Tgas_cy_weighted_core_excision",
            "Tgas_cy_weighted_core_excision_no_agn",
            "SpectroscopicLikeTemperature_core_excision",
            "SpectroscopicLikeTemperature_no_agn_core_excision",
            "Xraylum_core_excision",
            "Xraylum_no_agn_core_excision",
            "Xrayphlum_core_excision",
            "Xrayphlum_no_agn_core_excision",
            "Xraylum_restframe_core_excision",
            "Xraylum_restframe_no_agn_core_excision",
            "Xrayphlum_restframe_core_excision",
            "Xrayphlum_restframe_no_agn_core_excision",
        ]
    }

    def __init__(
        self,
        cellgrid,
        parameters: ParameterFile,
        recently_heated_gas_filter,
        category_filter,
        halo_filter: str,
        SOval,
        type="mean",
        core_excision_fraction=None,
    ):
        # Store the excision fraction
        self.core_excision_fraction = core_excision_fraction

        # initialise the SOProperties object
        super().__init__(
            cellgrid,
            parameters,
            recently_heated_gas_filter,
            category_filter,
            halo_filter,
            SOval,
            type,
        )


class RadiusMultipleSOProperties(SOProperties):
    """
    SOProperties specialisation for SOs that use a multiple of another SO radius
    as aperture radius.

    This class is responsible for retrieving the corresponding radius from
    the other SO result provided that exists.
    """

    # since the halo_result dictionary contains the name of the dataset as it
    # appears in the output, we have to get that name from the property table
    # to access the radius
    radius_name = PropertyTable.full_property_list["r"].name

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        category_filter: CategoryFilter,
        halo_filter: str,
        SOval: float,
        multiple: float,
        type: str = "mean",
    ):
        """
        Constructor.

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology and the dataset metadata.
         - parameters: ParameterFile
           Parameter file object containing the parameters from the parameter
           file.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
         - SOval: float
           SO threshold value. The precise meaning of this parameter depends on
           the selected SO type. Note that this value determines the "parent"
           SO, i.e. the SO whose radius is used to determine the actual radius
           of this SO.
         - multiple: float
           Radius multiple to use. The actual radius is determined by multiplying
           the normal SO determined by the type and SOval with this value.
         - type: str
           SO type. Possible values are "mean" or "crit", for respectively a multiple
           of the mean density or a multiple of the critical density.
        """
        if not type in ["mean", "crit"]:
            raise AttributeError(
                "SOs with a radius that is a multiple of another SO radius are only allowed for type mean or crit!"
            )

        # initialise the SOProperties object
        super().__init__(
            cellgrid,
            parameters,
            recently_heated_gas_filter,
            category_filter,
            halo_filter,
            0,
            "physical",
        )

        # overwrite the name, SO_name and label
        self.SO_name = f"{multiple:.0f}xR_{SOval:.0f}_{type}"
        self.group_name = f"SO/{self.SO_name}"
        self.label = f"with a radius that is {self.SO_name}"
        self.name = f"SO_{self.SO_name}"

        self.requested_type = type
        self.requested_SOval = SOval
        self.multiple = multiple

    def calculate(
        self,
        input_halo: Dict,
        search_radius: unyt.unyt_quantity,
        data: Dict,
        halo_result: Dict,
    ):
        """
        Calculate the properties of an SO of which the radius is the multiple of
        another SO radius.

        Parameters:
         - input_halo: Dict
           Properties of the VR subhalo, as they are read from the catalogue.
         - search_radius: unyt.unyt_quantity
           Current search radius. Particles are guaranteed to be included up to
           this radius.
         - data: Dict
           Particle data as it is read from the snapshot.
         - halo_result: Dict
           Dictionary in which halo properties for this halo are stored. Should
           contain a valid result for the "parent" SO, i.e. the SO that determines
           the radius of this SO.

        Throws a RuntimeError if the "parent" SO radius cannot be obtained from
        halo_result.

        Throws a SearchRadiusTooSmallError if the current search radius was too small
        to guarantee a correct result.
        """

        # find the actual physical radius we want
        key = f"SO/{self.requested_SOval:.0f}_{self.requested_type}/{self.radius_name}"
        if not key in halo_result:
            raise RuntimeError(
                f"Trying to obtain {key}, but the corresponding SO radius has not been calculated!"
            )
        self.physical_radius_mpc = self.multiple * (halo_result[key][0].to("Mpc").value)

        # Check that we read in a large enough radius
        if self.multiple * halo_result[key][0] > search_radius:
            raise SearchRadiusTooSmallError(
                "SO radius multiple estimate was too small!"
            )

        super().calculate(input_halo, search_radius, data, halo_result)
        return


def test_SO_properties_random_halo():
    """
    Unit test for the SO property calculation.

    We generate 100 random halos and check that the various SO halo
    calculations return the expected results and do not lead to any
    errors.
    """
    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(4251)
    gas_filter = dummy_halos.get_recently_heated_gas_filter()
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 100}))
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
                "PartType0/XrayLuminositiesRestframe": "PartType0/XrayLuminositiesRestframe",
                "PartType0/XrayPhotonLuminositiesRestframe": "PartType0/XrayPhotonLuminositiesRestframe",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
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
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        50.0,
        "physical",
    )
    property_calculator_2500mean = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        "mean",
    )
    property_calculator_2500crit = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        "crit",
    )
    property_calculator_BN98 = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        0.0,
        "BN98",
    )
    property_calculator_5x2500mean = RadiusMultipleSOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        2500.0,
        5.0,
        "mean",
    )

    # Create a filter that no halos will satisfy
    fail_filter = CategoryFilter(dummy_halos.get_filters({"general": 10000000}))
    property_calculator_filter_test = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        fail_filter,
        "general",
        200.0,
        "crit",
    )
    property_calculator_filter_test.SO_name = "filter_test"
    property_calculator_filter_test.group_name = "SO/filter_test"

    for i in range(100):
        (
            input_halo,
            data,
            rmax,
            Mtot,
            Npart,
            particle_numbers,
        ) = dummy_halos.get_random_halo([2, 10, 100, 1000, 10000], has_neutrinos=True)
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax ** 3)

        # force the SO radius to be outside the search sphere and check that
        # we get a SearchRadiusTooSmallError
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
            except SearchRadiusTooSmallError:
                fail = True
            # 1 particle halos don't fail, since we always assume that the first
            # particle is at the centre of potential (which means we exclude it
            # in the SO calculation)
            # non-centrals don't fail, since we do not calculate any SO
            # properties and simply return zeros in this case

            # TODO: This can fail due to how we calculate the SO if the
            # first particle is a neutrino with negative mass. In that case
            # we linearly interpolate the mass of the first non-negative particle
            # outwards.
            # TODO
            # assert (Npart == 1) or input_halo["is_central"] == 0 or fail

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
        except SearchRadiusTooSmallError:
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
            ("filter_test", property_calculator_filter_test),
        ]:
            halo_result = dict(halo_result_template)
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            # TODO: remove this
            # Adding Restframe luminosties as they are calculated in halo_tasks
            if "PartType0" in input_data:
                for dset in [
                    "XrayLuminositiesRestframe",
                    "XrayPhotonLuminositiesRestframe",
                ]:
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                halo_result[
                    f"SO/2500_mean/{property_calculator_5x2500mean.radius_name}"
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            # TODO: SearchRadiusTooSmallError
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list.values():
                outputname = prop.name
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                assert result.units.same_dimensions_as(unit.units)

            # Check properties were not calculated for filtered halos
            if SO_name == "filter_test":
                for prop in prop_calc.property_list.values():
                    outputname = prop.name
                    size = prop.shape
                    full_name = f"SO/{SO_name}/{outputname}"
                    assert np.all(halo_result[full_name][0].value == np.zeros(size))

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
            gas_filter,
            cat_filter,
            "basic",
            50.0,
            "physical",
        )
        property_calculator_2500mean = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            "mean",
        )
        property_calculator_2500crit = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            "crit",
        )
        property_calculator_BN98 = SOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            0.0,
            "BN98",
        )
        property_calculator_5x2500mean = RadiusMultipleSOProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            gas_filter,
            cat_filter,
            "basic",
            2500.0,
            5.0,
            "mean",
        )

        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax ** 3)

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
                ] = (0.1 * rmax, "Dummy value to force correct behaviour")
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            # Adding Restframe luminosties as they are calculated in halo_tasks
            if "PartType0" in input_data:
                for dset in [
                    "XrayLuminositiesRestframe",
                    "XrayPhotonLuminositiesRestframe",
                ]:
                    input_data["PartType0"][dset] = data["PartType0"][dset]
                    input_data["PartType0"][dset] = data["PartType0"][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for prop in prop_calc.property_list.values():
                outputname = prop.name
                if not outputname == property:
                    continue
                size = prop.shape
                dtype = prop.dtype
                unit_string = prop.unit
                physical = prop.output_physical
                a_exponent = prop.a_scale_exponent
                full_name = f"SO/{SO_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                if not physical:
                    unit = (
                        unit
                        * unyt.Unit("a", registry=dummy_halos.unit_registry)
                        ** a_exponent
                    )
                assert result.units == unit.units

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


def calculate_SO_properties_nfw_halo(seed, num_part, c):
    """
    Generates a halo with an NFW profile, and calculates SO properties for it
    """
    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(seed)
    gas_filter = dummy_halos.get_recently_heated_gas_filter()
    cat_filter = CategoryFilter(dummy_halos.get_filters({"general": 100}))
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
                "PartType0/XrayLuminositiesRestframe": "PartType0/XrayLuminositiesRestframe",
                "PartType0/XrayPhotonLuminositiesRestframe": "PartType0/XrayPhotonLuminositiesRestframe",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
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

    property_calculator_200crit = SOProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        gas_filter,
        cat_filter,
        "basic",
        200.0,
        "crit",
    )

    (input_halo, data, rmax, Mtot, Npart, particle_numbers) = dummy_halos.gen_nfw_halo(
        100, c, num_part
    )

    halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

    property_calculator_200crit.cosmology["nu_density"] *= 0
    property_calculator_200crit.calculate(input_halo, rmax, data, halo_result_template)

    return halo_result_template


def test_concentration_nfw_halo():
    """
    Test if the calculated concentration is close to the input value.
    Only tests halos with for 10000 particles.
    Fails due to noise for small particle numbers.
    """
    n_part = 10000
    for seed in range(10):
        for concentration in [5, 10]:
            halo_result = calculate_SO_properties_nfw_halo(seed, n_part, concentration)
            calculated = halo_result["SO/200_crit/Concentration"][0]
            delta = np.abs(calculated - concentration) / concentration
            assert delta < 0.1


if __name__ == "__main__":
    """
    Standalone mode for running tests.
    """
    print("Calling test_SO_properties_random_halo()...")
    test_SO_properties_random_halo()
    print("Calling test_concentration_nfw_halo()...")
    test_concentration_nfw_halo()
    print("Tests passed.")
