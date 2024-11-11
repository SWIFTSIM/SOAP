#! /usr/bin/env python

"""
aperture_properties.py

Halo properties within 3D apertures. These include either all the particles
(inclusive) or all the gravitionally bound particles (exclusive) of a subhalo,
within a fixed physical radius.

Just like the other HaloProperty implementations, the calculation of the
properties is done lazily: only calculations that are actually needed are
performed. To achieve this, we use a somewhat weird coding pattern: the
halo property calculations correspond to methods of an ApertureParticleData
object, decorated with the 'lazy_property' decorator. Consider the following
naive calculation of the stellar mass and stellar metal mass fraction:

  radius = data["PartType4/Radius"] # (this dataset does not actually exist)
  aperture_mask = radius < aperture_radius
  star_mass = data["PartType4/Masses"][aperture_mask]
  Mstar = star_mass.sum()
  metal_frac = data["PartType4/MetalMassFractions"][aperture_mask]
  star_metal_mass = (star_mass * metal_frac).sum()
  MetalFracStar = star_metal_mass / Mstar

In this code excerpt, every line corresponds to a new variable that will be
computed. The stellar mass and aperture mask are used multiple times. So far,
everything is fine. Problems arise however if we want to disable the calculation
of for example the stellar mass, based on some flag. We could write

  radius = data["PartType4/Radius"]
  aperture_mask = radius < aperture_radius
  if flag:
    star_mass = data["PartType4/Masses"][aperture_mask]
    Mstar = star_mass.sum()
  metal_frac = data["PartType4/MetalMassFractions"][aperture_mask]
  star_metal_mass = (star_mass * metal_frac).sum()
  MetalFracStar = star_metal_mass / Mstar

but this is obviously wrong, since we still need 'star_mass' and 'Mstar' to
compute the metal mass fraction. In a lot of cases, these dependencies are
not that clear, and it becomes very tricky to figure out how to disable some
properties without breaking other property calculations. It is possible, but
it is painful to do and very prone to mistakes.

Instead of figuring out all the depencies, we can instead use this:

  class PropertyCalculations:
    def __init__(self, data):
      self.data = data

    @lazy_property
    def aperture_mask(self):
      radius = self.data["PartType4/Radius"]
      return radius < aperture_radius

    @lazy_property
    def star_mass(self):
      return self.data["PartType4/Masses"][self.aperture_mask]

    @lazy_property
    def Mstar(self):
      return self.star_mass.sum()

    @lazy_property
    def star_metal_mass(self):
      metal_frac = self.data["PartType4/MetalMassFractions"][self.aperture_mask]
      return (self.star_mass * metal_frac).sum()

    @lazy_property
    def MetalFracStar(self):
      return self.star_metal_mass / self.Mstar

This looks the same as the previous code excerpt, but then a lot more
complicated. The key difference is that all of these methods are 'lazy', which
means they only get evaluated when they are actually used. The advantage becomes
clear when we consider the various scenarios:

1. We want to compute Mstar, but not MetalFracStar:
 - we call Mstar()
 - Mstar() has not been called before, so it is run
 - Mstar() calls star_mass()
 - star_mass() has not been called before, so it is run
 - star_mass() calls aperture_mask()
 - aperture_mask() has not been called before, so it is run
 - done.

2. We want to compute MetalFracStar, but not Mstar:
 - we call MetalFracStar()
 - MetalFracStar() has not been called before, so it is run
 - MetalFracStar() calls star_metal_mass() and Mstar()
 - star_metal_mass() has not been called before, so it is run
 - star_metal_mass() calls aperture_mask() and star_mass()
 - aperture_mask() has not been called before, so it is run
 - star_mass() has not been called before, so it is run
 - star_mass() calls aperture_mask(), but that has already run
 - Mstar() calls star_mass(), but that has already run
 - done.

3. We want to compute both Mstar and MetalFracStar:
 - we call Mstar()
 - Mstar() has not been called before, so it is run
 - Mstar() calls star_mass()
 - star_mass() has not been called before, so it is run
 - star_mass() calls aperture_mask()
 - aperture_mask() has not been called before, so it is run
 - we call MetalFracStar()
 - MetalFracStar() has not been called before, so it is run
 - MetalFracStar() calls star_metal_mass() and Mstar(), but that has already
    run
 - star_metal_mass() has not been called before, so it is run
 - star_metal_mass() calls aperture_mask() and star_mass(), both have already
   run
 - done.

Depending on what we want to calculate, we get a different order in which
variables are calculated (and methods are called), but only the variables that
are actually used are calculated. This way to evaluate methods when they are
needed dynamically adapts to the particular situation, without the need to
figure out the dependencies yourself.

In the HaloProperty implementation, we need at least one method for every
halo property in the table (property_table.py) that we want to compute. But that
does not eliminate the overhead of auxiliary variables (like aperture_mask) that
are needed by multiple properties. To make this lazy evaluation work, you
therefore need to determine which variables are used multiple times, and which
variables are not and can hence stay local to a particular lazy method. There is
still some decision making needed there.

On top of that, we also need to deal with borderline cases, like computing the
stellar mass for halos with no star particles. These need to be dealt with in
each lazy method separately, because you cannot/should not expect that a lazy
method will never be called in that case. That is why the implementation looks
very messy and complex. But it is in fact quite neat and powerful.
"""

import numpy as np
import unyt

from halo_properties import HaloProperty, SearchRadiusTooSmallError
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
)

from swift_cells import SWIFTCellGrid
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from stellar_age_calculator import StellarAgeCalculator
from cold_dense_gas_filter import ColdDenseGasFilter
from property_table import PropertyTable
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile
from snapshot_datasets import SnapshotDatasets
from typing import Dict, List, Tuple
from numpy.typing import NDArray


class ApertureParticleData:
    """
    Halo calculation class.

    All properties we want to compute in apertures are implemented as lazy
    methods of this class.

    Note that this class internally uses and requires two different masks:
     - *_mask_all: Mask that masks out particles belonging to this halo: either
         only gravitationally bound particles (exclusive apertures) or all
         particles (no mask -- inclusive apertures). This mask needs to be
         applied _first_ to raw "PartTypeX" datasets.
     - *_mask_ap: Mask that masks out particles that are inside the aperture
         radius. This mask can only be applied after *_mask_all has been applied.
    compute_basics() furthermore defines some arrays that contain variables
    (e.g. masses, positions) for all particles that belong to the halo (so
    after applying *_mask_all, but before applying *_mask_ap). To retrieve the
    variables for a single particle type, these have to be masked with
    "PartTypeX == 'type'".
    All of these masks have different lengths, so using the wrong mask will
    lead to errors. Those are captured by the unit tests, so make sure to run
    those after you implement a new property!
    """

    def __init__(
        self,
        input_halo: Dict,
        data: Dict,
        types_present: List[str],
        inclusive: bool,
        aperture_radius: unyt.unyt_quantity,
        stellar_age_calculator: StellarAgeCalculator,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        cold_dense_gas_filter: ColdDenseGasFilter,
        snapshot_datasets: SnapshotDatasets,
        softening_of_parttype: unyt.unyt_array,
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
         - inclusive: bool
           Whether or not to include particles not gravitationally bound to the subhalo
           in the property calculations.
         - aperture_radius: unyt.unyt_quantity
           Aperture radius.
         - stellar_age_calculator: StellarAgeCalculator
           Object used to compute stellar ages from the current cosmological scale factor
           and the birth scale factors of star particles.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - cold_dense_gas_filter: ColdDenseGasFilter
           Filter used to mask out gas particles containing cold, dense gas.
         - snapshot_datasets: SnapshotDatasets
           Object containing metadata about the datasets in the snapshot, like
           appropriate aliases and column names.
         - softening_of_parttype: unyt.unyt_array
           Softening length of each particle types
        """
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.inclusive = inclusive
        self.aperture_radius = aperture_radius
        self.stellar_age_calculator = stellar_age_calculator
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.cold_dense_gas_filter = cold_dense_gas_filter
        self.snapshot_datasets = snapshot_datasets
        self.softening_of_parttype = softening_of_parttype
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
        mass = []
        position = []
        radius = []
        velocity = []
        types = []
        softening = []
        for ptype in self.types_present:
            grnr = self.get_dataset(f"{ptype}/GroupNr_bound")
            if self.inclusive:
                in_halo = np.ones(grnr.shape, dtype=bool)
            else:
                in_halo = grnr == self.index
            mass.append(self.get_dataset(f"{ptype}/{mass_dataset(ptype)}")[in_halo])
            pos = (
                self.get_dataset(f"{ptype}/Coordinates")[in_halo, :]
                - self.centre[None, :]
            )
            position.append(pos)
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius.append(r)
            velocity.append(self.get_dataset(f"{ptype}/Velocities")[in_halo, :])
            typearr = int(ptype[-1]) * np.ones(r.shape, dtype=np.int32)
            types.append(typearr)
            s = np.ones(r.shape, dtype=np.float64) * self.softening_of_parttype[ptype]
            softening.append(s)

        self.mass = np.concatenate(mass)
        self.position = np.concatenate(position)
        self.radius = np.concatenate(radius)
        self.velocity = np.concatenate(velocity)
        self.types = np.concatenate(types)
        self.softening = np.concatenate(softening)

        self.mask = self.radius <= self.aperture_radius

        self.mass = self.mass[self.mask]
        self.position = self.position[self.mask]
        self.velocity = self.velocity[self.mask]
        self.radius = self.radius[self.mask]
        self.type = self.types[self.mask]
        self.softening = self.softening[self.mask]

    @lazy_property
    def gas_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out gas particles that are inside the aperture radius.
        This mask can be used on arrays of all gas particles that are included
        in the calculation (so either the raw "PartType0" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.mask[self.types == 0]

    @lazy_property
    def dm_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out DM particles that are inside the aperture radius.
        This mask can be used on arrays of all DM particles that are included
        in the calculation (so either the raw "PartType1" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.mask[self.types == 1]

    @lazy_property
    def star_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out star particles that are inside the aperture radius.
        This mask can be used on arrays of all star particles that are included
        in the calculation (so either the raw "PartType4" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.mask[self.types == 4]

    @lazy_property
    def bh_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out BH particles that are inside the aperture radius.
        This mask can be used on arrays of all BH particles that are included
        in the calculation (so either the raw "PartType5" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.mask[self.types == 5]

    @lazy_property
    def baryon_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out baryon particles that are inside the aperture radius.
        This mask can be used on arrays of all baryon particles that are included
        in the calculation. Note that baryons are gas and star particles,
        so "PartType0" and "PartType4".
        """
        return self.mask[(self.types == 0) | (self.types == 4)]

    @lazy_property
    def Ngas(self) -> int:
        """
        Number of gas particles in the aperture.
        """
        return self.gas_mask_ap.sum()

    @lazy_property
    def Ndm(self) -> int:
        """
        Number of DM particles in the aperture.
        """
        return self.dm_mask_ap.sum()

    @lazy_property
    def Nstar(self) -> int:
        """
        Number of star particles in the aperture.
        """
        return self.star_mask_ap.sum()

    @lazy_property
    def Nbh(self) -> int:
        """
        Number of BH particles in the aperture.
        """
        return self.bh_mask_ap.sum()

    @lazy_property
    def Nbaryon(self) -> int:
        """
        Number of baryon particles in the aperture.
        """
        return self.baryon_mask_ap.sum()

    @lazy_property
    def mass_gas(self) -> unyt.unyt_array:
        """
        Mass of the gas particles.
        """
        return self.mass[self.type == 0]

    @lazy_property
    def mass_dm(self) -> unyt.unyt_array:
        """
        Mass of the DM particles.
        """
        return self.mass[self.type == 1]

    @lazy_property
    def mass_star(self) -> unyt.unyt_array:
        """
        Mass of the star particles.
        """
        return self.mass[self.type == 4]

    @lazy_property
    def mass_baryons(self) -> unyt.unyt_array:
        """
        Mass of the baryon particles (gas + stars).
        """
        return self.mass[(self.type == 0) | (self.type == 4)]

    @lazy_property
    def pos_gas(self) -> unyt.unyt_array:
        """
        Position of the gas particles.
        """
        return self.position[self.type == 0]

    @lazy_property
    def pos_dm(self) -> unyt.unyt_array:
        """
        Position of the DM particles.
        """
        return self.position[self.type == 1]

    @lazy_property
    def pos_star(self) -> unyt.unyt_array:
        """
        Position of the star particles.
        """
        return self.position[self.type == 4]

    @lazy_property
    def pos_baryons(self) -> unyt.unyt_array:
        """
        Position of the baryon (gas+stars) particles.
        """
        return self.position[(self.type == 0) | (self.type == 4)]

    @lazy_property
    def vel_gas(self) -> unyt.unyt_array:
        """
        Velocity of the gas particles.
        """
        return self.velocity[self.type == 0]

    @lazy_property
    def vel_dm(self) -> unyt.unyt_array:
        """
        Velocity of the DM particles.
        """
        return self.velocity[self.type == 1]

    @lazy_property
    def vel_star(self) -> unyt.unyt_array:
        """
        Velocity of the star particles.
        """
        return self.velocity[self.type == 4]

    @lazy_property
    def vel_baryons(self) -> unyt.unyt_array:
        """
        Velocity of the baryon (gas+star) particles.
        """
        return self.velocity[(self.type == 0) | (self.type == 4)]

    @lazy_property
    def Mtot(self) -> unyt.unyt_quantity:
        """
        Total mass of all particles.
        """
        return self.mass.sum()

    @lazy_property
    def Mgas(self) -> unyt.unyt_quantity:
        """
        Total mass of gas particles.
        """
        return self.mass_gas.sum()

    @lazy_property
    def Mdm(self) -> unyt.unyt_quantity:
        """
        Total mass of DM particles.
        """
        return self.mass_dm.sum()

    @lazy_property
    def Mstar(self) -> unyt.unyt_quantity:
        """
        Total mass of star particles.
        """
        return self.mass_star.sum()
    
    @lazy_property
    def ChabrierInferredMstar(self) -> unyt.unyt_quantity:
        """
        Total mass of star particles, modified for a Chabrier IMF.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/ChabrierMasses")[self.star_mask_all][
            self.star_mask_ap
        ].sum()

    @lazy_property
    def Mbh_dynamical(self) -> unyt.unyt_quantity:
        """
        Total dynamical mass of BH particles.
        """
        return self.mass[self.type == 5].sum()

    @lazy_property
    def Mbaryons(self) -> unyt.unyt_quantity:
        """
        Total mass of baryon (gas+star) particles.
        """
        return self.Mgas + self.Mstar

    @lazy_property
    def star_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out star particles in raw PartType4 arrays.
        This is the mask that masks out unbound particles for exclusive halos.
        For inclusive halos, this mask does nothing.
        """
        if self.Nstar == 0:
            return None
        groupnr_bound = self.get_dataset("PartType4/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def Mstar_init(self) -> unyt.unyt_quantity:
        """
        Total initial mass of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/InitialMasses")[self.star_mask_all][
            self.star_mask_ap
        ].sum()

    @lazy_property
    def stellar_luminosities(self) -> unyt.unyt_array:
        """
        Stellar luminosities.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/Luminosities")[self.star_mask_all][
            self.star_mask_ap
        ]

    @lazy_property
    def StellarLuminosity(self) -> unyt.unyt_array:
        """
        Total luminosity of star particles.

        Note that this returns an array with total luminosities in multiple
        bands.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_luminosities.sum(axis=0)
    
    @lazy_property
    def CorrectedStellarLuminosity(self) -> unyt.unyt_array:
        """
        Total IMF-corrected luminosity of star particles.

        Note that this returns an array with total luminosities in multiple
        bands.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/CorrectedLuminosities")[self.star_mask_all][
            self.star_mask_ap].sum(axis=0)
    
    @lazy_property
    def CorrectedStellarLuminosityWithDust(self) -> unyt.unyt_array:
        """
        Total IMF-corrected luminosity of star particles, dust attenuation included.

        Note that this returns an array with total luminosities in multiple
        bands.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/CorrectedLuminositiesWithDust")[self.star_mask_all][
            self.star_mask_ap].sum(axis=0)

    @lazy_property
    def starmetalfrac(self) -> unyt.unyt_quantity:
        """
        Total metal mass fraction of star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.mass_star
            * self.get_dataset("PartType4/MetalMassFractions")[self.star_mask_all][
                self.star_mask_ap
            ]
        ).sum() / self.Mstar

    @lazy_property
    def star_element_fractions(self) -> unyt.unyt_array:
        """
        Element mass fractions of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/ElementMassFractions")[self.star_mask_all][
            self.star_mask_ap
        ]

    @lazy_property
    def star_mass_O(self) -> unyt.unyt_array:
        """
        Oxygen masses of star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
            * self.mass_star
        )

    @lazy_property
    def star_mass_Mg(self) -> unyt.unyt_array:
        """
        Magnesium mass fractions of star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Magnesium"
                ),
            ]
            * self.mass_star
        )

    @lazy_property
    def star_mass_Fe(self) -> unyt.unyt_array:
        """
        Iron mass fractions of star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron"),
            ]
            * self.mass_star
        )

    @lazy_property
    def starOfrac(self) -> unyt.unyt_quantity:
        """
        Total oxygen mass fraction of star particles.
        """
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_O.sum() / self.Mstar

    @lazy_property
    def starMgfrac(self) -> unyt.unyt_quantity:
        """
        Total magnesium mass fraction of star particles.
        """
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_Mg.sum() / self.Mstar

    @lazy_property
    def starFefrac(self) -> unyt.unyt_quantity:
        """
        Total iron mass fraction of star particles.
        """
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_Fe.sum() / self.Mstar

    @lazy_property
    def stellar_ages(self) -> unyt.unyt_array:
        """
        Ages of star particles.

        Note that these are computed from the birth scale factor using the
        provided StellarAgeCalculator (which uses the correct cosmology and
        snapshot redshift).
        """
        if self.Nstar == 0:
            return None
        birth_a = self.get_dataset("PartType4/BirthScaleFactors")[self.star_mask_all][
            self.star_mask_ap
        ]
        return self.stellar_age_calculator.stellar_age(birth_a)

    @lazy_property
    def star_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fraction of each star particle.

        Used to avoid numerical overflow in calculations like
          com = (mass_star * pos_star).sum() / Mstar
        by rewriting it as
          com = ((mass_star / Mstar) * pos_star).sum()
              = (star_mass_fraction * pos_star).sum()
        This is more accurate, since the stellar mass fractions are numbers
        of the order of 1e-5 or so, while the masses themselves can be much
        larger, if expressed in the wrong units (and that is up to unyt).
        """
        if self.Mstar == 0:
            return None
        return self.mass_star / self.Mstar

    @lazy_property
    def stellar_age_mw(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average stellar age.
        """
        if self.Nstar == 0 or self.Mstar == 0:
            return None
        return (self.star_mass_fraction * self.stellar_ages).sum()

    @lazy_property
    def stellar_age_lw(self) -> unyt.unyt_quantity:
        """
        (r-band) Luminosity-weighted average stellar age.
        """
        if self.Nstar == 0:
            return None
        Lr = self.stellar_luminosities[
            :, self.snapshot_datasets.get_column_index("Luminosities", "GAMA_r")
        ]
        Lrtot = Lr.sum()
        if Lrtot == 0:
            return None
        return ((Lr / Lrtot) * self.stellar_ages).sum()
    
    @lazy_property
    def stellar_age_uvlw(self) -> unyt.unyt_quantity:
        """
        FUV corrected luminosity-weighted average stellar age.
        """
        if self.Nstar == 0:
            return None
        Luv = self.stellar_luminosities[
            :, self.snapshot_datasets.get_column_index("CorrectedLuminosities", "GALEX_FUV")
        ]
        Luvtot = Luv.sum()
        if Luvtot == 0:
            return None
        
        return ((Luv / Luvtot) * self.stellar_ages).sum()

    @lazy_property
    def TotalSNIaRate(self) -> unyt.unyt_quantity:
        """
        Total SNIa rate.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/SNIaRates")[self.star_mask_all][
            self.star_mask_ap
        ].sum()

    @lazy_property
    def bh_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out BH particles in raw PartType5 arrays.
        This is the mask that masks out unbound particles for exclusive halos.
        For inclusive halos, this mask does nothing.
        """
        if self.Nbh == 0:
            return None
        groupnr_bound = self.get_dataset("PartType5/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def BH_subgrid_masses(self) -> unyt.unyt_array:
        """
        Subgrid masses of BH particles.
        """
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all][
            self.bh_mask_ap
        ]

    @lazy_property
    def Mbh_subgrid(self) -> unyt.unyt_quantity:
        """
        Total subgrid mass of BH particles.
        """
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses.sum()

    @lazy_property
    def agn_eventa(self) -> unyt.unyt_array:
        """
        Last AGN feedback event scale factors for BH particles.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
            self.bh_mask_all
        ][self.bh_mask_ap]

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
            self.get_dataset("PartType5/AGNTotalInjectedEnergies")[self.bh_mask_all][
                self.bh_mask_ap
            ]
        )

    @lazy_property
    def BlackHolesTotalInjectedJetEnergy(self) -> unyt.unyt_quantity:
        """
        Total jet energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.get_dataset("PartType5/InjectedJetEnergies")[self.bh_mask_all][
                self.bh_mask_ap
            ]
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
        return self.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxpos(self) -> unyt.unyt_array:
        """
        Position of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Coordinates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxvel(self) -> unyt.unyt_array:
        """
        Velocity of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Velocities")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxAR(self) -> unyt.unyt_quantity:
        """
        Accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionRates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleAveragedAccretionRate(self) -> unyt.unyt_quantity:
        """
        Averaged accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AveragedAccretionRates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Total energy injected into gas particles by the most massive
        BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AGNTotalInjectedEnergies")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleAccretionMode(self) -> unyt.unyt_quantity:
        """
        Accretion flow regime of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionModes")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleGWMassLoss(self) -> unyt.unyt_quantity:
        """
        Cumulative mass lost to GW of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/GWMassLosses")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleInjectedJetEnergyByMode(self) -> unyt.unyt_quantity:
        """
        Total energy injected in the kinetic jet AGN feedback mode, split by accretion mode,
        of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/InjectedJetEnergiesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleLastJetEventScalefactor(self) -> unyt.unyt_quantity:
        """
        Scale-factor of last jet event of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNJetScaleFactors")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNEvents(self) -> unyt.unyt_quantity:
        """
        Number of AGN events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNEvents")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNJetEvents(self) -> unyt.unyt_quantity:
        """
        Number of jet events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNJetEvents")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfMergers(self) -> unyt.unyt_quantity:
        """
        Number of mergers the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfMergers")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleRadiatedEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into radiation by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/RadiatedEnergiesByMode")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMassesByMode(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/TotalAccretedMassesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleWindEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into accretion disc winds by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/WindEnergiesByMode")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleSpin(self) -> unyt.unyt_quantity:
        """
        The spin of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Spins")[self.bh_mask_all][self.bh_mask_ap][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMass(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/TotalAccretedMasses")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleFormationScalefactor(self) -> unyt.unyt_quantity:
        """
        The formation scale factor of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/FormationScaleFactors")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxlasteventa(self) -> unyt.unyt_quantity:
        """
        Last feedback scale factor of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of all particles. See the documentation of star_mass_fraction
        for the rationale behind this.
        """
        if self.Mtot == 0:
            return None
        return self.mass / self.Mtot

    @lazy_property
    def com(self) -> unyt.unyt_array:
        """
        Centre of mass of all particles in the aperture.
        """
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.position).sum(axis=0) + self.centre

    @lazy_property
    def vcom(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles in the aperture.
        """
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def spin_parameter(self) -> unyt.unyt_quantity:
        """
        Spin parameter of all particles in the aperture.

        Computed as in Bullock et al. (2021):
          lambda = |Ltot| / (sqrt(2) * M * v_max * R)
        """
        if self.Mtot == 0:
            return None
        soft_r = np.maximum(self.softening, self.radius)
        _, vmax_soft = get_vmax(self.mass, soft_r)
        if vmax_soft > 0:
            vrel = self.velocity - self.vcom[None, :]
            Ltot = np.linalg.norm(
                (self.mass[:, None] * np.cross(self.position, vrel)).sum(axis=0)
            )
            return Ltot / (np.sqrt(2.0) * self.Mtot * self.aperture_radius * vmax_soft)
        return None

    @lazy_property
    def gas_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of gas particles. See the documentation of star_mass_fraction
        for the rationale behind this.
        """
        if self.Mgas == 0:
            return None
        return self.mass_gas / self.Mgas

    @lazy_property
    def vcom_gas(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of gas particles in the aperture.
        """
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.vel_gas).sum(axis=0)

    def compute_Lgas_props(self):
        """
        Compute the angular momentum and related properties for gas particles.

        We need this method because Lgas, kappa_gas and Mcountrot_gas are
        computed together.
        """
        (
            self.internal_Lgas,
            self.internal_kappa_gas,
            self.internal_Mcountrot_gas,
        ) = get_angular_momentum_and_kappa_corot(
            self.mass_gas,
            self.pos_gas,
            self.vel_gas,
            ref_velocity=self.vcom_gas,
            do_counterrot_mass=True,
        )

    @lazy_property
    def Lgas(self) -> unyt.unyt_array:
        """
        Angular momentum of gas particles.

        This is computed together with kappa_gas and Mcountrot_gas
        by compute_Lgas_props().
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Lgas"):
            self.compute_Lgas_props()
        return self.internal_Lgas

    @lazy_property
    def kappa_corot_gas(self) -> unyt.unyt_quantity:
        """
        Kinetic energy fraction of co-rotating gas particles.

        This is computed together with Lgas and Mcountrot_gas
        by compute_Lgas_props().
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_kappa_gas"):
            self.compute_Lgas_props()
        return self.internal_kappa_gas

    @lazy_property
    def DtoTgas(self) -> unyt.unyt_quantity:
        """
        Disk to total ratio of the gas.

        This is computed together with Lgas and kappa_corot_gas
        by compute_Lgas_props().
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_gas"):
            self.compute_Lgas_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_gas / self.Mgas

    @lazy_property
    def veldisp_matrix_gas(self) -> unyt.unyt_array:
        """
        Velocity dispersion matrix of the gas.
        """
        if self.Mgas == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.gas_mass_fraction, self.vel_gas, self.vcom_gas
        )

    @lazy_property
    def Ekin_gas(self) -> unyt.unyt_quantity:
        """
        Kinetic energy of the gas.
        """
        if self.Mgas == 0:
            return None
        # below we need to force conversion to np.float64 before summing
        # up particles to avoid overflow
        ekin_gas = self.mass_gas * ((self.vel_gas - self.vcom_gas) ** 2).sum(axis=1)
        ekin_gas = unyt.unyt_array(
            ekin_gas.value, dtype=np.float64, units=ekin_gas.units
        )
        return 0.5 * ekin_gas.sum()

    @lazy_property
    def dm_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of DM particles. See the documentation of star_mass_fraction
        for the rationale behind this.
        """
        if self.Mdm == 0:
            return None
        return self.mass_dm / self.Mdm

    @lazy_property
    def vcom_dm(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of DM particles.
        """
        if self.Mdm == 0:
            return None
        return (self.dm_mass_fraction[:, None] * self.vel_dm).sum(axis=0)

    @lazy_property
    def veldisp_matrix_dm(self) -> unyt.unyt_array:
        """
        Velocity dispersion matrix of DM particles.
        """
        if self.Mdm == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.dm_mass_fraction, self.vel_dm, self.vcom_dm
        )

    @lazy_property
    def Ldm(self) -> unyt.unyt_array:
        """
        Angular momentum of DM particles.
        """
        if self.Mdm == 0:
            return None
        return get_angular_momentum(
            self.mass_dm, self.pos_dm, self.vel_dm, ref_velocity=self.vcom_dm
        )

    @lazy_property
    def com_star(self) -> unyt.unyt_array:
        """
        Centre of mass of star particles in the aperture.
        """
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.pos_star).sum(
            axis=0
        ) + self.centre

    @lazy_property
    def vcom_star(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of star particles.
        """
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.vel_star).sum(axis=0)

    def compute_Lstar_props(self):
        """
        Compute the angular momentum and related properties for star particles.

        We need this method because Lstar, kappa_star and Mcountrot_star are
        computed together.
        """
        (
            self.internal_Lstar,
            self.internal_kappa_star,
            self.internal_Mcountrot_star,
        ) = get_angular_momentum_and_kappa_corot(
            self.mass_star,
            self.pos_star,
            self.vel_star,
            ref_velocity=self.vcom_star,
            do_counterrot_mass=True,
        )

    @lazy_property
    def Lstar(self) -> unyt.unyt_array:
        """
        Angular momentum of star particles.

        This is computed together with kappa_star and Mcountrot_star
        by compute_Lstar_props().
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Lstar"):
            self.compute_Lstar_props()
        return self.internal_Lstar

    @lazy_property
    def kappa_corot_star(self) -> unyt.unyt_quantity:
        """
        Kinetic energy fraction of co-rotating star particles.

        This is computed together with Lstar and Mcountrot_star
        by compute_Lstar_props().
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_kappa_star"):
            self.compute_Lstar_props()
        return self.internal_kappa_star

    @lazy_property
    def DtoTstar(self) -> unyt.unyt_quantity:
        """
        Disk to total ratio of the stars.

        This is computed together with Lstar and kappa_corot_star
        by compute_Lstar_props().
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_star"):
            self.compute_Lstar_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_star / self.Mstar

    @lazy_property
    def veldisp_matrix_star(self) -> unyt.unyt_array:
        """
        Velocity dispersion matrix of the stars.
        """
        if self.Mstar == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.star_mass_fraction, self.vel_star, self.vcom_star
        )

    @lazy_property
    def Ekin_star(self) -> unyt.unyt_quantity:
        """
        Kinetic energy of star particles.
        """
        if self.Mstar == 0:
            return None
        # below we need to force conversion to np.float64 before summing
        # up particles to avoid overflow
        ekin_star = self.mass_star * ((self.vel_star - self.vcom_star) ** 2).sum(axis=1)
        ekin_star = unyt.unyt_array(
            ekin_star.value, dtype=np.float64, units=ekin_star.units
        )
        return 0.5 * ekin_star.sum()

    @lazy_property
    def baryon_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of baryon particles. See the documentation of star_mass_fraction
        for the rationale behind this.
        """
        if self.Mbaryons == 0:
            return None
        return self.mass_baryons / self.Mbaryons

    @lazy_property
    def vcom_bar(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of baryons (gas + stars).
        """
        if self.Mbaryons == 0:
            return None
        return (self.baryon_mass_fraction[:, None] * self.vel_baryons).sum(axis=0)

    def compute_Lbar_props(self):
        """
        Compute the angular momentum and related properties for baryon particles.

        We need this method because Lbaryon, kappa_baryon and Mcountrot_baryon are
        computed together.
        """
        (
            self.internal_Lbar,
            self.internal_kappa_bar,
        ) = get_angular_momentum_and_kappa_corot(
            self.mass_baryons,
            self.pos_baryons,
            self.vel_baryons,
            ref_velocity=self.vcom_bar,
        )

    @lazy_property
    def Lbaryons(self) -> unyt.unyt_array:
        """
        Angular momentum of baryon (gas + stars) particles.

        This is computed together with kappa_baryon and Mcountrot_baryon
        by compute_Lbaryon_props().
        """
        if self.Mbaryons == 0:
            return None
        if not hasattr(self, "internal_Lbar"):
            self.compute_Lbar_props()
        return self.internal_Lbar

    @lazy_property
    def kappa_corot_baryons(self) -> unyt.unyt_quantity:
        """
        Kinetic energy fraction of co-rotating baryon (gas + stars) particles.

        This is computed together with Lbaryon and Mcountrot_baryon
        by compute_Lbaryon_props().
        """
        if self.Mbaryons == 0:
            return None
        if not hasattr(self, "internal_kappa_bar"):
            self.compute_Lbar_props()
        return self.internal_kappa_bar

    @lazy_property
    def gas_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out gas particles in raw PartType0 arrays.
        This is the mask that masks out unbound particles for exclusive halos.
        For inclusive halos, this mask does nothing.
        """
        if self.Ngas == 0:
            return None
        groupnr_bound = self.get_dataset("PartType0/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def gas_SFR(self) -> unyt.unyt_array:
        """
        Star formation rates of star particles.

        Note that older versions of SWIFT would hijack this dataset to also encode
        other information, so that negative SFR values (which are unphysical) would
        correspond to the last scale factor or time the gas was star-forming.
        We need to mask out these negative values and set them to 0.
        """
        if self.Ngas == 0:
            return None
        raw_SFR = self.get_dataset("PartType0/StarFormationRates")[self.gas_mask_all][
            self.gas_mask_ap
        ]
        # Negative SFR are not SFR at all!
        raw_SFR[raw_SFR < 0] = 0
        return raw_SFR

    @lazy_property
    def is_SFR(self) -> NDArray[bool]:
        """
        Mask to select only star-forming gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_SFR > 0

    @lazy_property
    def SFR(self) -> unyt.unyt_quantity:
        """
        Total star formation rate of the gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def Mgas_SF(self) -> unyt.unyt_quantity:
        """
        Mass of star-forming gas.
        """
        if self.Ngas == 0:
            return None
        return self.mass_gas[self.is_SFR].sum()

    @lazy_property
    def gas_metal_mass_fractions(self) -> unyt.unyt_array:
        """
        Metal mass fractions of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/MetalMassFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_Mgasmetal(self) -> unyt.unyt_array:
        """
        Metal masses of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.mass_gas * self.gas_metal_mass_fractions

    @lazy_property
    def gas_Mgasmetal_diffuse(self) -> unyt.unyt_array:
        """
        Metal masses of gas particles, without metals locked up in dust.
        """
        if self.Ngas == 0:
            return None
        return self.mass_gas * (
            self.gas_metal_mass_fractions - self.gas_dust_mass_fractions.sum(axis=1)
        )

    @lazy_property
    def GasMassInColdDenseDiffuseMetals(self) -> unyt.unyt_quantity:
        """
        Mass of metals in cold, dense gas, excluding metals locked up in dust.
        """
        if self.Ngas == 0:
            return None
        return self.gas_Mgasmetal_diffuse[self.gas_is_cold_dense].sum()

    @lazy_property
    def gasmetalfrac_SF(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction of star-forming gas.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_Mgasmetal[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasmetalfrac(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction of gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_Mgasmetal.sum() / self.Mgas

    @lazy_property
    def gas_MgasO(self) -> unyt.unyt_array:
        """
        Oxygen mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.mass_gas
            * self.get_dataset("PartType0/ElementMassFractions")[self.gas_mask_all][
                self.gas_mask_ap
            ][
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
        )

    @lazy_property
    def gasOfrac_SF(self) -> unyt.unyt_quantity:
        """
        Oxgen mass fraction of star-forming gas.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_MgasO[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasOfrac(self) -> unyt.unyt_quantity:
        """
        Oxygen mass fraction of gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_MgasO.sum() / self.Mgas

    @lazy_property
    def gas_MgasFe(self) -> unyt.unyt_array:
        """
        Iron mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.mass_gas
            * self.get_dataset("PartType0/ElementMassFractions")[self.gas_mask_all][
                self.gas_mask_ap
            ][
                :,
                self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron"),
            ]
        )

    @lazy_property
    def gasFefrac_SF(self) -> unyt.unyt_quantity:
        """
        Iron mass fraction of star-forming gas.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_MgasFe[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasFefrac(self) -> unyt.unyt_quantity:
        """
        Oxgen mass fraction of gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_MgasFe.sum() / self.Mgas

    @lazy_property
    def gas_temp(self) -> unyt.unyt_array:
        """
        Temperature of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Temperatures")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_rho(self) -> unyt.unyt_array:
        """
        Density of gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Densities")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_no_agn(self) -> NDArray[bool]:
        """
        Create a mask for gas particles that wer not recently heated by AGN.

        The mask is created by negating the mask returned by the RecentlyHeatedGasFilter.
        """
        if self.Ngas == 0:
            return None
        last_agn_gas = self.get_dataset("PartType0/LastAGNFeedbackScaleFactors")[
            self.gas_mask_all
        ][self.gas_mask_ap]
        return ~self.recently_heated_gas_filter.is_recently_heated(
            last_agn_gas, self.gas_temp
        )

    @lazy_property
    def Tgas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of the gas.
        """
        if self.Mgas == 0 or self.Ngas == 0:
            return None
        return (self.gas_mass_fraction * self.gas_temp).sum()

    @lazy_property
    def Tgas_no_agn(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of the gas, excluding gas that was
        recently heated by AGN feedback.
        """
        if self.Ngas == 0:
            return None
        if np.any(self.gas_no_agn):
            mass_gas_no_agn = self.mass_gas[self.gas_no_agn]
            Mgas_no_agn = mass_gas_no_agn.sum()
            if Mgas_no_agn > 0:
                return (
                    (mass_gas_no_agn / Mgas_no_agn) * self.gas_temp[self.gas_no_agn]
                ).sum()
        return None

    @lazy_property
    def gas_element_fractions(self) -> unyt.unyt_array:
        """
        Element fractions of the gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/ElementMassFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_mass_H(self) -> unyt.unyt_array:
        """
        Hydrogen mass in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Hydrogen"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_mass_He(self) -> unyt.unyt_array:
        """
        Helium mass in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Helium"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_species_fractions(self) -> unyt.unyt_array:
        """
        Ion/molecule fractions in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/SpeciesFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_mass_HI(self) -> unyt.unyt_array:
        """
        Atomic hydrogen mass in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_mass_H
            * self.gas_species_fractions[
                :, self.snapshot_datasets.get_column_index("SpeciesFractions", "HI")
            ]
        )

    @lazy_property
    def gas_mass_H2(self) -> unyt.unyt_array:
        """
        Molecular hydrogen mass in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_mass_H
            * self.gas_species_fractions[
                :, self.snapshot_datasets.get_column_index("SpeciesFractions", "H2")
            ]
            * 2.0
        )

    @lazy_property
    def HydrogenMass(self) -> unyt.unyt_quantity:
        """
        Hydrogen mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_mass_H.sum()

    @lazy_property
    def HeliumMass(self) -> unyt.unyt_quantity:
        """
        Helium mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_mass_He.sum()

    @lazy_property
    def MolecularHydrogenMass(self) -> unyt.unyt_quantity:
        """
        Molecular hydrogen mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_mass_H2.sum()

    @lazy_property
    def AtomicHydrogenMass(self) -> unyt.unyt_quantity:
        """
        Atomic hydrogen mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_mass_HI.sum()

    @lazy_property
    def gas_dust_mass_fractions(self) -> unyt.unyt_array:
        """
        Dust mass fractions in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/DustMassFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_dust_mass_fractions_graphite_large(self) -> unyt.unyt_array:
        """
        Dust mass fractions of large graphite grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.gas_dust_mass_fractions[
            :,
            self.snapshot_datasets.get_column_index(
                "DustMassFractions", "GraphiteLarge"
            ),
        ]

    @lazy_property
    def gas_dust_mass_fractions_silicates_large(self) -> unyt.unyt_array:
        """
        Dust mass fractions of large silicates grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "DustMassFractions", "MgSilicatesLarge"
                ),
            ]
            + self.gas_dust_mass_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "DustMassFractions", "FeSilicatesLarge"
                ),
            ]
        )

    @lazy_property
    def gas_dust_mass_fractions_graphite_small(self) -> unyt.unyt_array:
        """
        Dust mass fractions of small graphite grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.gas_dust_mass_fractions[
            :,
            self.snapshot_datasets.get_column_index(
                "DustMassFractions", "GraphiteSmall"
            ),
        ]

    @lazy_property
    def gas_dust_mass_fractions_silicates_small(self) -> unyt.unyt_array:
        """
        Dust mass fractions of small silicates grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "DustMassFractions", "MgSilicatesSmall"
                ),
            ]
            + self.gas_dust_mass_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "DustMassFractions", "FeSilicatesSmall"
                ),
            ]
        )

    @lazy_property
    def gas_graphite_mass_fractions(self) -> unyt.unyt_array:
        """
        Dust mass fractions of graphite grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_large
            + self.gas_dust_mass_fractions_graphite_small
        )

    @lazy_property
    def gas_silicates_mass_fractions(self) -> unyt.unyt_array:
        """
        Dust mass fractions of silicates grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_silicates_large
            + self.gas_dust_mass_fractions_silicates_small
        )

    @lazy_property
    def gas_large_dust_mass_fractions(self) -> unyt.unyt_array:
        """
        Dust mass fractions of large grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_large
            + self.gas_dust_mass_fractions_silicates_large
        )

    @lazy_property
    def gas_small_dust_mass_fractions(self) -> unyt.unyt_array:
        """
        Dust mass fractions of small grains in gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_small
            + self.gas_dust_mass_fractions_silicates_small
        )

    @lazy_property
    def gas_is_cold_dense(self) -> NDArray[bool]:
        """
        Mask for gas particles containing cold, dense gas.

        The mask is created by the ColdDenseGasFilter.
        """
        if self.Ngas == 0:
            return None
        return self.cold_dense_gas_filter.is_cold_and_dense(self.gas_temp, self.gas_rho)

    @lazy_property
    def DustGraphiteMass(self) -> unyt.unyt_quantity:
        """
        Graphite dust mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustGraphiteMassInAtomicGas(self) -> unyt.unyt_quantity:
        """
        Graphite dust mass in atomic gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.gas_mass_HI).sum()

    @lazy_property
    def DustGraphiteMassInMolecularGas(self) -> unyt.unyt_quantity:
        """
        Graphite dust mass in molecular gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustGraphiteMassInColdDenseGas(self) -> unyt.unyt_quantity:
        """
        Graphite dust mass in cold, dense gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_graphite_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustSilicatesMass(self) -> unyt.unyt_quantity:
        """
        Silicates dust mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustSilicatesMassInAtomicGas(self) -> unyt.unyt_quantity:
        """
        Silicates dust mass in atomic gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.gas_mass_HI).sum()

    @lazy_property
    def DustSilicatesMassInMolecularGas(self) -> unyt.unyt_quantity:
        """
        Silicates dust mass in molecular gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustSilicatesMassInColdDenseGas(self) -> unyt.unyt_quantity:
        """
        Silicates dust mass in cold, dense gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_silicates_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustLargeGrainMass(self) -> unyt.unyt_quantity:
        """
        Large dust grain mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_large_dust_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustLargeGrainMassInMolecularGas(self) -> unyt.unyt_quantity:
        """
        Large dust grain mass in molecular gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_large_dust_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustLargeGrainMassInColdDenseGas(self) -> unyt.unyt_quantity:
        """
        Large dust grain mass in cold, dense gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_large_dust_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustSmallGrainMass(self) -> unyt.unyt_quantity:
        """
        Small dust grain mass in gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_small_dust_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustSmallGrainMassInMolecularGas(self) -> unyt.unyt_quantity:
        """
        Small dust grain mass in molecular gas.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_small_dust_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustSmallGrainMassInColdDenseGas(self) -> unyt.unyt_quantity:
        """
        Small dust grain mass in cold, dense gas.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_small_dust_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def GasMassInColdDenseGas(self) -> unyt.unyt_quantity:
        """
        Mass of cold, dense gas.
        """
        if self.Ngas == 0:
            return None
        return self.mass_gas[self.gas_is_cold_dense].sum()

    @lazy_property
    def gas_diffuse_element_fractions(self) -> unyt.unyt_array:
        """
        Diffuse element fractions of gas particles.

        Diffuse means the contribution from dust has been removed.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/ElementMassFractionsDiffuse")[
            self.gas_mask_all
        ][self.gas_mask_ap]

    @lazy_property
    def gas_diffuse_carbon_mass(self) -> unyt.unyt_array:
        """
        Diffuse carbon mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_diffuse_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Carbon"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_diffuse_oxygen_mass(self) -> unyt.unyt_array:
        """
        Diffuse oxygen mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_diffuse_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_diffuse_magnesium_mass(self) -> unyt.unyt_array:
        """
        Diffuse magnesium mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_diffuse_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Magnesium"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_diffuse_silicon_mass(self) -> unyt.unyt_array:
        """
        Diffuse silicon mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_diffuse_element_fractions[
                :,
                self.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Silicon"
                ),
            ]
            * self.mass_gas
        )

    @lazy_property
    def gas_diffuse_iron_mass(self) -> unyt.unyt_array:
        """
        Diffuse iron mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_diffuse_element_fractions[
                :,
                self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron"),
            ]
            * self.mass_gas
        )

    @lazy_property
    def DiffuseCarbonMass(self) -> unyt.unyt_quantity:
        """
        Diffuse carbon mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_diffuse_carbon_mass.sum()

    @lazy_property
    def DiffuseOxygenMass(self) -> unyt.unyt_quantity:
        """
        Diffuse oxygen mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_diffuse_oxygen_mass.sum()

    @lazy_property
    def DiffuseMagnesiumMass(self) -> unyt.unyt_quantity:
        """
        Diffuse magnesium mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_diffuse_magnesium_mass.sum()

    @lazy_property
    def DiffuseSiliconMass(self) -> unyt.unyt_quantity:
        """
        Diffuse silicon mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_diffuse_silicon_mass.sum()

    @lazy_property
    def DiffuseIronMass(self) -> unyt.unyt_quantity:
        """
        Diffuse iron mass in gas.
        """
        if self.Ngas == 0:
            return None
        return self.gas_diffuse_iron_mass.sum()

    @lazy_property
    def gas_O_over_H_total(self) -> unyt.unyt_array:
        """
        Total oxygen over hydrogen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        nH = self.gas_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Hydrogen"),
        ]
        nO = self.gas_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        return nO / (16.0 * nH)

    @lazy_property
    def gas_N_over_O_total(self) -> unyt.unyt_array:
        """
        Total nitrogen over oxygen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        nN = self.gas_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Nitrogen"),
        ]
        nO = self.gas_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        ratio = np.zeros_like(nN)
        ratio[nO != 0] = (16.0 * nN[nO != 0]) / (14.0 * nO[nO != 0])
        return ratio

    @lazy_property
    def gas_C_over_O_total(self) -> unyt.unyt_array:
        """
        Total carbon over oxygen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        nC = self.gas_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Carbon")
        ]
        nO = self.gas_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        ratio = np.zeros_like(nC)
        ratio[nO != 0] = (16.0 * nC[nO != 0]) / (12.011 * nO[nO != 0])
        return ratio

    @lazy_property
    def gas_N_over_O_diffuse(self) -> unyt.unyt_array:
        """
        Diffuse nitrogen over oxygen ratio of gas particles.
        Keep in mind this does not consider the metals in dust.
        """
        if self.Ngas == 0:
            return None
        nN = self.gas_diffuse_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Nitrogen"),
        ]
        nO = self.gas_diffuse_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        ratio = np.zeros_like(nN)
        ratio[nO != 0] = (16.0 * nN[nO != 0]) / (14.0 * nO[nO != 0])
        return ratio

    @lazy_property
    def gas_C_over_O_diffuse(self) -> unyt.unyt_array:
        """
        Diffuse carbon over oxygen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        nC = self.gas_diffuse_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Carbon")
        ]
        nO = self.gas_diffuse_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        ratio = np.zeros_like(nC)
        ratio[nO != 0] = (16.0 * nC[nO != 0]) / (12.011 * nO[nO != 0])
        return ratio

    @lazy_property
    def gas_O_over_H_diffuse(self) -> unyt.unyt_array:
        """
        Diffuse oxygen over hydrogen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        nH = self.gas_diffuse_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Hydrogen"),
        ]
        nO = self.gas_diffuse_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Oxygen")
        ]
        return nO / (16.0 * nH)

    @lazy_property
    def gas_log10_N_over_O_diffuse_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse nitrogen over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-4 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_N_over_O_diffuse,
                self.snapshot_datasets.get_defined_constant("N_O_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_N_over_O_diffuse_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse nitrogen over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-3 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_N_over_O_diffuse,
                self.snapshot_datasets.get_defined_constant("N_O_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_N_over_O_total_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the total nitrogen over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-4 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_N_over_O_total,
                self.snapshot_datasets.get_defined_constant("N_O_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_N_over_O_total_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the total nitrogen over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-3 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_N_over_O_total,
                self.snapshot_datasets.get_defined_constant("N_O_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_C_over_O_diffuse_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse carbon over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-4 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_C_over_O_diffuse,
                self.snapshot_datasets.get_defined_constant("C_O_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_C_over_O_diffuse_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse carbon over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-3 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_C_over_O_diffuse,
                self.snapshot_datasets.get_defined_constant("C_O_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_C_over_O_total_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the total carbon over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-4 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_C_over_O_total,
                self.snapshot_datasets.get_defined_constant("C_O_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_C_over_O_total_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the total carbon over oxygen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-3 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_C_over_O_total,
                self.snapshot_datasets.get_defined_constant("C_O_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_O_over_H_diffuse_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse oxygen over hydrogen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-4 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_O_over_H_diffuse,
                self.snapshot_datasets.get_defined_constant("O_H_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def gas_log10_O_over_H_diffuse_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the diffuse oxygen over hydrogen ratio of gas particles.

        Uses a lower limit on the ratio of 1.e-3 times the solar ratio,
        which is set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return np.log10(
            np.clip(
                self.gas_O_over_H_diffuse,
                self.snapshot_datasets.get_defined_constant("O_H_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def LinearMassWeightedOxygenOverHydrogenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the total oxygen over hydrogen ratio of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_O_over_H_total[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LinearMassWeightedNitrogenOverOxygenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the total nitrogen over oxygen ratio of gas particles.
        This includes the contribution from dust!
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_N_over_O_total[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LinearMassWeightedDiffuseNitrogenOverOxygenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the diffuse nitrogen over oxygen ratio of gas particles.
        This excludes the contribution from dust!
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_N_over_O_diffuse[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LinearMassWeightedCarbonOverOxygenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the total carbon over oxygen ratio of gas particles.
        This includes the contribution from dust!
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_C_over_O_total[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LinearMassWeightedDiffuseCarbonOverOxygenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the diffuse carbon over oxygen ratio of gas particles.
        This excludes the contribution from dust.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_C_over_O_diffuse[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LinearMassWeightedDiffuseOxygenOverHydrogenOfGas(self) -> unyt.unyt_quantity:
        """
        Mass-weigthed sum of the diffuse oxygen over hydrogen ratio of gas particles,
        excluding the contribution from dust.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_O_over_H_diffuse[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_low_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_high_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse nitrogen over oxygen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_N_over_O_diffuse_low_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse nitrogen over oxygen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_N_over_O_diffuse_high_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse carbon over oxygen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_C_over_O_diffuse_low_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the diffuse carbon over oxygen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_C_over_O_diffuse_high_limit[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Atomic mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_low_limit[self.gas_is_cold_dense]
            * self.gas_mass_HI[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Atomic mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_high_limit[self.gas_is_cold_dense]
            * self.gas_mass_HI[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Molecular mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_low_limit[self.gas_is_cold_dense]
            * self.gas_mass_H2[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Molecular mass-weighted sum of the logarithm of the diffuse oxygen over hydrogen ratio of gas
        particles, excluding the contribution from dust and using a lower limit on the ratio
        of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Ngas == 0:
            return None
        return (
            self.gas_log10_O_over_H_diffuse_high_limit[self.gas_is_cold_dense]
            * self.gas_mass_H2[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def star_Fe_over_H(self) -> unyt.unyt_array:
        """
        Iron over hydrogen ratio of star particles.
        """
        if self.Nstar == 0:
            return None
        nH = self.star_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Hydrogen"),
        ]
        nFe = self.star_element_fractions[
            :, self.snapshot_datasets.get_column_index("ElementMassFractions", "Iron")
        ]
        return nFe / (55.845 * nH)

    @lazy_property
    def star_Fe_from_SNIa_over_H(self) -> unyt.unyt_array:
        """
        Iron over hydrogen ratio of star particles, only taking into account iron produced
        by SNIa.
        """
        if self.Nstar == 0:
            return None
        nH = self.star_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Hydrogen"),
        ]
        nFe = self.get_dataset("PartType4/IronMassFractionsFromSNIa")[
            self.star_mask_all
        ][self.star_mask_ap]
        return nFe / (55.845 * nH)

    @lazy_property
    def star_log10_Fe_over_H_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the iron over hydrogen ratio of star particles, using a lower limit
        on the ratio of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return np.log10(
            np.clip(
                self.star_Fe_over_H,
                self.snapshot_datasets.get_defined_constant("Fe_H_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def star_log10_Fe_from_SNIa_over_H_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the iron over hydrogen ratio of star particles, using a lower limit
        on the ratio of 1.e-4 times the solar ratio, set in the parameter file, and only
        taking into account iron produced by SNIa.
        """
        if self.Nstar == 0:
            return None
        return np.log10(
            np.clip(
                self.star_Fe_from_SNIa_over_H,
                self.snapshot_datasets.get_defined_constant("Fe_H_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def star_log10_Fe_over_H_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the iron over hydrogen ratio of star particles, using a lower limit
        on the ratio of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return np.log10(
            np.clip(
                self.star_Fe_over_H,
                self.snapshot_datasets.get_defined_constant("Fe_H_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def LinearMassWeightedIronOverHydrogenOfStars(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the iron over hydrogen ratio for star particles.
        """
        if self.Nstar == 0:
            return None
        return (self.star_Fe_over_H * self.mass_star).sum()

    @lazy_property
    def LogarithmicMassWeightedIronOverHydrogenOfStarsLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the iron over hydrogen ratio for star particles,
        using a lower limit of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return (self.star_log10_Fe_over_H_low_limit * self.mass_star).sum()

    @lazy_property
    def LogarithmicMassWeightedIronOverHydrogenOfStarsHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the iron over hydrogen ratio for star particles,
        using a lower limit of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return (self.star_log10_Fe_over_H_high_limit * self.mass_star).sum()

    @lazy_property
    def LogarithmicMassWeightedIronFromSNIaOverHydrogenOfStarsLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the iron over hydrogen ratio for star particles,
        using a lower limit of 1.e-4 times the solar ratio, set in the parameter file, and
        only taking into account iron produced by SNIa.
        """
        if self.Nstar == 0:
            return None
        return (self.star_log10_Fe_from_SNIa_over_H_low_limit * self.mass_star).sum()

    @lazy_property
    def LinearMassWeightedIronFromSNIaOverHydrogenOfStars(self,) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the iron over hydrogen ratio for star particles,
        times the solar ratio, set in the parameter file, and
        only taking into account iron produced by SNIa.
        """
        if self.Nstar == 0:
            return None
        return (self.star_Fe_from_SNIa_over_H * self.mass_star).sum()

    @lazy_property
    def star_Mg_over_H(self) -> unyt.unyt_array:
        """
        Magnesium over hydrogen ratio of star particles.
        """
        if self.Nstar == 0:
            return None
        nH = self.star_element_fractions[
            :,
            self.snapshot_datasets.get_column_index("ElementMassFractions", "Hydrogen"),
        ]
        nMg = self.star_element_fractions[
            :,
            self.snapshot_datasets.get_column_index(
                "ElementMassFractions", "Magnesium"
            ),
        ]
        return nMg / (24.305 * nH)

    @lazy_property
    def star_log10_Mg_over_H_low_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the iron over hydrogen ratio of star particles, using a lower limit
        on the ratio of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return np.log10(
            np.clip(
                self.star_Mg_over_H,
                self.snapshot_datasets.get_defined_constant("Mg_H_sun") * 1.0e-4,
                np.inf,
            )
        )

    @lazy_property
    def star_log10_Mg_over_H_high_limit(self) -> unyt.unyt_array:
        """
        Logarithm of the iron over hydrogen ratio of star particles, using a lower limit
        on the ratio of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return np.log10(
            np.clip(
                self.star_Mg_over_H,
                self.snapshot_datasets.get_defined_constant("Mg_H_sun") * 1.0e-3,
                np.inf,
            )
        )

    @lazy_property
    def LinearMassWeightedMagnesiumOverHydrogenOfStars(self) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the iron over hydrogen ratio for star particles.
        """
        if self.Nstar == 0:
            return None
        return (self.star_Mg_over_H * self.mass_star).sum()

    @lazy_property
    def LogarithmicMassWeightedMagnesiumOverHydrogenOfStarsLowLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the iron over hydrogen ratio for star particles,
        using a lower limit of 1.e-4 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return (self.star_log10_Mg_over_H_low_limit * self.mass_star).sum()

    @lazy_property
    def LogarithmicMassWeightedMagnesiumOverHydrogenOfStarsHighLimit(
        self,
    ) -> unyt.unyt_quantity:
        """
        Mass-weighted sum of the logarithm of the iron over hydrogen ratio for star particles,
        using a lower limit of 1.e-3 times the solar ratio, set in the parameter file.
        """
        if self.Nstar == 0:
            return None
        return (self.star_log10_Mg_over_H_high_limit * self.mass_star).sum()

    @lazy_property
    def HalfMassRadiusGas(self) -> unyt.unyt_quantity:
        """
        Half mass radius of gas.
        """
        return get_half_mass_radius(
            self.radius[self.type == 0], self.mass_gas, self.Mgas
        )

    @lazy_property
    def HalfMassRadiusDM(self) -> unyt.unyt_quantity:
        """
        Half mass radius of dark matter.
        """
        return get_half_mass_radius(self.radius[self.type == 1], self.mass_dm, self.Mdm)

    @lazy_property
    def HalfMassRadiusStar(self) -> unyt.unyt_quantity:
        """
        Half mass radius of stars.
        """
        return get_half_mass_radius(
            self.radius[self.type == 4], self.mass_star, self.Mstar
        )

    @lazy_property
    def HalfMassRadiusBaryon(self) -> unyt.unyt_quantity:
        """
        Half mass radius of baryons (gas + stars).
        """
        return get_half_mass_radius(
            self.radius[(self.type == 0) | (self.type == 4)],
            self.mass_baryons,
            self.Mbaryons,
        )


class ApertureProperties(HaloProperty):
    """
    Compute aperture properties for halos.

    The aperture has a fixed radius and optionally only includes particles that
    are bound to the halo.
    """

    """
    List of properties from the table that we want to compute.
    Each property should have a corresponding method/property/lazy_property in
    the ApertureParticleData class above.
    """
    property_list: List[Tuple] = [
        (prop, *PropertyTable.full_property_list[prop])
        for prop in [
            "Mtot",
            "Mgas",
            "Mdm",
            "Mstar",
            "Mstar_init",
            "Mbh_dynamical",
            "Mbh_subgrid",
            "Ngas",
            "Ndm",
            "Nstar",
            "Nbh",
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
            "com",
            "com_star",
            "vcom",
            "vcom_star",
            "Lgas",
            "Ldm",
            "Lstar",
            "kappa_corot_gas",
            "kappa_corot_star",
            "Lbaryons",
            "kappa_corot_baryons",
            "veldisp_matrix_gas",
            "veldisp_matrix_dm",
            "veldisp_matrix_star",
            "Ekin_gas",
            "Ekin_star",
            "Mgas_SF",
            "gasmetalfrac",
            "gasmetalfrac_SF",
            "gasOfrac",
            "gasOfrac_SF",
            "gasFefrac",
            "gasFefrac_SF",
            "Tgas",
            "Tgas_no_agn",
            "SFR",
            "StellarLuminosity",
            "CorrectedStellarLuminosity",
            "CorrectedStellarLuminosityWithDust",
            "ChabrierInferredMstar",
            "starmetalfrac",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "HalfMassRadiusBaryon",
            "spin_parameter",
            "DtoTgas",
            "DtoTstar",
            "starOfrac",
            "starFefrac",
            "stellar_age_mw",
            "stellar_age_lw",
            "stellar_age_uvlw",
            "TotalSNIaRate",
            "HydrogenMass",
            "HeliumMass",
            "MolecularHydrogenMass",
            "AtomicHydrogenMass",
            "starMgfrac",
            "DustGraphiteMass",
            "DustGraphiteMassInAtomicGas",
            "DustGraphiteMassInMolecularGas",
            "DustGraphiteMassInColdDenseGas",
            "DustLargeGrainMass",
            "DustLargeGrainMassInMolecularGas",
            "DustLargeGrainMassInColdDenseGas",
            "DustSilicatesMass",
            "DustSilicatesMassInAtomicGas",
            "DustSilicatesMassInMolecularGas",
            "DustSilicatesMassInColdDenseGas",
            "DustSmallGrainMass",
            "DustSmallGrainMassInMolecularGas",
            "DustSmallGrainMassInColdDenseGas",
            "GasMassInColdDenseGas",
            "DiffuseCarbonMass",
            "DiffuseOxygenMass",
            "DiffuseMagnesiumMass",
            "DiffuseSiliconMass",
            "DiffuseIronMass",
            "LinearMassWeightedOxygenOverHydrogenOfGas",
            "LinearMassWeightedNitrogenOverOxygenOfGas",
            "LinearMassWeightedCarbonOverOxygenOfGas",
            "LinearMassWeightedDiffuseOxygenOverHydrogenOfGas",
            "LinearMassWeightedDiffuseNitrogenOverOxygenOfGas",
            "LinearMassWeightedDiffuseCarbonOverOxygenOfGas",
            "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasLowLimit",
            "LogarithmicMassWeightedDiffuseNitrogenOverOxygenOfGasHighLimit",
            "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasLowLimit",
            "LogarithmicMassWeightedDiffuseCarbonOverOxygenOfGasHighLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasLowLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfAtomicGasHighLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasLowLimit",
            "LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfMolecularGasHighLimit",
            "LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "LogarithmicMassWeightedMagnesiumOverHydrogenOfStarsLowLimit",
            "LogarithmicMassWeightedMagnesiumOverHydrogenOfStarsHighLimit",
            "LinearMassWeightedIronOverHydrogenOfStars",
            "LogarithmicMassWeightedIronOverHydrogenOfStarsLowLimit",
            "LogarithmicMassWeightedIronOverHydrogenOfStarsHighLimit",
            "GasMassInColdDenseDiffuseMetals",
            "LogarithmicMassWeightedIronFromSNIaOverHydrogenOfStarsLowLimit",
            "LinearMassWeightedIronFromSNIaOverHydrogenOfStars",
        ]
    ]

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        physical_radius_kpc: float,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        stellar_age_calculator: StellarAgeCalculator,
        cold_dense_gas_filter: ColdDenseGasFilter,
        category_filter: CategoryFilter,
        halo_filter: str,
        inclusive: bool = False,
    ):
        """
        Construct an ApertureProperties object with the given physical
        radius (in kpc) that uses the given filter to filter out recently
        heated gas particles.

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology and the dataset metadata.
         - parameters: ParameterFile
           Parameter file object containing the parameters from the parameter
           file.
         - physical_radius_kpc: float
           Physical radius of the aperture. Unitless and assumed to be expressed
           in units of kpc.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - stellar_age_calculator: StellarAgeCalculator
           Object used to calculate stellar ages from the current cosmological
           scale factor and the birth scale factor of the star particles.
         - cold_dense_gas_filter: ColdDenseGasFilter
           Filter used to mask out gas particles that represent cold, dense gas.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
         - inclusive: bool
           Should properties include particles that are not gravitationally bound to the
           subhalo?
        """

        super().__init__(cellgrid)

        self.property_mask = parameters.get_property_mask(
            "ApertureProperties", [prop[1] for prop in self.property_list]
        )

        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.stellar_ages = stellar_age_calculator
        self.cold_dense_gas_filter = cold_dense_gas_filter
        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets
        self.halo_filter = halo_filter

        # no density criterion for these properties
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.inclusive = inclusive

        if self.inclusive:
            self.name = f"inclusive_sphere_{physical_radius_kpc:.0f}kpc"
            self.group_name = f"InclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        else:
            self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"
            self.group_name = f"ExclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        self.mask_metadata = self.category_filter.get_filter_metadata(halo_filter)

        # List of particle properties we need to read in
        # Coordinates, Masses and Velocities are always required, as is
        # GroupNr_bound.
        self.particle_properties = {
            "PartType0": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
            "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
            "PartType4": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "GroupNr_bound",
                "Velocities",
            ],
        }
        # add additional particle properties based on the selected halo
        # properties in the parameter file
        for prop in self.property_list:
            outputname = prop[1]
            if not self.property_mask[outputname]:
                continue
            is_dmo = prop[8]
            if self.category_filter.dmo and not is_dmo:
                continue
            partprops = prop[9]
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
        Compute centre of mass etc of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius out to which the particle data is guaranteed to
                           be complete
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        The halo_result dictionary is updated with the properties computed by this function.
        """

        aperture_sphere = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        registry = input_halo["cofp"].units.registry
        for prop in self.property_list:
            outputname = prop[1]
            # skip properties that are masked
            if not self.property_mask[outputname]:
                continue
            # skip non-DMO properties in DMO run mode
            is_dmo = prop[8]
            if self.category_filter.dmo and not is_dmo:
                continue
            name = prop[0]
            shape = prop[2]
            dtype = prop[3]
            unit = unyt.Unit(prop[4], registry=registry)
            physical = prop[10]
            a_exponent = prop[11]
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            if not physical:
                unit = unit * unyt.Unit("a", registry=registry) ** a_exponent
            aperture_sphere[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=registry
            )

        do_calculation = self.category_filter.get_do_calculation(halo_result)

        # Determine whether to skip this halo because of filter
        if do_calculation[self.halo_filter]:
            if search_radius < self.physical_radius_mpc * unyt.Mpc:
                raise SearchRadiusTooSmallError(
                    "Search radius is smaller than aperture"
                )

            types_present = [type for type in self.particle_properties if type in data]
            part_props = ApertureParticleData(
                input_halo,
                data,
                types_present,
                self.inclusive,
                self.physical_radius_mpc * unyt.Mpc,
                self.stellar_ages,
                self.recently_heated_gas_filter,
                self.cold_dense_gas_filter,
                self.snapshot_datasets,
                self.softening_of_parttype,
            )

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
                unit = unyt.Unit(prop[4], registry=registry)
                category = prop[6]
                physical = prop[10]
                a_exponent = prop[11]
                if not physical:
                    unit = unit * unyt.Unit("a", registry=registry) ** a_exponent
                if do_calculation[category]:
                    val = getattr(part_props, name)
                    if val is not None:
                        assert (
                            aperture_sphere[name].shape == val.shape
                        ), f"Attempting to store {name} with wrong dimensions"
                        if unit == unyt.Unit("dimensionless"):
                            if hasattr(val, "units"):
                                assert (
                                    val.units == unyt.dimensionless
                                ), f"{name} is not dimensionless"
                            aperture_sphere[name] = unyt.unyt_array(
                                val.astype(dtype),
                                dtype=dtype,
                                units=unit,
                                registry=registry,
                            )
                        else:
                            err = f'Overflow for halo {input_halo["index"]} when'
                            err += f"calculating {name} in aperture_properties"
                            assert np.max(np.abs(val.to(unit).value)) < float(
                                "inf"
                            ), err
                            aperture_sphere[name] += val

        # add the new properties to the halo_result dictionary
        for prop in self.property_list:
            outputname = prop[1]
            # skip properties that are masked
            if not self.property_mask[outputname]:
                continue
            # skip non-DMO properties in DMO run mode
            is_dmo = prop[8]
            if self.category_filter.dmo and not is_dmo:
                continue
            name = prop[0]
            description = prop[5]
            physical = prop[10]
            a_exponent = prop[11]
            halo_result.update(
                {
                    f"{self.group_name}/{outputname}": (
                        aperture_sphere[name],
                        description,
                        physical,
                        a_exponent,
                    )
                }
            )

        return


class ExclusiveSphereProperties(ApertureProperties):
    """
    ApertureProperties specialization for exclusive apertures,
    i.e. excluding particles not gravitationally bound to the
    subhalo.
    """

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        physical_radius_kpc: float,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        stellar_age_calculator: StellarAgeCalculator,
        cold_dense_gas_filter: ColdDenseGasFilter,
        category_filter: CategoryFilter,
        halo_filter: str,
    ):
        """
        Construct an ExclusiveSphereProperties object with the given physical
        radius (in Mpc) that uses the given filter to filter out recently
        heated gas particles.

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology and the dataset metadata.
         - parameters: ParameterFile
           Parameter file object containing the parameters from the parameter
           file.
         - physical_radius_kpc: float
           Physical radius of the aperture. Unitless and assumed to be expressed
           in units of kpc.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - stellar_age_calculator: StellarAgeCalculator
           Object used to calculate stellar ages from the current cosmological
           scale factor and the birth scale factor of the star particles.
         - cold_dense_gas_filter: ColdDenseGasFilter
           Filter used to mask out gas particles that represent cold, dense gas.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the Bound subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
        """
        super().__init__(
            cellgrid,
            parameters,
            physical_radius_kpc,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            category_filter,
            halo_filter,
            False,
        )


class InclusiveSphereProperties(ApertureProperties):
    """
    ApertureProperties specialization for inclusive apertures,
    i.e. including particles not gravitationally bound to the
    subhalo.
    """

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        physical_radius_kpc: float,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        stellar_age_calculator: StellarAgeCalculator,
        cold_dense_gas_filter: ColdDenseGasFilter,
        category_filter: CategoryFilter,
        halo_filter: str,
    ):
        """
        Construct an InclusiveSphereProperties object with the given physical
        radius (in Mpc) that uses the given filter to filter out recently
        heated gas particles.

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology and the dataset metadata.
         - parameters: ParameterFile
           Parameter file object containing the parameters from the parameter
           file.
         - physical_radius_kpc: float
           Physical radius of the aperture. Unitless and assumed to be expressed
           in units of kpc.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - stellar_age_calculator: StellarAgeCalculator
           Object used to calculate stellar ages from the current cosmological
           scale factor and the birth scale factor of the star particles.
         - cold_dense_gas_filter: ColdDenseGasFilter
           Filter used to mask out gas particles that represent cold, dense gas.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the Bound subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
        """
        super().__init__(
            cellgrid,
            parameters,
            physical_radius_kpc,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            category_filter,
            halo_filter,
            True,
        )


def test_aperture_properties():
    """
    Unit test for the aperture property calculations.

    We generate 100 random "dummy" halos and feed them to
    ExclusiveSphereProperties::calculate() and
    InclusiveSphereProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    import pytest
    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(3256)
    recently_heated_filter = dummy_halos.get_recently_heated_gas_filter()
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())
    cold_dense_gas_filter = dummy_halos.get_cold_dense_gas_filter()
    cat_filter = CategoryFilter(
        dummy_halos.get_filters(
            {"general": 0, "gas": 0, "dm": 0, "star": 0, "baryon": 0}
        )
    )
    parameters = ParameterFile(
        parameter_dictionary={
            "aliases": {
                "PartType0/ElementMassFractions": "PartType0/SmoothedElementMassFractions",
                "PartType4/ElementMassFractions": "PartType4/SmoothedElementMassFractions",
            }
        }
    )
    dummy_halos.get_cell_grid().snapshot_datasets.setup_aliases(
        parameters.get_aliases()
    )
    parameters.get_halo_type_variations(
        "ApertureProperties",
        {
            "exclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": False},
            "inclusive_50_kpc": {"radius_in_kpc": 50.0, "inclusive": True},
        },
    )

    pc_exclusive = ExclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
        "basic",
    )
    pc_inclusive = InclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
        "basic",
    )

    # Create a filter that no halos will satisfy
    fail_filter = CategoryFilter(dummy_halos.get_filters({"general": 10000000}))
    pc_filter_test = ExclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        recently_heated_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        fail_filter,
        "general",
    )

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _, particle_numbers = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )
        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for pc_name, pc_type, pc_calc in [
            ("ExclusiveSphere", "ExclusiveSphere", pc_exclusive),
            ("InclusiveSphere", "InclusiveSphere", pc_inclusive),
            ("filter_test", "ExclusiveSphere", pc_filter_test),
        ]:
            input_data = {}
            for ptype in pc_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in pc_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()

            # Check halo fails if search radius is too small
            halo_result = dict(halo_result_template)
            if pc_name != "filter_test":
                with pytest.raises(SearchRadiusTooSmallError):
                    pc_calc.calculate(
                        input_halo, 10 * unyt.kpc, input_data, halo_result
                    )
            # Skipped halos shouldn't ever require a larger search radius
            else:
                pc_calc.calculate(input_halo, 10 * unyt.kpc, input_data, halo_result)

            halo_result = dict(halo_result_template)
            pc_calc.calculate(input_halo, 100 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in pc_calc.property_list:
                outputname = prop[1]
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                physical = prop[10]
                a_exponent = prop[11]
                full_name = f"{pc_type}/50kpc/{outputname}"
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

            # Check properties were not calculated for filtered halos
            if pc_name == "filter_test":
                for prop in pc_calc.property_list:
                    outputname = prop[1]
                    size = prop[2]
                    full_name = f"{pc_type}/50kpc/{outputname}"
                    assert np.all(halo_result[full_name][0].value == np.zeros(size))

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    # we reuse the last random halo for this
    all_parameters = parameters.get_parameters()
    for property in all_parameters["ApertureProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["ApertureProperties"]["properties"]:
            single_property["ApertureProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)
        pc_exclusive = ExclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            50.0,
            recently_heated_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
        )
        pc_inclusive = InclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            50.0,
            recently_heated_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
            "basic",
        )

        halo_result_template = dummy_halos.get_halo_result_template(particle_numbers)

        for pc_type, pc_calc in [
            ("ExclusiveSphere", pc_exclusive),
            ("InclusiveSphere", pc_inclusive),
        ]:
            input_data = {}
            for ptype in pc_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in pc_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            halo_result = dict(halo_result_template)
            pc_calc.calculate(input_halo, 100 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in pc_calc.property_list:
                outputname = prop[1]
                if not outputname == property:
                    continue
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"{pc_type}/50kpc/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string, registry=dummy_halos.unit_registry)
                assert result.units.same_dimensions_as(unit.units)

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


if __name__ == "__main__":
    """
    Standalone version of the program: just run the unit test.

    Note that this can also be achieved by running "pytest *.py" in the folder.
    """

    print("Running test_aperture_properties()...")
    test_aperture_properties()
    print("Test passed.")
