#!/bin/env python

"""
subhalo_properties.py

Calculate subhalo properties using all the particles deemed to be
a member of that subhalo.

Note that all the membership information used to determine which particles are
a member of a subhalo comes directly from the halo finder, SOAP does not
perform any FOF algorithm or boundedness calculations.

Just like the other HaloProperty implementations, the calculation of the
properties is done lazily: only calculations that are actually needed are
performed. See aperture_properties.py for a fully documented example.
"""

import time
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
import unyt

from .halo_properties import HaloProperty, SearchRadiusTooSmallError
from SOAP.property_calculation.half_mass_radius import get_half_mass_radius
from SOAP.property_calculation.kinematic_properties import (
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_inertia_tensor,
    get_velocity_dispersion_matrix,
)
from SOAP.particle_filter.recently_heated_gas_filter import RecentlyHeatedGasFilter
from SOAP.property_calculation.stellar_age_calculator import StellarAgeCalculator
from SOAP.property_table import PropertyTable
from SOAP.core.dataset_names import mass_dataset
from SOAP.core.lazy_properties import lazy_property
from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.core.snapshot_datasets import SnapshotDatasets
from SOAP.core.swift_cells import SWIFTCellGrid


class SubhaloParticleData:
    """
    Halo calculation class.

    All properties we want to compute in apertures are implemented as lazy
    methods of this class.

    Note that unlike aperture-based halo property calculations, subhalo calculations
    only require a single mask, since their membership is purely based on VR
    membership information and is irrespective of the particle position.
    That said, we still require a types==PartTypeX mask
    (see aperture_properties.py) to access some arrays that have been
    precomputed for all particles.
    """

    def __init__(
        self,
        input_halo: Dict,
        data: Dict,
        types_present: List[str],
        stellar_age_calculator: StellarAgeCalculator,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        snapshot_datasets: SnapshotDatasets,
        softening_of_parttype: unyt.unyt_array,
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
         - stellar_age_calculator: StellarAgeCalculator
           Object used to compute stellar ages from the current cosmological scale factor
           and the birth scale factors of star particles.
         - recently_heated_gas_filter: RecentlyHeatedGasFilter
           Filter used to mask out gas particles that were recently heated by
           AGN feedback.
         - snapshot_datasets: SnapshotDatasets
           Object containing metadata about the datasets in the snapshot, like
           appropriate aliases and column names.
         - boxsize: unyt.unyt_quantity
           Boxsize for correcting periodic boundary conditions
        """
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.stellar_age_calculator = stellar_age_calculator
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.snapshot_datasets = snapshot_datasets
        self.softening_of_parttype = softening_of_parttype
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

        mass = []
        position = []
        radius = []
        velocity = []
        types = []
        softening = []
        for ptype in self.types_present:
            grnr = self.get_dataset(f"{ptype}/GroupNr_bound")
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

    @lazy_property
    def gas_mask_sh(self) -> NDArray[bool]:
        """
        Mask used to mask out gas particles that belong to this subhalo in
        arrays containing all particles, e.g. self.mass.
        """
        return self.types == 0

    @lazy_property
    def dm_mask_sh(self) -> NDArray[bool]:
        """
        Mask used to mask out dark matter particles that belong to this subhalo in
        arrays containing all particles, e.g. self.mass.
        """
        return self.types == 1

    @lazy_property
    def star_mask_sh(self) -> NDArray[bool]:
        """
        Mask used to mask out star particles that belong to this subhalo in
        arrays containing all particles, e.g. self.mass.
        """
        return self.types == 4

    @lazy_property
    def bh_mask_sh(self) -> NDArray[bool]:
        """
        Mask used to mask out black hole particles that belong to this subhalo in
        arrays containing all particles, e.g. self.mass.
        """
        return self.types == 5

    @lazy_property
    def baryons_mask_sh(self) -> NDArray[bool]:
        """
        Mask used to mask out baryon (gas + star) particles that belong to this subhalo in
        arrays containing all particles, e.g. self.mass.
        """
        return self.gas_mask_sh | self.star_mask_sh

    @lazy_property
    def Ngas(self) -> int:
        """
        Number of gas particles in the subhalo.
        """
        return self.gas_mask_sh.sum()

    @lazy_property
    def Ndm(self) -> int:
        """
        Number of dark matter particles in the subhalo.
        """
        return self.dm_mask_sh.sum()

    @lazy_property
    def Nstar(self) -> int:
        """
        Number of star particles in the subhalo.
        """
        return self.star_mask_sh.sum()

    @lazy_property
    def Nbh(self) -> int:
        """
        Number of black hole particles in the subhalo.
        """
        return self.bh_mask_sh.sum()

    @lazy_property
    def mass_gas(self) -> unyt.unyt_array:
        """
        Masses of the gas particles in the subhalo.
        """
        return self.mass[self.gas_mask_sh]

    @lazy_property
    def mass_dust(self) -> unyt.unyt_array:
        """
        Masses of the dust particles in the subhalo.
        """
        return self.gas_total_dust_mass_fractions * self.mass_gas

    @lazy_property
    def mass_dm(self) -> unyt.unyt_array:
        """
        Masses of the dark matter particles in the subhalo.
        """
        return self.mass[self.dm_mask_sh]

    @lazy_property
    def mass_star(self) -> unyt.unyt_array:
        """
        Masses of the star particles in the subhalo.
        """
        return self.mass[self.star_mask_sh]

    @lazy_property
    def mass_baryons(self) -> unyt.unyt_array:
        """
        Masses of the baryon (gas + star) particles in the subhalo.
        """
        return self.mass[self.baryons_mask_sh]

    @lazy_property
    def pos_gas(self) -> unyt.unyt_array:
        """
        Positions of the gas particles in the subhalo.
        """
        return self.position[self.gas_mask_sh]

    @lazy_property
    def pos_dm(self) -> unyt.unyt_array:
        """
        Positions of the dark matter particles in the subhalo.
        """
        return self.position[self.dm_mask_sh]

    @lazy_property
    def pos_star(self) -> unyt.unyt_array:
        """
        Positions of the star particles in the subhalo.
        """
        return self.position[self.star_mask_sh]

    @lazy_property
    def pos_baryons(self) -> unyt.unyt_array:
        """
        Positions of the baryon (gas + star) particles in the subhalo.
        """
        return self.position[self.baryons_mask_sh]

    @lazy_property
    def vel_gas(self) -> unyt.unyt_array:
        """
        Velocities of the gas particles in the subhalo.
        """
        return self.velocity[self.gas_mask_sh]

    @lazy_property
    def vel_dm(self) -> unyt.unyt_array:
        """
        Velocities of the dark matter particles in the subhalo.
        """
        return self.velocity[self.dm_mask_sh]

    @lazy_property
    def vel_star(self) -> unyt.unyt_array:
        """
        Velocities of the star particles in the subhalo.
        """
        return self.velocity[self.star_mask_sh]

    @lazy_property
    def vel_baryons(self) -> unyt.unyt_array:
        """
        Velocities of the baryon (gas + star) particles in the subhalo.
        """
        return self.velocity[self.baryons_mask_sh]

    @lazy_property
    def Mtot(self) -> unyt.unyt_quantity:
        """
        Total mass of the particles in the subhalo.
        """
        return self.mass.sum()

    @lazy_property
    def Mgas(self) -> unyt.unyt_quantity:
        """
        Total mass of the gas particles in the subhalo.
        """
        return self.mass_gas.sum()

    @lazy_property
    def DustMass(self) -> unyt.unyt_quantity:
        """
        Total dust mass of the gas particles in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return self.mass_dust.sum()

    @lazy_property
    def gas_total_dust_mass_fractions(self) -> unyt.unyt_array:
        """
        Total dust mass fractions in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/TotalDustMassFractions")[self.gas_mask_all]

    @lazy_property
    def Mdm(self) -> unyt.unyt_quantity:
        """
        Total mass of the dark matter particles in the subhalo.
        """
        return self.mass_dm.sum()

    @lazy_property
    def Mstar(self) -> unyt.unyt_quantity:
        """
        Total mass of the star particles in the subhalo.
        """
        return self.mass_star.sum()

    @lazy_property
    def Mbh_dynamical(self) -> unyt.unyt_quantity:
        """
        Total dynamical mass of the black hole particles in the subhalo.
        """
        return self.mass[self.bh_mask_sh].sum()

    @lazy_property
    def star_mask_all(self) -> NDArray[bool]:
        """
        Mask that can be used to filter out star particles belonging to this
        subhalo in raw particle arrays, e.g. PartType4/Masses.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset(f"PartType4/GroupNr_bound") == self.index

    @lazy_property
    def mass_star_init(self) -> unyt.unyt_array:
        """
        Initial stellar masses of star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/InitialMasses")[self.star_mask_all]

    @lazy_property
    def Mstar_init(self) -> unyt.unyt_quantity:
        """
        Total initial stellar mass of star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.mass_star_init.sum()

    @lazy_property
    def stellar_luminosities(self) -> unyt.unyt_array:
        """
        Stellar luminosities of star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/Luminosities")[self.star_mask_all]

    @lazy_property
    def StellarLuminosity(self) -> unyt.unyt_array:
        """
        Total stellar luminosity of star particles in the subhalo.

        Note that this is an array, since there are multiple luminosity bands.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_luminosities.sum(axis=0)

    @lazy_property
    def starmetalfrac(self) -> unyt.unyt_quantity:
        """
        Total metal mass fraction in star particles in the subhalo.

        Given as a fraction of the total mass in star particles.
        """
        if self.Nstar == 0:
            return None
        return (
            self.mass_star
            * self.get_dataset("PartType4/MetalMassFractions")[self.star_mask_all]
        ).sum() / self.Mstar

    @lazy_property
    def stellar_ages(self) -> unyt.unyt_array:
        """
        Stellar ages of star particles.

        Uses the StellarAgeCalculator to convert the stellar birth scale factor
        into a stellar age.
        """
        if self.Nstar == 0:
            return None
        birth_a = self.get_dataset("PartType4/BirthScaleFactors")[self.star_mask_all]
        return self.stellar_age_calculator.stellar_age(birth_a)

    @lazy_property
    def stellar_age_mw(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average stellar age of star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return ((self.mass_star / self.Mstar) * self.stellar_ages).sum()

    @lazy_property
    def stellar_age_lw(self) -> unyt.unyt_array:
        """
        R-band luminosity weighted average stellar age of star particles in the
        subhalo.

        This assumes that the Luminosities have a named column with the name
        "GAMA_r".
        """
        if self.Nstar == 0:
            return None
        Lr = self.stellar_luminosities[
            :, self.snapshot_datasets.get_column_index("Luminosities", "GAMA_r")
        ]
        Lrtot = Lr.sum()
        return ((Lr / Lrtot) * self.stellar_ages).sum()

    @lazy_property
    def bh_mask_all(self) -> NDArray[bool]:
        """
        Mask that can be used to filter out black hole particles belonging to this
        subhalo in raw particle arrays, e.g. PartType5/DynamicalMasses.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset(f"PartType5/GroupNr_bound") == self.index

    @lazy_property
    def Mbh_subgrid(self) -> unyt.unyt_quantity:
        """
        Total sub-grid mass of black hole particles in the subhalo.
        """
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses.sum()

    @lazy_property
    def agn_eventa(self) -> unyt.unyt_array:
        """
        Last AGN feedback scale factor of black hole particles in the subhalo.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
            self.bh_mask_all
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
    def BH_subgrid_masses(self) -> unyt.unyt_array:
        """
        Sub-grid masses of black hole particles in the subhalo.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all]

    @lazy_property
    def BlackHolesTotalInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Total thermal energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.get_dataset("PartType5/AGNTotalInjectedEnergies")[self.bh_mask_all]
        )

    @lazy_property
    def BlackHolesTotalInjectedJetEnergy(self) -> unyt.unyt_quantity:
        """
        Total jet energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.get_dataset("PartType5/InjectedJetEnergies")[self.bh_mask_all]
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
        return self.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxpos(self) -> unyt.unyt_array:
        """
        Position of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Coordinates")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxvel(self) -> unyt.unyt_array:
        """
        Velocity of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Velocities")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxAR(self) -> unyt.unyt_quantity:
        """
        Accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionRates")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleAveragedAccretionRate(self) -> unyt.unyt_quantity:
        """
        Averaged accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AveragedAccretionRates")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Total thermal energy injected into gas particles by the most massive
        BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AGNTotalInjectedEnergies")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleAccretionMode(self) -> unyt.unyt_quantity:
        """
        Accretion flow regime of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionModes")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleGWMassLoss(self) -> unyt.unyt_quantity:
        """
        Cumulative mass lost to GW of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/GWMassLosses")[self.bh_mask_all][self.iBHmax]

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
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleLastJetEventScalefactor(self) -> unyt.unyt_quantity:
        """
        Scale-factor of last jet event of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNJetScaleFactors")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNEvents(self) -> unyt.unyt_quantity:
        """
        Number of AGN events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNEvents")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNJetEvents(self) -> unyt.unyt_quantity:
        """
        Number of jet events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfAGNJetEvents")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleNumberOfMergers(self) -> unyt.unyt_quantity:
        """
        Number of mergers the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/NumberOfMergers")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleRadiatedEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into radiation by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/RadiatedEnergiesByMode")[self.bh_mask_all][
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
            self.bh_mask_all
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleWindEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into accretion disc winds by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/WindEnergiesByMode")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleSpin(self) -> unyt.unyt_quantity:
        """
        The spin of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Spins")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMass(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/TotalAccretedMasses")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def MostMassiveBlackHoleFormationScalefactor(self) -> unyt.unyt_quantity:
        """
        The formation scale factor of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/FormationScaleFactors")[self.bh_mask_all][
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
    def total_mass_fraction(self) -> unyt.unyt_array:
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
        if self.Mtot == 0:
            return None
        return self.mass / self.Mtot

    @lazy_property
    def com(self) -> unyt.unyt_array:
        """
        Centre of mass of all particles in the subhalo.
        """
        if self.Mtot == 0:
            return None
        return (
            (self.total_mass_fraction[:, None] * self.position).sum(axis=0)
            + self.centre
        ) % self.boxsize

    @lazy_property
    def com_star(self) -> unyt.unyt_array:
        """
        Centre of mass of star particles in the subhalo.
        """
        if self.Mstar == 0:
            return None
        return (
            (self.star_mass_fraction[:, None] * self.pos_star).sum(axis=0) + self.centre
        ) % self.boxsize

    @lazy_property
    def vcom(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles in the subhalo.
        """
        if self.Mtot == 0:
            return None
        return (self.total_mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def R_vmax_unsoft(self) -> unyt.unyt_quantity:
        """
        Radius at which the maximum circular velocity of the halo is reached.
        Particles are not constrained to be at least one softening length away
        from the centre.

        This includes contributions from all particle types.
        """
        if self.Mtot == 0:
            return None
        if not hasattr(self, "r_vmax_unsoft"):
            self.r_vmax_unsoft, self.vmax_unsoft = get_vmax(
                self.mass, self.radius, nskip=1
            )
        return self.r_vmax_unsoft

    @lazy_property
    def Vmax_unsoft(self) -> unyt.unyt_quantity:
        """
        Maximum circular velocity of the halo.
        Particles are not constrained to be at least one softening length away
        from the centre.

        This includes contributions from all particle types.
        """
        if self.Mtot == 0:
            return None
        if not hasattr(self, "vmax_unsoft"):
            self.r_vmax_unsoft, self.vmax_unsoft = get_vmax(
                self.mass, self.radius, nskip=1
            )
        return self.vmax_unsoft

    @lazy_property
    def R_vmax_soft(self) -> unyt.unyt_quantity:
        """
        Radius at which the maximum circular velocity of the halo is reached.
        Particles are set to have minimum radius equal to their softening length.

        This includes contributions from all particle types.
        """
        if self.Mtot == 0:
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
        if self.Mtot == 0:
            return None
        if not hasattr(self, "vmax_soft"):
            soft_r = np.maximum(self.softening, self.radius)
            self.r_vmax_soft, self.vmax_soft = get_vmax(self.mass, soft_r)
        return self.vmax_soft

    @lazy_property
    def spin_parameter(self) -> unyt.unyt_quantity:
        """
        Spin parameter of all particles in the subhalo.

        Computed as in Bullock et al. (2021):
          lambda = |Ltot| / (sqrt(2) * M * v_max * R)

        Since a subhalo does not have a characteristic radius, R, we instead use
        the radius at which v_max is reached (and the corresponding mass).
        """
        if self.Mtot == 0:
            return None
        if self.R_vmax_soft > 0 and self.Vmax_soft > 0:
            mask_r_vmax = self.radius <= self.R_vmax_soft
            vrel = self.velocity[mask_r_vmax, :] - self.vcom[None, :]
            Ltot = np.linalg.norm(
                (
                    self.mass[mask_r_vmax, None]
                    * np.cross(self.position[mask_r_vmax, :], vrel)
                ).sum(axis=0)
            )
            M_r_vmax = self.mass[mask_r_vmax].sum()
            if M_r_vmax > 0:
                return Ltot / (
                    np.sqrt(2.0) * M_r_vmax * self.Vmax_soft * self.R_vmax_soft
                )
        return None

    @lazy_property
    def TotalInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusTot. Only considers bound particles.
        """
        if self.Mtot == 0:
            return None
        return get_inertia_tensor(self.mass, self.position, self.HalfMassRadiusTot)

    @lazy_property
    def TotalInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusTot. Only considers bound particles.
        """
        if self.Mtot == 0:
            return None
        return get_inertia_tensor(
            self.mass, self.position, self.HalfMassRadiusTot, reduced=True
        )

    @lazy_property
    def TotalInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution.
        Computed using all bound particles within HalfMassRadiusTot.
        """
        if self.Mtot == 0:
            return None
        return get_inertia_tensor(
            self.mass, self.position, self.HalfMassRadiusTot, max_iterations=1
        )

    @lazy_property
    def TotalInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution.
        Computed using all bound particles within HalfMassRadiusTot.
        """
        if self.Mtot == 0:
            return None
        return get_inertia_tensor(
            self.mass,
            self.position,
            self.HalfMassRadiusTot,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def gas_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of gas particles in the subhalo.

        See total_mass_fraction() for the rationale behind this function.
        """
        if self.Mgas == 0:
            return None
        return self.mass_gas / self.Mgas

    @lazy_property
    def vcom_gas(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of gas particles in the subhalo.
        """
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.vel_gas).sum(axis=0)

    def compute_Lgas_props(self):
        """
        Auxiliary function used to compute a number of properties that depend
        on Lgas. It is more efficient to compute these properties together.
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
        Total angular momentum of gas particles in the subhalo.

        Calls compute_Lgas_props() if required.
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Lgas"):
            self.compute_Lgas_props()
        return self.internal_Lgas

    @lazy_property
    def kappa_corot_gas(self) -> unyt.unyt_quantity:
        """
        Ratio of the kinetic energy in counter-rotating rotation to the total
        kinetic energy for gas particles in the subhalo.

        Calls compute_Lgas_props() if required.
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_kappa_gas"):
            self.compute_Lgas_props()
        return self.internal_kappa_gas

    @lazy_property
    def DtoTgas(self) -> unyt.unyt_quantity:
        """
        Disc to total mass ratio for gas particles in the subhalo.

        Calls compute_Lgas_props() if required.
        """
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_gas"):
            self.compute_Lgas_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_gas / self.Mgas

    @lazy_property
    def GasInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusGas. Only considers bound particles.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(self.mass_gas, self.pos_gas, self.HalfMassRadiusGas)

    @lazy_property
    def GasInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusGas. Only considers bound particles.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(
            self.mass_gas, self.pos_gas, self.HalfMassRadiusGas, reduced=True
        )

    @lazy_property
    def GasInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution.
        Computed using all bound gas particles within HalfMassRadiusGas.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(
            self.mass_gas, self.pos_gas, self.HalfMassRadiusGas, max_iterations=1
        )

    @lazy_property
    def GasInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution.
        Computed using all bound gas particles within HalfMassRadiusGas.
        """
        if self.Mgas == 0:
            return None
        return get_inertia_tensor(
            self.mass_gas,
            self.pos_gas,
            self.HalfMassRadiusGas,
            reduced=True,
            max_iterations=1,
        )

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
    def dm_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of dark matter particles in the subhalo.

        See total_mass_fraction() for the rationale behind this function.
        """
        if self.Mdm == 0:
            return None
        return self.mass_dm / self.Mdm

    @lazy_property
    def vcom_dm(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of dark matter particles in the subhalo.
        """
        if self.Mdm == 0:
            return None
        return (self.dm_mass_fraction[:, None] * self.vel_dm).sum(axis=0)

    @lazy_property
    def Ldm(self) -> unyt.unyt_array:
        """
        Total angular momentum of dark matter particles in the subhalo.
        """
        if self.Mdm == 0:
            return None
        return get_angular_momentum(
            self.mass_dm, self.pos_dm, self.vel_dm, ref_velocity=self.vcom_dm
        )

    @lazy_property
    def DarkMatterInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the dark matter mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusDM. Only considers bound particles.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(self.mass_dm, self.pos_dm, self.HalfMassRadiusDM)

    @lazy_property
    def DarkMatterInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the dark matter mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusDM. Only considers bound particles.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(
            self.mass_dm, self.pos_dm, self.HalfMassRadiusDM, reduced=True
        )

    @lazy_property
    def DarkMatterInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the dark matter mass distribution.
        Computed using all bound DM particles within HalfMassRadiusDM.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(
            self.mass_dm, self.pos_dm, self.HalfMassRadiusDM, max_iterations=1
        )

    @lazy_property
    def DarkMatterInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the dark matter mass distribution.
        Computed using all bound DM particles within HalfMassRadiusDM.
        """
        if self.Mdm == 0:
            return None
        return get_inertia_tensor(
            self.mass_dm,
            self.pos_dm,
            self.HalfMassRadiusDM,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def veldisp_matrix_dm(self) -> unyt.unyt_array:
        """
        Velocity dispersion matrix of dark matter particles in the subhalo.
        """
        if self.Mdm == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.dm_mass_fraction, self.vel_dm, self.vcom_dm
        )

    @lazy_property
    def DM_Vmax_soft(self) -> unyt.unyt_quantity:
        """
        Maximum circular velocity of the dark matter particles in the subhalo.
        Particles are set to have minimum radius equal to their softening length.
        """
        if self.Ndm == 0:
            return None
        if not hasattr(self, "DM_r_vmax_soft"):
            soft_r = np.maximum(
                self.softening[self.dm_mask_sh], self.radius[self.dm_mask_sh]
            )
            self.DM_r_vmax_soft, self.DM_vmax_soft = get_vmax(self.mass_dm, soft_r)
        return self.DM_vmax_soft

    @lazy_property
    def DM_R_vmax_soft(self) -> unyt.unyt_quantity:
        """
        Radius for which the maximum circular velocity of dark matter particles
        is reached.
        Particles are set to have minimum radius equal to their softening length.
        """
        if self.Ndm == 0:
            return None
        if not hasattr(self, "DM_r_vmax_soft"):
            soft_r = np.maximum(
                self.softening[self.dm_mask_sh], self.radius[self.dm_mask_sh]
            )
            self.DM_r_vmax_soft, self.DM_vmax_soft = get_vmax(self.mass_dm, soft_r)
        return self.DM_r_vmax_soft

    @lazy_property
    def star_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fraction of star particles in the subhalo.

        See total_mass_fraction() for the rationale behind this function.
        """
        if self.Mstar == 0:
            return None
        return self.mass_star / self.Mstar

    @lazy_property
    def vcom_star(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of star particles in the subhalo.
        """
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.vel_star).sum(axis=0)

    def compute_Lstar_props(self):
        """
        Auxiliary function used to compute some properties that depend on Lstar.
        It is more efficient to compute these properties together.
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
        Total angular momentum of star particles in the subhalo.

        Calls compute_Lstar_props() if required.
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Lstar"):
            self.compute_Lstar_props()
        return self.internal_Lstar

    @lazy_property
    def kappa_corot_star(self) -> unyt.unyt_quantity:
        """
        Ratio of the kinetic energy in counter-rotating rotation and the total
        kinetic energy for star particles in the subhalo.

        Calls compute_Lstar_props() if required.
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_kappa_star"):
            self.compute_Lstar_props()
        return self.internal_kappa_star

    @lazy_property
    def DtoTstar(self) -> unyt.unyt_quantity:
        """
        Disc to total mass ratio for star particles in the subhalo.

        Calls compute_Lstar_props() if required.
        """
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Mcountrot_star"):
            self.compute_Lstar_props()
        return 1.0 - 2.0 * self.internal_Mcountrot_star / self.Mstar

    @lazy_property
    def StellarInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusStar. Only considers bound particles.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.mass_star, self.pos_star, self.HalfMassRadiusStar
        )

    @lazy_property
    def StellarInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution.
        Computed iteratively using an ellipsoid with volume equal to that of
        a sphere with radius HalfMassRadiusStar. Only considers bound particles.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.mass_star, self.pos_star, self.HalfMassRadiusStar, reduced=True
        )

    @lazy_property
    def StellarInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution.
        Computed using all bound star particles within HalfMassRadiusStar.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.mass_star, self.pos_star, self.HalfMassRadiusStar, max_iterations=1
        )

    @lazy_property
    def StellarInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution.
        Computed using all bound star particles within HalfMassRadiusStar.
        """
        if self.Mstar == 0:
            return None
        return get_inertia_tensor(
            self.mass_star,
            self.pos_star,
            self.HalfMassRadiusStar,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def veldisp_matrix_star(self) -> unyt.unyt_array:
        """
        Velocity dispersion matrix of the star particles in the subhalo.
        """
        if self.Mstar == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.star_mass_fraction, self.vel_star, self.vcom_star
        )

    @lazy_property
    def Mbaryon(self) -> unyt.unyt_quantity:
        """
        Total mass of baryon (gas + star) particles in the subhalo.
        """
        return self.Mgas + self.Mstar

    @lazy_property
    def baryon_mass_fraction(self) -> unyt.unyt_array:
        """
        Mass fractions of baryon (gas + star) particles in the subhalo.

        See total_mass_fraction() for the rationale behind this function.
        """
        if self.Mbaryon == 0:
            return None
        return self.mass_baryons / self.Mbaryon

    @lazy_property
    def vcom_bar(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of baryon (gas + star) particles in the subhalo.
        """
        if self.Mbaryon == 0:
            return None
        return (self.baryon_mass_fraction[:, None] * self.vel_baryons).sum(axis=0)

    def compute_Lbar_props(self):
        """
        Auxiliary function used to compute a number of properties that depend on
        Lbar. It is more efficient to compute these properties together.
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
        Total angular momentum of baryon (gas + star) particles in the subhalo.

        Calls compute_Lbar_props() if required.
        """
        if self.Mbaryon == 0:
            return None
        if not hasattr(self, "internal_Lbar"):
            self.compute_Lbar_props()
        return self.internal_Lbar

    @lazy_property
    def kappa_corot_baryons(self) -> unyt.unyt_quantity:
        """
        Ratio of the total kinetic energy in counter-rotating rotation and the
        total kinetic energy for baryon (gas + star) particles in the subhalo.

        Calls compute_Lbar_props() if required.
        """
        if self.Mbaryon == 0:
            return None
        if not hasattr(self, "internal_kappa_bar"):
            self.compute_Lbar_props()
        return self.internal_kappa_bar

    @lazy_property
    def gas_mask_all(self) -> NDArray[bool]:
        """
        Mask that can be used to filter out gas particles that belong to this
        subhalo in raw particle arrays, like PartType0/Masses.
        """
        return self.get_dataset(f"PartType0/GroupNr_bound") == self.index

    @lazy_property
    def gas_SFR(self) -> unyt.unyt_array:
        """
        Star formation rates of gas particles.

        Note that older versions of SWIFT would use negative values to store
        the last star formation scale factor, so we need to filter these out.
        """
        if self.Ngas == 0:
            return None
        # remember: SFR < 0. is not SFR at all!
        all_SFR = self.get_dataset("PartType0/StarFormationRates")[self.gas_mask_all]
        all_SFR[all_SFR < 0.0] = 0.0
        return all_SFR

    @lazy_property
    def SFR(self) -> unyt.unyt_quantity:
        """
        Total star formation rate of gas particles in the subhalo.
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
            self.gas_mask_all
        ]
        return np.sum(avg_SFR, axis=0)

    @lazy_property
    def gas_metal_mass(self) -> unyt.unyt_array:
        """
        Metal masses of gas particles in the subhalo.

        Includes contributions from dust.
        """
        if self.Ngas == 0:
            return None
        return (
            self.mass_gas
            * self.get_dataset("PartType0/MetalMassFractions")[self.gas_mask_all]
        )

    @lazy_property
    def gasmetalfrac(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction of gas particles in the subhalo.

        Given as a fraction of the total gas mass.

        Includes contributions from dust.
        """
        if self.Ngas == 0:
            return None
        return self.gas_metal_mass.sum() / self.Mgas

    @lazy_property
    def Mgas_SF(self) -> unyt.unyt_quantity:
        """
        Total mass of star-forming gas in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return self.mass_gas[self.gas_SFR > 0.0].sum()

    @lazy_property
    def gasmetalfrac_SF(self) -> unyt.unyt_quantity:
        """
        Metal mass fraction of star-forming gas in the subhalo.

        Given as a fraction of the total star-forming gas mass.

        Includes contributions from dust.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_metal_mass[self.gas_SFR > 0.0].sum() / self.Mgas_SF

    @lazy_property
    def gas_temp(self) -> unyt.unyt_array:
        """
        Temperatures of gas particles in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Temperatures")[self.gas_mask_all]

    @lazy_property
    def last_agn_gas(self) -> unyt.unyt_array:
        """
        Last AGN feedback scale factors of gas particles in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/LastAGNFeedbackScaleFactors")[
            self.gas_mask_all
        ]

    @lazy_property
    def gas_no_agn(self) -> NDArray[bool]:
        """
        Mask to filter out gas particles that were recently heated by AGN
        feedback.

        Obtains the mask from the RecentlyHeatedGasFilter.
        """
        if self.Ngas == 0:
            return None
        return ~self.recently_heated_gas_filter.is_recently_heated(
            self.last_agn_gas, self.gas_temp
        )

    @lazy_property
    def gas_no_cool(self) -> NDArray[bool]:
        """
        Mask to filter out cool gas particles (i.e. temperature < 1.e5 K).
        """
        if self.Ngas == 0:
            return None
        return self.gas_temp >= 1.0e5 * unyt.K

    @lazy_property
    def Tgas(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (self.gas_mass_fraction * self.gas_temp).sum()

    @lazy_property
    def Tgas_no_cool(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of non-cool gas particles (i.e.
        excluding gas particles with temperature < 1.e5 K).
        """
        if self.Ngas == 0:
            return None
        if np.any(self.gas_no_cool):
            mass_gas_no_cool = self.mass_gas[self.gas_no_cool]
            Mgas_no_cool = mass_gas_no_cool.sum()
            if Mgas_no_cool > 0:
                return (
                    (mass_gas_no_cool / Mgas_no_cool) * self.gas_temp[self.gas_no_cool]
                ).sum()
        return None

    @lazy_property
    def Tgas_no_agn(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of gas particles, excluding gas that
        was recently heated by AGN feedback.
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
    def Tgas_no_cool_no_agn(self) -> unyt.unyt_quantity:
        """
        Mass-weighted average temperature of gas particles, excluding gas that
        was recently heated by AGN feedback, or non-cool gas (i.e.
        temperature < 1.e5 K).
        """
        if self.Ngas == 0:
            return None
        no_cool_no_agn = self.gas_no_agn & self.gas_no_cool
        if np.any(no_cool_no_agn):
            mass_gas_no_cool_no_agn = self.mass_gas[no_cool_no_agn]
            Mgas_no_cool_no_agn = mass_gas_no_cool_no_agn.sum()
            if Mgas_no_cool_no_agn > 0:
                return (
                    (mass_gas_no_cool_no_agn / Mgas_no_cool_no_agn)
                    * self.gas_temp[no_cool_no_agn]
                ).sum()
        return None

    @lazy_property
    def stellar_birth_density(self) -> unyt.unyt_array:
        """
        Birth densities of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/BirthDensities")[self.star_mask_all]

    @lazy_property
    def MedianStellarBirthDensity(self) -> unyt.unyt_quantity:
        """
        Median birth density of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return np.median(self.stellar_birth_density)

    @lazy_property
    def MinimumStellarBirthDensity(self) -> unyt.unyt_quantity:
        """
        Minimum birth density of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_density.min()

    @lazy_property
    def MaximumStellarBirthDensity(self) -> unyt.unyt_quantity:
        """
        Maximum birth density of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_density.max()

    @lazy_property
    def stellar_birth_pressure(self) -> unyt.unyt_array:
        """
        Birth pressures of star particles.
        """
        if self.Nstar == 0:
            return None
        # Pressure in physical units can overflow float32
        birth_densities = self.stellar_birth_density.astype(np.float64) / unyt.mh
        return birth_densities * self.stellar_birth_temperature

    @lazy_property
    def MedianStellarBirthPressure(self) -> unyt.unyt_quantity:
        """
        Median birth pressure of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return np.median(self.stellar_birth_pressure)

    @lazy_property
    def MinimumStellarBirthPressure(self) -> unyt.unyt_quantity:
        """
        Minimum birth pressure of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_pressure.min()

    @lazy_property
    def MaximumStellarBirthPressure(self) -> unyt.unyt_quantity:
        """
        Maximum birth pressure of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_pressure.max()

    @lazy_property
    def stellar_birth_temperature(self) -> unyt.unyt_array:
        """
        Birth temperatures of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/BirthTemperatures")[self.star_mask_all]

    @lazy_property
    def MedianStellarBirthTemperature(self) -> unyt.unyt_quantity:
        """
        Median birth temperature of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return np.median(self.stellar_birth_temperature)

    @lazy_property
    def MinimumStellarBirthTemperature(self) -> unyt.unyt_quantity:
        """
        Minimum birth temperature of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_temperature.min()

    @lazy_property
    def MaximumStellarBirthTemperature(self) -> unyt.unyt_quantity:
        """
        Maximum birth temperature of the star particles in the subhalo.
        """
        if self.Nstar == 0:
            return None
        return self.stellar_birth_temperature.max()

    @lazy_property
    def LastSupernovaEventMaximumGasDensity(self) -> unyt.unyt_quantity:
        """
        Maximum gas density during the last feedback event of the gas particles.

        To clarify: each gas particle carries a tracer for the density it had
        when it was last hit by feedback. This property simply returns the
        maximum of these tracer values of all the gas particles. This ignores
        the fact that not all feedback events happened at the same time, and the
        fact that a gas particle could have received feedback when it had a
        higher density in its past, before being hit by the final feedback
        event that was registered.

        Since we have tracers for both thermal and kinetic SNII feedback, we
        use the maximum of these two types of feedback.
        """
        if self.Ngas == 0:
            return None
        rho_thermal = self.get_dataset("PartType0/LastSNIIThermalFeedbackDensities")[
            self.gas_mask_all
        ]
        rho_kinetic = self.get_dataset("PartType0/LastSNIIKineticFeedbackDensities")[
            self.gas_mask_all
        ]
        return max(rho_thermal.max(), rho_kinetic.max())

    @lazy_property
    def HalfMassRadiusTot(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the total mass distribution in the subhalo.
        """
        return get_half_mass_radius(self.radius, self.mass, self.Mtot)

    @lazy_property
    def HalfMassRadiusDust(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the dust particle distribution in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return get_half_mass_radius(
            self.radius[self.gas_mask_sh], self.mass_dust, self.DustMass
        )

    @lazy_property
    def HalfMassRadiusGas(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the gas particle distribution in the subhalo.
        """
        return get_half_mass_radius(
            self.radius[self.gas_mask_sh], self.mass_gas, self.Mgas
        )

    @lazy_property
    def HalfMassRadiusDM(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the dark matter particle distribution in the subhalo.
        """
        return get_half_mass_radius(
            self.radius[self.dm_mask_sh], self.mass_dm, self.Mdm
        )

    @lazy_property
    def HalfMassRadiusStar(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the star particle distribution in the subhalo.
        """
        return get_half_mass_radius(
            self.radius[self.star_mask_sh], self.mass_star, self.Mstar
        )

    @lazy_property
    def HalfMassRadiusBaryon(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the baryon (gas + star) particle distribution in the
        subhalo.
        """
        return get_half_mass_radius(
            self.radius[self.gas_mask_sh | self.star_mask_sh],
            self.mass[self.gas_mask_sh | self.star_mask_sh],
            self.Mgas + self.Mstar,
        )

    @lazy_property
    def EncloseRadius(self) -> unyt.unyt_quantity:
        """
        Maximum radius of particles in the subhalo.
        """
        if self.Mtot == 0:
            return None
        return np.max(self.radius)


class SubhaloProperties(HaloProperty):
    """
    Compute subhalo properties for halos.

    The subhalo contains all the particles that were identified to be members
    of this subhalo by VR, either as part of the original FOF group, or as
    gravitationally bound particles.
    """

    """
    List of properties from the table that we want to compute.
    Each property should have a corresponding method/property/lazy_property in
    the SubhaloParticleData class above.
    """
    base_halo_type = "SubhaloProperties"
    property_list = {
        prop: PropertyTable.full_property_list[prop]
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
            "Lgas",
            "Ldm",
            "Lstar",
            "kappa_corot_gas",
            "kappa_corot_star",
            "Lbaryons",
            "kappa_corot_baryons",
            "gasmetalfrac",
            "Tgas",
            "Tgas_no_cool",
            "Tgas_no_agn",
            "Tgas_no_cool_no_agn",
            "SFR",
            "AveragedStarFormationRate",
            "StellarLuminosity",
            "starmetalfrac",
            "Vmax_unsoft",
            "Vmax_soft",
            "R_vmax_unsoft",
            "DM_Vmax_soft",
            "DM_R_vmax_soft",
            "spin_parameter",
            "DustMass",
            "HalfMassRadiusTot",
            "HalfMassRadiusDust",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "HalfMassRadiusBaryon",
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
            "veldisp_matrix_gas",
            "veldisp_matrix_dm",
            "veldisp_matrix_star",
            "DtoTgas",
            "DtoTstar",
            "stellar_age_mw",
            "stellar_age_lw",
            "Mgas_SF",
            "gasmetalfrac_SF",
            "MedianStellarBirthDensity",
            "MinimumStellarBirthDensity",
            "MaximumStellarBirthDensity",
            "MedianStellarBirthTemperature",
            "MinimumStellarBirthTemperature",
            "MaximumStellarBirthTemperature",
            "MedianStellarBirthPressure",
            "MinimumStellarBirthPressure",
            "MaximumStellarBirthPressure",
            "LastSupernovaEventMaximumGasDensity",
            "EncloseRadius",
        ]
    }

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        recently_heated_gas_filter: RecentlyHeatedGasFilter,
        stellar_age_calculator: StellarAgeCalculator,
        category_filter: CategoryFilter,
    ):
        """
        Construct a SubhaloProperties object.

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
         - stellar_age_calculator: StellarAgeCalculator
           Object used to calculate stellar ages from the current cosmological
           scale factor and the birth scale factor of the star particles.
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the subhalo and the category
           of each property.
        """

        super().__init__(cellgrid)

        self.property_filters = parameters.get_property_filters(
            "SubhaloProperties", [prop.name for prop in self.property_list.values()]
        )

        self.filter = recently_heated_gas_filter
        self.stellar_ages = stellar_age_calculator
        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets
        self.record_timings = parameters.record_property_timings
        self.boxsize = cellgrid.boxsize

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.0

        # Give this calculation a name so we can select it on the command line
        # Save mask metadata and name of group in the final output file
        self.name = "bound_subhalo"
        self.group_name = "BoundSubhalo"
        self.mask_metadata = {"Masked": False}
        self.halo_filter = "basic"

        # Arrays which must be read in for this calculation.
        # Note that if there are no particles of a given type in the
        # snapshot, that type will not be read in and will not have
        # an entry in the data argument to calculate(), below.
        # (E.g. gas, star or BH particles in DMO runs)
        # Most arrays are linked to a particular property and are only
        # read if that particular property is actually requested
        # Some basic properties are always required; these are added below
        self.particle_properties = {
            "PartType0": ["Coordinates", "Masses", "Velocities", "GroupNr_bound"],
            "PartType1": ["Coordinates", "Masses", "Velocities", "GroupNr_bound"],
            "PartType4": ["Coordinates", "Masses", "Velocities", "GroupNr_bound"],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "Velocities",
                "GroupNr_bound",
            ],
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

    def calculate(self, input_halo, search_radius, data, halo_result):
        """
        Compute centre of mass etc of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius in which we have all particles
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        types_present = [type for type in self.particle_properties if type in data]

        part_props = SubhaloParticleData(
            input_halo,
            data,
            types_present,
            self.stellar_ages,
            self.filter,
            self.snapshot_datasets,
            self.softening_of_parttype,
            self.boxsize,
        )

        # this is the halo type that we use for the filter particle numbers,
        # so we have to pass the numbers for the category filters manually
        do_calculation = self.category_filter.get_do_calculation(
            halo_result,
            {
                "BoundSubhalo/NumberOfDarkMatterParticles": part_props.Ndm,
                "BoundSubhalo/NumberOfGasParticles": part_props.Ngas,
                "BoundSubhalo/NumberOfStarParticles": part_props.Nstar,
                "BoundSubhalo/NumberOfBlackHoleParticles": part_props.Nbh,
            },
        )

        subhalo = {}
        timings = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        registry = part_props.mass.units.registry
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
            subhalo[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=registry
            )
            if do_calculation[filter_name]:
                t0_calc = time.time()
                val = getattr(part_props, name)
                if val is not None:
                    assert (
                        subhalo[name].shape == val.shape
                    ), f"Attempting to store {name} with wrong dimensions"
                    if unit == unyt.Unit("dimensionless"):
                        if hasattr(val, "units"):
                            assert (
                                val.units == unyt.dimensionless
                            ), f"{name} is not dimensionless"
                        subhalo[name] = unyt.unyt_array(
                            val.astype(dtype),
                            dtype=dtype,
                            units=unit,
                            registry=registry,
                        )
                    else:
                        err = f'Overflow for halo {input_halo["index"]} when'
                        err += f"calculating {name} in subhalo_properties"
                        assert np.max(np.abs(val.to(unit).value)) < float("inf"), err
                        subhalo[name] += val
                    timings[name] = time.time() - t0_calc

        # Check that we found the expected number of halo member particles:
        # If not, we need to try again with a larger search radius.
        # For HBT this should not happen since we use the radius of the most distant
        # bound particle.
        Ntot = part_props.Ngas + part_props.Ndm + part_props.Nstar + part_props.Nbh
        Nexpected = input_halo["nr_bound_part"]
        if Ntot < Nexpected:
            # Try again with a larger search radius
            # print(f"Ntot = {Ntot}, Nexpected = {Nexpected}, search_radius = {search_radius}")
            raise SearchRadiusTooSmallError(
                "Search radius does not contain expected number of particles!"
            )
        elif Ntot > Nexpected:
            # This would indicate a bug somewhere
            raise RuntimeError(
                f'Found more particles than expected for halo {input_halo["index"]}'
            )

        # Add these properties to the output
        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            if not self.property_filters[outputname]:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            # Add data array and metadata to halo_result
            halo_result.update(
                {
                    f"{self.group_name}/{outputname}": (
                        subhalo[name],
                        prop.description,
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
                            "Time taken in seconds",
                            True,
                            None,
                        )
                    }
                )
