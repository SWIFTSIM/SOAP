##!/bin/env python

"""
projected_aperture_properties.py

Halo properties within projected 2D apertures. These only include
the gravitionally bound particles (the equivalent of the exclusive
3D apertures) of a subhalo, within a fixed physical radius.

Just like the other HaloProperty implementations, the calculation of the
properties is done lazily: only calculations that are actually needed are
performed. A fully documented explanation can be found in
aperture_properties.py.

Note that for the projected apertures we use a design that adds another
level of complexity: apart from the ProjectedApertureParticleData equivalent
of ApertureParticleData, we now also need a
SingleProjectionProjectedApertureParticleData object to deal with individual
projections. Besides this difference, the approach is very similar.
"""

import time
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
import unyt

from .halo_properties import HaloProperty, SearchRadiusTooSmallError
from SOAP.core.swift_cells import SWIFTCellGrid
from SOAP.core.lazy_properties import lazy_property
from SOAP.core.category_filter import CategoryFilter
from SOAP.core.parameter_file import ParameterFile
from SOAP.core.snapshot_datasets import SnapshotDatasets
from SOAP.core.dataset_names import mass_dataset
from SOAP.property_calculation.half_mass_radius import get_half_mass_radius, get_half_light_radius
from SOAP.property_table import PropertyTable
from SOAP.property_calculation.kinematic_properties import get_projected_inertia_tensor


class ProjectedApertureParticleData:
    """
    Halo calculation class.

    Only used to obtain some basic particle properties; actual property
    calculations are done by SingleProjectionProjectedApertureParticleData
    objects based on this object.
    """

    def __init__(
        self,
        input_halo: Dict,
        data: Dict,
        types_present: List[str],
        aperture_radius: unyt.unyt_quantity,
        snapshot_datasets: SnapshotDatasets,
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
         - aperture_radius: unyt.unyt_quantity
           Aperture radius.
         - snapshot_datasets: SnapshotDatasets
           Object containing metadata about the datasets in the snapshot, like
           appropriate aliases and column names.
         - boxsize: unyt.unyt_quantity
           Boxsize for correcting periodic boundary conditions
        """
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.aperture_radius = aperture_radius
        self.snapshot_datasets = snapshot_datasets
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
        properties and projection we actually want to compute.
        """
        self.centre = self.input_halo["cofp"]
        self.index = self.input_halo["index"]

        mass = []
        position = []
        radius_projx = []
        radius_projy = []
        radius_projz = []
        velocity = []
        types = []
        for ptype in self.types_present:
            grnr = self.get_dataset(f"{ptype}/GroupNr_bound")
            in_halo = grnr == self.index
            mass.append(self.get_dataset(f"{ptype}/{mass_dataset(ptype)}")[in_halo])
            pos = (
                self.get_dataset(f"{ptype}/Coordinates")[in_halo, :]
                - self.centre[None, :]
            )
            position.append(pos)
            rprojx = np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius_projx.append(rprojx)
            rprojy = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
            radius_projy.append(rprojy)
            rprojz = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            radius_projz.append(rprojz)
            velocity.append(self.get_dataset(f"{ptype}/Velocities")[in_halo, :])
            typearr = int(ptype[-1]) * np.ones(rprojx.shape, dtype=np.int32)
            types.append(typearr)

        self.mass = np.concatenate(mass)
        self.position = np.concatenate(position)
        self.radius_projx = np.concatenate(radius_projx)
        self.radius_projy = np.concatenate(radius_projy)
        self.radius_projz = np.concatenate(radius_projz)
        self.velocity = np.concatenate(velocity)
        self.types = np.concatenate(types)

        self.mask_projx = self.radius_projx <= self.aperture_radius
        self.mask_projy = self.radius_projy <= self.aperture_radius
        self.mask_projz = self.radius_projz <= self.aperture_radius


class SingleProjectionProjectedApertureParticleData:
    """
    Halo calculation class for individual projections.

    All properties we want to compute in apertures are implemented as lazy
    methods of this class.

    Note that the aperture is applied in projection, which means that there
    is no restriction on the coordinates of the particles parallel to the
    projection axis.

    Note that this class internally uses and requires two different masks:
     - *_mask_all: Mask that masks out particles belonging to this halo:
         gravitationally bound particles. This mask needs to be
         applied _first_ to raw "PartTypeX" datasets.
     - *_mask_ap: Mask that masks out particles that are inside the projected aperture
         radius. This mask can only be applied after *_mask_all has been applied.
    compute_basics() of ProjectedApertureParticleData furthermore defines
    some arrays that contain variables (e.g. masses, positions) for all
    particles that belong to the halo (so after applying *_mask_all, but before
    applying *_mask_ap). To retrieve the variables for a single particle type,
    these have to be masked with "PartTypeX == 'type'".
    All of these masks have different lengths, so using the wrong mask will
    lead to errors. Those are captured by the unit tests, so make sure to run
    those after you implement a new property!
    """

    def __init__(self, part_props: ProjectedApertureParticleData, projection: str):
        """
        Constructor.

        Parameters:
         - part_props: ProjectedApertureParticleData
           ProjectedApertureParticleData object that precomputed some quantities
           for this halo.
         - projection: str
           Projection axis for this particular projection. Needs to be one of
           "projx", "projy", "projz".
        """
        self.part_props = part_props
        self.index = part_props.index
        self.centre = part_props.centre
        self.types = part_props.types
        self.aperture_radius = part_props.aperture_radius

        self.iproj = {"projx": 0, "projy": 1, "projz": 2}[projection]
        self.projmask = getattr(part_props, f"mask_{projection}")
        self.projr = getattr(part_props, f"radius_{projection}")

        self.proj_mass = part_props.mass[self.projmask]
        self.proj_position = part_props.position[self.projmask]
        self.proj_velocity = part_props.velocity[self.projmask]
        self.proj_radius = self.projr[self.projmask]
        self.proj_type = part_props.types[self.projmask]

    @lazy_property
    def gas_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out gas particles that are inside the aperture radius.
        This mask can be used on arrays of all gas particles that are included
        in the calculation (so either the raw "PartType0" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.projmask[self.types == 0]

    @lazy_property
    def dm_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out DM particles that are inside the aperture radius.
        This mask can be used on arrays of all DM particles that are included
        in the calculation (so either the raw "PartType1" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.projmask[self.types == 1]

    @lazy_property
    def star_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out star particles that are inside the aperture radius.
        This mask can be used on arrays of all star particles that are included
        in the calculation (so either the raw "PartType4" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.projmask[self.types == 4]

    @lazy_property
    def bh_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out BH particles that are inside the aperture radius.
        This mask can be used on arrays of all BH particles that are included
        in the calculation (so either the raw "PartType5" array for inclusive
        apertures, or only the bound particles in that array for exclusive
        apertures).
        """
        return self.projmask[self.types == 5]

    @lazy_property
    def baryon_mask_ap(self) -> NDArray[bool]:
        """
        Mask that filters out baryon particles that are inside the aperture radius.
        This mask can be used on arrays of all baryon particles that are included
        in the calculation. Note that baryons are gas and star particles,
        so "PartType0" and "PartType4".
        """
        return self.projmask[(self.types == 0) | (self.types == 4)]

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
    def proj_mass_gas(self) -> unyt.unyt_array:
        """
        Mass of the gas particles.
        """
        return self.proj_mass[self.proj_type == 0]

    @lazy_property
    def proj_mass_dm(self) -> unyt.unyt_array:
        """
        Mass of the DM particles.
        """
        return self.proj_mass[self.proj_type == 1]

    @lazy_property
    def proj_mass_star(self) -> unyt.unyt_array:
        """
        Mass of the star particles.
        """
        return self.proj_mass[self.proj_type == 4]

    @lazy_property
    def proj_mass_baryons(self) -> unyt.unyt_array:
        """
        Mass of the baryon particles (gas + stars).
        """
        return self.proj_mass[(self.proj_type == 0) | (self.proj_type == 4)]

    @lazy_property
    def proj_pos_gas(self) -> unyt.unyt_array:
        """
        Projected position of the gas particles.
        """
        return self.proj_position[self.proj_type == 0]

    @lazy_property
    def proj_pos_dm(self) -> unyt.unyt_array:
        """
        Projected position of the DM particles.
        """
        return self.proj_position[self.proj_type == 1]

    @lazy_property
    def proj_pos_star(self) -> unyt.unyt_array:
        """
        Projected position of the star particles.
        """
        return self.proj_position[self.proj_type == 4]

    @lazy_property
    def proj_pos_baryons(self) -> unyt.unyt_array:
        """
        Projected position of the baryon (gas + stars) particles.
        """
        return self.proj_position[(self.proj_type == 0) | (self.proj_type == 4)]

    @lazy_property
    def Mtot(self) -> unyt.unyt_quantity:
        """
        Total mass of all particles.
        """
        return self.proj_mass.sum()

    @lazy_property
    def Mgas(self) -> unyt.unyt_quantity:
        """
        Total mass of gas particles.
        """
        return self.proj_mass_gas.sum()

    @lazy_property
    def Mdm(self) -> unyt.unyt_quantity:
        """
        Total mass of DM particles.
        """
        return self.proj_mass_dm.sum()

    @lazy_property
    def Mstar(self) -> unyt.unyt_quantity:
        """
        Total mass of star particles.
        """
        return self.proj_mass_star.sum()

    @lazy_property
    def Mbh_dynamical(self) -> unyt.unyt_quantity:
        """
        Total dynamical mass of BH particles.
        """
        return self.proj_mass[self.proj_type == 5].sum()

    @lazy_property
    def Mbaryons(self) -> unyt.unyt_quantity:
        """
        Total mass of baryon (gas+star) particles.
        """
        return self.proj_mass_baryons.sum()

    @lazy_property
    def star_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out star particles in raw PartType4 arrays.
        This is the mask that masks out unbound particles.
        """
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/GroupNr_bound") == self.index

    @lazy_property
    def Mstar_init(self) -> unyt.unyt_quantity:
        """
        Total initial mass of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/InitialMasses")[
            self.star_mask_all
        ][self.star_mask_ap].sum()

    @lazy_property
    def stellar_luminosities(self) -> unyt.unyt_array:
        """
        Stellar luminosities.
        """
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/Luminosities")[
            self.star_mask_all
        ][self.star_mask_ap]

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
    def bh_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out BH particles in raw PartType5 arrays.
        This is the mask that masks out unbound particles.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/GroupNr_bound") == self.index

    @lazy_property
    def BH_subgrid_masses(self) -> unyt.unyt_array:
        """
        Subgrid masses of BH particles.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all][
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
        return self.part_props.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
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
        Thermal energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.part_props.get_dataset("PartType5/AGNTotalInjectedEnergies")[
                self.bh_mask_all
            ][self.bh_mask_ap]
        )

    @lazy_property
    def BlackHolesTotalInjectedJetEnergy(self) -> unyt.unyt_quantity:
        """
        Jet energy injected into gas particles by all BH particles.
        """
        if self.Nbh == 0:
            return None
        return np.sum(
            self.part_props.get_dataset("PartType5/InjectedJetEnergies")[
                self.bh_mask_all
            ][self.bh_mask_ap]
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
        return self.part_props.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxpos(self) -> unyt.unyt_array:
        """
        Position of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/Coordinates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxvel(self) -> unyt.unyt_array:
        """
        Velocity of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/Velocities")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxAR(self) -> unyt.unyt_quantity:
        """
        Accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/AccretionRates")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleAveragedAccretionRate(self) -> unyt.unyt_quantity:
        """
        Averaged accretion rate of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/AveragedAccretionRates")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleInjectedThermalEnergy(self) -> unyt.unyt_quantity:
        """
        Total thermal energy injected into gas particles by the most massive
        BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/AGNTotalInjectedEnergies")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleAccretionMode(self) -> unyt.unyt_quantity:
        """
        Accretion flow regime of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/AccretionModes")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleGWMassLoss(self) -> unyt.unyt_quantity:
        """
        Cumulative mass lost to GW of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/GWMassLosses")[self.bh_mask_all][
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
        return self.part_props.get_dataset("PartType5/InjectedJetEnergiesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleLastJetEventScalefactor(self) -> unyt.unyt_quantity:
        """
        Scale-factor of last jet event of the most massive BH particle (largest sub-grid mass).
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/LastAGNJetScaleFactors")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNEvents(self) -> unyt.unyt_quantity:
        """
        Number of AGN events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/NumberOfAGNEvents")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfAGNJetEvents(self) -> unyt.unyt_quantity:
        """
        Number of jet events the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/NumberOfAGNJetEvents")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleNumberOfMergers(self) -> unyt.unyt_quantity:
        """
        Number of mergers the most massive black hole has had so far.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/NumberOfMergers")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleRadiatedEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into radiation by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/RadiatedEnergiesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMassesByMode(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/TotalAccretedMassesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleWindEnergyByMode(self) -> unyt.unyt_quantity:
        """
        The total energy launched into accretion disc winds by the most massive black hole, split by accretion mode.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/WindEnergiesByMode")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleSpin(self) -> unyt.unyt_quantity:
        """
        The spin of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/Spins")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleTotalAccretedMass(self) -> unyt.unyt_quantity:
        """
        The total mass accreted by the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/TotalAccretedMasses")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def MostMassiveBlackHoleFormationScalefactor(self) -> unyt.unyt_quantity:
        """
        The formation scale factor of the most massive black hole.
        """
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/FormationScaleFactors")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

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
        return self.proj_mass / self.Mtot

    @lazy_property
    def com(self) -> unyt.unyt_array:
        """
        Centre of mass of all particles in the projected aperture.
        """
        if self.Mtot == 0:
            return None
        return (
            (self.mass_fraction[:, None] * self.proj_position).sum(axis=0) + self.centre
        ) % self.part_props.boxsize

    @lazy_property
    def com_star(self) -> unyt.unyt_array:
        """
        Centre of mass of star particles in the subhalo.
        """
        if self.Mstar == 0:
            return None
        return (
            (self.star_mass_fraction[:, None] * self.proj_pos_star).sum(axis=0)
            + self.centre
        ) % self.part_props.boxsize

    @lazy_property
    def vcom(self) -> unyt.unyt_array:
        """
        Centre of mass velocity of all particles in the projected aperture.
        """
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.proj_velocity).sum(axis=0)

    @lazy_property
    def ProjectedTotalInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius.
        """
        if self.Mtot == 0:
            return None
        return get_projected_inertia_tensor(
            self.part_props.mass,
            self.part_props.position,
            self.iproj,
            self.aperture_radius,
        )

    @lazy_property
    def ProjectedTotalInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius.
        """
        if self.Mtot == 0:
            return None
        return get_projected_inertia_tensor(
            self.part_props.mass,
            self.part_props.position,
            self.iproj,
            self.aperture_radius,
            reduced=True,
        )

    @lazy_property
    def ProjectedTotalInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the total mass distribution in projection.
        Computed using all bound particles within the projected aperture.
        """
        if self.Mtot == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass,
            self.proj_position,
            self.iproj,
            self.aperture_radius,
            max_iterations=1,
        )

    @lazy_property
    def ProjectedTotalInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the total mass distribution in projection.
        Computed using all bound particles within the projected aperture.
        """
        if self.Mtot == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass,
            self.proj_position,
            self.iproj,
            self.aperture_radius,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def gas_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of gas particles. See the documentation of mass_fraction
        for the rationale behind this.
        """
        if self.Mgas == 0:
            return None
        return self.proj_mass_gas / self.Mgas

    @lazy_property
    def proj_veldisp_gas(self) -> unyt.unyt_quantity:
        """
        Projected velocity dispersion of gas particles along the projection
        axis. Unlike the 3D aperture counterpart, the velocity dispersion
        along the projection axis is a single number.
        """
        if self.Mgas == 0:
            return None
        proj_vgas = self.proj_velocity[self.proj_type == 0, self.iproj]
        vcom_gas = (self.gas_mass_fraction * proj_vgas).sum()
        return np.sqrt((self.gas_mass_fraction * (proj_vgas - vcom_gas) ** 2).sum())

    def gas_inertia_tensor(self, **kwargs) -> unyt.unyt_array:
        """
        Helper function for calculating projected gas inertia tensors
        """
        mass = self.part_props.mass[self.part_props.types == 0]
        position = self.part_props.position[self.part_props.types == 0]
        return get_projected_inertia_tensor(
            mass, position, self.iproj, self.aperture_radius, **kwargs
        )

    @lazy_property
    def ProjectedGasInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius. Only considers bound particles within the projected aperture.
        """
        if self.Mgas == 0:
            return None
        return self.gas_inertia_tensor()

    @lazy_property
    def ProjectedGasInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius. Only considers bound particles within the projected aperture.
        """
        if self.Mgas == 0:
            return None
        return self.gas_inertia_tensor(reduced=True)

    @lazy_property
    def ProjectedGasInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the gas mass distribution in projection.
        Computed using all bound gas particles within the projected aperture.
        """
        if self.Mgas == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass_gas,
            self.proj_pos_gas,
            self.iproj,
            self.aperture_radius,
            max_iterations=1,
        )

    @lazy_property
    def ProjectedGasInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the gas mass distribution in projection.
        Computed using all bound gas particles within the projected aperture.
        """
        if self.Mgas == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass_gas,
            self.proj_pos_gas,
            self.iproj,
            self.aperture_radius,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def dm_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of DM particles. See the documentation of mass_fraction
        for the rationale behind this.
        """
        if self.Mdm == 0:
            return None
        return self.proj_mass_dm / self.Mdm

    @lazy_property
    def proj_veldisp_dm(self) -> unyt.unyt_quantity:
        """
        Projected velocity dispersion of DM particles along the projection
        axis. Unlike the 3D aperture counterpart, the velocity dispersion
        along the projection axis is a single number.
        """
        if self.Mdm == 0:
            return None
        proj_vdm = self.proj_velocity[self.proj_type == 1, self.iproj]
        vcom_dm = (self.dm_mass_fraction * proj_vdm).sum()
        return np.sqrt((self.dm_mass_fraction * (proj_vdm - vcom_dm) ** 2).sum())

    @lazy_property
    def star_mass_fraction(self) -> unyt.unyt_array:
        """
        Fractional mass of star particles. See the documentation of mass_fraction
        for the rationale behind this.
        """
        if self.Mstar == 0:
            return None
        return self.proj_mass_star / self.Mstar

    @lazy_property
    def proj_veldisp_star(self) -> unyt.unyt_quantity:
        """
        Projected velocity dispersion of star particles along the projection
        axis. Unlike the 3D aperture counterpart, the velocity dispersion
        along the projection axis is a single number.
        """
        if self.Mstar == 0:
            return None
        proj_vstar = self.proj_velocity[self.proj_type == 4, self.iproj]
        vcom_star = (self.star_mass_fraction * proj_vstar).sum()
        return np.sqrt((self.star_mass_fraction * (proj_vstar - vcom_star) ** 2).sum())

    def stellar_inertia_tensor(self, **kwargs) -> unyt.unyt_array:
        """
        Helper function for calculating projected stellar inertia tensors
        """
        mass = self.part_props.mass[self.part_props.types == 4]
        position = self.part_props.position[self.part_props.types == 4]
        return get_projected_inertia_tensor(
            mass, position, self.iproj, self.aperture_radius, **kwargs
        )

    @lazy_property
    def ProjectedStellarInertiaTensor(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius. Only considers bound particles within the projected aperture.
        """
        if self.Mstar == 0:
            return None
        return self.stellar_inertia_tensor()

    @lazy_property
    def ProjectedStellarInertiaTensorReduced(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution in projection.
        Computed iteratively using an ellipse with area equal to that of a circle with radius
        equal to the aperture radius. Only considers bound particles within the projected aperture.
        """
        if self.Mstar == 0:
            return None
        return self.stellar_inertia_tensor(reduced=True)

    @lazy_property
    def ProjectedStellarInertiaTensorNoniterative(self) -> unyt.unyt_array:
        """
        Inertia tensor of the stellar mass distribution in projection.
        Computed using all bound star particles within the projected aperture.
        """
        if self.Mstar == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass_star,
            self.proj_pos_star,
            self.iproj,
            self.aperture_radius,
            max_iterations=1,
        )

    @lazy_property
    def ProjectedStellarInertiaTensorReducedNoniterative(self) -> unyt.unyt_array:
        """
        Reduced inertia tensor of the stellar mass distribution in projection.
        Computed using all bound star particles within the projected aperture.
        """
        if self.Mstar == 0:
            return None
        return get_projected_inertia_tensor(
            self.proj_mass_star,
            self.proj_pos_star,
            self.iproj,
            self.aperture_radius,
            reduced=True,
            max_iterations=1,
        )

    @lazy_property
    def gas_mask_all(self) -> NDArray[bool]:
        """
        Mask for masking out gas particles in raw PartType0 arrays.
        This is the mask that masks out unbound particles.
        """
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/GroupNr_bound") == self.index

    @lazy_property
    def gas_total_dust_mass_fractions(self) -> unyt.unyt_array:
        """
        Total dust mass fractions in gas particles in the projection.
        """
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/TotalDustMassFractions")[
            self.gas_mask_all
        ][self.gas_mask_ap]

    @lazy_property
    def proj_mass_dust(self) -> unyt.unyt_array:
        """
        Dust masses of the gas particles in the subhalo.
        """
        return self.gas_total_dust_mass_fractions * self.proj_mass_gas

    @lazy_property
    def DustMass(self) -> unyt.unyt_quantity:
        """
        Total dust mass of the gas particles in the subhalo.
        """
        if self.Ngas == 0:
            return None
        return self.proj_mass_dust.sum()

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
        raw_SFR = self.part_props.get_dataset("PartType0/StarFormationRates")[
            self.gas_mask_all
        ][self.gas_mask_ap]
        # Negative SFR are not SFR at all!
        raw_SFR[raw_SFR < 0] = 0
        return raw_SFR

    @lazy_property
    def SFR(self) -> unyt.unyt_quantity:
        """
        Total star formation rate of the gas.
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
        avg_SFR = self.part_props.get_dataset("PartType0/AveragedStarFormationRates")[
            self.gas_mask_all
        ][self.gas_mask_ap]
        return np.sum(avg_SFR, axis=0)

    @lazy_property
    def gas_metal_mass(self) -> unyt.unyt_array:
        """
        Metal mass of gas particles.
        """
        if self.Ngas == 0:
            return None
        return (
            self.part_props.get_dataset("PartType0/MetalMassFractions")[
                self.gas_mask_all
            ][self.gas_mask_ap]
            * self.proj_mass_gas
        )

    @lazy_property
    def gasmetalfrac(self) -> unyt.unyt_array:
        """
        Metal mass fraction of gas.
        """
        if self.Ngas == 0 or self.Mgas == 0.0:
            return None
        return self.gas_metal_mass.sum() / self.Mgas

    @lazy_property
    def gas_is_star_forming(self) -> NDArray[bool]:
        """
        Mask for gas particles that are star-forming
        (SFR > 0).
        """
        if self.Ngas == 0:
            return None
        return self.gas_SFR > 0

    @lazy_property
    def Mgas_SF(self) -> unyt.unyt_quantity:
        """
        Mass of star-forming gas.
        """
        if self.Ngas == 0:
            return None
        return self.proj_mass_gas[self.gas_is_star_forming].sum()

    @lazy_property
    def gasmetalfrac_SF(self) -> unyt.unyt_array:
        """
        Metal mass fraction of star-forming gas.
        """
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_metal_mass[self.gas_is_star_forming].sum() / self.Mgas_SF

    @lazy_property
    def gas_element_fractions(self) -> unyt.unyt_array:
        """
        Element fractions of the gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/ElementMassFractions")[
            self.gas_mask_all
        ][self.gas_mask_ap]

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
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType0/ElementMassFractions", "Hydrogen"
                ),
            ]
            * self.proj_mass_gas
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
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType0/ElementMassFractions", "Helium"
                ),
            ]
            * self.proj_mass_gas
        )

    @lazy_property
    def gas_species_fractions(self) -> unyt.unyt_array:
        """
        Ion/molecule fractions in gas particles.
        """
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/SpeciesFractions")[
            self.gas_mask_all
        ][self.gas_mask_ap]

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
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType0/SpeciesFractions", "HI"
                ),
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
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType0/SpeciesFractions", "H2"
                ),
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
    def star_element_fractions(self) -> unyt.unyt_array:
        """
        Element mass fractions of star particles.
        """
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/ElementMassFractions")[
            self.star_mask_all
        ][self.star_mask_ap]

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
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType4/ElementMassFractions", "Oxygen"
                ),
            ]
            * self.proj_mass_star
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
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType4/ElementMassFractions", "Magnesium"
                ),
            ]
            * self.proj_mass_star
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
                self.part_props.snapshot_datasets.get_column_index(
                    "PartType4/ElementMassFractions", "Iron"
                ),
            ]
            * self.proj_mass_star
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
    def starmetalfrac(self) -> unyt.unyt_quantity:
        """
        Total metal mass fraction of star particles.
        """
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return (
            self.part_props.get_dataset("PartType4/MetalMassFractions")[
                self.star_mask_all
            ][self.star_mask_ap]
            * self.proj_mass_star
        ).sum() / self.Mstar

    @lazy_property
    def HalfMassRadiusGas(self) -> unyt.unyt_quantity:
        """
        Half mass radius of gas.
        """
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == 0], self.proj_mass_gas, self.Mgas
        )

    @lazy_property
    def HalfMassRadiusDust(self) -> unyt.unyt_quantity:
        """
        Half-mass radius of the dust.
        """
        if self.Ngas == 0:
            return None
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == 0], self.proj_mass_dust, self.DustMass
        )

    @lazy_property
    def HalfMassRadiusDM(self) -> unyt.unyt_quantity:
        """
        Half mass radius of dark matter.
        """
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == 1], self.proj_mass_dm, self.Mdm
        )

    @lazy_property
    def HalfMassRadiusStar(self) -> unyt.unyt_quantity:
        """
        Half mass radius of stars.
        """
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == 4], self.proj_mass_star, self.Mstar
        )

    @lazy_property
    def HalfLightRadiusStar(self) -> unyt.unyt_array:
        """
        Half light radius of stars for the 9 GAMA bands.
        """
        return get_half_light_radius(
            self.proj_radius[self.proj_type == 4], self.stellar_luminosities, self.StellarLuminosity
        )

    @lazy_property
    def HalfMassRadiusBaryon(self) -> unyt.unyt_quantity:
        """
        Half mass radius of baryons (gas + stars).
        """
        return get_half_mass_radius(
            self.proj_radius[(self.proj_type == 0) | (self.proj_type == 4)],
            self.proj_mass_baryons,
            self.Mbaryons,
        )


class ProjectedApertureProperties(HaloProperty):
    """
    Calculate projected aperture properties.

    These contain all particles bound to a halo. For projections along the three
    principal coordinate axes, all particles within a given fixed aperture
    radius are used. The depth of the projection is always the full extent of
    the halo along the projection axis.
    """

    base_halo_type = "ProjectedApertureProperties"
    # Properties to calculate. The key is the name of the property,
    # the value indicates the property has a direct dependence on aperture size.
    # This is needed since for larger apertures we sometimes copy across the
    # values computed by the previous aperture (if the number of particles was
    # the same for both apertures), but we can't do this for all properties
    property_names = {
        "Mtot": False,
        "Mgas": False,
        "Mdm": False,
        "Mstar": False,
        "Mstar_init": False,
        "Mbh_dynamical": False,
        "Mbh_subgrid": False,
        "DustMass": True,
        "Ngas": False,
        "Ndm": False,
        "Nstar": False,
        "Nbh": False,
        "com": False,
        "com_star": False,
        "vcom": False,
        "SFR": False,
        "AveragedStarFormationRate": False,
        "StellarLuminosity": False,
        "HalfMassRadiusGas": False,
        "HalfMassRadiusDust": False,
        "HalfMassRadiusDM": False,
        "HalfMassRadiusStar": False,
        "HalfLightRadiusStar": False,
        "HalfMassRadiusBaryon": False,
        "proj_veldisp_gas": False,
        "proj_veldisp_dm": False,
        "proj_veldisp_star": False,
        "BHmaxAR": False,
        "BHmaxM": False,
        "BHmaxID": False,
        "BHmaxpos": False,
        "BHmaxvel": False,
        "BHlasteventa": False,
        "BHmaxlasteventa": False,
        "BlackHolesTotalInjectedThermalEnergy": False,
        "BlackHolesTotalInjectedJetEnergy": False,
        "MostMassiveBlackHoleAveragedAccretionRate": False,
        "MostMassiveBlackHoleNumberOfAGNEvents": False,
        "MostMassiveBlackHoleAccretionMode": False,
        "MostMassiveBlackHoleGWMassLoss": False,
        "MostMassiveBlackHoleInjectedJetEnergyByMode": False,
        "MostMassiveBlackHoleLastJetEventScalefactor": False,
        "MostMassiveBlackHoleNumberOfAGNJetEvents": False,
        "MostMassiveBlackHoleNumberOfMergers": False,
        "MostMassiveBlackHoleRadiatedEnergyByMode": False,
        "MostMassiveBlackHoleTotalAccretedMassesByMode": False,
        "MostMassiveBlackHoleWindEnergyByMode": False,
        "MostMassiveBlackHoleInjectedThermalEnergy": False,
        "MostMassiveBlackHoleSpin": False,
        "MostMassiveBlackHoleTotalAccretedMass": False,
        "MostMassiveBlackHoleFormationScalefactor": False,
        "ProjectedTotalInertiaTensor": True,
        "ProjectedGasInertiaTensor": True,
        "ProjectedStellarInertiaTensor": True,
        "ProjectedTotalInertiaTensorReduced": True,
        "ProjectedGasInertiaTensorReduced": True,
        "ProjectedStellarInertiaTensorReduced": True,
        "ProjectedTotalInertiaTensorNoniterative": False,
        "ProjectedGasInertiaTensorNoniterative": False,
        "ProjectedStellarInertiaTensorNoniterative": False,
        "ProjectedTotalInertiaTensorReducedNoniterative": False,
        "ProjectedGasInertiaTensorReducedNoniterative": False,
        "ProjectedStellarInertiaTensorReducedNoniterative": False,
        "HydrogenMass": False,
        "HeliumMass": False,
        "MolecularHydrogenMass": False,
        "AtomicHydrogenMass": False,
        "starFefrac": False,
        "starMgfrac": False,
        "starOfrac": False,
        "starmetalfrac": False,
        "gasmetalfrac": False,
        "gasmetalfrac_SF": False,
    }
    property_list = {
        name: PropertyTable.full_property_list[name] for name in property_names
    }

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        parameters: ParameterFile,
        physical_radius_kpc: float,
        category_filter: CategoryFilter,
        halo_filter: str,
        all_radii_kpc: list,
    ):
        """
        Construct an ProjectedApertureProperties object with the given physical
        radius.

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
         - category_filter: CategoryFilter
           Filter used to determine which properties can be calculated for this halo.
           This depends on the number of particles in the subhalo and the category
           of each property.
         - halo_filter: str
           The filter to apply to this halo type. Halos which do not fulfil the
           filter requirements will be skipped.
         - all_radii_kpc: list
           A list of all the radii for which we compute a ProjectedAperture. This
           can allow us to skip the property calculation for larger apertures
        """
        super().__init__(cellgrid)

        self.property_filters = parameters.get_property_filters(
            "ProjectedApertureProperties",
            [prop.name for prop in self.property_list.values()],
        )

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets
        self.halo_filter = halo_filter
        self.record_timings = parameters.record_property_timings
        self.all_radii_kpc = all_radii_kpc
        self.strict_halo_copy = parameters.strict_halo_copy()
        self.boxsize = cellgrid.boxsize

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"
        self.group_name = f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc"
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
        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            if not self.property_filters[outputname]:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue
            for partprop in prop.particle_properties:
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

        do_calculation = self.category_filter.get_do_calculation(halo_result)
        registry = input_halo["cofp"].units.registry

        projected_aperture = {}
        timings = {}

        skip_gt_enclose_radius = False
        # Determine if the previous aperture already enclosed
        # all the bound particles of the subhalo
        r_enclose = halo_result["BoundSubhalo/EncloseRadius"][0]
        i_radius = self.all_radii_kpc.index(1000 * self.physical_radius_mpc)
        if i_radius != 0:
            r_previous_kpc = self.all_radii_kpc[i_radius - 1]
            if r_previous_kpc * unyt.kpc > r_enclose:
                skip_gt_enclose_radius = True

        # loop over the different projections
        for projname in ["projx", "projy", "projz"]:
            projected_aperture[projname] = {}
            # declare all the variables we will compute
            # we set them to 0 in case a particular variable cannot be computed
            # all variables are defined with physical units and an appropriate dtype
            # we need to use the custom unit registry so that everything can be converted
            # back to snapshot units in the end
            for name, prop in self.property_list.items():
                outputname = prop.name
                # Skip if this property is disabled in the parameter file
                if not self.property_filters[outputname]:
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
                projected_aperture[projname][name] = unyt.unyt_array(
                    val, dtype=dtype, units=unit, registry=registry
                )

                if skip_gt_enclose_radius:
                    prev_group_name = f"ProjectedAperture/{r_previous_kpc:.0f}kpc"
                    prev_prop = f"{prev_group_name}/{projname}/{outputname}"
                    # Skip if this property has a direct dependence on
                    # aperture size (and so would have a different value)
                    if self.strict_halo_copy and self.property_names[name]:
                        continue
                    projected_aperture[projname][name] = halo_result[prev_prop][0]

        # Determine whether to skip this halo (because of the filter or because we
        # have copied over the values from the previous aperture)
        if do_calculation[self.halo_filter] and (not skip_gt_enclose_radius):
            # For projected apertures we are only using bound particles
            # Therefore we don't need to check if the serach_radius is large enough,
            # because all particles will have been loaded

            types_present = [type for type in self.particle_properties if type in data]
            part_props = ProjectedApertureParticleData(
                input_halo,
                data,
                types_present,
                self.physical_radius_mpc * unyt.Mpc,
                self.snapshot_datasets,
                self.boxsize,
            )
            for projname in ["projx", "projy", "projz"]:
                proj_part_props = SingleProjectionProjectedApertureParticleData(
                    part_props, projname
                )
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
                    if not physical:
                        unit = unit * unyt.Unit("a", registry=registry) ** a_exponent
                    if do_calculation[filter_name]:
                        t0_calc = time.time()
                        val = getattr(proj_part_props, name)
                        if val is not None:
                            assert (
                                projected_aperture[projname][name].shape == val.shape
                            ), f"Attempting to store {name} with wrong dimensions"
                            if unit == unyt.Unit("dimensionless"):
                                if hasattr(val, "units"):
                                    assert (
                                        val.units == unyt.dimensionless
                                    ), f"{name} is not dimensionless"
                                projected_aperture[projname][name] = unyt.unyt_array(
                                    val.astype(dtype),
                                    dtype=dtype,
                                    units=unit,
                                    registry=registry,
                                )
                            else:
                                err = f'Overflow for halo {input_halo["index"]} when'
                                err += f"calculating {name} in projected_properties"
                                assert np.max(np.abs(val.to(unit).value)) < float(
                                    "inf"
                                ), err
                                projected_aperture[projname][name] += val
                            # Include the time from previous projection calculations
                            timings[name] = timings.get(name, 0) + time.time() - t0_calc

        for name, prop in self.property_list.items():
            outputname = prop.name
            # Skip if this property is disabled in the parameter file
            if not self.property_filters[outputname]:
                continue
            # Skip non-DMO properties when in DMO run mode
            if self.category_filter.dmo and not prop.dmo_property:
                continue

            for projname in ["projx", "projy", "projz"]:
                # add the new properties to the halo_result dictionary
                halo_result.update(
                    {
                        f"{self.group_name}/{projname}/{outputname}": (
                            projected_aperture[projname][name],
                            prop.description,
                            prop.output_physical,
                            prop.a_scale_exponent,
                        )
                    }
                )

            # Storing total time for each calculation over the three projections
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

        return
