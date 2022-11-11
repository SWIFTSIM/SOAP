#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_axis_lengths,
    get_velocity_dispersion_matrix,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from stellar_age_calculator import StellarAgeCalculator
from property_table import PropertyTable
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile


class SubhaloParticleData:
    def __init__(
        self,
        input_halo,
        data,
        types_present,
        grnr,
        stellar_age_calculator,
        recently_heated_gas_filter,
        snapshot_datasets,
    ):
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.grnr = grnr
        self.stellar_age_calculator = stellar_age_calculator
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.snapshot_datasets = snapshot_datasets
        self.compute_basics()

    def get_dataset(self, name):
        return self.snapshot_datasets.get_dataset(name, self.data)

    def compute_basics(self):
        self.centre = self.input_halo["cofp"]
        self.index = self.input_halo["index"]

        mass = []
        position = []
        radius = []
        velocity = []
        types = []
        for ptype in self.types_present:
            grnr = self.get_dataset(f"{ptype}/{self.grnr}")
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
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        self.mass = unyt.array.uconcatenate(mass)
        self.position = unyt.array.uconcatenate(position)
        self.radius = unyt.array.uconcatenate(radius)
        self.velocity = unyt.array.uconcatenate(velocity)
        self.types = np.concatenate(types)

    @lazy_property
    def gas_mask_sh(self):
        return self.types == "PartType0"

    @lazy_property
    def dm_mask_sh(self):
        return self.types == "PartType1"

    @lazy_property
    def star_mask_sh(self):
        return self.types == "PartType4"

    @lazy_property
    def bh_mask_sh(self):
        return self.types == "PartType5"

    @lazy_property
    def baryons_mask_sh(self):
        return (self.types == "PartType0") | (self.types == "PartType4")

    @lazy_property
    def Ngas(self):
        return self.gas_mask_sh.sum()

    @lazy_property
    def Ndm(self):
        return self.dm_mask_sh.sum()

    @lazy_property
    def Nstar(self):
        return self.star_mask_sh.sum()

    @lazy_property
    def Nbh(self):
        return self.bh_mask_sh.sum()

    @lazy_property
    def mass_gas(self):
        return self.mass[self.gas_mask_sh]

    @lazy_property
    def mass_dm(self):
        return self.mass[self.dm_mask_sh]

    @lazy_property
    def mass_star(self):
        return self.mass[self.star_mask_sh]

    @lazy_property
    def mass_baryons(self):
        return self.mass[self.baryons_mask_sh]

    @lazy_property
    def pos_gas(self):
        return self.position[self.gas_mask_sh]

    @lazy_property
    def pos_dm(self):
        return self.position[self.dm_mask_sh]

    @lazy_property
    def pos_star(self):
        return self.position[self.star_mask_sh]

    @lazy_property
    def pos_baryons(self):
        return self.position[self.baryons_mask_sh]

    @lazy_property
    def vel_gas(self):
        return self.velocity[self.gas_mask_sh]

    @lazy_property
    def vel_dm(self):
        return self.velocity[self.dm_mask_sh]

    @lazy_property
    def vel_star(self):
        return self.velocity[self.star_mask_sh]

    @lazy_property
    def vel_baryons(self):
        return self.velocity[self.baryons_mask_sh]

    @lazy_property
    def Mtot(self):
        return self.mass.sum()

    @lazy_property
    def Mgas(self):
        return self.mass_gas.sum()

    @lazy_property
    def Mdm(self):
        return self.mass_dm.sum()

    @lazy_property
    def Mstar(self):
        return self.mass_star.sum()

    @lazy_property
    def Mbh_dynamical(self):
        return self.mass[self.bh_mask_sh].sum()

    @lazy_property
    def star_mask_all(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset(f"PartType4/{self.grnr}") == self.index

    @lazy_property
    def mass_star_init(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/InitialMasses")[self.star_mask_all]

    @lazy_property
    def Mstar_init(self):
        if self.Nstar == 0:
            return None
        return self.mass_star_init.sum()

    @lazy_property
    def stellar_luminosities(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/Luminosities")[self.star_mask_all]

    @lazy_property
    def StellarLuminosity(self):
        if self.Nstar == 0:
            return None
        return self.stellar_luminosities.sum(axis=0)

    @lazy_property
    def starmetalfrac(self):
        if self.Nstar == 0:
            return None
        return (
            self.mass_star
            * self.get_dataset("PartType4/MetalMassFractions")[self.star_mask_all]
        ).sum() / self.Mstar

    @lazy_property
    def stellar_ages(self):
        if self.Nstar == 0:
            return None
        birth_a = self.get_dataset("PartType4/BirthScaleFactors")[self.star_mask_all]
        return self.stellar_age_calculator.stellar_age(birth_a)

    @lazy_property
    def stellar_age_mw(self):
        if self.Nstar == 0:
            return None
        return ((self.mass_star / self.Mstar) * self.stellar_ages).sum()

    @lazy_property
    def stellar_age_lw(self):
        if self.Nstar == 0:
            return None
        Lr = self.stellar_luminosities[
            :, self.snapshot_datasets.get_column_index("Luminosities", "GAMA_r")
        ]
        Lrtot = Lr.sum()
        return ((Lr / Lrtot) * self.stellar_ages).sum()

    @lazy_property
    def bh_mask_all(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset(f"PartType5/{self.grnr}") == self.index

    @lazy_property
    def Mbh_subgrid(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all].sum()

    @lazy_property
    def agn_eventa(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
            self.bh_mask_all
        ]

    @lazy_property
    def BHlasteventa(self):
        if self.Nbh == 0:
            return None
        return np.max(self.agn_eventa)

    @lazy_property
    def BH_subgrid_masses(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all]

    @lazy_property
    def iBHmax(self):
        if self.Nbh == 0:
            return None
        return np.argmax(self.BH_subgrid_masses)

    @lazy_property
    def BHmaxM(self):
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses[self.iBHmax]

    @lazy_property
    def BHmaxID(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxpos(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Coordinates")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxvel(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Velocities")[self.bh_mask_all][self.iBHmax]

    @lazy_property
    def BHmaxAR(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionRates")[self.bh_mask_all][
            self.iBHmax
        ]

    @lazy_property
    def BHmaxlasteventa(self):
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def total_mass_fraction(self):
        if self.Mtot == 0:
            return None
        return self.mass / self.Mtot

    @lazy_property
    def com(self):
        if self.Mtot == 0:
            return None
        return (self.total_mass_fraction[:, None] * self.position).sum(
            axis=0
        ) + self.centre

    @lazy_property
    def vcom(self):
        if self.Mtot == 0:
            return None
        return (self.total_mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def R_vmax(self):
        if self.Mtot == 0:
            return None
        if not hasattr(self, "r_vmax"):
            self.r_vmax, self.vmax = get_vmax(self.mass, self.radius)
        return self.r_vmax

    @lazy_property
    def Vmax(self):
        if self.Mtot == 0:
            return None
        if not hasattr(self, "vmax"):
            self.r_vmax, self.vmax = get_vmax(self.mass, self.radius)
        return self.vmax

    @lazy_property
    def spin_parameter(self):
        if self.Mtot == 0:
            return None
        if self.R_vmax > 0 and self.Vmax > 0:
            mask_r_vmax = self.radius <= self.R_vmax
            vrel = self.velocity[mask_r_vmax, :] - self.vcom[None, :]
            Ltot = unyt.array.unorm(
                (
                    self.mass[mask_r_vmax, None]
                    * unyt.array.ucross(self.position[mask_r_vmax, :], vrel)
                ).sum(axis=0)
            )
            M_r_vmax = self.mass[mask_r_vmax].sum()
            if M_r_vmax > 0:
                return Ltot / (np.sqrt(2.0) * M_r_vmax * self.Vmax * self.R_vmax)
        return None

    @lazy_property
    def TotalAxisLengths(self):
        if self.Mtot == 0:
            return None
        return get_axis_lengths(self.mass, self.position)

    @lazy_property
    def gas_mass_fraction(self):
        if self.Mgas == 0:
            return None
        return self.mass_gas / self.Mgas

    @lazy_property
    def vcom_gas(self):
        if self.Mgas == 0:
            return None
        return (self.gas_mass_fraction[:, None] * self.vel_gas).sum(axis=0)

    def compute_Lgas_props(self):
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
    def Lgas(self):
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_Lgas"):
            self.compute_Lgas_props()
        return self.internal_Lgas

    @lazy_property
    def kappa_corot_gas(self):
        if self.Mgas == 0:
            return None
        if not hasattr(self, "internal_kappa_gas"):
            self.compute_Lgas_props()
        return self.internal_kappa_gas

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
        return get_axis_lengths(self.mass_gas, self.pos_gas)

    @lazy_property
    def veldisp_matrix_gas(self):
        if self.Mgas == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.gas_mass_fraction, self.vel_gas, self.vcom_gas
        )

    @lazy_property
    def dm_mass_fraction(self):
        if self.Mdm == 0:
            return None
        return self.mass_dm / self.Mdm

    @lazy_property
    def vcom_dm(self):
        if self.Mdm == 0:
            return None
        return (self.dm_mass_fraction[:, None] * self.vel_dm).sum(axis=0)

    @lazy_property
    def Ldm(self):
        if self.Mdm == 0:
            return None
        return get_angular_momentum(
            self.mass_dm, self.pos_dm, self.vel_dm, ref_velocity=self.vcom_dm
        )

    @lazy_property
    def DMAxisLengths(self):
        if self.Mdm == 0:
            return None
        return get_axis_lengths(self.mass_dm, self.pos_dm)

    @lazy_property
    def veldisp_matrix_dm(self):
        if self.Mdm == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.dm_mass_fraction, self.vel_dm, self.vcom_dm
        )

    @lazy_property
    def DM_Vmax(self):
        if self.Ndm == 0:
            return None
        if not hasattr(self, "DM_r_vmax"):
            self.DM_r_vmax, self.DM_vmax = get_vmax(
                self.mass_dm, self.radius[self.dm_mask_sh]
            )
        return self.DM_vmax

    @lazy_property
    def DM_R_vmax(self):
        if self.Ndm == 0:
            return None
        if not hasattr(self, "DM_r_vmax"):
            self.DM_r_vmax, self.DM_vmax = get_vmax(
                self.mass_dm, self.radius[self.dm_mask_sh]
            )
        return self.DM_r_vmax

    @lazy_property
    def star_mass_fraction(self):
        if self.Mstar == 0:
            return None
        return self.mass_star / self.Mstar

    @lazy_property
    def vcom_star(self):
        if self.Mstar == 0:
            return None
        return (self.star_mass_fraction[:, None] * self.vel_star).sum(axis=0)

    def compute_Lstar_props(self):
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
    def Lstar(self):
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_Lstar"):
            self.compute_Lstar_props()
        return self.internal_Lstar

    @lazy_property
    def kappa_corot_star(self):
        if self.Mstar == 0:
            return None
        if not hasattr(self, "internal_kappa_star"):
            self.compute_Lstar_props()
        return self.internal_kappa_star

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
        return get_axis_lengths(self.mass_star, self.pos_star)

    @lazy_property
    def veldisp_matrix_star(self):
        if self.Mstar == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.star_mass_fraction, self.vel_star, self.vcom_star
        )

    @lazy_property
    def Mbaryon(self):
        return self.Mgas + self.Mstar

    @lazy_property
    def baryon_mass_fraction(self):
        if self.Mbaryon == 0:
            return None
        return self.mass_baryons / self.Mbaryon

    @lazy_property
    def vcom_bar(self):
        if self.Mbaryon == 0:
            return None
        return (self.baryon_mass_fraction[:, None] * self.vel_baryons).sum(axis=0)

    def compute_Lbar_props(self):
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
    def Lbaryons(self):
        if self.Mbaryon == 0:
            return None
        if not hasattr(self, "internal_Lbar"):
            self.compute_Lbar_props()
        return self.internal_Lbar

    @lazy_property
    def kappa_corot_baryons(self):
        if self.Mbaryon == 0:
            return None
        if not hasattr(self, "internal_kappa_bar"):
            self.compute_Lbar_props()
        return self.internal_kappa_bar

    @lazy_property
    def BaryonAxisLengths(self):
        if self.Mbaryon == 0:
            return None
        return get_axis_lengths(self.mass_baryons, self.pos_baryons)

    @lazy_property
    def gas_mask_all(self):
        return self.get_dataset(f"PartType0/{self.grnr}") == self.index

    @lazy_property
    def gas_SFR(self):
        if self.Ngas == 0:
            return None
        # remember: SFR < 0. is not SFR at all!
        all_SFR = self.get_dataset("PartType0/StarFormationRates")[self.gas_mask_all]
        all_SFR[all_SFR < 0.0] = 0.0
        return all_SFR

    @lazy_property
    def SFR(self):
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def gas_metal_mass(self):
        if self.Ngas == 0:
            return None
        return (
            self.mass_gas
            * self.get_dataset("PartType0/MetalMassFractions")[self.gas_mask_all]
        )

    @lazy_property
    def gasmetalfrac(self):
        if self.Ngas == 0:
            return None
        return self.gas_metal_mass.sum() / self.Mgas

    @lazy_property
    def Mgas_SF(self):
        if self.Ngas == 0:
            return None
        return self.mass_gas[self.gas_SFR > 0.0].sum()

    @lazy_property
    def gasmetalfrac_SF(self):
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_metal_mass[self.gas_SFR > 0.0].sum() / self.Mgas_SF

    @lazy_property
    def gas_temp(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Temperatures")[self.gas_mask_all]

    @lazy_property
    def last_agn_gas(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/LastAGNFeedbackScaleFactors")[
            self.gas_mask_all
        ]

    @lazy_property
    def gas_no_agn(self):
        if self.Ngas == 0:
            return None
        return ~self.recently_heated_gas_filter.is_recently_heated(
            self.last_agn_gas, self.gas_temp
        )

    @lazy_property
    def gas_no_cool(self):
        if self.Ngas == 0:
            return None
        return self.gas_temp >= 1.0e5 * unyt.K

    @lazy_property
    def Tgas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_mass_fraction * self.gas_temp).sum()

    @lazy_property
    def Tgas_no_cool(self):
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
    def Tgas_no_agn(self):
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
    def Tgas_no_cool_no_agn(self):
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
    def stellar_birth_density(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/BirthDensities")[self.star_mask_all]

    @lazy_property
    def LogarithmicallyAveragedStellarBirthDensity(self):
        if self.Nstar == 0:
            return None
        # note: the stellar birth density cannot be zero for physical reasons
        log_birth_density = np.log(
            self.stellar_birth_density / self.stellar_birth_density.units
        )
        return np.exp(log_birth_density.mean()) * self.stellar_birth_density.units

    @lazy_property
    def MinimumStellarBirthDensity(self):
        if self.Nstar == 0:
            return None
        return self.stellar_birth_density.min()

    @lazy_property
    def MaximumStellarBirthDensity(self):
        if self.Nstar == 0:
            return None
        return self.stellar_birth_density.max()

    @lazy_property
    def HalfMassRadiusTot(self):
        return get_half_mass_radius(self.radius, self.mass, self.Mtot)

    @lazy_property
    def HalfMassRadiusGas(self):
        return get_half_mass_radius(
            self.radius[self.gas_mask_sh], self.mass_gas, self.Mgas
        )

    @lazy_property
    def HalfMassRadiusDM(self):
        return get_half_mass_radius(
            self.radius[self.dm_mask_sh], self.mass_dm, self.Mdm
        )

    @lazy_property
    def HalfMassRadiusStar(self):
        return get_half_mass_radius(
            self.radius[self.star_mask_sh], self.mass_star, self.Mstar
        )

    @lazy_property
    def HalfMassRadiusBaryon(self):
        return get_half_mass_radius(
            self.radius[self.gas_mask_sh | self.star_mask_sh],
            self.mass[self.gas_mask_sh | self.star_mask_sh],
            self.Mgas + self.Mstar,
        )


class SubhaloProperties(HaloProperty):

    # get the properties we want from the table
    property_list = [
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
            "com",
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
            "StellarLuminosity",
            "starmetalfrac",
            "Vmax",
            "R_vmax",
            "DM_Vmax",
            "DM_R_vmax",
            "spin_parameter",
            "HalfMassRadiusTot",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "HalfMassRadiusBaryon",
            "TotalAxisLengths",
            "GasAxisLengths",
            "DMAxisLengths",
            "StellarAxisLengths",
            "BaryonAxisLengths",
            "veldisp_matrix_gas",
            "veldisp_matrix_dm",
            "veldisp_matrix_star",
            "DtoTgas",
            "DtoTstar",
            "stellar_age_mw",
            "stellar_age_lw",
            "Mgas_SF",
            "gasmetalfrac_SF",
            "LogarithmicallyAveragedStellarBirthDensity",
            "MinimumStellarBirthDensity",
            "MaximumStellarBirthDensity",
        ]
    ]

    def __init__(
        self,
        cellgrid,
        parameters,
        recently_heated_gas_filter,
        stellar_age_calculator,
        category_filter,
        bound_only=True,
    ):
        super().__init__(cellgrid)

        self.property_mask = parameters.get_property_mask(
            "SubhaloProperties", [prop[1] for prop in self.property_list]
        )

        self.bound_only = bound_only
        self.filter = recently_heated_gas_filter
        self.stellar_ages = stellar_age_calculator
        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.0

        # Give this calculation a name so we can select it on the command line
        if bound_only:
            self.grnr = "GroupNr_bound"
            self.name = "bound_subhalo_properties"
        else:
            self.grnr = "GroupNr_all"
            self.name = "fof_subhalo_properties"

        # Arrays which must be read in for this calculation.
        # Note that if there are no particles of a given type in the
        # snapshot, that type will not be read in and will not have
        # an entry in the data argument to calculate(), below.
        # (E.g. gas, star or BH particles in DMO runs)
        # Most arrays are linked to a particular property and are only
        # read if that particular property is actually requested
        # Some basic properties are always required; these are added below
        self.particle_properties = {
            "PartType0": [
                "Coordinates",
                "Masses",
                "Velocities",
                self.grnr,
            ],
            "PartType1": ["Coordinates", "Masses", "Velocities", self.grnr],
            "PartType4": [
                "Coordinates",
                "Masses",
                "Velocities",
                self.grnr,
            ],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "Velocities",
                self.grnr,
            ],
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
            self.grnr,
            self.stellar_ages,
            self.filter,
            self.snapshot_datasets,
        )

        if not self.bound_only:
            # this is the halo that we use for the filter particle numbers,
            # so we have the get the numbers for the category filters manually
            Ngas = part_props.Ngas
            Ndm = part_props.Ndm
            Nstar = part_props.Nstar
            Nbh = part_props.Nbh
            do_calculation = self.category_filter.get_filters_direct(
                Ngas, Ndm, Nstar, Nbh
            )
        else:
            do_calculation = self.category_filter.get_filters(halo_result)

        subhalo = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        registry = part_props.mass.units.registry
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
            category = prop[6]
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            subhalo[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=registry
            )
            if do_calculation[category]:
                val = getattr(part_props, name)
                if val is not None:
                    if unit == "dimensionless":
                        subhalo[name] = unyt.unyt_array(
                            val.astype(dtype),
                            dtype=dtype,
                            units=unit,
                            registry=registry,
                        )
                    else:
                        subhalo[name] += val

        # Add these properties to the output
        if self.bound_only:
            prefix = "BoundSubhaloProperties"
        else:
            prefix = "FOFSubhaloProperties"
        for prop in self.property_list:
            outputname = prop[1]
            # skip properties that are masked
            if not self.property_mask[outputname]:
                continue
            is_dmo = prop[8]
            if do_calculation["DMO"] and not is_dmo:
                continue
            name = prop[0]
            description = prop[5]
            halo_result.update(
                {
                    f"{prefix}/{outputname}": (
                        subhalo[name],
                        description,
                    )
                }
            )


def test_subhalo_properties():
    """
    Unit test for the subhalo property calculations.

    We generate 100 random "dummy" halos and feed them to
    SubhaloProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(16902)
    cat_filter = CategoryFilter(
        {"general": 0, "gas": 0, "dm": 0, "star": 0, "baryon": 0}
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
        "SubhaloProperties",
        {"FOF": {"bound_only": False}, "Bound": {"bound_only": True}},
    )

    recently_heated_gas_filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())

    property_calculator_bound = SubhaloProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cat_filter,
    )
    property_calculator_both = SubhaloProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cat_filter,
        False,
    )
    parameters.write_parameters("subhalo.used_parameters.yml")

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _, _ = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )

        halo_result = {}
        for subhalo_name, prop_calc in [
            ("FOFSubhaloProperties", property_calculator_both),
            ("BoundSubhaloProperties", property_calculator_bound),
        ]:
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in prop_calc.property_list:
                outputname = prop[1]
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"{subhalo_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["SubhaloProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["SubhaloProperties"]["properties"]:
            single_property["SubhaloProperties"]["properties"][other_property] = (
                other_property == property
            ) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)
        property_calculator_bound = SubhaloProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cat_filter,
        )
        property_calculator_both = SubhaloProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cat_filter,
            False,
        )
        halo_result = {}
        for subhalo_name, prop_calc in [
            ("FOFSubhaloProperties", property_calculator_both),
            ("BoundSubhaloProperties", property_calculator_bound),
        ]:
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in prop_calc.property_list:
                outputname = prop[1]
                if not outputname == property:
                    continue
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"{subhalo_name}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


if __name__ == "__main__":
    """
    Standalone version of the program: just run the unit test.

    Note that this can also be achieved by running "pytest *.py" in the folder.
    """
    print("Running test_subhalo_properties()...")
    test_subhalo_properties()
    print("Test passed.")
