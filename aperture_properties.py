#! /usr/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
    get_vmax,
    get_axis_lengths,
)
from recently_heated_gas_filter import RecentlyHeatedGasFilter
from stellar_age_calculator import StellarAgeCalculator
from cold_dense_gas_filter import ColdDenseGasFilter
from property_table import PropertyTable
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile


class ApertureParticleData:
    def __init__(
        self,
        input_halo,
        data,
        types_present,
        inclusive,
        aperture_radius,
        stellar_age_calculator,
        recently_heated_gas_filter,
        cold_dense_gas_filter,
        snapshot_datasets,
    ):
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.inclusive = inclusive
        self.aperture_radius = aperture_radius
        self.stellar_age_calculator = stellar_age_calculator
        self.recently_heated_gas_filter = recently_heated_gas_filter
        self.cold_dense_gas_filter = cold_dense_gas_filter
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
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        self.mass = unyt.array.uconcatenate(mass)
        self.position = unyt.array.uconcatenate(position)
        self.radius = unyt.array.uconcatenate(radius)
        self.velocity = unyt.array.uconcatenate(velocity)
        self.types = np.concatenate(types)

        self.mask = self.radius <= self.aperture_radius

        self.mass = self.mass[self.mask]
        self.position = self.position[self.mask]
        self.velocity = self.velocity[self.mask]
        self.radius = self.radius[self.mask]
        self.type = self.types[self.mask]

    @lazy_property
    def gas_mask_ap(self):
        return self.mask[self.types == "PartType0"]

    @lazy_property
    def dm_mask_ap(self):
        return self.mask[self.types == "PartType1"]

    @lazy_property
    def star_mask_ap(self):
        return self.mask[self.types == "PartType4"]

    @lazy_property
    def bh_mask_ap(self):
        return self.mask[self.types == "PartType5"]

    @lazy_property
    def baryon_mask_ap(self):
        return self.mask[(self.types == "PartType0") | (self.types == "PartType4")]

    @lazy_property
    def Ngas(self):
        return self.gas_mask_ap.sum()

    @lazy_property
    def Ndm(self):
        return self.dm_mask_ap.sum()

    @lazy_property
    def Nstar(self):
        return self.star_mask_ap.sum()

    @lazy_property
    def Nbh(self):
        return self.bh_mask_ap.sum()

    @lazy_property
    def Nbaryon(self):
        return self.baryon_mask_ap.sum()

    @lazy_property
    def mass_gas(self):
        return self.mass[self.type == "PartType0"]

    @lazy_property
    def mass_dm(self):
        return self.mass[self.type == "PartType1"]

    @lazy_property
    def mass_star(self):
        return self.mass[self.type == "PartType4"]

    @lazy_property
    def mass_baryons(self):
        return self.mass[(self.type == "PartType0") | (self.type == "PartType4")]

    @lazy_property
    def pos_gas(self):
        return self.position[self.type == "PartType0"]

    @lazy_property
    def pos_dm(self):
        return self.position[self.type == "PartType1"]

    @lazy_property
    def pos_star(self):
        return self.position[self.type == "PartType4"]

    @lazy_property
    def pos_baryons(self):
        return self.position[(self.type == "PartType0") | (self.type == "PartType4")]

    @lazy_property
    def vel_gas(self):
        return self.velocity[self.type == "PartType0"]

    @lazy_property
    def vel_dm(self):
        return self.velocity[self.type == "PartType1"]

    @lazy_property
    def vel_star(self):
        return self.velocity[self.type == "PartType4"]

    @lazy_property
    def vel_baryons(self):
        return self.velocity[(self.type == "PartType0") | (self.type == "PartType4")]

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
        return self.mass[self.type == "PartType5"].sum()

    @lazy_property
    def Mbaryons(self):
        return self.Mgas + self.Mstar

    @lazy_property
    def star_mask_all(self):
        if self.Nstar == 0:
            return None
        groupnr_bound = self.get_dataset("PartType4/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def Mstar_init(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/InitialMasses")[self.star_mask_all][
            self.star_mask_ap
        ].sum()

    @lazy_property
    def stellar_luminosities(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/Luminosities")[self.star_mask_all][
            self.star_mask_ap
        ]

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
            * self.get_dataset("PartType4/MetalMassFractions")[self.star_mask_all][
                self.star_mask_ap
            ]
        ).sum() / self.Mstar

    @lazy_property
    def star_element_fractions(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/ElementMassFractions")[self.star_mask_all][
            self.star_mask_ap
        ]

    @lazy_property
    def star_mass_O(self):
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
    def star_mass_Mg(self):
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
    def star_mass_Fe(self):
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
    def starOfrac(self):
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_O.sum() / self.Mstar

    @lazy_property
    def starMgfrac(self):
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_Mg.sum() / self.Mstar

    @lazy_property
    def starFefrac(self):
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return self.star_mass_Fe.sum() / self.Mstar

    @lazy_property
    def stellar_ages(self):
        if self.Nstar == 0:
            return None
        birth_a = self.get_dataset("PartType4/BirthScaleFactors")[self.star_mask_all][
            self.star_mask_ap
        ]
        return self.stellar_age_calculator.stellar_age(birth_a)

    @lazy_property
    def star_mass_fraction(self):
        if self.Mstar == 0:
            return None
        return self.mass_star / self.Mstar

    @lazy_property
    def stellar_age_mw(self):
        if self.Nstar == 0 or self.Mstar == 0:
            return None
        return (self.star_mass_fraction * self.stellar_ages).sum()

    @lazy_property
    def stellar_age_lw(self):
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
    def TotalSNIaRate(self):
        if self.Nstar == 0:
            return None
        return self.get_dataset("PartType4/SNIaRates")[self.star_mask_all][
            self.star_mask_ap
        ].sum()

    @lazy_property
    def bh_mask_all(self):
        if self.Nbh == 0:
            return None
        groupnr_bound = self.get_dataset("PartType5/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def BH_subgrid_masses(self):
        return self.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all][
            self.bh_mask_ap
        ]

    @lazy_property
    def Mbh_subgrid(self):
        if self.Nbh == 0:
            return None
        return self.BH_subgrid_masses.sum()

    @lazy_property
    def agn_eventa(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
            self.bh_mask_all
        ][self.bh_mask_ap]

    @lazy_property
    def BHlasteventa(self):
        if self.Nbh == 0:
            return None
        return np.max(self.agn_eventa)

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
        return self.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxpos(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Coordinates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxvel(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/Velocities")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxAR(self):
        if self.Nbh == 0:
            return None
        return self.get_dataset("PartType5/AccretionRates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxlasteventa(self):
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def mass_fraction(self):
        if self.Mtot == 0:
            return None
        return self.mass / self.Mtot

    @lazy_property
    def com(self):
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.position).sum(axis=0) + self.centre

    @lazy_property
    def vcom(self):
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.velocity).sum(axis=0)

    @lazy_property
    def spin_parameter(self):
        if self.Mtot == 0:
            return None
        _, vmax = get_vmax(self.mass, self.radius)
        if vmax == 0:
            return None
        vrel = self.velocity - self.vcom[None, :]
        Ltot = unyt.array.unorm(
            (self.mass[:, None] * unyt.array.ucross(self.position, vrel)).sum(axis=0)
        )
        return Ltot / (np.sqrt(2.0) * self.Mtot * self.aperture_radius * vmax)

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
    def veldisp_matrix_gas(self):
        if self.Mgas == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.gas_mass_fraction, self.vel_gas, self.vcom_gas
        )

    @lazy_property
    def Ekin_gas(self):
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
    def GasAxisLengths(self):
        if self.Mgas == 0:
            return None
        return get_axis_lengths(self.mass_gas, self.pos_gas)

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
    def veldisp_matrix_dm(self):
        if self.Mdm == 0:
            return None
        return get_velocity_dispersion_matrix(
            self.dm_mass_fraction, self.vel_dm, self.vcom_dm
        )

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
    def Ekin_star(self):
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
    def baryon_mass_fraction(self):
        if self.Mbaryons == 0:
            return None
        return self.mass_baryons / self.Mbaryons

    @lazy_property
    def vcom_bar(self):
        if self.Mbaryons == 0:
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
        if self.Mbaryons == 0:
            return None
        if not hasattr(self, "internal_Lbar"):
            self.compute_Lbar_props()
        return self.internal_Lbar

    @lazy_property
    def kappa_corot_baryons(self):
        if self.Mbaryons == 0:
            return None
        if not hasattr(self, "internal_kappa_bar"):
            self.compute_Lbar_props()
        return self.internal_kappa_bar

    @lazy_property
    def BaryonAxisLengths(self):
        if self.Mbaryons == 0:
            return None
        return get_axis_lengths(self.mass_baryons, self.pos_baryons)

    @lazy_property
    def gas_mask_all(self):
        if self.Ngas == 0:
            return None
        groupnr_bound = self.get_dataset("PartType0/GroupNr_bound")
        if self.inclusive:
            return np.ones(groupnr_bound.shape, dtype=bool)
        else:
            return groupnr_bound == self.index

    @lazy_property
    def gas_SFR(self):
        if self.Ngas == 0:
            return None
        raw_SFR = self.get_dataset("PartType0/StarFormationRates")[self.gas_mask_all][
            self.gas_mask_ap
        ]
        # Negative SFR are not SFR at all!
        raw_SFR[raw_SFR < 0] = 0
        return raw_SFR

    @lazy_property
    def is_SFR(self):
        if self.Ngas == 0:
            return None
        return self.gas_SFR > 0

    @lazy_property
    def SFR(self):
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def Mgas_SF(self):
        if self.Ngas == 0:
            return None
        return self.mass_gas[self.is_SFR].sum()

    @lazy_property
    def gas_Mgasmetal(self):
        if self.Ngas == 0:
            return None
        return (
            self.mass_gas
            * self.get_dataset("PartType0/MetalMassFractions")[self.gas_mask_all][
                self.gas_mask_ap
            ]
        )

    @lazy_property
    def gasmetalfrac_SF(self):
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_Mgasmetal[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasmetalfrac(self):
        if self.Ngas == 0:
            return None
        return self.gas_Mgasmetal.sum() / self.Mgas

    @lazy_property
    def gas_MgasO(self):
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
    def gasOfrac_SF(self):
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_MgasO[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasOfrac(self):
        if self.Ngas == 0:
            return None
        return self.gas_MgasO.sum() / self.Mgas

    @lazy_property
    def gas_MgasFe(self):
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
    def gasFefrac_SF(self):
        if self.Ngas == 0 or self.Mgas_SF == 0.0:
            return None
        return self.gas_MgasFe[self.is_SFR].sum() / self.Mgas_SF

    @lazy_property
    def gasFefrac(self):
        if self.Ngas == 0:
            return None
        return self.gas_MgasFe.sum() / self.Mgas

    @lazy_property
    def gas_temp(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Temperatures")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_rho(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/Densities")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_no_agn(self):
        if self.Ngas == 0:
            return None
        last_agn_gas = self.get_dataset("PartType0/LastAGNFeedbackScaleFactors")[
            self.gas_mask_all
        ][self.gas_mask_ap]
        return ~self.recently_heated_gas_filter.is_recently_heated(
            last_agn_gas, self.gas_temp
        )

    @lazy_property
    def Tgas(self):
        if self.Mgas == 0 or self.Ngas == 0:
            return None
        return (self.gas_mass_fraction * self.gas_temp).sum()

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
    def gas_element_fractions(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/ElementMassFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_mass_H(self):
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
    def gas_mass_He(self):
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
    def gas_species_fractions(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/SpeciesFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_mass_HI(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_mass_H
            * self.gas_species_fractions[
                :,
                self.snapshot_datasets.get_column_index("SpeciesFractions", "HI"),
            ]
        )

    @lazy_property
    def gas_mass_H2(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_mass_H
            * self.gas_species_fractions[
                :,
                self.snapshot_datasets.get_column_index("SpeciesFractions", "H2"),
            ]
        )

    @lazy_property
    def HydrogenMass(self):
        if self.Ngas == 0:
            return None
        return self.gas_mass_H.sum()

    @lazy_property
    def HeliumMass(self):
        if self.Ngas == 0:
            return None
        return self.gas_mass_He.sum()

    @lazy_property
    def MolecularHydrogenMass(self):
        if self.Ngas == 0:
            return None
        return self.gas_mass_H2.sum()

    @lazy_property
    def AtomicHydrogenMass(self):
        if self.Ngas == 0:
            return None
        return self.gas_mass_HI.sum()

    @lazy_property
    def gas_dust_mass_fractions(self):
        if self.Ngas == 0:
            return None
        return self.get_dataset("PartType0/DustMassFractions")[self.gas_mask_all][
            self.gas_mask_ap
        ]

    @lazy_property
    def gas_dust_mass_fractions_graphite_large(self):
        if self.Ngas == 0:
            return None
        return self.gas_dust_mass_fractions[
            :,
            self.snapshot_datasets.get_column_index(
                "DustMassFractions", "GraphiteLarge"
            ),
        ]

    @lazy_property
    def gas_dust_mass_fractions_silicates_large(self):
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
    def gas_dust_mass_fractions_graphite_small(self):
        if self.Ngas == 0:
            return None
        return self.gas_dust_mass_fractions[
            :,
            self.snapshot_datasets.get_column_index(
                "DustMassFractions", "GraphiteSmall"
            ),
        ]

    @lazy_property
    def gas_dust_mass_fractions_silicates_small(self):
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
    def gas_graphite_mass_fractions(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_large
            + self.gas_dust_mass_fractions_graphite_small
        )

    @lazy_property
    def gas_silicates_mass_fractions(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_silicates_large
            + self.gas_dust_mass_fractions_silicates_small
        )

    @lazy_property
    def gas_large_dust_mass_fractions(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_large
            + self.gas_dust_mass_fractions_silicates_large
        )

    @lazy_property
    def gas_small_dust_mass_fractions(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_dust_mass_fractions_graphite_small
            + self.gas_dust_mass_fractions_silicates_small
        )

    @lazy_property
    def gas_is_cold_dense(self):
        if self.Ngas == 0:
            return None
        return self.cold_dense_gas_filter.is_cold_and_dense(self.gas_temp, self.gas_rho)

    @lazy_property
    def DustGraphiteMass(self):
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustGraphiteMassInAtomicGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.gas_mass_HI).sum()

    @lazy_property
    def DustGraphiteMassInMolecularGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_graphite_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustGraphiteMassInColdDenseGas(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_graphite_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustSilicatesMass(self):
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustSilicatesMassInAtomicGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.gas_mass_HI).sum()

    @lazy_property
    def DustSilicatesMassInMolecularGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_silicates_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustSilicatesMassInColdDenseGas(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_silicates_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustLargeGrainMass(self):
        if self.Ngas == 0:
            return None
        return (self.gas_large_dust_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustLargeGrainMassInMolecularGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_large_dust_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustLargeGrainMassInColdDenseGas(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_large_dust_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def DustSmallGrainMass(self):
        if self.Ngas == 0:
            return None
        return (self.gas_small_dust_mass_fractions * self.mass_gas).sum()

    @lazy_property
    def DustSmallGrainMassInMolecularGas(self):
        if self.Ngas == 0:
            return None
        return (self.gas_small_dust_mass_fractions * self.gas_mass_H2).sum()

    @lazy_property
    def DustSmallGrainMassInColdDenseGas(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_small_dust_mass_fractions[self.gas_is_cold_dense]
            * self.mass_gas[self.gas_is_cold_dense]
        ).sum()

    @lazy_property
    def HalfMassRadiusGas(self):
        return get_half_mass_radius(
            self.radius[self.type == "PartType0"], self.mass_gas, self.Mgas
        )

    @lazy_property
    def HalfMassRadiusDM(self):
        return get_half_mass_radius(
            self.radius[self.type == "PartType1"], self.mass_dm, self.Mdm
        )

    @lazy_property
    def HalfMassRadiusStar(self):
        return get_half_mass_radius(
            self.radius[self.type == "PartType4"], self.mass_star, self.Mstar
        )

    @lazy_property
    def HalfMassRadiusBaryon(self):
        return get_half_mass_radius(
            self.radius[(self.type == "PartType0") | (self.type == "PartType4")],
            self.mass_baryons,
            self.Mbaryons,
        )


class ApertureProperties(HaloProperty):
    """
    Compute aperture properties for halos.

    The aperture has a fixed radius and optionally only includes particles that
    are bound to the halo.
    """

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
            "starmetalfrac",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "HalfMassRadiusBaryon",
            "spin_parameter",
            "GasAxisLengths",
            "DMAxisLengths",
            "StellarAxisLengths",
            "BaryonAxisLengths",
            "DtoTgas",
            "DtoTstar",
            "starOfrac",
            "starFefrac",
            "stellar_age_mw",
            "stellar_age_lw",
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
        ]
    ]

    def __init__(
        self,
        cellgrid,
        parameters,
        physical_radius_kpc,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        category_filter,
        inclusive=False,
    ):
        """
        Construct an ApertureProperties object with the given physical
        radius (in Mpc) that uses the given filter to filter out recently
        heated gas particles.
        """

        super().__init__(cellgrid)

        self.property_mask = parameters.get_property_mask(
            "ApertureProperties", [prop[1] for prop in self.property_list]
        )

        self.filter = recently_heated_gas_filter
        self.stellar_ages = stellar_age_calculator
        self.cold_dense_gas_filter = cold_dense_gas_filter
        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets

        # no density criterion for these properties
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.inclusive = inclusive

        if self.inclusive:
            self.name = f"inclusive_sphere_{physical_radius_kpc:.0f}kpc"
        else:
            self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"

        # List of particle properties we need to read in
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
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        types_present = [type for type in self.particle_properties if type in data]

        part_props = ApertureParticleData(
            input_halo,
            data,
            types_present,
            self.inclusive,
            self.physical_radius_mpc * unyt.Mpc,
            self.stellar_ages,
            self.filter,
            self.cold_dense_gas_filter,
            self.snapshot_datasets,
        )

        do_calculation = self.category_filter.get_filters(halo_result)

        aperture_sphere = {}
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
            aperture_sphere[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=registry
            )
            if do_calculation[category]:
                val = getattr(part_props, name)
                if val is not None:
                    if unit == "dimensionless":
                        aperture_sphere[name] = unyt.unyt_array(
                            val.astype(dtype),
                            dtype=dtype,
                            units=unit,
                            registry=registry,
                        )
                    else:
                        aperture_sphere[name] += val

        if self.inclusive:
            prefix = f"InclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        else:
            prefix = f"ExclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
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
                    f"{prefix}/{outputname}": (
                        aperture_sphere[name],
                        description,
                    )
                }
            )

        return


class ExclusiveSphereProperties(ApertureProperties):
    def __init__(
        self,
        cellgrid,
        parameters,
        physical_radius_kpc,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        category_filter,
    ):
        super().__init__(
            cellgrid,
            parameters,
            physical_radius_kpc,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            category_filter,
            False,
        )


class InclusiveSphereProperties(ApertureProperties):
    def __init__(
        self,
        cellgrid,
        parameters,
        physical_radius_kpc,
        recently_heated_gas_filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        category_filter,
    ):
        super().__init__(
            cellgrid,
            parameters,
            physical_radius_kpc,
            recently_heated_gas_filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            category_filter,
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

    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(3256)
    filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())
    stellar_age_calculator = StellarAgeCalculator(dummy_halos.get_cell_grid())
    cold_dense_gas_filter = ColdDenseGasFilter()
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
        filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
    )
    pc_inclusive = InclusiveSphereProperties(
        dummy_halos.get_cell_grid(),
        parameters,
        50.0,
        filter,
        stellar_age_calculator,
        cold_dense_gas_filter,
        cat_filter,
    )

    parameters.write_parameters("aperture.used_parameters.yml")

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _, particle_numbers = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
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
            pc_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
            assert input_halo == input_halo_copy
            assert input_data == input_data_copy

            # check that the calculation returns the correct values
            for prop in pc_calc.property_list:
                outputname = prop[1]
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"{pc_type}/50kpc/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
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
            filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
        )
        pc_inclusive = InclusiveSphereProperties(
            dummy_halos.get_cell_grid(),
            single_parameters,
            50.0,
            filter,
            stellar_age_calculator,
            cold_dense_gas_filter,
            cat_filter,
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
            pc_calc.calculate(input_halo, 0.0 * unyt.kpc, input_data, halo_result)
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
                unit = unyt.Unit(unit_string)
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
