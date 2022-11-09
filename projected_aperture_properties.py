##!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from property_table import PropertyTable
from kinematic_properties import get_projected_axis_lengths
from lazy_properties import lazy_property
from category_filter import CategoryFilter
from parameter_file import ParameterFile


class ProjectedApertureParticleData:
    def __init__(
        self,
        input_halo,
        data,
        types_present,
        aperture_radius,
        snapshot_datasets,
    ):
        self.input_halo = input_halo
        self.data = data
        self.types_present = types_present
        self.aperture_radius = aperture_radius
        self.snapshot_datasets = snapshot_datasets
        self.compute_basics()

    def get_dataset(self, name):
        return self.snapshot_datasets.get_dataset(name, self.data)

    def compute_basics(self):
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
            typearr = np.zeros(rprojx.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        self.mass = unyt.array.uconcatenate(mass)
        self.position = unyt.array.uconcatenate(position)
        self.radius_projx = unyt.array.uconcatenate(radius_projx)
        self.radius_projy = unyt.array.uconcatenate(radius_projy)
        self.radius_projz = unyt.array.uconcatenate(radius_projz)
        self.velocity = unyt.array.uconcatenate(velocity)
        self.types = np.concatenate(types)

        self.mask_projx = self.radius_projx <= self.aperture_radius
        self.mask_projy = self.radius_projy <= self.aperture_radius
        self.mask_projz = self.radius_projz <= self.aperture_radius


class SingleProjectionProjectedApertureParticleData:
    def __init__(self, part_props, projection):
        self.part_props = part_props
        self.index = part_props.index
        self.centre = part_props.centre
        self.types = part_props.types

        self.iproj = {"projx": 0, "projy": 1, "projz": 2}[projection]
        self.projmask = getattr(part_props, f"mask_{projection}")
        self.projr = getattr(part_props, f"radius_{projection}")

        self.proj_mass = part_props.mass[self.projmask]
        self.proj_position = part_props.position[self.projmask]
        self.proj_velocity = part_props.velocity[self.projmask]
        self.proj_radius = self.projr[self.projmask]
        self.proj_type = part_props.types[self.projmask]

    @lazy_property
    def gas_mask_ap(self):
        return self.projmask[self.types == "PartType0"]

    @lazy_property
    def dm_mask_ap(self):
        return self.projmask[self.types == "PartType1"]

    @lazy_property
    def star_mask_ap(self):
        return self.projmask[self.types == "PartType4"]

    @lazy_property
    def bh_mask_ap(self):
        return self.projmask[self.types == "PartType5"]

    @lazy_property
    def baryon_mask_ap(self):
        return self.projmask[(self.types == "PartType0") | (self.types == "PartType4")]

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
    def proj_mass_gas(self):
        return self.proj_mass[self.proj_type == "PartType0"]

    @lazy_property
    def proj_mass_dm(self):
        return self.proj_mass[self.proj_type == "PartType1"]

    @lazy_property
    def proj_mass_star(self):
        return self.proj_mass[self.proj_type == "PartType4"]

    @lazy_property
    def proj_mass_baryons(self):
        return self.proj_mass[
            (self.proj_type == "PartType0") | (self.proj_type == "PartType4")
        ]

    @lazy_property
    def proj_pos_gas(self):
        return self.proj_position[self.proj_type == "PartType0"]

    @lazy_property
    def proj_pos_dm(self):
        return self.proj_position[self.proj_type == "PartType1"]

    @lazy_property
    def proj_pos_star(self):
        return self.proj_position[self.proj_type == "PartType4"]

    @lazy_property
    def proj_pos_baryons(self):
        return self.proj_position[
            (self.proj_type == "PartType0") | (self.proj_type == "PartType4")
        ]

    @lazy_property
    def Mtot(self):
        return self.proj_mass.sum()

    @lazy_property
    def Mgas(self):
        return self.proj_mass_gas.sum()

    @lazy_property
    def Mdm(self):
        return self.proj_mass_dm.sum()

    @lazy_property
    def Mstar(self):
        return self.proj_mass_star.sum()

    @lazy_property
    def Mbh_dynamical(self):
        return self.proj_mass[self.proj_type == "PartType5"].sum()

    @lazy_property
    def Mbaryons(self):
        return self.proj_mass_baryons.sum()

    @lazy_property
    def star_mask_all(self):
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/GroupNr_bound") == self.index

    @lazy_property
    def Mstar_init(self):
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/InitialMasses")[
            self.star_mask_all
        ][self.star_mask_ap].sum()

    @lazy_property
    def stellar_luminosities(self):
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/Luminosities")[
            self.star_mask_all
        ][self.star_mask_ap]

    @lazy_property
    def StellarLuminosity(self):
        if self.Nstar == 0:
            return None
        return self.stellar_luminosities.sum(axis=0)

    @lazy_property
    def bh_mask_all(self):
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/GroupNr_bound") == self.index

    @lazy_property
    def BH_subgrid_masses(self):
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/SubgridMasses")[self.bh_mask_all][
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
        return self.part_props.get_dataset("PartType5/LastAGNFeedbackScaleFactors")[
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
        return self.part_props.get_dataset("PartType5/ParticleIDs")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxpos(self):
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/Coordinates")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxvel(self):
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/Velocities")[self.bh_mask_all][
            self.bh_mask_ap
        ][self.iBHmax]

    @lazy_property
    def BHmaxAR(self):
        if self.Nbh == 0:
            return None
        return self.part_props.get_dataset("PartType5/AccretionRates")[
            self.bh_mask_all
        ][self.bh_mask_ap][self.iBHmax]

    @lazy_property
    def BHmaxlasteventa(self):
        if self.Nbh == 0:
            return None
        return self.agn_eventa[self.iBHmax]

    @lazy_property
    def mass_fraction(self):
        if self.Mtot == 0:
            return None
        return self.proj_mass / self.Mtot

    @lazy_property
    def com(self):
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.proj_position).sum(
            axis=0
        ) + self.centre

    @lazy_property
    def vcom(self):
        if self.Mtot == 0:
            return None
        return (self.mass_fraction[:, None] * self.proj_velocity).sum(axis=0)

    @lazy_property
    def gas_mass_fraction(self):
        if self.Mgas == 0:
            return None
        return self.proj_mass_gas / self.Mgas

    @lazy_property
    def proj_veldisp_gas(self):
        if self.Mgas == 0:
            return None
        proj_vgas = self.proj_velocity[self.proj_type == "PartType0", self.iproj]
        vcom_gas = (self.gas_mass_fraction * proj_vgas).sum()
        return np.sqrt(((proj_vgas - vcom_gas) ** 2).sum())

    @lazy_property
    def ProjectedGasAxisLengths(self):
        if self.Mgas == 0:
            return None
        return get_projected_axis_lengths(
            self.proj_mass_gas, self.proj_pos_gas, self.iproj
        )

    @lazy_property
    def dm_mass_fraction(self):
        if self.Mdm == 0:
            return None
        return self.proj_mass_dm / self.Mdm

    @lazy_property
    def proj_veldisp_dm(self):
        if self.Mdm == 0:
            return None
        proj_vdm = self.proj_velocity[self.proj_type == "PartType1", self.iproj]
        vcom_dm = (self.dm_mass_fraction * proj_vdm).sum()
        return np.sqrt(((proj_vdm - vcom_dm) ** 2).sum())

    @lazy_property
    def star_mass_fraction(self):
        if self.Mstar == 0:
            return None
        return self.proj_mass_star / self.Mstar

    @lazy_property
    def proj_veldisp_star(self):
        if self.Mstar == 0:
            return None
        proj_vstar = self.proj_velocity[self.proj_type == "PartType4", self.iproj]
        vcom_star = (self.star_mass_fraction * proj_vstar).sum()
        return np.sqrt(((proj_vstar - vcom_star) ** 2).sum())

    @lazy_property
    def ProjectedStellarAxisLengths(self):
        if self.Mstar == 0:
            return None
        return get_projected_axis_lengths(
            self.proj_mass_star, self.proj_pos_star, self.iproj
        )

    @lazy_property
    def ProjectedBaryonAxisLengths(self):
        if self.Mbaryons == 0:
            return None
        return get_projected_axis_lengths(
            self.proj_mass_baryons, self.proj_pos_baryons, self.iproj
        )

    @lazy_property
    def gas_mask_all(self):
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/GroupNr_bound") == self.index

    @lazy_property
    def gas_SFR(self):
        if self.Ngas == 0:
            return None
        raw_SFR = self.part_props.get_dataset("PartType0/StarFormationRates")[
            self.gas_mask_all
        ][self.gas_mask_ap]
        # Negative SFR are not SFR at all!
        raw_SFR[raw_SFR < 0] = 0
        return raw_SFR

    @lazy_property
    def SFR(self):
        if self.Ngas == 0:
            return None
        return self.gas_SFR.sum()

    @lazy_property
    def gas_element_fractions(self):
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/ElementMassFractions")[
            self.gas_mask_all
        ][self.gas_mask_ap]

    @lazy_property
    def gas_mass_H(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_element_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Hydrogen"
                ),
            ]
            * self.proj_mass_gas
        )

    @lazy_property
    def gas_mass_He(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_element_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Helium"
                ),
            ]
            * self.proj_mass_gas
        )

    @lazy_property
    def gas_species_fractions(self):
        if self.Ngas == 0:
            return None
        return self.part_props.get_dataset("PartType0/SpeciesFractions")[
            self.gas_mask_all
        ][self.gas_mask_ap]

    @lazy_property
    def gas_mass_HI(self):
        if self.Ngas == 0:
            return None
        return (
            self.gas_mass_H
            * self.gas_species_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "SpeciesFractions", "HI"
                ),
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
                self.part_props.snapshot_datasets.get_column_index(
                    "SpeciesFractions", "H2"
                ),
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
    def star_element_fractions(self):
        if self.Nstar == 0:
            return None
        return self.part_props.get_dataset("PartType4/ElementMassFractions")[
            self.star_mask_all
        ][self.star_mask_ap]

    @lazy_property
    def star_mass_O(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Oxygen"
                ),
            ]
            * self.proj_mass_star
        )

    @lazy_property
    def star_mass_Mg(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Magnesium"
                ),
            ]
            * self.proj_mass_star
        )

    @lazy_property
    def star_mass_Fe(self):
        if self.Nstar == 0:
            return None
        return (
            self.star_element_fractions[
                :,
                self.part_props.snapshot_datasets.get_column_index(
                    "ElementMassFractions", "Iron"
                ),
            ]
            * self.proj_mass_star
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
    def starmetalfrac(self):
        if self.Nstar == 0 or self.Mstar == 0.0:
            return None
        return (
            self.part_props.get_dataset("PartType4/MetalMassFractions")[
                self.star_mask_all
            ][self.star_mask_ap]
            * self.proj_mass_star
        ).sum() / self.Mstar

    @lazy_property
    def HalfMassRadiusGas(self):
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == "PartType0"],
            self.proj_mass_gas,
            self.Mgas,
        )

    @lazy_property
    def HalfMassRadiusDM(self):
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == "PartType1"], self.proj_mass_dm, self.Mdm
        )

    @lazy_property
    def HalfMassRadiusStar(self):
        return get_half_mass_radius(
            self.proj_radius[self.proj_type == "PartType4"],
            self.proj_mass_star,
            self.Mstar,
        )

    @lazy_property
    def HalfMassRadiusBaryon(self):
        return get_half_mass_radius(
            self.proj_radius[
                (self.proj_type == "PartType0") | (self.proj_type == "PartType4")
            ],
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
            "com",
            "vcom",
            "SFR",
            "StellarLuminosity",
            "HalfMassRadiusGas",
            "HalfMassRadiusDM",
            "HalfMassRadiusStar",
            "HalfMassRadiusBaryon",
            "proj_veldisp_gas",
            "proj_veldisp_dm",
            "proj_veldisp_star",
            "BHmaxM",
            "BHmaxID",
            "BHmaxpos",
            "BHmaxvel",
            "BHlasteventa",
            "BHmaxlasteventa",
            "ProjectedGasAxisLengths",
            "ProjectedStellarAxisLengths",
            "ProjectedBaryonAxisLengths",
            "HydrogenMass",
            "HeliumMass",
            "MolecularHydrogenMass",
            "AtomicHydrogenMass",
            "starFefrac",
            "starMgfrac",
            "starOfrac",
            "starmetalfrac",
        ]
    ]

    def __init__(self, cellgrid, parameters, physical_radius_kpc, category_filter):
        super().__init__(cellgrid)

        self.property_mask = parameters.get_property_mask(
            "ProjectedApertureProperties", [prop[1] for prop in self.property_list]
        )

        # No density criterion
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.category_filter = category_filter
        self.snapshot_datasets = cellgrid.snapshot_datasets

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"

        # Particle properties that are used
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

        part_props = ProjectedApertureParticleData(
            input_halo,
            data,
            types_present,
            self.physical_radius_mpc * unyt.Mpc,
            self.snapshot_datasets,
        )

        do_calculation = self.category_filter.get_filters(halo_result)

        registry = part_props.mass.units.registry
        for projname in ["projx", "projy", "projz"]:
            proj_part_props = SingleProjectionProjectedApertureParticleData(
                part_props, projname
            )

            projected_aperture = {}
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
                category = prop[6]
                if shape > 1:
                    val = [0] * shape
                else:
                    val = 0
                projected_aperture[name] = unyt.unyt_array(
                    val, dtype=dtype, units=unit, registry=registry
                )
                if do_calculation[category]:
                    val = getattr(proj_part_props, name)
                    if val is not None:
                        if unit == "dimensionless":
                            projected_aperture[name] = unyt.unyt_array(
                                val.astype(dtype),
                                dtype=dtype,
                                units=unit,
                                registry=registry,
                            )
                        else:
                            projected_aperture[name] += val

            prefix = (
                f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc/{projname}"
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
                description = prop[5]
                halo_result.update(
                    {f"{prefix}/{outputname}": (projected_aperture[name], description)}
                )

        return


def test_projected_aperture_properties():
    """
    Unit test for the projected aperture calculation.

    Generates 100 random halos and passes them on to
    ProjectedApertureProperties::calculate().
    Tests that all expected return values are computed and have the right size,
    dtype and units.
    """

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(127)
    category_filter = CategoryFilter(
        {"general": 100, "gas": 100, "dm": 100, "star": 100, "baryon": 100}
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
        "ProjectedApertureProperties", {"30_kpc": {"radius_in_kpc": 30.0}}
    )

    property_calculator = ProjectedApertureProperties(
        dummy_halos.get_cell_grid(), parameters, 30.0, category_filter
    )

    parameters.write_parameters("projected_apertures.used_parameters.yml")

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

        input_data = {}
        for ptype in property_calculator.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        halo_result = dict(halo_result_template)
        property_calculator.calculate(
            input_halo, 0.0 * unyt.kpc, input_data, halo_result
        )
        assert input_halo == input_halo_copy
        assert input_data == input_data_copy

        for proj in ["projx", "projy", "projz"]:
            for prop in property_calculator.property_list:
                outputname = prop[1]
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    # Now test the calculation for each property individually, to make sure that
    # all properties read all the datasets they require
    all_parameters = parameters.get_parameters()
    for property in all_parameters["ProjectedApertureProperties"]["properties"]:
        print(f"Testing only {property}...")
        single_property = dict(all_parameters)
        for other_property in all_parameters["ProjectedApertureProperties"][
            "properties"
        ]:
            single_property["ProjectedApertureProperties"]["properties"][
                other_property
            ] = (other_property == property) or other_property.startswith("NumberOf")
        single_parameters = ParameterFile(parameter_dictionary=single_property)

        property_calculator = ProjectedApertureProperties(
            dummy_halos.get_cell_grid(), single_parameters, 30.0, category_filter
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

        input_data = {}
        for ptype in property_calculator.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        halo_result = dict(halo_result_template)
        property_calculator.calculate(
            input_halo, 0.0 * unyt.kpc, input_data, halo_result
        )
        assert input_halo == input_halo_copy
        assert input_data == input_data_copy

        for proj in ["projx", "projy", "projz"]:
            for prop in property_calculator.property_list:
                outputname = prop[1]
                if not outputname == property:
                    continue
                size = prop[2]
                dtype = prop[3]
                unit_string = prop[4]
                full_name = f"ProjectedAperture/30kpc/{proj}/{outputname}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)

    dummy_halos.get_cell_grid().snapshot_datasets.print_dataset_log()


if __name__ == "__main__":
    """
    Standalone mode: simply run the unit test.

    Note that this can also be achieved by running
    python3 -m pytest *.py
    in the main folder.
    """
    print("Calling test_projected_aperture_properties()...")
    test_projected_aperture_properties()
    print("Test passed.")
