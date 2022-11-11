#! /usr/bin/env python3

import numpy as np
import unyt
import types
from swift_units import unit_registry_from_snapshot
from snapshot_datasets import SnapshotDatasets


class DummySnapshot:
    def __init__(self):
        """
        Values extracted from a 400 Mpc FLAMINGO snapshot at z=3.
        """
        self.metadata = {
            "PhysicalConstants/CGS": {
                "T_CMB_0": np.array([2.7255]),
                "astronomical_unit": np.array([1.49597871e13]),
                "avogadro_number": np.array([6.02214076e23]),
                "boltzmann_k": np.array([1.380649e-16]),
                "caseb_recomb": np.array([2.6e-13]),
                "earth_mass": np.array([5.97217e27]),
                "electron_charge": np.array([1.60217663e-19]),
                "electron_mass": np.array([9.1093837e-28]),
                "electron_volt": np.array([1.60217663e-12]),
                "light_year": np.array([9.46063e17]),
                "newton_G": np.array([6.6743e-08]),
                "parsec": np.array([3.08567758e18]),
                "planck_h": np.array([6.62607015e-27]),
                "planck_hbar": np.array([1.05457182e-27]),
                "primordial_He_fraction": np.array([0.248]),
                "proton_mass": np.array([1.67262192e-24]),
                "reduced_hubble": np.array([3.24077929e-18]),
                "solar_mass": np.array([1.98841e33]),
                "speed_light_c": np.array([2.99792458e10]),
                "stefan_boltzmann": np.array([5.67037442e-05]),
                "thomson_cross_section": np.array([6.65245873e-25]),
                "year": np.array([31556925.1]),
            },
            "Cosmology": {
                "Cosmological run": np.array([1], dtype=np.int32),
                "Critical density [internal units]": np.array([17.58736923]),
                "H [internal units]": np.array([79.60499176]),
                "H0 [internal units]": np.array([68.09999997]),
                "Hubble time [internal units]": np.array([0.01468429]),
                "Lookback time [internal units]": np.array([0.00358892]),
                "M_nu_eV": np.array([0.06]),
                "N_eff": np.array([3.04400163]),
                "N_nu": np.array([1.0]),
                "N_ur": np.array([2.0308]),
                "Omega_b": np.array([0.0486]),
                "Omega_cdm": np.array([0.256011]),
                "Omega_g": np.array([5.33243487e-05]),
                "Omega_k": np.array([2.5212783e-09]),
                "Omega_lambda": np.array([0.693922]),
                "Omega_m": np.array([0.304611]),
                "Omega_nu": np.array([0.00106856]),
                "Omega_nu_0": np.array([0.00138908]),
                "Omega_r": np.array([7.79180471e-05]),
                "Omega_ur": np.array([2.45936984e-05]),
                "Redshift": np.array([0.3]),
                "Scale-factor": np.array([0.76923077]),
                "T_CMB_0 [K]": np.array([2.7255]),
                "T_CMB_0 [internal units]": np.array([2.7255]),
                "T_nu_0 [eV]": np.array([0.00016819]),
                "T_nu_0 [internal units]": np.array([1.9517578]),
                "Universe age [internal units]": np.array([0.01048484]),
                "a_beg": np.array([0.03125]),
                "a_end": np.array([1.0]),
                "deg_nu": np.array([1.0]),
                "deg_nu_tot": np.array([1.0]),
                "h": np.array([0.681]),
                "time_beg [internal units]": np.array([9.66296122e-05]),
                "time_end [internal units]": np.array([0.01407376]),
                "w": np.array([-1.0]),
                "w_0": np.array([-1.0]),
                "w_a": np.array([0.0]),
            },
            "Units": {
                "Unit current in cgs (U_I)": np.array([1.0]),
                "Unit length in cgs (U_L)": np.array([3.08567758e24]),
                "Unit mass in cgs (U_M)": np.array([1.98841e43]),
                "Unit temperature in cgs (U_T)": np.array([1.0]),
                "Unit time in cgs (U_t)": np.array([3.08567758e19]),
            },
            "InternalCodeUnits": {
                "Unit current in cgs (U_I)": np.array([1.0]),
                "Unit length in cgs (U_L)": np.array([3.08567758e24]),
                "Unit mass in cgs (U_M)": np.array([1.98841e43]),
                "Unit temperature in cgs (U_T)": np.array([1.0]),
                "Unit time in cgs (U_t)": np.array([3.08567758e19]),
            },
        }

    def __getitem__(self, name):
        if not name in self.metadata:
            raise AttributeError(f"No {name} in dummy snapshot file!")
        x = types.SimpleNamespace()
        x.attrs = self.metadata[name]
        return x


class DummySnapshotDatasets(SnapshotDatasets):
    def __init__(self):
        self.datasets_in_file = {
            "PartType0": [
                "Coordinates",
                "Masses",
                "Velocities",
                "MetalMassFractions",
                "Temperatures",
                "LastAGNFeedbackScaleFactors",
                "StarFormationRates",
                "XrayLuminosities",
                "XrayPhotonLuminosities",
                "ComptonYParameters",
                "Pressures",
                "Densities",
                "ElectronNumberDensities",
                "SpeciesFractions",
                "DustMassFractions",
                "LastSNIIKineticFeedbackDensities",
                "LastSNIIThermalFeedbackDensities",
                "ElementMassFractionsDiffuse",
                "SmoothedElementMassFractions",
            ],
            "PartType1": [
                "Coordinates",
                "Masses",
                "Velocities",
            ],
            "PartType4": [
                "Coordinates",
                "Masses",
                "Velocities",
                "InitialMasses",
                "Luminosities",
                "MetalMassFractions",
                "BirthScaleFactors",
                "SNIaRates",
                "BirthDensities",
                "SmoothedElementMassFractions",
            ],
            "PartType5": [
                "Coordinates",
                "DynamicalMasses",
                "Velocities",
                "SubgridMasses",
                "LastAGNFeedbackScaleFactors",
                "ParticleIDs",
                "AccretionRates",
            ],
            "PartType6": ["Coordinates", "Masses", "Weights"],
        }

        self.named_columns = {
            "Luminosities": {"GAMA_r": 2},
            "SmoothedElementMassFractions": {
                "Hydrogen": 0,
                "Helium": 1,
                "Carbon": 2,
                "Nitrogen": 3,
                "Oxygen": 4,
                "Neon": 5,
                "Magnesium": 6,
                "Silicon": 7,
                "Iron": 8,
            },
            "SpeciesFractions": {
                "elec": 0,
                "HI": 1,
                "HII": 2,
                "Hm": 3,
                "HeI": 4,
                "HeII": 5,
                "HeIII": 6,
                "H2": 7,
                "H2p": 8,
                "H3p": 9,
            },
            "DustMassFractions": {
                "GraphiteLarge": 0,
                "MgSilicatesLarge": 1,
                "FeSilicatesLarge": 2,
                "GraphiteSmall": 3,
                "MgSilicatesSmall": 4,
                "FeSilicatesSmall": 5,
            },
        }

        self.dust_grain_composition = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.45487377,
                    0.0,
                    0.3455038,
                    0.19962244,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.31406304,
                    0.0,
                    0.0,
                    0.13782732,
                    0.54810965,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.45487377,
                    0.0,
                    0.3455038,
                    0.19962244,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.31406304,
                    0.0,
                    0.0,
                    0.13782732,
                    0.54810965,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        self.datasets_used = set()
        self.columns_used = set()

    def get_dataset(self, name, data):
        self.datasets_used.add(name)
        return super().get_dataset(name, data)

    def get_column_index(self, name, column):
        self.columns_used.add(f"{name}/{column}")
        return super().get_column_index(name, column)

    def print_dataset_log(self):
        print(f"Datasets used: {self.datasets_used}")
        print(f"Columns used: {self.columns_used}")


class DummyCellGrid:
    """
    Minimal CellGrid, that contains just enough information to be passed on to
    the HaloProperty and RecentlyHeatedGasFilter constructors.
    """

    def get_unit(self, name, reg):
        return unyt.Unit(name, registry=reg)

    def __init__(self, reg, snap):
        self.a_unit = self.get_unit("a", reg)
        self.a = self.a_unit.base_value
        self.z = 1.0 / self.a - 1.0
        self.cosmology = {}
        cosmology_attrs = snap["Cosmology"].attrs
        for name in cosmology_attrs:
            self.cosmology[name] = cosmology_attrs[name][0]
        self.snap_unit_registry = reg
        critical_density = float(self.cosmology["Critical density [internal units]"])
        internal_density_unit = self.get_unit("code_mass", reg) / (
            self.get_unit("code_length", reg) ** 3
        )
        self.critical_density = unyt.unyt_quantity(
            critical_density, units=internal_density_unit
        )
        self.mean_density = self.critical_density * self.cosmology["Omega_m"]
        # Compute the BN98 critical density multiple
        Omega_k = self.cosmology["Omega_k"]
        Omega_Lambda = self.cosmology["Omega_lambda"]
        Omega_m = self.cosmology["Omega_m"]
        bnx = -(Omega_k / self.a**2 + Omega_Lambda) / (
            Omega_k / self.a**2 + Omega_m / self.a**3 + Omega_Lambda
        )
        self.virBN98 = 18.0 * np.pi**2 + 82.0 * bnx - 39.0 * bnx**2
        if self.virBN98 < 50.0 or self.virBN98 > 1000.0:
            raise RuntimeError("Invalid value for virBN98!")

        # Get the box size. Assume it's comoving with no h factors.
        comoving_length_unit = self.get_unit("snap_length", reg) * self.a_unit
        self.boxsize = unyt.unyt_quantity(100.0, units=comoving_length_unit)
        self.observer_position = unyt.unyt_array([50.0] * 3, units=comoving_length_unit)

        self.snapshot_datasets = DummySnapshotDatasets()


class DummyHaloGenerator:
    """
    Object used to generate random halos.

    The random halos contain all the variables a real halo would get, expressed
    in the right units and with realistic values.
    """

    def __init__(self, seed):
        """
        Set up an artificial snapshot and extract the unit system.
        Seed the random number generator.
        """
        self.dummy_snapshot = DummySnapshot()
        self.unit_registry = unit_registry_from_snapshot(self.dummy_snapshot)
        self.dummy_cellgrid = DummyCellGrid(self.unit_registry, self.dummy_snapshot)
        np.random.seed(seed)

    def get_cell_grid(self):
        """
        Return a minimal cell grid that is consistent with the random halos
        that are generated.
        """
        return self.dummy_cellgrid

    def get_random_halo(self, npart, has_neutrinos=False):
        """
        Generate a random halo, with the given number of particles.
        If npart is a list, a random element of the list is chosen.

        To get a rough idea of the ranges found in a typical halo, this is the
        input for one halo from a test run on a 400 Mpc FLAMINGO box (we later
        added values from COLIBRE runs as well):
        (input_type, units, min value, max value)
        types = [
        "PartType0": {
          "ComptonYParameters": (np.float32, snap_length**2, 0., 5.e-9),
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "Densities": (np.float32, snap_mass/(a**3*snap_length**3), 0.1, 1.e8),
          "DustMassFractions":
            (np.float32, dimensionless,
             [0., 0., 0., 0., 0., 0.],
             [6.7e-3, 5.3e-3, 1.1e-2, 4.4e-3, 4.1e-3, 1.1e-2]),
          "ElectronNumberDensities": (np.float64, 1/snap_length**3, 0., 3.4e73),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "LastAGNFeedbackScaleFactors": (np.float32, dimensionless, 0., 1.),
          "LastSNIIKineticFeedbackDensities": (
            np.float32, snap_mass/snap_length**3, 5.84e1, 1.56e10),
          "LastSNIIThermalFeedbackDensities": (
            np.float32, snap_mass/snap_length**3, 5.84e1, 1.56e10),
          "Masses": (np.float32, snap_mass, 0.1, 0.1),
          "MetalMassFractions": (np.float32, dimensionless, 0., 0.06),
          "Pressures": (np.float32, snap_mass/(a**5*snap_length*snap_time**2),
            2.8, 1.e9),
          "SmoothedElementMassFractions":
            (np.float32, dimensionless,
             [0.68, 0.24, 0., 0., 0., 0., 0., 0., 0.],
             [0.75, 0.29, 0.006, 0.001, 0.01, 0.002, 0.0008, 0.002, 0.002]),
          "SpeciesFractions":
            (np.float32, dimensionless,
             [3.94e-5, 0., 1.78e-5, 0., 0., 2.05e-10, 0., 0., 0., 0.],
             [1.26, 1., 1., 2.53e-8, 0.134, 0.122, 0.125, 0.5, 9.67e-6, 2.14e-5]),
          "StarFormationRates": (np.float32, snap_mass/snap_time, -0.99, 246.5),
          "Temperatures": (np.float32, snap_temperature, 1.e3, 1.e10),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
          "XrayLuminosities":
            (np.float64, snap_length**2*snap_mass/snap_time**3,
             [0., 0., 0.], [1.e7, 1.e7, 1.e7]),
          "XrayPhotonLuminosities":
            (np.float64, 1/snap_time,
             [0., 0., 0.], [1.e70, 1.e70, 1.e70]),
        },
        "PartType1": {
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "Masses": (np.float32, snap_mass, 0.5, 0.5),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        "PartType4": {
          "BirthDensities": (np.float32, snap_mass/snap_length**3, 2.5e5, 3.38e11),
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "InitialMasses": (np.float32, snap_mass, 0.1, 0.3),
          "Luminosities": (np.float32, dimensionless, 1.e5, 1.e10),
          "Masses": (np.float32, snap_mass, 0.06, 0.1),
          "MetalMassFractions": (np.float32, dimensionless, 0., 0.075),
          "SNIaRates": (np.float64, 1/snap_time, 0., 5.36e7),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        "PartType5": {
          "AccretionRates": (np.float32, snap_mass/snap_time, 0., 0.07),
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "DynamicalMasses": (np.float32, snap_mass, 0.1, 0.1),
          "GroupNr_bound": (np.int32, dimensionless, N/A),
          "LastAGNFeedbackScaleFactors": (np.float32, dimensionless, 0., 1.),
          "ParticleIDs": (np.int64, dimensionless, N/A),
          "SubgridMasses": (np.float32, snap_mass, 0.00001, 0.1),
          "Velocities": (np.float32, snap_length/snap_time, -1.e3, 1.e3),
        }
        "PartType6": (based on the L1000N1800/HYDRO_FIDUCIAL snapshot) {
          "Coordinates": (np.float64, a*snap_length, 0., boxsize),
          "Masses": (np.float32, snap_mass, 0.018, 0.018),
          "Weights": (np.float64, dimensionless, -0.46, 0.71),
        }
        """

        if isinstance(npart, list):
            npart = np.random.choice(npart)

        reg = self.unit_registry

        centre = unyt.unyt_array(
            100.0 * np.random.random(3),
            dtype=np.float64,
            units="snap_length",
            registry=reg,
        )
        # the random halo always gets GroupNr 1
        groupnr_halo = 1
        # structure type: mostly centrals, but some satellites
        structuretype = np.random.choice([10, 20], p=[0.99, 0.01])

        # Generate a random radius from an exponential distribution.
        # The chosen beta parameter should ensure that ~90% of the values is
        # below 50 kpc.
        radius = np.random.exponential(1.0 / 60.0, npart)
        # force the first particle to be at the centre
        radius[0] = 0.0
        # generate a random direction to convert the radius into an actual
        # coordinate
        phi = 2.0 * np.pi * np.random.random(npart)
        sintheta = 2.0 * np.random.random(npart) - 1.0
        costheta = np.sqrt((1.0 - sintheta) * (1.0 + sintheta))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        coords = np.zeros((npart, 3))
        coords[:, 0] = radius * cosphi * sintheta
        coords[:, 1] = radius * sinphi * sintheta
        coords[:, 2] = radius * costheta
        coords = unyt.unyt_array(
            coords, dtype=np.float64, units="snap_length", registry=reg
        )
        rmax = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2).max()
        # add the (random) halo centre
        coords += centre
        mass = unyt.unyt_array(
            0.1 + 0.4 * np.random.random(npart),
            dtype=np.float32,
            units="snap_mass",
            registry=reg,
        )
        vs = unyt.unyt_array(
            1000.0 * (np.random.random((npart, 3)) - 0.5),
            dtype=np.float32,
            units="snap_length/snap_time",
            registry=reg,
        )

        # randomly allocate particle types
        # we expect halos to be dominated by DM and star particles and have
        # relatively little BH particles
        possible_types = ["PartType0", "PartType1", "PartType4", "PartType5"]
        probability_for_type = [0.2, 0.4, 0.39, 0.01]
        if has_neutrinos:
            possible_types.append("PartType6")
            probability_for_type.append(0.1)
            probability_for_type = np.array(probability_for_type)
            probability_for_type /= probability_for_type.sum()
        types = np.random.choice(possible_types, size=npart, p=probability_for_type)
        # make sure we have at least 1 non neutrino particle
        if (types == "PartType6").sum() == npart:
            types[0] = "PartType1"
        # randomly assign bound particles to the halo
        # we make sure most particles will be bound, and use two different
        # alternative values just in case that matters
        groupnr_all = unyt.unyt_array(
            np.random.choice([groupnr_halo, 2, 3], size=npart, p=[0.6, 0.2, 0.2]),
            dtype=np.int32,
            units=unyt.dimensionless,
            registry=reg,
        )
        # randomly unbind 10% of the particles
        index = np.random.choice(groupnr_all.shape[0], npart // 10, replace=False)
        groupnr_bound = groupnr_all.copy()
        groupnr_bound[index] = -1

        Mtot = 0.0
        data = {}
        # gas particle variables
        gas_mask = types == "PartType0"
        Ngas = int(gas_mask.sum())
        if Ngas > 0:
            data["PartType0"] = {}
            data["PartType0"]["ComptonYParameters"] = unyt.unyt_array(
                5.0e-9 * np.random.random(Ngas),
                dtype=np.float32,
                units="snap_length**2",
                registry=reg,
            )
            data["PartType0"]["Coordinates"] = coords[gas_mask]
            data["PartType0"]["Densities"] = unyt.unyt_array(
                10.0 ** (10.0 * np.random.random(Ngas) - 2.0),
                dtype=np.float32,
                units="snap_mass/(a**3*snap_length**3)",
                registry=reg,
            )
            dmf = np.zeros((Ngas, 6))
            dmf[:, 0] = 6.7e-3 * np.random.random(Ngas)
            dmf[:, 1] = 5.3e-3 * np.random.random(Ngas)
            dmf[:, 2] = 1.1e-2 * np.random.random(Ngas)
            dmf[:, 3] = 4.4e-3 * np.random.random(Ngas)
            dmf[:, 4] = 4.1e-3 * np.random.random(Ngas)
            dmf[:, 5] = 1.1e-2 * np.random.random(Ngas)
            data["PartType0"]["DustMassFractions"] = unyt.unyt_array(
                dmf, dtype=np.float32, units=unyt.dimensionless, registry=reg
            )
            data["PartType0"]["ElectronNumberDensities"] = unyt.unyt_array(
                10.0 ** (65.0 + 8.0 * np.random.random(Ngas)),
                dtype=np.float64,
                units="1/snap_length**3",
                registry=reg,
            )
            idx0 = np.random.choice(np.arange(Ngas), size=Ngas // 10, replace=False)
            data["PartType0"]["ElectronNumberDensities"][idx0] = 0.0
            data["PartType0"]["GroupNr_all"] = groupnr_all[gas_mask]
            data["PartType0"]["GroupNr_bound"] = groupnr_bound[gas_mask]
            # we assume a fixed "snapshot" redshift of 0.1, so we make sure
            # the random values span a range of scale factors that is lower
            data["PartType0"]["LastAGNFeedbackScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["LastSNIIKineticFeedbackDensities"] = unyt.unyt_array(
                10.0 ** (1.77 + (10.2 - 1.77) * np.random.random(Ngas)),
                dtype=np.float32,
                units="snap_mass/snap_length**3",
                registry=reg,
            )
            data["PartType0"]["LastSNIIThermalFeedbackDensities"] = unyt.unyt_array(
                10.0 ** (1.77 + (10.2 - 1.77) * np.random.random(Ngas)),
                dtype=np.float32,
                units="snap_mass/snap_length**3",
                registry=reg,
            )
            # randomly set some values to -1
            data["PartType0"]["LastAGNFeedbackScaleFactors"][
                np.random.random() > 0.9
            ] = -1
            data["PartType0"]["LastSNIIKineticFeedbackDensities"][
                np.random.random() > 0.9
            ] = -1
            data["PartType0"]["LastSNIIThermalFeedbackDensities"][
                np.random.random() > 0.9
            ] = -1
            data["PartType0"]["Masses"] = mass[gas_mask]
            Mtot += data["PartType0"]["Masses"].sum()
            data["PartType0"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-2 * np.random.random(Ngas),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["Pressures"] = unyt.unyt_array(
                10.0 ** (10.0 * np.random.random(Ngas)),
                dtype=np.float32,
                units="snap_mass/(a**5*snap_length*snap_time**2)",
                registry=reg,
            )
            # all entries in the element mass fractions have their own limits,
            # so we need to generate those separately if we want a realistic
            # sample
            semf = np.zeros((Ngas, 9))
            semf[:, 0] = 0.68 + 0.07 * np.random.random(Ngas)
            semf[:, 1] = 0.25 + 0.04 * np.random.random(Ngas)
            semf[:, 2] = 0.006 * np.random.random(Ngas)
            semf[:, 3] = 0.001 * np.random.random(Ngas)
            semf[:, 4] = 0.01 * np.random.random(Ngas)
            semf[:, 5] = 0.002 * np.random.random(Ngas)
            semf[:, 6] = 0.001 * np.random.random(Ngas)
            semf[:, 7] = 0.002 * np.random.random(Ngas)
            semf[:, 8] = 0.002 * np.random.random(Ngas)
            data["PartType0"]["SmoothedElementMassFractions"] = unyt.unyt_array(
                semf,
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType0"]["ElementMassFractionsDiffuse"] = data["PartType0"][
                "SmoothedElementMassFractions"
            ].copy()
            # same for the species fractions
            specfrac = np.zeros((Ngas, 10))
            specfrac[:, 0] = 3.94e-5 + 1.25 * np.random.random(Ngas)
            specfrac[:, 1] = np.random.random(Ngas)
            specfrac[:, 2] = 1.78e-5 + (1.0 - 1.78e-5) * np.random.random(Ngas)
            specfrac[:, 3] = 2.53e-8 * np.random.random(Ngas)
            specfrac[:, 4] = 0.134 * np.random.random(Ngas)
            specfrac[:, 5] = 2.05e-10 + 0.122 * np.random.random(Ngas)
            specfrac[:, 6] = 0.125 * np.random.random(Ngas)
            specfrac[:, 7] = 0.5 * np.random.random()
            specfrac[:, 8] = 9.67e-6 * np.random.random()
            specfrac[:, 9] = 2.14e-5 * np.random.random()
            data["PartType0"]["SpeciesFractions"] = unyt.unyt_array(
                specfrac,
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            # negative StarFormationRates are actually scale factors, again
            # limited by the highest value at z=0.1
            data["PartType0"]["StarFormationRates"] = unyt.unyt_array(
                300.0 * np.random.random(Ngas) - 1.0 / 1.1,
                dtype=np.float32,
                units="snap_mass/snap_time",
                registry=reg,
            )
            data["PartType0"]["Temperatures"] = unyt.unyt_array(
                10.0 ** (10.0 * np.random.random(Ngas)),
                dtype=np.float32,
                units="snap_temperature",
                registry=reg,
            )
            data["PartType0"]["Velocities"] = vs[gas_mask]
            data["PartType0"]["XrayLuminosities"] = unyt.unyt_array(
                10.0 ** (7.0 * np.random.random(Ngas)),
                dtype=np.float64,
                units="snap_length**2*snap_mass/snap_time**3",
                registry=reg,
            )
            data["PartType0"]["XrayPhotonLuminosities"] = unyt.unyt_array(
                10.0 ** (70.0 * np.random.random(Ngas)),
                dtype=np.float64,
                units="1/snap_time",
                registry=reg,
            )

        # DM properties
        dm_mask = types == "PartType1"
        Ndm = int(dm_mask.sum())
        if Ndm > 0:
            data["PartType1"] = {}
            data["PartType1"]["Coordinates"] = coords[dm_mask]
            data["PartType1"]["GroupNr_all"] = groupnr_all[dm_mask]
            data["PartType1"]["GroupNr_bound"] = groupnr_bound[dm_mask]
            data["PartType1"]["Masses"] = mass[dm_mask]
            Mtot += data["PartType1"]["Masses"].sum()
            data["PartType1"]["Velocities"] = vs[dm_mask]

        # star properties
        star_mask = types == "PartType4"
        Nstar = int(star_mask.sum())
        if Nstar > 0:
            data["PartType4"] = {}
            data["PartType4"]["BirthDensities"] = unyt.unyt_array(
                10.0 ** (4.4 + (11.5 - 4.4) * np.random.random(Nstar)),
                dtype=np.float32,
                units="snap_mass/snap_length**3",
                registry=reg,
            )
            data["PartType4"]["BirthScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Nstar),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Coordinates"] = coords[star_mask]
            data["PartType4"]["GroupNr_all"] = groupnr_all[star_mask]
            data["PartType4"]["GroupNr_bound"] = groupnr_bound[star_mask]
            # initial masses are always larger than the actual mass
            data["PartType4"]["InitialMasses"] = unyt.unyt_array(
                mass[star_mask].value * (1.0 + np.random.random(Nstar)),
                dtype=np.float32,
                units="snap_mass",
                registry=reg,
            )
            data["PartType4"]["Luminosities"] = unyt.unyt_array(
                1.0e10 * np.random.random((Nstar, 9)),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["Masses"] = mass[star_mask]
            Mtot += data["PartType4"]["Masses"].sum()
            data["PartType4"]["MetalMassFractions"] = unyt.unyt_array(
                1.0e-2 * np.random.random(Nstar),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            # all entries in the element mass fractions have their own limits,
            # so we need to generate those separately if we want a realistic
            # sample
            semf = np.zeros((Nstar, 9))
            semf[:, 0] = 0.68 + 0.07 * np.random.random(Nstar)
            semf[:, 1] = 0.25 + 0.04 * np.random.random(Nstar)
            semf[:, 2] = 0.006 * np.random.random(Nstar)
            semf[:, 3] = 0.001 * np.random.random(Nstar)
            semf[:, 4] = 0.01 * np.random.random(Nstar)
            semf[:, 5] = 0.002 * np.random.random(Nstar)
            semf[:, 6] = 0.001 * np.random.random(Nstar)
            semf[:, 7] = 0.002 * np.random.random(Nstar)
            semf[:, 8] = 0.002 * np.random.random(Nstar)
            data["PartType4"]["SmoothedElementMassFractions"] = unyt.unyt_array(
                semf,
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            data["PartType4"]["SNIaRates"] = unyt.unyt_array(
                5.36e7 * np.random.random(Nstar),
                dtype=np.float64,
                units="1/snap_time",
                registry=reg,
            )
            data["PartType4"]["Velocities"] = vs[star_mask]

        # BH properties
        bh_mask = types == "PartType5"
        Nbh = int(bh_mask.sum())
        if Nbh > 0:
            data["PartType5"] = {}
            data["PartType5"]["AccretionRates"] = unyt.unyt_array(
                0.1 * np.random.random(Nbh),
                dtype=np.float32,
                units="snap_mass/snap_time",
                registry=reg,
            )
            data["PartType5"]["Coordinates"] = coords[bh_mask]
            data["PartType5"]["DynamicalMasses"] = mass[bh_mask]
            Mtot += data["PartType5"]["DynamicalMasses"].sum()
            data["PartType5"]["GroupNr_all"] = groupnr_all[bh_mask]
            data["PartType5"]["GroupNr_bound"] = groupnr_bound[bh_mask]
            data["PartType5"]["LastAGNFeedbackScaleFactors"] = unyt.unyt_array(
                1.0 / 1.1 + 0.01 * np.random.random(Nbh),
                dtype=np.float32,
                units=unyt.dimensionless,
                registry=reg,
            )
            # randomly set some values to -1
            data["PartType5"]["LastAGNFeedbackScaleFactors"][
                np.random.random() > 0.9
            ] = -1
            # no need to do anything random for the IDs; we simply make sure
            # the IDs are non zero
            data["PartType5"]["ParticleIDs"] = unyt.unyt_array(
                np.arange(Nbh, dtype=np.uint64) + 1,
                dtype=np.int64,
                units=unyt.dimensionless,
                registry=reg,
            )
            # the sub-grid mass is always less than the dynamical mass
            data["PartType5"]["SubgridMasses"] = unyt.unyt_array(
                mass[bh_mask].value * (1.0e-5 + 0.99 * np.random.random(Nbh)),
                dtype=np.float32,
                units="snap_mass",
                registry=reg,
            )
            data["PartType5"]["Velocities"] = vs[bh_mask]

        # Neutrino properties
        nu_mask = types == "PartType6"
        Nnu = int(nu_mask.sum())
        if Nnu > 0:
            data["PartType6"] = {}
            data["PartType6"]["Coordinates"] = coords[nu_mask]
            # make sure neutrino masses are lower, since that is also the case
            # in the snapshots
            data["PartType6"]["Masses"] = 0.1 * mass[nu_mask]
            data["PartType6"]["Weights"] = unyt.unyt_array(
                1.17 * np.random.random(Nnu) - 0.46,
                dtype=np.float64,
                units="dimensionless",
                registry=reg,
            )
            Mtot += (data["PartType6"]["Masses"] * data["PartType6"]["Weights"]).sum()

        # set the required halo properties
        input_halo = {}
        input_halo["cofp"] = centre
        input_halo["index"] = groupnr_halo
        input_halo["Structuretype"] = structuretype

        nu_density = (
            self.dummy_cellgrid.cosmology["Omega_nu_0"]
            * self.dummy_cellgrid.critical_density
            * (
                self.dummy_cellgrid.cosmology["H0 [internal units]"]
                / self.dummy_cellgrid.cosmology["H [internal units]"]
            )
            ** 2
            / self.dummy_cellgrid.a**3
        )

        Mtot += nu_density * 4.0 * np.pi / 3.0 * rmax**3

        particle_numbers = {
            "PartType0": Ngas,
            "PartType1": Ndm,
            "PartType4": Nstar,
            "PartType5": Nbh,
        }

        return input_halo, data, rmax, Mtot, npart, particle_numbers
