import h5py
import numpy as np
from numba import jit
from unyt import g, cm, mp, erg, s


class XrayCalculator:
    def __init__(self, redshift, table_path, bands, observing_types, recalculate):
        self.z_now = redshift
        self.recalculate = recalculate
        if not recalculate:
            return

        if bands == None:
            print(
                'Please specify the band you would like to generate emissivities for\n \
                Using the "band = " keyword\n\n \
                Available options are:\n \
                "erosita-low" (0.2-2.3 keV)\n \
                "erosita-high" (2.3-8.0 keV)\n \
                "ROSAT" (0.5-2.0 keV)'
            )
            raise KeyError

        if observing_types == None:
            print(
                'Please specify whether you would like to generate photon or energie emissivities\n \
                Using the "observing_type = " keyword\n\n \
                Available options are:\n \
                "energies_intrinsic"\n \
                "photons_intrinsic"'
            )
            raise KeyError

        if (bands != None) & (observing_types != None):
            assert len(bands) == len(observing_types)

        self.tables = self.load_all_tables(redshift, table_path, bands, observing_types)
        # Always only read the nearest 2 redshift bins of the table
        self.idx_z = np.array([0, 1]).astype(int)

    def load_all_tables(self, redshift, table_path, bands, observing_types):
        """
        Load the x-ray tables for the specified bands and observing-types
        Only read the 2 redshifts closest to the redshift of the snapshot being processed by SOAP
        """
        try:
            table = h5py.File(table_path, "r")
        except ValueError as e:
            raise Exception("You must pass an x-ray table path") from e

        self.redshift_bins = table["/Bins/Redshift_bins"][()].astype(np.float32)
        idx_z, self.dx_z = self.get_index_1d(self.redshift_bins, np.array([redshift]))
        self.dx_z = self.dx_z[0]

        self.He_bins = table["/Bins/He_bins"][()].astype(np.float32)
        self.missing_elements = table["/Bins/Missing_element"][()]
        self.element_masses = table["Bins/Element_masses"][()].astype(np.float32)

        self.density_bins = table["/Bins/Density_bins/"][()].astype(np.float32)
        self.temperature_bins = table["/Bins/Temperature_bins/"][()].astype(np.float32)
        self.redshift_bins = table["/Bins/Redshift_bins"][()].astype(np.float32)

        self.log10_solar_metallicity = table["/Bins/Solar_metallicities/"][()].astype(
            np.float32
        )
        self.solar_metallicity = np.power(10, self.log10_solar_metallicity)

        tables = {}
        for band in bands:
            tables[band] = {}
            for observing_type in observing_types:
                temp = table[band][observing_type][
                    np.array([idx_z[0], idx_z[0] + 1]).astype(int), :, :, :, :
                ].astype(np.float32)
                tables[band][observing_type] = temp

        return tables

    @staticmethod
    @jit(nopython=True)
    def get_index_1d(bins, subdata):
        """
        Find the closest bin index below the specified value, and the relative offset compared to that bin.
        """
        eps = 1e-4
        delta = (len(bins) - 1) / (bins[-1] - bins[0])

        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype=np.float32)
        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                # We are below the first element
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                # Normal case
                idx[i] = int((x - bins[0]) * delta)
                dx[i] = (x - bins[int(idx[i])]) * delta
            else:
                # We are after the last element
                idx[i] = len(bins) - 2
                dx[i] = 1

        return idx, dx

    @staticmethod
    @jit(nopython=True)
    def get_index_1d_irregular(bins, subdata):
        """
        Find the closest bin index below the specified value, and the relative offset compared to that bin.
        Unlike get_index_1d, this allows for irregular bin spacing
        """
        eps = 1e-6
        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype=np.float32)

        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                min_idx = -1

                """
                Do this the hard way: Search the table
                for the smallest index i in bins[i] such
                that table[i] < x
                """
                for j in range(len(bins)):
                    if x - bins[j] <= 0:
                        # Found the first entry that is larger than x, go back by 1
                        min_idx = j - 1
                        break

                idx[i] = min_idx
                dx[i] = (x - bins[min_idx]) / (bins[min_idx + 1] - bins[min_idx])
            else:
                idx[i] = len(bins) - 2
                dx[i] = 1

        return idx, dx

    @staticmethod
    # @jit(nopython = True)
    def get_table_interp(
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        X_Ray,
        abundance_to_solar,
    ):
        """
        4D interpolate the x-ray table for each traced metal
        Scale the metals with their respective relative solar abundances
        Add the metals to the background case without metals
        """

        f_n_T = np.zeros((t_nH.shape[0], X_Ray.shape[1]), dtype=np.float32)

        f_n_T += (t_nH * t_He * t_T * t_z)[:, None] * X_Ray[0, idx_he, :, idx_T, idx_n]
        f_n_T += (t_nH * t_He * d_T * t_z)[:, None] * X_Ray[
            0, idx_he, :, idx_T + 1, idx_n
        ]
        f_n_T += (t_nH * d_He * t_T * t_z)[:, None] * X_Ray[
            0, idx_he + 1, :, idx_T, idx_n
        ]
        f_n_T += (d_nH * t_He * t_T * t_z)[:, None] * X_Ray[
            0, idx_he, :, idx_T, idx_n + 1
        ]

        f_n_T += (t_nH * d_He * d_T * t_z)[:, None] * X_Ray[
            0, idx_he + 1, :, idx_T + 1, idx_n
        ]
        f_n_T += (d_nH * t_He * d_T * t_z)[:, None] * X_Ray[
            0, idx_he, :, idx_T + 1, idx_n + 1
        ]
        f_n_T += (d_nH * d_He * t_T * t_z)[:, None] * X_Ray[
            0, idx_he + 1, :, idx_T, idx_n + 1
        ]
        f_n_T += (d_nH * d_He * d_T * t_z)[:, None] * X_Ray[
            0, idx_he + 1, :, idx_T + 1, idx_n + 1
        ]

        f_n_T += (t_nH * t_He * t_T * d_z)[:, None] * X_Ray[1, idx_he, :, idx_T, idx_n]
        f_n_T += (t_nH * t_He * d_T * d_z)[:, None] * X_Ray[
            1, idx_he, :, idx_T + 1, idx_n
        ]
        f_n_T += (t_nH * d_He * t_T * d_z)[:, None] * X_Ray[
            1, idx_he + 1, :, idx_T, idx_n
        ]
        f_n_T += (d_nH * t_He * t_T * d_z)[:, None] * X_Ray[
            1, idx_he, :, idx_T, idx_n + 1
        ]

        f_n_T += (t_nH * d_He * d_T * d_z)[:, None] * X_Ray[
            1, idx_he + 1, :, idx_T + 1, idx_n
        ]
        f_n_T += (d_nH * t_He * d_T * d_z)[:, None] * X_Ray[
            1, idx_he, :, idx_T + 1, idx_n + 1
        ]
        f_n_T += (d_nH * d_He * t_T * d_z)[:, None] * X_Ray[
            1, idx_he + 1, :, idx_T, idx_n + 1
        ]
        f_n_T += (d_nH * d_He * d_T * d_z)[:, None] * X_Ray[
            1, idx_he + 1, :, idx_T + 1, idx_n + 1
        ]

        # Add each metal contribution individually
        f_n_T_Z_temp = np.power(10, f_n_T[:, -1], dtype=np.float64)
        for j in range(f_n_T.shape[1] - 1):
            f_n_T_Z_temp += np.power(10, f_n_T[:, j]) * abundance_to_solar[:, j]

        f_n_T_Z = np.log10(f_n_T_Z_temp)

        return f_n_T_Z

    def find_indices(
        self, densities, temperatures, element_mass_fractions, masses, fill_value=0
    ):
        """
        Check interpolation table bounds
        Compute all interpolation bin indices, and the offsets from those bins
        Compute all the indices for the flattened x-ray table
        """
        redshift = self.z_now
        scale_factor = 1 / (1 + redshift)
        data_n = np.log10(
            element_mass_fractions[:, 0] * densities.to(g * cm ** -3) / mp
        )
        data_T = np.log10(temperatures)
        volumes = (masses.astype(np.float64) / densities.astype(np.float64)).to(cm ** 3)

        # Create density mask, round to avoid numerical errors
        density_mask = (data_n >= np.round(self.density_bins.min(), 1)) & (
            data_n <= np.round(self.density_bins.max(), 1)
        )
        # Create temperature mask, round to avoid numerical errors
        temperature_mask = (data_T >= np.round(self.temperature_bins.min(), 1)) & (
            data_T <= np.round(self.temperature_bins.max(), 1)
        )

        # Combine masks
        joint_mask = density_mask & temperature_mask

        # Check if within density and temperature bounds
        density_bounds = np.sum(density_mask) == density_mask.shape[0]
        temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
        if ~(density_bounds & temperature_bounds):
            # If no fill_value is set, return an error with some explanation
            if fill_value == None:
                raise ValueError(
                    "Temperature or density are outside of the interpolation range and no fill_value is supplied\n \
                                Temperature ranges between log(T) = 5 and log(T) = 9.5\n \
                                Density ranges between log(nH) = -8 and log(nH) = 6\n \
                                Set the kwarg 'fill_value = some value' to set all particles outside of the interpolation range to 'some value'\n \
                                Or limit your particle data set to be within the interpolation range"
                )
            else:
                pass

        # get individual mass fraction
        mass_fraction = element_mass_fractions[joint_mask]

        # Find density offsets
        idx_n, dx_n = self.get_index_1d(self.density_bins, data_n[joint_mask])
        idx_n = idx_n.astype(int)

        # Find temperature offsets
        idx_T, dx_T = self.get_index_1d(self.temperature_bins, data_T[joint_mask])
        idx_T = idx_T.astype(int)

        # Calculate the abundance wrt to solar
        abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis=1)) * (
            self.element_masses[0] / np.array(self.element_masses)
        )

        # Calculate abundance offsets using solar abundances
        abundance_to_solar = abundances / self.solar_metallicity

        # Add columns for Calcium and Sulphur and move Iron to the end
        abundance_to_solar = np.c_[
            abundance_to_solar[:, :-1],
            abundance_to_solar[:, -2],
            abundance_to_solar[:, -2],
            abundance_to_solar[:, -1],
        ]

        # Find helium offsets
        idx_he, dx_he = self.get_index_1d_irregular(
            self.He_bins, np.log10(abundances[:, 1])
        )
        idx_he = idx_he.astype(int)

        t_z = 1 - self.dx_z
        d_z = self.dx_z

        # Compute temperature offset relative to bin
        t_T = 1 - dx_T
        d_T = dx_T

        # Compute density offset relative to bin
        t_nH = 1 - dx_n
        d_nH = dx_n

        # Compute Helium offset relative to bin
        t_He = 1 - dx_he
        d_He = dx_he

        return (
            idx_he,
            idx_T,
            idx_n,
            t_z,
            d_z,
            t_T,
            d_T,
            t_nH,
            d_nH,
            t_He,
            d_He,
            abundance_to_solar,
            joint_mask,
            volumes,
            data_n,
        )

    def interpolate_X_Ray(
        self,
        idx_he,
        idx_T,
        idx_n,
        t_z,
        d_z,
        t_T,
        d_T,
        t_nH,
        d_nH,
        t_He,
        d_He,
        abundance_to_solar,
        joint_mask,
        volumes,
        data_n,
        bands=None,
        observing_types=None,
        fill_value=None,
    ):
        """
        Main function
        Calculate the particle emissivities through interpolation
        Convert to luminosity using the particle volume
        """
        # Initialise the emissivity array which will be returned
        emissivities = np.zeros((joint_mask.shape[0], len(bands)), dtype=float)
        luminosities = np.zeros_like(emissivities)
        emissivities[~joint_mask] = fill_value

        # Interpolate the table for each specified band
        for i_interp, band, observing_type in zip(
            range(len(bands)), bands, observing_types
        ):
            emissivities[joint_mask, i_interp] = self.get_table_interp(
                idx_he,
                idx_T,
                idx_n,
                t_z,
                d_z,
                t_T,
                d_T,
                t_nH,
                d_nH,
                t_He,
                d_He,
                self.tables[band][observing_type],
                abundance_to_solar[:, 2:],
            )

            # Convert from erg cm^3 s^-1 to erg cm^-3 s^-1
            # To do so we multiply by nH^2, this is the actual nH not the nearest bin
            # It allows to extrapolate in density space without too much worry
            # log(emissivity * nH^2) = log(emissivity) + 2*log(nH)
            emissivities[joint_mask, i_interp] += 2 * data_n[joint_mask]

            luminosities[joint_mask, i_interp] = (
                np.power(10, emissivities[joint_mask, i_interp]) * volumes[joint_mask]
            )

        if "energies" in observing_types[0]:
            return luminosities * erg * s ** -1
        elif "photon" in observing_types[0]:
            return luminosities * s ** -1
