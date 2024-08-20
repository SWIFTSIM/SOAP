#!/bin/env python

"""
cold_dense_gas_filter.py

Filter used to mask out gas particles that contain cold, dense gas.

The mask is based on the gas density and temperature, i.e.
"PartType0/Densities" and "PartType0/Temperatures". The threshold
values are passed on as constructor arguments and should be read from
the parameter file.

Note that we assume a hydrogen-only gas when converting the mass
density to a hydrogen number density, i.e.
  n_H = rho / m_H
"""

import unyt
from numpy.typing import NDArray


class ColdDenseGasFilter:
    """
    Filter used to determine whether gas particles should be considered to be
    "cold and dense".
    """

    def __init__(
        self,
        maximum_temperature: unyt.unyt_quantity,
        minimum_hydrogen_number_density: unyt.unyt_quantity,
        initialised: bool,
    ):
        """
        Construct the filter.

        Parameters:
         - maximum_temperature: unyt.unyt_quantity
           Maximum temperature below which gas is considered to be cold.
         - minimum_hydrogen_number_density: unyt.unyt_quantity
           Minimum hydrogen number density above which gas is considered to
           be dense.
         - initialised: bool
           If the parameters required for the filter where found in the parameter
           file. If this is false and the filter is called, it will throw an error.
        """
        self.maximum_temperature = maximum_temperature
        self.minimum_hydrogen_number_density = minimum_hydrogen_number_density
        self.initialised = initialised
        self.metadata = {
            'maximum_temperature_in_K': maximum_temperature.to('K').value,
            'minimum_hydrogen_number_density_in_cm-3': minimum_hydrogen_number_density.to('1/cm**3').value,
        }

    def is_cold_and_dense(
        self, temperature: unyt.unyt_array, mass_density: unyt.unyt_array
    ) -> NDArray[bool]:
        """
        Compute the mask for cold, dense gas particles.

        Parameters:
         - temperature: unyt.unyt_array
           Temperatures of the gas particles, i.e. "PartType0/Temperatures".
         - mass_density: unyt.unyt_array
           Mass densities of the gas particles, i.e. "PartType0/Densities".

        Returns a mask array that can be used to index other particle arrays.
        """
        if not self.initialised:
            raise RuntimeError('ColdDenseGasFilter was not initialised')
        hydrogen_number_density = mass_density / unyt.mh
        return (temperature < self.maximum_temperature.to(temperature.units)) & (
            hydrogen_number_density
            > self.minimum_hydrogen_number_density.to(hydrogen_number_density.units)
        )

    def get_metadata(self):
        return self.metadata
