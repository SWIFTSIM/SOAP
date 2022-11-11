#!/bin/env python

import numpy as np
import unyt


class ColdDenseGasFilter:
    """
    Filter used to determine whether gas particles should be considered to be
    "cold and dense".
    """

    def __init__(
        self,
        maximum_temperature=10.0**4.5 * unyt.K,
        minimum_hydrogen_number_density=0.1 / unyt.cm**3,
    ):
        self.maximum_temperature = maximum_temperature
        self.minimum_hydrogen_number_density = minimum_hydrogen_number_density

    def is_cold_and_dense(self, temperature, mass_density):
        hydrogen_number_density = mass_density / unyt.mh
        return (temperature < self.maximum_temperature.to(temperature.units)) & (
            hydrogen_number_density
            > self.minimum_hydrogen_number_density.to(hydrogen_number_density.units)
        )
