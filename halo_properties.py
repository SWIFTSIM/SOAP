#!/bin/env python

import numpy as np
import unyt


class ReadRadiusTooSmallError(Exception):
    pass


class HaloProperty:
    def __init__(self, cellgrid):

        # Store parameters needed for halo property calculations
        self.unit_registry = (
            cellgrid.snap_unit_registry
        )  # unyt registry with snapshot units
        self.critical_density = (
            cellgrid.critical_density
        )  # critical density as unyt_quantity
        self.mean_density = cellgrid.mean_density  # mean density as unyt_quantity
        self.a = cellgrid.a  # expansion factor of this snapshot
        self.a_unit = (
            cellgrid.a_unit
        )  # Dimensionless unit used to define comoving quantities
        self.z = cellgrid.z  # redshift of this snapshot
        self.boxsize = cellgrid.boxsize  # boxsize as unyt_quantity
