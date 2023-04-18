#!/bin/env python

import numpy as np
import unyt

from astropy.cosmology import w0waCDM, z_at_value
import astropy.constants as const
import astropy.units as astropy_units


class StellarAgeCalculator:
    """
    Auxiliary object used to calculate stellar ages.

    Since the snapshots contain the stellar birth scale factor, obtaining the
    stellar age requires knowledge about the simulation cosmology.

    Upon construction, we extract the relevant cosmology and the current scale
    factor from the cellgrid to create a convenient function that can directly
    calculate the stellar age from the stellar birth scale factor.
    """

    def __init__(self, cellgrid):
        H0 = unyt.unyt_quantity(
            cellgrid.cosmology["H0 [internal units]"],
            units="1/snap_time",
            registry=cellgrid.snap_unit_registry,
        ).to("1/s")

        Omega_b = cellgrid.cosmology["Omega_b"]
        Omega_lambda = cellgrid.cosmology["Omega_lambda"]
        Omega_r = cellgrid.cosmology["Omega_r"]
        Omega_m = cellgrid.cosmology["Omega_m"]
        w_0 = cellgrid.cosmology["w_0"]
        w_a = cellgrid.cosmology["w_a"]
        z_now = cellgrid.cosmology["Redshift"]

        # expressions taken directly from astropy, since they do no longer
        # allow access to these attributes (since version 5.1+)
        critdens_const = (3.0 / (8.0 * np.pi * const.G)).cgs.value
        a_B_c2 = (4.0 * const.sigma_sb / const.c ** 3).cgs.value

        # SWIFT provides Omega_r, but we need a consistent Tcmb0 for astropy.
        # This is an exact inversion of the procedure performed in astropy.
        critical_density_0 = astropy_units.Quantity(
            critdens_const * H0.to("1/s").value ** 2,
            astropy_units.g / astropy_units.cm ** 3,
        )

        Tcmb0 = (Omega_r * critical_density_0.value / a_B_c2) ** (1.0 / 4.0)

        self.cosmology = w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
        )

        self.t_now = unyt.unyt_quantity.from_astropy(
            self.cosmology.lookback_time(z_now)
        ).to("Myr")

    def stellar_age(self, birth_a):
        birth_z = 1.0 / birth_a - 1.0
        t_birth = unyt.unyt_array.from_astropy(
            self.cosmology.lookback_time(birth_z.value)
        ).to("Myr")
        # remember: we use lookback time, which runs backwards!
        return t_birth - self.t_now
