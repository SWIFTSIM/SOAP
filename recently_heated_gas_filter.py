#!/bin/env python

"""
recently_heated_gas_filter.py

Filter to mask out gas particles that were recently heated by AGN feedback.

"Recently heated" is defined in terms of two variables: the last time the
gas particle was heated by AGN feedback needs to be close enough in time
to the current time, and the temperature of the particle needs to still be
high enough. For consistency, we parametrise these two conditions in the
same way as done in SWIFT.

Since we store last feedback scale factors rather than times, calculating
the time interval between the current time and the last feedback time
requires knowledge of the cosmology.
"""

import numpy as np
import unyt

from astropy.cosmology import w0waCDM, z_at_value
import astropy.constants as const
import astropy.units as astropy_units

from swift_cells import SWIFTCellGrid
from numpy.typing import NDArray


class RecentlyHeatedGasFilter:
    """
    Filter used to determine whether gas particles should be considered to be
    "recently heated".

    This corresponds to the lightcone map filter used in SWIFT itself, which
    filters out gas particles for which LastAGNFeedbackScaleFactors is less
    than some time interval, and within some temperature bracket.

    Since the conversion from a time difference to a scale factor is not
    trivial, we compute the corresponding scale factor limit only once using
    the correct astropy.cosmology.
    """

    # lower limit on the scale factor, below which events are no
    # longer considered to be recent
    a_limit: unyt.unyt_quantity
    # temperature limits. Within this interval, a gas particle is
    # considered to be "heated by AGN".
    Tmin: unyt.unyt_quantity
    Tmax: unyt.unyt_quantity

    def __init__(
        self,
        cellgrid: SWIFTCellGrid,
        delta_time: unyt.unyt_quantity,
        use_AGN_delta_T: bool,
        initialised: bool,
        delta_logT_min: float = -1.0,
        delta_logT_max: float = 0.3,
    ):
        """
        Constructor.

        Precomputes the cutoff scale factor below which feedback
        events are no longer considered to be "recent".

        Parameters:
         - cellgrid: SWIFTCellGrid
           Container object containing global information about the snapshot,
           like the cosmology.
         - delta_time: unyt.unyt_quantity
           Time interval considered to be "recent".
         - use_AGN_delta_T: bool
           Whether to filter particles based on their temperature. If used then
           the value of AGN_delta_T is read from the snapshot.
         - delta_logT_min: float
           Lower limit on the temperature (dex below AGN_delta_T) below which
           gas is no longer considered to be "heated".
         - delta_logT_max: float
           Upper limit on the temperature (dex above AGN_delta_T) above which
           gas is too hot to be considered "heated by AGN".
         - initialised: bool
           If the parameters required for the filter where found in the parameter
           file. If this is false and the filter is called, it will throw an error.
        """
        self.initialised = initialised

        H0 = unyt.unyt_quantity(
            cellgrid.cosmology["H0 [internal units]"],
            units="1/snap_time",
            registry=cellgrid.snap_unit_registry,
        ).to("1/s")

        Omega_b = cellgrid.cosmology["Omega_b"]
        Omega_lambda = cellgrid.cosmology["Omega_lambda"]
        Omega_g = cellgrid.cosmology["Omega_g"]
        Omega_m = cellgrid.cosmology["Omega_m"]
        w_0 = cellgrid.cosmology["w_0"]
        w_a = cellgrid.cosmology["w_a"]
        z_now = cellgrid.cosmology["Redshift"]

        # expressions taken directly from astropy, since they do no longer
        # allow access to these attributes (since version 5.1+)
        critdens_const = (3.0 / (8.0 * np.pi * const.G)).cgs.value
        a_B_c2 = (4.0 * const.sigma_sb / const.c ** 3).cgs.value

        # SWIFT provides Omega_g, but we need a consistent Tcmb0 for astropy.
        # This is an exact inversion of the procedure performed in astropy.
        critical_density_0 = astropy_units.Quantity(
            critdens_const * H0.to("1/s").value ** 2,
            astropy_units.g / astropy_units.cm ** 3,
        )

        Tcmb0 = (Omega_g * critical_density_0.value / a_B_c2) ** (1.0 / 4.0)

        cosmology = w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
        )

        lookback_time_now = cosmology.lookback_time(z_now)
        lookback_time_limit = lookback_time_now + delta_time.to_astropy()
        z_limit = z_at_value(cosmology.lookback_time, lookback_time_limit)

        # for some reason, the return type of z_at_value has changed between
        # astropy versions. We make sure it is not some astropy quantity
        # before using it.
        if hasattr(z_limit, "value"):
            z_limit = z_limit.value

        self.a_limit = 1.0 / (1.0 + z_limit) * unyt.dimensionless
        self.metadata = {
            "delta_time_in_Myr": delta_time.to("Myr").value,
            "a_limit": self.a_limit.value,
            "use_AGN_delta_T": use_AGN_delta_T,
        }

        self.use_AGN_delta_T = use_AGN_delta_T
        if use_AGN_delta_T:
            AGN_delta_T = cellgrid.AGN_delta_T
            self.Tmin = AGN_delta_T * 10.0 ** delta_logT_min
            self.Tmax = AGN_delta_T * 10.0 ** delta_logT_max
            self.metadata["AGN_delta_T_in_K"] = (AGN_delta_T.to("K").value,)
            self.metadata["delta_logT_min"] = (delta_logT_min,)
            self.metadata["delta_logT_max"] = (delta_logT_max,)
            self.metadata["Tmin_in_K"] = (self.Tmin.to("K").value,)
            self.metadata["Tmax_in_K"] = (self.Tmax.to("K").value,)

    def is_recently_heated(
        self, lastAGNfeedback: unyt.unyt_array, temperature: unyt.unyt_array
    ) -> NDArray[bool]:
        """
        Get a mask to mask out gas particles that were recently heated
        by AGN feedback.

        Parameters:
         - lastAGNfeedback: unyt.unyt_array
           Last AGN feedback scale factors, as read from the snapshot.
         - temperature: unyt.unyt_array
           Temperatures of the gas particles, as read from the snapshot.

        Returns a mask that can be used to index particle arrays.
        """
        if not self.initialised:
            raise RuntimeError("RecentlyHeatedGasFilter was not initialised")
        mask = lastAGNfeedback >= self.a_limit
        if self.use_AGN_delta_T:
            mask = mask & (temperature >= self.Tmin) & (temperature <= self.Tmax)
        return mask

    def get_metadata(self):
        return self.metadata
