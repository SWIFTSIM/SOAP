#!/bin/env python

"""
category_filter.py

Filter used to determine which halo properties should be computed.
This decision is based on the number of particles in the FOF subhalo and
the category a particular halo property belongs to.

There are 6 categories:
 - basic: Always computed.
 - general: Only computed if the total number of particles exceeds a threshold.
 - gas: Only computed if the number of gas particles exceeds a threshold.
 - dm: Only computed if the number of dark matter particles exceeds a threshold.
 - star: Only computed if the number of star particles exceeds a threshold.
 - baryon: Only computed if the number of baryon (gas + star) particles exceeds a threshold.

Additionally, this object also marks properties that should not be computed for DMO runs.

The filter thresholds for the 5 categories that use a threshold are read from the parameter
file. The corresponding particle numbers are hardcoded to be read from the
FOFSubhaloProperties properties.
"""

from property_table import PropertyTable
from typing import Dict

# hardcoded names of the particle number data to use:
# FOFSubhaloProperties/<correct name for Ngas/Ndm/Nstar/Nbh>
gas_filter_name = f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ngas'][0]}"
dm_filter_name = f"FOFSubhaloProperties/{PropertyTable.full_property_list['Ndm'][0]}"
star_filter_name = (
    f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nstar'][0]}"
)
bh_filter_name = f"FOFSubhaloProperties/{PropertyTable.full_property_list['Nbh'][0]}"


class CategoryFilter:
    """
    Filter used to determine whether properties need to be calculated for a
    certain halo or not.

    This decision is always based on the number of particles in the 6D FOF
    group, and requires the calculation of FOFSubhaloProperties for each halo.
    """

    def __init__(
        self,
        filter_values: Dict,
        dmo: bool = False,
    ):
        """
        Construct the filter with the requested filter thresholds.

        Parameters:
         - filter_values: Dict
           Dictionary containing the filter thresholds for the 5 categories that
           require such a threshold. These should be read from the parameter file.
         - dmo: bool
           Whether or not SOAP is run in DMO mode, in which case only properties that
           are marked for DMO calculation are actually computed.
        """
        self.Ngeneral = filter_values["general"]
        self.Ngas = filter_values["gas"]
        self.Ndm = filter_values["dm"]
        self.Nstar = filter_values["star"]
        self.Nbaryon = filter_values["baryon"]
        self.dmo = dmo

    def get_filters_direct(self, Ngas: int, Ndm: int, Nstar: int, Nbh: int) -> Dict:
        """
        Get the mask for each category, directly based on the particle numbers of
        gas, dm, stars and bh.

        Parameters:
         - Nx: int
           Particle numbers for the FOF subhalo.
        Returns a dictionary containing True/False for each property category.
        """
        return {
            "basic": True,
            "general": Ngas + Ndm + Nstar + Nbh >= self.Ngeneral,
            "gas": Ngas >= self.Ngas,
            "dm": Ndm >= self.Ndm,
            "star": Nstar >= self.Nstar,
            "baryon": Ngas + Nstar >= self.Nbaryon,
            "DMO": self.dmo,
        }

    def get_filters(self, halo_result: Dict) -> Dict:
        """
        Get the mask for each category, based on the particle numbers of
        the FOF subhalo group.

        Parameters:
         - halo_result: Dict
           Halo result dictionary that contains the particle numbers for the FOF subhalo.
        Returns a dictionary containing True/False for each property category.
        """
        Ndm = halo_result[dm_filter_name][0].value
        if self.dmo:
            Ngas = 0
            Nstar = 0
            Nbh = 0
        else:
            Ngas = halo_result[gas_filter_name][0].value
            Nstar = halo_result[star_filter_name][0].value
            Nbh = halo_result[bh_filter_name][0].value
        return self.get_filters_direct(Ngas, Ndm, Nstar, Nbh)

    def get_compression_metadata(self, property_output_name):
        base_output_name = property_output_name.split("/")[-1]
        compression = None
        for _, prop in PropertyTable.full_property_list.items():
            if prop[0] == base_output_name:
                compression = prop[6]
        if compression is None:
            return {"Lossy Compression Algorithm": "None", "Is Compressed": False}
        else:
            return {"Lossy Compression Algorithm": compression, "Is Compressed": False}

    def get_filter_metadata(self, property_output_name):
        base_output_name = property_output_name.split("/")[-1]
        category = None
        for _, prop in PropertyTable.full_property_list.items():
            if prop[0] == base_output_name:
                category = prop[5]
        # category=None corresponds to quantities outside the table
        # (e.g. "density_in_search_radius")
        if category is None or category == "basic":
            return {"Masked": False}
        elif category == "general":
            return {
                "Masked": True,
                "Mask Datasets": [
                    gas_filter_name,
                    dm_filter_name,
                    star_filter_name,
                    bh_filter_name,
                ],
                "Mask Threshold": self.Ngeneral,
            }
        elif category == "gas":
            return {
                "Masked": True,
                "Mask Datasets": [gas_filter_name],
                "Mask Threshold": self.Ngas,
            }
        elif category == "dm":
            return {
                "Masked": True,
                "Mask Datasets": [dm_filter_name],
                "Mask Threshold": self.Ndm,
            }
        elif category == "star":
            return {
                "Masked": True,
                "Mask Datasets": [star_filter_name],
                "Mask Threshold": self.Nstar,
            }
        elif category == "baryon":
            return {
                "Masked": True,
                "Mask Datasets": [gas_filter_name, star_filter_name],
                "Mask Threshold": self.Nbaryon,
            }
        else:
            # if we don't know the category, we cannot mask it
            # (e.g. "VR")
            return {"Masked": False}
