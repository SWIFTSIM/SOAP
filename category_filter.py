#!/bin/env python

from property_table import PropertyTable

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
        Ngeneral=100,
        Ngas=50,
        Ndm=100,
        Nstar=50,
        Nbaryon=100,
        dmo=False,
    ):
        self.Ngeneral = Ngeneral
        self.Ngas = Ngas
        self.Ndm = Ndm
        self.Nstar = Nstar
        self.Nbaryon = Nbaryon
        self.dmo = dmo

    def get_filters_direct(self, Ngas, Ndm, Nstar, Nbh):
        return {
            "basic": True,
            "general": Ngas + Ndm + Nstar + Nbh >= self.Ngeneral,
            "gas": Ngas >= self.Ngas,
            "dm": Ndm >= self.Ndm,
            "star": Nstar >= self.Nstar,
            "baryon": Ngas + Nstar >= self.Nbaryon,
            "DMO": self.dmo,
        }

    def get_filters(self, halo_result):
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
