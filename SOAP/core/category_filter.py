#!/bin/env python

"""
category_filter.py

Filter used to determine which halo properties should be computed.
This decision is based on the number of particles in the subhalo and
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
BoundSubhalo properties.
"""

from typing import Dict

from SOAP.property_table import PropertyTable

class CategoryFilter:
    """
    Filter used to determine whether properties need to be calculated for a
    certain halo or not.

    This decision is always based on the number of particles in the subhalo, 
    and requires the calculation of BoundSubhalo for each halo.
    """

    def __init__(self, filters: Dict, dmo: bool = False):
        """
        Construct the filter with the requested filter thresholds.

        Parameters:
         - filter: Dict
           Dictionary where each key the the name of the filter (the category),
           and each value is a dictionary describing the filter. An example filter
           description dictionary is:
               {
                   'limit': 100,
                   'properties': [
                       'BoundSubhalo/NumberOfGasParticles',
                       'BoundSubhalo/NumberOfStarParticles',
                    ],
                   'combine_properties': 'sum'
               }
           "limit" gives the threshold value above which the filter is satisfied
           "properties" lists the properties used to compare with the limit
           "combine_properties" is only required if there are multiple properties
           listed, and describes how to reduce the different property values before
           comparing the the threshold value
         - dmo: bool
           Whether or not SOAP is run in DMO mode, in which case only properties that
           are marked for DMO calculation are actually computed.
        """
        self.filters = filters
        self.dmo = dmo
        self.property_filters = {}

    def get_do_calculation(
        self, halo_result: Dict, precomputed_properties: Dict = {}
    ) -> Dict:
        """
        Get a mask for each category, depending on the properties of the subhalo.

        Parameters:
         - halo_result: Dict
           Halo result dictionary that contains the properties for the subhalo.
         - precomputed_properties: Dict
           Helper dictionary that can be passed if necessary properties
           have not yet been added to the halo result dictionary

        Returns a dictionary containing True/False for each filter category.
        """
        do_calculation = {"basic": True}
        if self.dmo:
            precomputed_properties["BoundSubhalo/NumberOfGasParticles"] = 0
            precomputed_properties["BoundSubhalo/NumberOfStarParticles"] = 0
            precomputed_properties["BoundSubhalo/NumberOfBlackHoleParticles"] = 0
            precomputed_properties["SO/200_crit/NumberOfGasParticles"] = 0
        for name, filter_info in self.filters.items():
            if (len(filter_info["properties"]) == 1) or (
                filter_info["combine_properties"] == "sum"
            ):
                v = 0
                for prop in filter_info["properties"]:
                    # Try to find the property in precomputed_properties
                    if prop in precomputed_properties:
                        v += precomputed_properties[prop]
                    # If property is also not present in halo_result throw an error
                    else:
                        v += halo_result[prop][0].value
                do_calculation[name] = v >= filter_info["limit"]
            else:
                msg = f"Invalid combine_properties function for filter {name}"
                raise NotImplementedError(msg)
        return do_calculation

    def get_compression_metadata(self, property_output_name: str) -> Dict:
        """
        Get the dictionary with compression metadata for a particular property.
        This function should only be called for properties that exist in the
        property table.

        Parameters:
         - property_output_name: str
           Name of a halo property as it appears in the output file.

        Returns a dictionary with (lossy) compression metadata. Currently, this
        dictionary contains the following keys:
         - Lossy Compression Algorithm: name of a lossy compression algorithm
           (we support the same algorithms as SWIFT), or None for no lossy
           compression.
         - Is Compressed: whether or not the lossy compression algorithm was
           actually applied to the data. Currently, this is always "False", since
           the lossy compression is done by a post-processing script.
        """
        # Output names can have / in them (for halo finder properties), so
        # we need to loop over that case
        split_name = property_output_name.split("/")
        for i in range(len(split_name)):
            output_name = '/'.join(split_name[-1-i:])
            for _, prop in PropertyTable.full_property_list.items():
                if prop.name == output_name:
                    return {"Lossy compression filter": prop.lossy_compression_filter, "Is Compressed": False}
        raise RuntimeError(f"Compression filter not found for {split_name}")

    def set_property_filters(self, property_filters: Dict) -> None:
        self.property_filters = property_filters

    def get_filter_metadata_for_property(self, property_output_name: str) -> Dict:
        """
        Get the dictionary with category filter metadata for a particular property.

        Parameters:
         - property_output_name: str
           Full path to halo property as it appears in the output file.

        Returns a dictionary with the same format as those output
        by get_filter_metadata
        """
        # Convert from output file group name to parameter file name
        # Return None for groups (e.g. InputHalos) which are never masked
        base_halo_type = {
            'BoundSubhalo': 'SubhaloProperties',
            'ProjectedAperture': 'ProjectedApertureProperties',
            'SO': 'SOProperties',
            'ExclusiveSphere': 'ApertureProperties',
            'InclusiveSphere': 'ApertureProperties',
        }.get(property_output_name.split('/')[0], None)

        if base_halo_type is not None:
            output_name = property_output_name.split("/")[-1]
            category = self.property_filters[base_halo_type][output_name]
        else:
            category = None

        return self.get_filter_metadata(category)

    def get_filter_metadata(self, category: str) -> Dict:
        """
        Return a dictionary with metadata for the input category filter.

        Parameters:
         - category: str
           Name of the filter category to get metadata for.

        Returns a dictionary with category filter metadata. Currently, this
        dictionary contains the following keys:
         - Masked: Whether or not some of the elements in the output have been
           masked. This is purely based on the category of the property: if this
           is "VR" or "basic", the flag is "False", otherwise it is "True" (even
           if the threshold is set to 0, so that there is effectively no masking).
         - Mask Datasets: Particle number datasets that were used for masking. This
           is a list of dataset names as they appear in the SOAP output (full path),
           e.g. [
             "BoundSubhalo/NumberOfGasParticles",
             "BoundSubhalo/NumberOfDarkMatterParticles"
           ]
           Only present if Masked==True.
         - Mask Threshold: Threshold value used for masking. A row in the output is
           masked (i.e. set to zero) if the sum of the elements in Mask Datasets is
           strictly below this value.
         - Mask Dataset Combination: Only present if Mask Datasets contains multiple
           values. Describes how the datasets were combined before comparing to the
           threshold value.

        Note that it is hence always possible to reproduce the mask for a dataset using
        this metadata.
        """
        if category is None or category == "basic":
            return {"Masked": False}
        elif category in self.filters:
            metadata = {
                "Masked": True,
                "Mask Datasets": self.filters[category]["properties"],
                "Mask Threshold": self.filters[category]["limit"],
            }
            if len(self.filters[category]["properties"]) > 1:
                metadata["Mask Dataset Combination"] = self.filters[category][
                    "combine_properties"
                ]
            return metadata
        else:
            # if we don't know the category, we cannot mask it
            # (e.g. "VR")
            return {"Masked": False}

    def print_filters(self):
        if self.dmo:
            print("Run in DMO mode")
        print("Category filters :")
        for name, filter_info in self.filters.items():
            print(f"  {name.ljust(10)}{filter_info['limit']}")
