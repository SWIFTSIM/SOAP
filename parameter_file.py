#!/bin/env python

"""
parameter_file.py

(Hacky) support for parameter files.

The parameter file is a simple YAML dictionary
with the following structure:

ApertureProperties:
  properties:
    TotalMass: true
    ...
  variations:
    exclusive_100_kpc:
      inclusive: false
      radius_in_kpc: 100.0
    inclusive_100_kpc:
      inclusive: true
      radius_in_kpc: 100.0
    ...
ProjectedApertures:
  properties:
    ...
  variations:
    100_kpc:
      radius_in_kpc: 100.0
    ...
SOProperties:
  properties:
    ...
  variations:
    200_crit:
      type: crit
      value: 200.0
    200_mean:
      type: mean
      value: 200.0
    BN98:
      type: BN98
      value: 0.0
    5xR200_crit:
      type: crit
      value: 200.0
      radius_multiple: 5.0
    ...
SubhaloProperties:
  properties:
    ...
  variations:
    Bound:
      bound_only: true
    FOF:
      bound_only: false
aliases:
  PartType0/ElementMassFractions: PartType0/SmoothedElementMassFractions
  ...
defined_constants:
  Fe_H_sun: 2.82e-5
  ...
filters:
  baryon: 100
  dm: 100
  gas: 50
  star: 50
  general: 100

where non relevant repetitive parameters have been omitted. Note that all
parameters are optional; if a block/parameter is missing, default values
are used.

For the proper working of SOAP, it is important that
SubhaloProperties:variations:FOF:bound_only:false is present, since this
halo type is used for the category filtering. Note that the variation
names are not used by SOAP; their name can be used to create meaningful
structure in the parameter file. The list of properties for a particular
halo type is fixed for all variations of that halo type.

The parameter file object keeps track of the parameters that are requested,
and can output this information in the form of a ".used_parameters" file,
similar to the file produced by SWIFT.
"""

from typing import Dict, Union, List, Tuple
import yaml

import property_table

class ParameterFile:
    """
    Internal representation of the parameter file.

    Acts as a meaningful wrapper around the bare parameter dictionary.
    """

    # parameter dictionary
    parameters: Dict

    def __init__(
        self,
        file_name: Union[None, str] = None,
        parameter_dictionary: Union[None, Dict] = None,
    ):
        """
        Constructor.

        Parameters:
         - file_name: str or None
           Name of the parameter file. If None, a parameter dictionary should be
           provided instead.
         - parameter_dictionary: Dict or None
           Dictionary of parameters. Only used if file_name is None.
           Useful for creating dummy parameter file objects in unit testing.
        """
        if file_name is not None:
            with open(file_name, "r") as handle:
                self.parameters = yaml.safe_load(handle)
                if self.calculate_missing_properties():
                    self.unregistered_parameters = set()
                else:
                    self.unregistered_parameters = None
        else:
            self.unregistered_parameters = None
            if parameter_dictionary is not None:
                self.parameters = parameter_dictionary
            else:
                self.parameters = {}

    def get_parameters(self) -> Dict:
        """
        Get a copy of the parameter dictionary.
        """
        return dict(self.parameters)

    def write_parameters(self, file_name: str = "SOAP.used_parameters.yml"):
        """
        Write the (used) parameters to a file.

        Parameters:
         - file_name: str
           Name of the file to write.
        """
        with open(file_name, "w") as handle:
            yaml.safe_dump(self.parameters, handle)

    def get_property_mask(self, halo_type: str, full_list: List[str]) -> Dict:
        """
        Get a dictionary with True/False values indicating which properties
        should actually be computed for the given halo type. The dictionary
        keys are based on the contents of the given list of properties. If
        a property in the list is missing from the parameter file, it is
        assumed that this property needs to be calculated.

        Note that we currently do not check for properties in the parameter
        file that are not in the list.

        Parameters:
         - halo_type: str
           Halo type identifier in the parameter file, can be one of
           ApertureProperties, ProjectedApertureProperties, SOProperties
           or SubhaloProperties.
         - full_list: List[str]
           List of all the properties that can be calculated by this
           particular halo type (as defined in the corresponding HaloProperty
           specialisation).

        Returns a dictionary with True or False for each property in full_list.
        """
        if not halo_type in self.parameters:
            self.parameters[halo_type] = {}
        if not "properties" in self.parameters[halo_type]:
            self.parameters[halo_type]["properties"] = {}
            for property in full_list:
                self.parameters[halo_type]["properties"][property] = True
        mask = {}
        for property in full_list:
            if property in self.parameters[halo_type]["properties"]:
                mask[property] = self.parameters[halo_type]["properties"][property]
            elif self.calculate_missing_properties():
                mask[property] = True
                self.parameters[halo_type]["properties"][property] = True
                if self.unregistered_parameters is not None:
                    self.unregistered_parameters.add((halo_type, property))
            else:
                mask[property] = False
        return mask

    def print_unregistered_properties(self) -> None:
        """
        Prints a list of any properties that will be calculated that are not present in the parameter file
        """
        if not self.calculate_missing_properties():
            print("Properties not present in the parameter file will not be calculated")
        elif (self.unregistered_parameters is not None) and (
            len(self.unregistered_parameters) != 0
        ):
            print(
                "The following properties were not found in the parameter file, but will be calculated:"
            )
            for halo_type, property in self.unregistered_parameters:
                print(f"  {halo_type.ljust(30)}{property}")

    def print_invalid_properties(self) -> None:
        """
        Print a list of any properties in the parameter file that are not present in
        the property table. This doesn't check if the property is defined for a specific
        halo type.
        """
        invalid_properties = []
        full_property_list = property_table.PropertyTable.full_property_list
        valid_properties = [prop[0] for prop in full_property_list.values()]
        for key in self.parameters:
            # Skip keys which aren't halo types
            if 'properties' not in self.parameters[key]:
                continue
            for prop in self.parameters[key]['properties']:
                if prop not in valid_properties:
                    invalid_properties.append(prop)
        if len(invalid_properties):
            print('The following properties were found in the parameter file, but are invalid:')
            for prop in invalid_properties:
                print(f"  {prop}")

    def get_halo_type_variations(
        self, halo_type: str, default_variations: Dict
    ) -> Dict:
        """
        Get a dictionary of variations for the given halo type.

        Different variations are for example bound/unbound SubhaloProperties or
        aperture properties with different aperture sizes.

        If the given halo type is not found in the parameter file, or no
        variations are specified, the default variations are used.

        Parameters:
         - halo_type: str
           Halo type identifier in the parameter file, can be one of
           ApertureProperties, ProjectedApertureProperties, SOProperties
           or SubhaloProperties.
         - default_variations: Dict
           Dictionary with default variations that will be used if the
           halo type variations are not provided in the parameter file.

        Returns a dictionary from which different versions of the
        corresponding HaloProperty specialisation can be constructed.
        """
        if not halo_type in self.parameters:
            self.parameters[halo_type] = {}
        if not "variations" in self.parameters[halo_type]:
            self.parameters[halo_type]["variations"] = {}
            for variation in default_variations:
                self.parameters[halo_type]["variations"][variation] = dict(
                    default_variations[variation]
                )
        return dict(self.parameters[halo_type]["variations"])

    def get_particle_property(self, property_name: str) -> Tuple[str, str]:
        """
        Get the particle type and name in the snapshot of the given generic
        particle property name, taking into account aliases.

        An alias is useful if a dataset has a different name than expected
        internally. For example, in FLAMINGO the ElementMassFractions were
        only output in their smoothed form, so the following alias is
        required:
         PartType0/ElementMassFractions: PartType0/SmoothedElementMassFractions

        Parameters:
         - property_name: str
           (Full) path to a generic dataset in the snapshot.

        Returns a tuple with the path of the actual dataset in the snapshot,
        e.g. ("PartType4", "Masses").
        """
        if "aliases" in self.parameters:
            if property_name in self.parameters["aliases"]:
                property_name = self.parameters["aliases"][property_name]
        parts = property_name.split("/")
        if not len(parts) == 2:
            raise RuntimeError(
                f'Unable to parse particle property name "{property_name}"!'
            )
        return parts[0], parts[1]

    def get_aliases(self) -> Dict:
        """
        Get all the aliases defined in the parameter file.

        Returns the dictionary of aliases or an empty dictionary if no
        aliases were defined (there are no default aliases).
        """
        if "aliases" in self.parameters:
            return dict(self.parameters["aliases"])
        else:
            return dict()

    def get_filters(self, default_filters: Dict) -> Dict:
        """
        Get a dictionary with filters to use for SOAP.

        Parameters:
         - default_filter: Dict
           Dictionary with default filters, which are used
           if no filters are found in the parameter file or if a particular
           category is missing.

        Returns a dictionary with a threshold value for each category present,
        the properties to use for the filter, and how to combine the properties
        if multiple are listed.
        """
        filters = dict(default_filters)
        if "filters" in self.parameters:
            for category in default_filters:
                if category in self.parameters["filters"]:
                    filters[category] = self.parameters["filters"][category]
                else:
                    self.parameters["filters"][category] = filters[category]
        else:
            self.parameters["filters"] = dict(default_filters)
        return filters

    def get_defined_constants(self) -> Dict:
        """
        Get the dictionary with defined constants from the parameter file.

        Returns an empty dictionary if no defined constants are found in the
        parameter file (there are no default constants).
        """
        if "defined_constants" in self.parameters:
            return dict(self.parameters["defined_constants"])
        else:
            return dict()

    def recalculate_xrays(self) -> bool:
        """
        Returns a bool indicating if xray properties should be recomputed.
        Defaults to true.
        """
        return self.parameters.get("calculations", {}).get("recalculate_xrays", True)

    def get_xray_table_path(self) -> bool:
        """
        Returns the table to use for calculating xray properties.
        Defaults to "", which will cause code to crash.
        """
        return self.parameters.get("calculations", {}).get("xray_table_path", "")

    def calculate_missing_properties(self) -> bool:
        """
        Returns a bool indicating if properties missing from parameter file
        should be computed. Defaults to true.
        """
        calculations = self.parameters.get("calculations", {})
        return calculations.get("calculate_missing_properties", True)
