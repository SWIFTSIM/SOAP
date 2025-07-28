#!/bin/env python

"""
parameter_file.py

Support for parameter files.

The parameter file object keeps track of the parameters that are requested,
and can output this information in the form of a ".used_parameters" file,
similar to the file produced by SWIFT.
"""

from typing import Dict, Union, List, Tuple
import yaml

from SOAP import property_table


class ParameterFile:
    """
    Internal representation of the parameter file.

    Acts as a meaningful wrapper around the bare parameter dictionary.
    """

    # parameter dictionary
    parameters: Dict

    # Whether to record timings for calculations
    record_property_timings: bool = False

    def __init__(
        self,
        file_name: Union[None, str] = None,
        parameter_dictionary: Union[None, Dict] = None,
        snipshot: bool = False,
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

        self.snipshot = snipshot
        self.aliases = None

        self.property_filters = {}

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

    def get_property_filters(self, base_halo_type: str, full_list: List[str]) -> Dict:
        """
        Get a dictionary with the filter that should be applied to each
        property for the given halo type. If a property should be not be
        computed for this halo type then False is return. The dictionary
        keys are based on the contents of the given list of properties.

        Parameters:
         - base_halo_type: str
           Halo type identifier in the parameter file, can be one of
           ApertureProperties, ProjectedApertureProperties, SOProperties
           or SubhaloProperties.
         - full_list: List[str]
           List of all the properties that can be calculated by this
           particular halo type (as defined in the corresponding HaloProperty
           specialisation).

        Returns a dictionary where the keys are each property in full_list. The
        values are either False (if the property should not be calculated) or a
        string (the name of the filter to apply to the property).
        """
        # Save the filters as they are needed in combine chunks
        self.property_filters[base_halo_type] = self.property_filters.get(
            base_halo_type, {}
        )

        if not base_halo_type in self.parameters:
            self.parameters[base_halo_type] = {}
        # Handle the case where no properties are listed for the halo type
        if not "properties" in self.parameters[base_halo_type]:
            self.parameters[base_halo_type]["properties"] = {}
            for property in full_list:
                self.parameters[base_halo_type]["properties"][
                    property
                ] = self.calculate_missing_properties()
        filters = {}
        for property in full_list:
            # Check if property is listed in the parameter file for this base_halo_type
            if property in self.parameters[base_halo_type]["properties"]:
                filter_name = self.parameters[base_halo_type]["properties"][property]
                # filter_name will a dict if we want different behaviour
                # for snapshots/snipshots
                if isinstance(filter_name, dict):
                    if self.snipshot:
                        filter_name = filter_name["snipshot"]
                    else:
                        filter_name = filter_name["snapshot"]
                # if a filter is not specified in the snapshots
                # then we default to "basic"
                if filter_name == True:
                    filter_name = "basic"
                filters[property] = filter_name
            # Property is not listed in the parameter file for this base_halo_type
            else:
                if self.calculate_missing_properties():
                    filters[property] = "basic"
                    self.parameters[base_halo_type]["properties"][property] = "basic"
                    if self.unregistered_parameters is not None:
                        self.unregistered_parameters.add((base_halo_type, property))
                else:
                    filters[property] = False
            if isinstance(filters[property], str):
                assert (filters[property] in self.parameters.get("filters", {})) or (
                    filters[property] == "basic"
                ), f'Filter "{filters[property]}" is not defined in paramter file'
            else:
                assert filters[property] == False

            self.property_filters[base_halo_type][property] = filters[property]
        return filters

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
            for base_halo_type, property in self.unregistered_parameters:
                print(f"  {base_halo_type.ljust(30)}{property}")

    def print_invalid_properties(self, halo_prop_list) -> None:
        """
        Print a list of any properties in the parameter file that are not present in
        the property table. This doesn't check if the property is defined for a specific
        halo type.
        """
        invalid_properties = set()
        for key in self.parameters:
            # Skip keys which aren't halo types
            if "properties" not in self.parameters[key]:
                continue
            # Add all properties to the invalid list
            for prop in self.parameters[key]["properties"]:
                invalid_properties.add((key, prop))
            # Remove those which are valid for this particle halo type
            for halo_type in halo_prop_list:
                if key != halo_type.base_halo_type:
                    continue
                valid_properties = [
                    prop.name for prop in halo_type.property_list.values()
                ]
                for prop in self.parameters[key]["properties"]:
                    if prop in valid_properties:
                        invalid_properties.discard((key, prop))
        if len(invalid_properties):
            invalid_properties = sorted(invalid_properties, key=lambda x: (x[0], x[1]))
            print(
                "The following properties were found in the parameter file, but are invalid:"
            )
            for base_halo_type, prop in invalid_properties:
                print(f"  {base_halo_type}  {prop}")

    def get_halo_type_variations(
        self, base_halo_type: str, default_variations: Dict
    ) -> Dict:
        """
        Get a dictionary of variations for the given halo type.

        Different variations are for example bound/unbound SubhaloProperties or
        aperture properties with different aperture sizes.

        If the given halo type is not found in the parameter file, or no
        variations are specified, the default variations are used.

        Parameters:
         - base_halo_type: str
           Halo type identifier in the parameter file, can be one of
           ApertureProperties, ProjectedApertureProperties, SOProperties
           or SubhaloProperties.
         - default_variations: Dict
           Dictionary with default variations that will be used if the
           halo type variations are not provided in the parameter file.

        Returns a dictionary from which different versions of the
        corresponding HaloProperty specialisation can be constructed.
        """
        if not base_halo_type in self.parameters:
            self.parameters[base_halo_type] = {}
        if not "variations" in self.parameters[base_halo_type]:
            self.parameters[base_halo_type]["variations"] = {}
            for variation in default_variations:
                self.parameters[base_halo_type]["variations"][variation] = dict(
                    default_variations[variation]
                )
        return dict(self.parameters[base_halo_type]["variations"])

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
        aliases = self.get_aliases()
        if property_name in aliases:
            property_name = aliases[property_name]
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
        if self.aliases is None:
            if "aliases" in self.parameters:
                if "snipshot" in self.parameters["aliases"]:
                    if self.snipshot:
                        self.aliases = dict(self.parameters["aliases"]["snipshot"])
                    else:
                        aliases = dict(self.parameters["aliases"])
                        del aliases["snipshot"]
                        self.aliases = aliases
                else:
                    self.aliases = dict(self.parameters["aliases"])
            else:
                self.aliases = dict()
        return self.aliases

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

    def calculate_missing_properties(self) -> bool:
        """
        Returns a bool indicating if properties missing from parameter file
        should be computed. Defaults to true.
        """
        calculations = self.parameters.get("calculations", {})
        return calculations.get("calculate_missing_properties", True)

    def strict_halo_copy(self) -> bool:
        """
        Returns a bool indicating if approximate properties should be copied
        over from small ExclusiveSphere/ProjectedApertures. Defaults to false
        """
        calculations = self.parameters.get("calculations", {})
        return calculations.get("strict_halo_copy", False)

    def renclose_enabled(self) -> bool:
        """
        Returns a bool indicating if BoundSubhalo/EncloseRadius is enabled
        """
        return self.parameters["SubhaloProperties"]["properties"].get(
            "EncloseRadius", False
        )

    def get_cold_dense_params(self) -> Dict:
        """
        Returns a dict of the parameters required for initialising
        the ColdDenseGasFilter object
        """

        try:
            raw_params = self.parameters["calculations"]["cold_dense_gas_filter"]
            return {
                "maximum_temperature_K": float(raw_params["maximum_temperature_K"]),
                "minimum_hydrogen_number_density_cm3": float(
                    raw_params["minimum_hydrogen_number_density_cm3"]
                ),
                "initialised": True,
            }
        except KeyError as e:
            return {
                "maximum_temperature_K": 0,
                "minimum_hydrogen_number_density_cm3": 0,
                "initialised": False,
            }
