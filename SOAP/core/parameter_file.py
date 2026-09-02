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

# Known parameter file structure, used by check_schema to flag typos. A value
# of None means the keys directly under that section are user-named or free-form
# and are not checked; a set lists the only keys allowed directly under that
# section. Add a key here when a new option is introduced.
_ALLOWED_KEYS = {
    "Parameters": None,
    "Snapshots": {"filename", "fof_filename"},
    "HaloFinder": {
        "type",
        "filename",
        "fof_filename",
        "fof_radius_filename",
        "read_potential_energies",
    },
    "GroupMembership": {"filename"},
    "ExtraInput": None,
    "HaloProperties": {"filename", "chunk_dir"},
    "SubhaloProperties": {"properties"},
    "ApertureProperties": {"properties", "variations"},
    "ProjectedApertureProperties": {"properties", "variations"},
    "SOProperties": {"properties", "variations"},
    "aliases": None,
    "filters": None,
    "defined_constants": None,
    "calculations": {
        "calculate_missing_properties",
        "min_read_radius_cmpc",
        "strict_halo_copy",
        "reduced_snapshots",
        "recently_heated_gas_filter",
        "cold_dense_gas_filter",
        "separate_chunks",
    },
}


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

        # Warnings generated while resolving halo type variations, printed later
        # on a single rank via print_variation_warnings()
        self.variation_warnings = []

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

    def _validate_filter_name(self, filter_name, context: str) -> None:
        """
        Check that a filter referenced in the parameter file is one that SOAP
        can apply.

        The only always-available filter is "basic" (computed for every halo).
        Any other filter must be defined in the "filters" section of the
        parameter file; there are no default filters.

        Parameters:
         - filter_name:
           The filter as read from the parameter file.
         - context: str
           Human readable description of where the filter is used, included in
           the error message.

        Raises ValueError if the filter is not "basic" and is not defined in the
        "filters" section.
        """
        if filter_name == "basic":
            return
        if filter_name not in self.parameters.get("filters", {}):
            raise ValueError(
                f'{context} uses filter "{filter_name}" which is not defined in '
                f'the "filters" section of the parameter file'
            )

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
                self._validate_filter_name(
                    filters[property], f"{base_halo_type}/{property}"
                )
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
            # Skip halo types which have a variations key that is an empty dict.
            # SubhaloProperties never has a variations key, so is still checked.
            variations = self.parameters[key].get("variations", None)
            if isinstance(variations, dict) and len(variations) == 0:
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

    def has_enabled_properties(self, base_halo_type: str) -> bool:
        """
        Return True if the parameter file enables at least one property for the
        given halo type, taking the current snapshot/snipshot mode into account.
        """
        section = self.parameters.get(base_halo_type) or {}
        properties = section.get("properties") or {}
        for value in properties.values():
            # value may be a dict specifying different behaviour for
            # snapshots/snipshots
            if isinstance(value, dict):
                value = value["snipshot"] if self.snipshot else value["snapshot"]
            if value:
                return True
        return False

    def get_halo_type_variations(self, base_halo_type: str) -> Dict:
        """
        Get a dictionary of variations for the given halo type.

        Different variations are for example aperture properties with different
        aperture sizes, or spherical overdensities with different definitions.

        There are no default variations. A missing section, a missing or empty
        "variations", and a "variations: {}" are all treated identically. The
        behaviour depends on whether any properties are enabled for the halo
        type (see has_enabled_properties):

         - No variations, no properties enabled: nothing is computed, no message.
         - No variations, but properties enabled: nothing is computed, a warning
           is recorded.
         - Variations set, no properties enabled, calculate_missing_properties
           is False: nothing is computed, the variations are cleared and a
           warning is recorded.
         - Variations set, no properties enabled, calculate_missing_properties
           is True: all properties are computed for the variations.
         - Variations set and properties enabled: the variations are computed.

        Parameters:
         - base_halo_type: str
           Halo type identifier in the parameter file, can be one of
           ApertureProperties, ProjectedApertureProperties or SOProperties.

        Returns a dictionary from which different versions of the
        corresponding HaloProperty specialisation can be constructed.
        """
        section = self.parameters.get(base_halo_type)
        if not isinstance(section, dict):
            section = {}
            self.parameters[base_halo_type] = section
        if not isinstance(section.get("variations"), dict):
            section["variations"] = {}

        has_variations = len(section["variations"]) > 0
        has_properties = self.has_enabled_properties(base_halo_type)

        if not has_variations:
            if has_properties:
                self.variation_warnings.append(
                    f"{base_halo_type}: properties are enabled but no variations "
                    f"are set, so nothing will be computed for {base_halo_type}."
                )
            return {}

        if not has_properties and not self.calculate_missing_properties():
            self.variation_warnings.append(
                f"{base_halo_type}: variations are set but no properties are "
                f"enabled and calculate_missing_properties is False, so nothing "
                f"will be computed for {base_halo_type}."
            )
            section["variations"] = {}
            return {}

        # Check that any filters referenced by the variations are defined
        for name, variation in section["variations"].items():
            self._validate_filter_name(
                variation.get("filter", "basic"),
                f"{base_halo_type} variation '{name}'",
            )

        return dict(section["variations"])

    def print_variation_warnings(self) -> None:
        """
        Print any warnings recorded while resolving halo type variations, for
        example a halo type section that lists properties but no variations.
        """
        for warning in self.variation_warnings:
            print(warning)

    def check_schema(self) -> None:
        """
        Abort if the parameter file has an unrecognised section, or a mistyped
        key directly under a section which has a fixed set of keys (see
        _ALLOWED_KEYS). This catches typos which would otherwise be silently
        ignored. It does not check value types, or keys nested more deeply.
        """
        errors = []
        for section, block in self.parameters.items():
            if section not in _ALLOWED_KEYS:
                errors.append(f'unknown section "{section}"')
            elif _ALLOWED_KEYS[section] is not None and isinstance(block, dict):
                errors += [
                    f'unknown key "{section}/{key}"'
                    for key in block
                    if key not in _ALLOWED_KEYS[section]
                ]
        if errors:
            raise ValueError("Invalid parameter file: " + "; ".join(errors))

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

    def get_filters(self) -> Dict:
        """
        Get the category filters defined in the parameter file.

        Returns the contents of the "filters" section, or an empty dictionary if
        the section is absent. There are no default filters: any filter
        referenced by a property or a halo type variation must be defined in the
        parameter file. The only exception is the implicit "basic" filter, which
        is always computed and is not listed in the "filters" section.
        """
        return dict(self.parameters.get("filters", {}))

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
