#!/bin/env python

import yaml


class ParameterFile:
    def __init__(self, file_name=None, parameter_dictionary=None):
        if file_name is not None:
            with open(file_name, "r") as handle:
                self.parameters = yaml.safe_load(handle)
        else:
            if parameter_dictionary is not None:
                self.parameters = parameter_dictionary
            else:
                self.parameters = {}

    def get_parameters(self):
        return dict(self.parameters)

    def write_parameters(self, file_name="SOAP.used_parameters.yml"):
        with open(file_name, "w") as handle:
            yaml.safe_dump(self.parameters, handle)

    def get_property_mask(self, halo_type, full_list):
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
            else:
                mask[property] = True
                self.parameters[halo_type]["properties"][property] = True
        return mask

    def get_halo_type_variations(self, halo_type, default_variations={}):
        if not halo_type in self.parameters:
            self.parameters[halo_type] = {}
        if not "variations" in self.parameters[halo_type]:
            self.parameters[halo_type]["variations"] = {}
            for variation in default_variations:
                self.parameters[halo_type]["variations"][variation] = dict(
                    default_variations[variation]
                )
        return dict(self.parameters[halo_type]["variations"])

    def get_particle_property(self, property_name):
        if "aliases" in self.parameters:
            if property_name in self.parameters["aliases"]:
                property_name = self.parameters["aliases"][property_name]
        parts = property_name.split("/")
        if not len(parts) == 2:
            raise RuntimeError(
                f'Unable to parse particle property name "{property_name}"!'
            )
        return parts[0], parts[1]
