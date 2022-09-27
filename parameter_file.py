#!/bin/env python

import yaml


class ParameterFile:
    def __init__(self, file_name=None):
        if file_name is not None:
            with open(file_name, "r") as handle:
                self.parameters = yaml.safe_load(handle)
        else:
            self.parameters = {}

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
