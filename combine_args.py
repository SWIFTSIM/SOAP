#!/bin/env python

import yaml
import argparse

from virgo.util.partial_formatter import PartialFormatter


def combine_arguments(command_line_args, config_file):
    """
    Given an argparse parser and a .yml config file, construct a dict
    object with the combined set of parameters. Command line arguments take
    precedence over the config file.

    Entries in the Parameters section of the .yml file are then substituted
    into the other sections.
    """

    command_line_args = vars(command_line_args)
    
    # Read the config file
    with open(config_file, "r") as infile:
        config_file_args = yaml.safe_load(infile)

    # Combine the two
    all_args = {
        "Parameters" : {},
    }
    for name in config_file_args["Parameters"]:
        all_args["Parameters"][name] = config_file_args["Parameters"][name]
    for arg_name in command_line_args:
        name = arg_name.replace("-","_")
        value =  command_line_args[arg_name]
        if value is not None or name not in all_args["Parameters"]:
            all_args["Parameters"][name] = value

    pf = PartialFormatter()

    # We don't want to substitute snap_nr or file_nr here
    format_values = {}
    for name in all_args["Parameters"]:
        if name not in ("snap_nr", "file_nr"):
            format_values[name] = all_args["Parameters"][name]
    format_values["snap_nr"] = None
    format_values["file_nr"] = None
    # Add halo_finder. Currently this must be passed through config_file
    format_values["halo_finder"] = config_file_args['HaloFinder']['type']
            
    # Now copy any extra sections from the config file while substituting in
    # parameters from the Parameters section
    for section in config_file_args:
        if section != "Parameters":
            if section not in all_args:
                all_args[section] = {}
            for name in config_file_args[section]:
                if isinstance(config_file_args[section][name], str):
                    all_args[section][name] = pf.format(config_file_args[section][name], **format_values)
                elif not isinstance(config_file_args[section][name], dict):
                    all_args[section][name] = config_file_args[section][name]

    return all_args
    
