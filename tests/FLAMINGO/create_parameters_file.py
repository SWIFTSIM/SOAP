"""
Creates a parameter file used for testing SOAP on FLAMINGO.

Loads input/output file paths from yaml file passed as argument
Loads properties to compute from 'parameter_files/FLAMINGO.yml'
"""

import sys

import yaml

with open(sys.argv[1], 'r') as file:
    test_params = yaml.safe_load(file)

with open('parameter_files/FLAMINGO.yml', 'r') as file:
    run_params = yaml.safe_load(file)

for k in run_params:
    if k not in test_params:
        test_params[k] = run_params[k]

with open('tests/FLAMINGO/test_parameters.yml', 'w') as file:
    yaml.dump(test_params, file)
