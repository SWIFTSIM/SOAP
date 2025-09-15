"""
Creates a parameter file used for testing SOAP on COLIBRE

Loads input/output file paths from 'tests/COLIBRE/parameters.yml'
Loads properties to compute from 'parameter_files/COLIBRE.yml'
"""

import yaml

with open('tests/COLIBRE/parameters.yml', 'r') as file:
    test_params = yaml.safe_load(file)

with open('parameter_files/COLIBRE_THERMAL.yml', 'r') as file:
    run_params = yaml.safe_load(file)

for k in run_params:
    if k not in test_params:
        test_params[k] = run_params[k]

with open('tests/COLIBRE/test_parameters.yml', 'w') as file:
    yaml.dump(test_params, file)
