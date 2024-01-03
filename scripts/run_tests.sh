#!/bin/bash
set -e

python3 -W error -m pytest aperture_properties.py
python3 -W error -m pytest half_mass_radius.py
python3 -W error -m pytest projected_aperture_properties.py
python3 -W error -m pytest SO_properties.py
python3 -W error -m pytest subhalo_properties.py

# Tests that won't run with pytest
python3 test_SO_radius_calculation.py
mpirun -np 8 python3 ./shared_mesh.py
mpirun -np 8 python3 ./io_test.py
