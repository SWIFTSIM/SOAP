#!/bin/bash
set -e

pytest -W error aperture_properties.py
pytest -W error half_mass_radius.py
pytest -W error projected_aperture_properties.py
pytest -W error SO_properties.py
pytest -W error subhalo_properties.py
pytest -W error test_SO_radius_calculation.py
rm test_SO_radius_*.png

mpirun -np 8 pytest -W error --with-mpi shared_mesh.py
mpirun -np 8 pytest -W error --with-mpi subhalo_rank.py
# Running without -W error since VirgoDC triggers a warning
mpirun -np 8 pytest --with-mpi read_vr.py
mpirun -np 8 pytest -W error --with-mpi io_test.py
rm io_test.png

# TODO: Add persistent data for these tests
#mpirun -np 8 pytest -W error --with-mpi read_subfind.py
#mpirun -np 8 pytest -W error --with-mpi read_rockstar.py
