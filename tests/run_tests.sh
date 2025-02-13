#!/bin/bash
set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

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


