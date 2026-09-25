#!/bin/bash
set -e

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

pytest -m "not mpi" -W error
rm test_SO_radius_*.png

mpirun -np 8 pytest -m mpi --with-mpi -W error

# Integration tests (full SOAP runs on the small volume)
bash tests/run_restart_test.sh
bash tests/run_chunk_count_test.sh


