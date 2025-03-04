#!/bin/bash

# Modules for cosma
module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4

# Clone and compile code
git clone https://github.com/SWIFTSIM/HBTplus.git
cd HBTplus
cmake -B$PWD/build -D HBT_USE_OPENMP=ON -D HBT_DM_ONLY=ON -D HBT_UNSIGNED_LONG_ID_OUTPUT=OFF
cd build
make -j 4
cd ../..

# Run HBT
export OMP_NUM_THREADS=8
mpirun -np 1 ./HBTplus/build/HBT HBT_config.txt

